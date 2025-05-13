import asyncio
import json
import os
from argparse import ArgumentParser
from asyncio import Queue, QueueEmpty, Task
from dataclasses import dataclass
from typing import Dict, List, Literal, Set, Tuple, TypedDict, cast

import dotenv
import google.genai.types as genai_types
import loguru
from google.genai.client import Client
from pydantic import BaseModel
from tqdm import tqdm

logger = loguru.logger

PI_VIDEO_DIR = "/lustre/fs12/portfolios/nvr/users/zijzhang/datasets/PhysGame/PhysInstruct-40k/PhysInstruct"
PI_ANNO_PATH = "/lustre/fs12/portfolios/nvr/users/zijzhang/datasets/PhysGame/PhysInstruct-40k/PhysInstruct_anno_40k.json"


@dataclass
class DatasetArgs:
    output_base_dir: str

    num_entries: int
    num_workers: int
    save_interval: int

    @property
    def output_dir(self) -> str:
        return os.path.join(
            self.output_base_dir,
            self.dataset_name,
        )

    @property
    def dataset_name(self) -> str:
        file_name = os.path.basename(__file__)
        file_name = file_name.replace(".py", "").replace("build_", "")
        return file_name


def parse_args() -> DatasetArgs:
    parser = ArgumentParser()
    parser.add_argument(
        "--output-base-dir",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--num-entries",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
    )

    args, _ = parser.parse_known_args()

    return DatasetArgs(
        output_base_dir=args.output_base_dir,
        num_entries=args.num_entries,
        num_workers=args.num_workers,
        save_interval=args.save_interval,
    )


async def main() -> None:
    dotenv.load_dotenv()

    dataset_args = parse_args()

    logger.info(f"Args: {dataset_args}")

    output_file_path = os.path.join(
        dataset_args.output_dir,
        f"physqa_anno.jsonl",
    )
    os.makedirs(dataset_args.output_dir, exist_ok=True)

    logger.info(f"Generated entries will be saved to {output_file_path}")

    # Prepare Gemini API client.
    client = Client()

    # Prepare seed dataset.
    class PhysInstructRawEntry(TypedDict):
        video: str
        QA: List[Dict[Literal["i", "q", "a"], str]]

    with open(PI_ANNO_PATH, "r", encoding="utf-8") as f:
        pi_anno: List[PhysInstructRawEntry] = json.load(f)

    pi_anno = pi_anno[: dataset_args.num_entries]

    if os.path.exists(output_file_path):
        existing_entry_idxs: Set[int] = set()

        with open(output_file_path, "r") as f:
            for line in f:
                idx = json.loads(line)["idx"]
                existing_entry_idxs.add(idx)

        pi_entries: List[Tuple[int, PhysInstructRawEntry]] = [
            (i, entry)
            for i, entry in enumerate(pi_anno)
            if i not in existing_entry_idxs
        ]

    else:
        pi_entries: List[Tuple[int, PhysInstructRawEntry]] = [
            (i, entry) for i, entry in enumerate(pi_anno)
        ]

    logger.info(f"#entries to generate: {len(pi_entries)}.")
    logger.info(f"#entries to skip: {len(pi_anno) - len(pi_entries)}.")

    # Generate QA entries.
    class ModelOutputItem(BaseModel):
        question: str
        A: str
        B: str
        C: str
        D: str
        answer: str

    class PhysQAEntry(TypedDict):
        idx: int
        video: str
        QA: List[Dict[Literal["question", "A", "B", "C", "D", "answer"], str]]

    pi_entry_queue: Queue[Tuple[int, PhysInstructRawEntry]] = Queue(
        maxsize=dataset_args.num_workers * 2
    )
    pqa_entry_queue: Queue[PhysQAEntry] = Queue()

    stopped = False

    async def gen_worker() -> None:
        while not stopped:
            await asyncio.sleep(0.1) # To avoid busy waiting.

            try:
                idx, pi_entry = pi_entry_queue.get_nowait()
            except QueueEmpty:
                await asyncio.sleep(0.1)
                continue

            pi_entry_queue.task_done()

            video_path = os.path.join(
                PI_VIDEO_DIR,
                pi_entry["video"],
            )

            try:
                with open(video_path, "rb") as f:
                    video_bytes = await asyncio.to_thread(f.read)

                # Skip if exceeding Gemini's limit (314572800 bytes) and preserve a safe redundancy.
                if len(video_bytes) > 16 * 1024 * 1024:
                    logger.warning(
                        f"Video of PhysInstruct entry {idx} is too large ({len(video_bytes) / 1024 / 1024:.2f}MB > 16MB). Skipping."
                    )
                    continue

                response = await client.aio.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[
                        genai_types.Part.from_bytes(
                            data=video_bytes,
                            mime_type="video/mp4",
                        ),
                        """
                        Based on the video and the physics glitch described in the ground truth answers, create a list of four-option multiple-choice questions. 
                        
                        Important requirements:
                        - DO NOT modify the question text from the original question.
                        - Use the provided answer as the correct option, though you may refine its language without changing its meaning.
                        - Create three plausible distractor options following these principles:
                        - Distractors should relate to actual objects and actions seen in the video
                        - Distractors should describe plausible but incorrect physics glitches
                        - All four options (A, B, C, D) should be of similar length to avoid bias
                        - Each option should be distinct and focus on different aspects of potential physics violations
                        
                        For the correct answer, incorporate all physics glitches visible in the video. Ensure your options are specific to the video content and require understanding the visual information to select correctly.
                        
                        Output your response as a list of JSON objects with the following format:
                        {
                        "question": "[original_question]",
                        "A": "[an_option]",
                        "B": "[an_option]",
                        "C": "[an_option]",
                        "D": "[an_option]",
                        "answer": "[correct_option (A, B, C, or D)]"
                        }
                        
                        Distribute the correct answer position (A, B, C, or D) randomly to ensure an equal distribution across questions.
                        """
                        + "\n\n".join(
                            [
                                f"""
                        <qa_pair>
                        Question: {entry['QA'][i]['q']}
                        Ground Truth Answer: {entry['QA'][i]['a']}
                        </qa_pair>
                        """
                                for i in range(len(entry["QA"]))
                            ]
                        ),
                    ],
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": list[ModelOutputItem],
                    },
                )

                model_outputs = cast(List[ModelOutputItem], response.parsed)

                physqa_entry = PhysQAEntry(
                    idx=idx,
                    video=entry["video"],
                    QA=[
                        {
                            "question": item.question,
                            "A": item.A,
                            "B": item.B,
                            "C": item.C,
                            "D": item.D,
                            "answer": item.answer,
                        }
                        for item in model_outputs
                    ],
                )

                # Save the entry to the queue.
                await pqa_entry_queue.put(physqa_entry)

            except Exception as e:
                logger.error(f"Error processing entry {idx} (video: {video_path}): {e}")

    async def save_pqa_entries() -> None:
        while not pqa_entry_queue.empty():
            await asyncio.sleep(0)
            pqa_entry = pqa_entry_queue.get_nowait()
            with open(output_file_path, "a") as f:
                json.dump(pqa_entry, f)
                f.write("\n")

    tasks: List[Task[None]] = []
    for _ in range(dataset_args.num_workers):
        task = asyncio.create_task(gen_worker())
        tasks.append(task)

    for idx, entry in tqdm(pi_entries):
        await pi_entry_queue.put((idx, entry))

        if pi_entry_queue.qsize() >= dataset_args.save_interval:
            await save_pqa_entries()

    logger.info("Waiting for all entries to be processed...")
    await pi_entry_queue.join()

    logger.info("All entries processed. Stopping workers...")
    stopped = True
    await asyncio.gather(*tasks)

    # Save any remaining entries in the queue.
    await save_pqa_entries()

    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
