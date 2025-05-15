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

from physgame.datasets.physdpo import PhysDPOEntry

logger = loguru.logger

PHYSDPO_ANNO_PATH = "/lustre/fs12/portfolios/nvr/users/zijzhang/datasets/PhysGame/PhysDPO-10k/PhysDPO_anno_10k.json"
PHYSDPO_VIDEO_DIR = (
    "/lustre/fs12/portfolios/nvr/users/zijzhang/datasets/PhysGame/PhysDPO-10k/PhysDPO"
)


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


class PhysPQAEntry(TypedDict):
    idx: int
    video: str
    question: str
    options: Dict[Literal["A", "B", "C", "D"], str]
    chosen_answer: str
    rejected_answer: str


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
        f"physpqa_anno.jsonl",
    )
    os.makedirs(dataset_args.output_dir, exist_ok=True)

    logger.info(f"Generated entries will be saved to {output_file_path}")

    # Prepare Gemini API client.
    client = Client()

    with open(PHYSDPO_ANNO_PATH, "r", encoding="utf-8") as f:
        physdpo_anno: List[PhysDPOEntry] = json.load(f)

    physdpo_anno = physdpo_anno[: dataset_args.num_entries]

    if os.path.exists(output_file_path):
        existing_entry_idxs: Set[int] = set()

        with open(output_file_path, "r") as f:
            for line in f:
                idx = json.loads(line)["idx"]
                existing_entry_idxs.add(idx)

        physdpo_entries: List[Tuple[int, PhysDPOEntry]] = [
            (i, entry)
            for i, entry in enumerate(physdpo_anno)
            if i not in existing_entry_idxs
        ]

    else:
        physdpo_entries: List[Tuple[int, PhysDPOEntry]] = [
            (i, entry) for i, entry in enumerate(physdpo_anno)
        ]

    logger.info(f"#entries to generate: {len(physdpo_entries)}.")
    logger.info(f"#entries to skip: {len(physdpo_anno) - len(physdpo_entries)}.")

    # Generate QA entries.
    class ModelOutputItem(BaseModel):
        question: str
        A: str
        B: str
        C: str
        D: str
        chosen_answer: str
        rejected_answer: str

    physdpo_entry_queue: Queue[Tuple[int, PhysDPOEntry]] = Queue(
        maxsize=dataset_args.num_workers * 2
    )
    physpqa_entry_queue: Queue[PhysPQAEntry] = Queue()

    stopped = False

    async def gen_worker() -> None:
        while not stopped:
            await asyncio.sleep(0.1)  # To avoid busy waiting.

            try:
                idx, physdpo_entry = physdpo_entry_queue.get_nowait()
            except QueueEmpty:
                await asyncio.sleep(0.1)
                continue

            physdpo_entry_queue.task_done()

            video_path = os.path.join(
                PHYSDPO_VIDEO_DIR,
                physdpo_entry["video"],
            )

            try:
                with open(video_path, "rb") as f:
                    video_bytes = await asyncio.to_thread(f.read)

                # Skip if exceeding Gemini's limit (20MB) and preserve a safe redundancy.
                if len(video_bytes) > 19 * 1024 * 1024:
                    logger.warning(
                        f"Video of PhysDPO entry {idx} is too large ({len(video_bytes) / 1024 / 1024:.2f}MB > 16MB). Skipping."
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
                        Based on the video, create a four-option multiple-choice question about the physics glitch shown.
                        
                        Important requirements:
                        - Use the original question text without modification.
                        - Include the "chosen answer" as the correct option (you may refine its language without changing meaning).
                        - Include the "rejected answer" as one of the distractors (you may refine its language).
                        - Create TWO additional distractor options following these principles:
                          - Distractors should relate to actual objects and actions seen in the video
                          - Distractors should describe plausible but incorrect physics glitches
                          - All four options should be of similar length to avoid bias
                          - Each option should be distinct and focus on different aspects of potential physics violations
                        
                        Output your response as a JSON object with the following format:
                        {
                        "question": "[original_question]",
                        "A": "[an_option]",
                        "B": "[an_option]",
                        "C": "[an_option]",
                        "D": "[an_option]",
                        "chosen_answer": "[option_of_chosen_answer (A, B, C, or D)]",
                        "rejected_answer": "[option_of_rejected_answer (A, B, C, or D)]"
                        }
                        
                        Distribute the options (A, B, C, or D) randomly to ensure no positional bias.
                        """
                        + f"""
                        Question: {physdpo_entry["prompt"]}
                        Chosen Answer: {physdpo_entry["chosen"]}
                        Rejected Answer: {physdpo_entry["rejected"]}
                        """,
                    ],
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": list[ModelOutputItem],
                    },
                )

                model_outputs = cast(List[ModelOutputItem], response.parsed)

                physpqa_entry = PhysPQAEntry(
                    idx=idx,
                    video=video_path,
                    question=model_outputs[0].question,
                    options={
                        "A": model_outputs[0].A,
                        "B": model_outputs[0].B,
                        "C": model_outputs[0].C,
                        "D": model_outputs[0].D,
                    },
                    chosen_answer=model_outputs[0].chosen_answer,
                    rejected_answer=model_outputs[0].rejected_answer,
                )

                # Save the entry to the queue.
                await physpqa_entry_queue.put(physpqa_entry)

            except Exception as e:
                logger.error(f"Error processing entry {idx} (video: {video_path}): {e}")

    async def save_output_entries() -> None:
        while not physpqa_entry_queue.empty():
            await asyncio.sleep(0)
            pqa_entry = physpqa_entry_queue.get_nowait()
            with open(output_file_path, "a") as f:
                json.dump(pqa_entry, f)
                f.write("\n")

    tasks: List[Task[None]] = []
    for _ in range(dataset_args.num_workers):
        task = asyncio.create_task(gen_worker())
        tasks.append(task)

    for idx, entry in tqdm(physdpo_entries):
        await physdpo_entry_queue.put((idx, entry))

        if physdpo_entry_queue.qsize() >= dataset_args.save_interval:
            await save_output_entries()

    logger.info("Waiting for all entries to be processed...")
    await physdpo_entry_queue.join()

    logger.info("All entries processed. Stopping workers...")
    stopped = True
    await asyncio.gather(*tasks)

    # Save any remaining entries in the queue.
    await save_output_entries()

    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
