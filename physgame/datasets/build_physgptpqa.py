import argparse
import asyncio
import base64
import json
import os
import random
import re
from argparse import ArgumentParser
from asyncio import Queue, QueueEmpty, Task
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Literal, Optional, Set, Tuple, TypedDict, cast

import dotenv
import google.genai.types as genai_types
import loguru
import numpy as np
import tqdm
import transformers.image_transforms as image_transforms
import transformers.image_utils as image_utils
from numpy.typing import NDArray
from openai import AsyncClient, AsyncOpenAI
from openai.types.chat import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_content_part_image_param import ImageURL
from openai.types.responses import (
    FunctionToolParam,
    ResponseFunctionToolCall,
    ResponseInputImageParam,
    ResponseInputItemParam,
    ResponseInputMessageContentListParam,
    ResponseInputParam,
    ResponseInputTextParam,
    ToolChoiceFunctionParam,
)
from openai.types.responses.response_input_param import Message
from PIL.Image import Image
from pydantic import BaseModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from physgame.datasets.physdpo import PhysDPOEntry

logger = loguru.logger

PHYSDPO_ANNO_PATH = ".dev/datasets/PhysGame/PhysDPO-10k/PhysDPO_anno_10k.json"
PHYSDPO_VIDEO_DIR = ".dev/datasets/PhysGame/PhysDPO-10k/PhysDPO"


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


async def make_conversation(
    entry: PhysDPOEntry,
) -> ResponseInputParam:
    video_path = os.path.join(
        PHYSDPO_VIDEO_DIR,
        entry["video"],
    )

    video, _ = await asyncio.to_thread(image_utils.load_video, video_path, num_frames=8)
    video: NDArray[np.uint8]

    images: List[Image] = await asyncio.gather(
        *[
            asyncio.to_thread(image_transforms.to_pil_image, video[i])
            for i in range(video.shape[0])
        ]
    )

    def convert_pil_image_to_base64_url(image: Image) -> str:
        image_bytes = BytesIO()
        image.save(image_bytes, format="JPEG")
        base64_image = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_image}"

    image_urls: List[str] = await asyncio.gather(
        *[asyncio.to_thread(convert_pil_image_to_base64_url, image) for image in images]
    )

    conversation: List[ResponseInputItemParam] = [
        Message(
            role="system",
            content=[
                ResponseInputTextParam(
                    type="input_text",
                    text="""\
You are an expert multiple-choice question writer.
Your goal is to create two plausible but incorrect answer choices (“distractors”) for a physics-glitch question about the attached video.
Follow the guidelines and output format exactly.""",
                ),
            ],
        ),
        Message(
            role="user",
            content=cast(
                ResponseInputMessageContentListParam,
                [
                    *[
                        ResponseInputImageParam(
                            type="input_image",
                            detail="auto",
                            image_url=image_url,
                        )
                        for image_url in image_urls
                    ],
                    ResponseInputTextParam(
                        type="input_text",
                        text="""\
You will be provided with a video, a question, a correct answer, and a example of a distractor.
Generate TWO new distractors that meet ALL of these constraints:
- Plausible yet wrong: Each distractor must describe an event that COULD appear to violate physics, but does NOT actually occur in the video.
- Video-grounded: Refer only to objects or actions clearly visible in the video frames. Do not invent characters, props, or camera motions that never appear.
- Distinctiveness: The two distractors must focus on DIFFERENT physical principles or objects. They must also differ clearly from both the correct answer and example distractor.
- Comparable length: Stay within 10%% of the length of the example distrator. Avoid making any distractor significantly longer or shorter than the others.
- No clues: Never hint at which option is correct. Make sure no one can guess the right answer without watching the video.

"""
                        + f"The question is: {entry['prompt']}\n"
                        + f"The correct answer is: {entry['chosen']}\n"
                        + f"The example distractor is: {entry['rejected']}\n",
                    ),
                ],
            ),
        ),
    ]

    return conversation


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
        default=10,
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


local_random = random.Random(0)


def shuffle_qa_options(
    options: Dict[Literal["A", "B", "C", "D"], str],
    chosen_answer: Literal["A", "B", "C", "D"],
    rejected_answer: Literal["A", "B", "C", "D"],
) -> Tuple[
    Dict[Literal["A", "B", "C", "D"], str],
    Literal["A", "B", "C", "D"],
    Literal["A", "B", "C", "D"],
]:
    all_option_contents = list(options.values())
    chosen_answer_content = options[chosen_answer]
    rejected_answer_content = options[rejected_answer]

    local_random.shuffle(all_option_contents)

    options_ids: List[Literal["A", "B", "C", "D"]] = ["A", "B", "C", "D"]

    return (
        {
            options_ids[i]: all_option_contents[i]
            for i in range(len(options))
        },
        options_ids[all_option_contents.index(chosen_answer_content)],
        options_ids[all_option_contents.index(rejected_answer_content)],
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
    client = AsyncClient()

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
                conversation = await make_conversation(physdpo_entry)

                response = await client.responses.create(
                    model="gpt-4o-2024-11-20",
                    input=conversation,
                    tools=[
                        FunctionToolParam(
                            type="function",
                            name="report_generated_entry",
                            description="Report the generated entry.",
                            parameters={
                                "type": "object",
                                "properties": {
                                    "question": {
                                        "type": "string",
                                        "description": "The orginal question.",
                                    },
                                    "chosen_answer": {
                                        "type": "string",
                                        "description": "The original correct answer.",
                                    },
                                    "example_distractor": {
                                        "type": "string",
                                        "description": "The original example distractor.",
                                    },
                                    "distractors": {
                                        "type": "array",
                                        "description": "The generated distractors.",
                                        "items": {
                                            "type": "string",
                                            "description": "The distractors.",
                                        },
                                    },
                                },
                                "required": [
                                    "question",
                                    "chosen_answer",
                                    "example_distractor",
                                    "distractors",
                                ],
                                "additionalProperties": False,
                            },
                            strict=True,
                        ),
                    ],
                    tool_choice=ToolChoiceFunctionParam(
                        type="function",
                        name="report_generated_entry",
                    ),
                )

                model_output = json.loads(
                    cast(ResponseFunctionToolCall, response.output[0]).arguments
                )

                distractors: List[str] = model_output["distractors"]

                if len(distractors) != 2:
                    raise ValueError(f"Expected 2 distractors, got {len(distractors)}")

                options, chosen_answer, rejected_answer = shuffle_qa_options(
                    {
                        "A": distractors[0],
                        "B": distractors[1],
                        "C": physdpo_entry["chosen"],
                        "D": physdpo_entry["rejected"],
                    },
                    chosen_answer="C",
                    rejected_answer="D",
                )

                # Push the generated entry to the queue.
                physpqa_entry = PhysPQAEntry(
                    idx=idx,
                    video=physdpo_entry["video"],
                    question=physdpo_entry["prompt"],
                    options=options,
                    chosen_answer=chosen_answer,
                    rejected_answer=rejected_answer,
                )

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
