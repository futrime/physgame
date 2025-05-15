import argparse
import asyncio
import base64
import json
import os
import re
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Optional, TypedDict

import dotenv
import loguru
import numpy as np
import tqdm
import transformers.image_transforms as image_transforms
import transformers.image_utils as image_utils
from numpy.typing import NDArray
from openai import AsyncOpenAI
from openai.types.chat import (ChatCompletionContentPartImageParam,
                               ChatCompletionContentPartTextParam,
                               ChatCompletionMessageParam,
                               ChatCompletionSystemMessageParam,
                               ChatCompletionUserMessageParam)
from openai.types.chat.chat_completion_content_part_image_param import ImageURL
from PIL.Image import Image
from torch.utils.data import DataLoader

from physgame.datasets.physgame_benchmark import (PhysGameBenchmarkDataset,
                                                  PhysGameBenchmarkEntry)

logger = loguru.logger


@dataclass
class EvalArgs:
    api_key: Optional[str]
    base_url: Optional[str]
    model: str
    output_base_dir: str

    batch_size: int

    @property
    def eval_name(self) -> str:
        file_name = os.path.basename(__file__)
        file_name = file_name.replace(".py", "")
        return file_name

    @property
    def output_dir(self) -> str:
        model_name = os.path.basename(os.path.normpath(self.model))

        return os.path.join(
            self.output_base_dir,
            f"{model_name}",
            f"{self.eval_name}",
        )


def parse_args() -> EvalArgs:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--api-key",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--base-url",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--output-base-dir",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parsed_args, _ = parser.parse_known_args()

    return EvalArgs(
        api_key=parsed_args.api_key,
        base_url=parsed_args.base_url,
        model=parsed_args.model,
        batch_size=parsed_args.batch_size,
        output_base_dir=parsed_args.output_base_dir,
    )


async def main() -> None:
    dotenv.load_dotenv(override=True)

    eval_args = parse_args()

    logger.info(f"Running {eval_args.eval_name} evaluation with args: {eval_args}")
    logger.info(f"Results will be saved to {eval_args.output_dir}")

    if os.path.exists(os.path.join(eval_args.output_dir, "metrics.json")):
        logger.info(f"Evaluation already completed. Skipping...")

    os.makedirs(eval_args.output_dir, exist_ok=True)

    # Load data.
    dataset = load_data()

    dataloader = DataLoader(
        dataset,
        batch_size=eval_args.batch_size,
        shuffle=False,
        collate_fn=lambda x: x,
    )

    # Prepare client.
    client = AsyncOpenAI(
        api_key=eval_args.api_key,
        base_url=eval_args.base_url,
    )

    all_model_outputs: List[ModelOutput] = []

    async def generate_from_entry(entry: DatasetEntry) -> str:
        conversation = await make_conversation(entry)

        chat_completion = await client.chat.completions.create(
            messages=conversation,
            model=eval_args.model,
            temperature=0,
        )

        return str(chat_completion.choices[0].message.content)

    # Generate outputs.
    for batch in tqdm.tqdm(dataloader, desc="Generating outputs"):
        batch: List[DatasetEntry]

        text_outputs = await asyncio.gather(
            *[generate_from_entry(entry) for entry in batch],
        )

        all_model_outputs.extend(
            [
                make_model_output(entry, text_output)
                for entry, text_output in zip(batch, text_outputs)
            ],
        )

    # Save results.
    logger.info(f"Saving results to {eval_args.output_dir}")

    with open(
        os.path.join(eval_args.output_dir, "outputs.jsonl"), "w", encoding="utf-8"
    ) as f:
        for model_output in all_model_outputs:
            f.write(json.dumps(model_output) + "\n")

    metrics = check_answers(all_model_outputs)

    with open(
        os.path.join(eval_args.output_dir, "metrics.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"Metrics: {metrics}")


##### Per-Eval Code Begin #####


type DatasetEntry = PhysGameBenchmarkEntry


class ModelOutput(TypedDict):
    question_id: str
    answer: str
    output: str


def check_answers(model_outputs: List[ModelOutput]) -> Dict[str, float]:
    correct = 0

    for model_output in tqdm.tqdm(model_outputs, desc="Checking answers"):
        match = re.search(r"\(?([A-D])\)", model_output["output"])
        if match and match.group(1) == model_output["answer"]:
            correct += 1

    accuracy = correct / len(model_outputs)

    return {
        "accuracy": accuracy,
    }


def load_data() -> PhysGameBenchmarkDataset:
    return PhysGameBenchmarkDataset()


async def make_conversation(
    entry: PhysGameBenchmarkEntry,
) -> List[ChatCompletionMessageParam]:
    video, _ = await asyncio.to_thread(
        image_utils.load_video, entry["video"], num_frames=16
    )
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

    conversation = [
        ChatCompletionSystemMessageParam(
            role="system",
            content=[
                ChatCompletionContentPartTextParam(
                    type="text",
                    text="Watch the video carefully and analyze the events and object movements, "
                    + "focusing on any inconsistencies with physical laws. "
                    + "Identify and highlight instances where the behavior deviates from expected real-world physics, "
                    + "and select the most accurate option to describe the detected glitch.",
                )
            ],
        ),
        ChatCompletionUserMessageParam(
            role="user",
            content=[
                *[
                    ChatCompletionContentPartImageParam(
                        type="image_url",
                        image_url=ImageURL(
                            url=await asyncio.to_thread(
                                convert_pil_image_to_base64_url, image
                            ),
                        ),
                    )
                    for image in images
                ],
                ChatCompletionContentPartTextParam(
                    type="text",
                    text=entry["question"]
                    + "\n"
                    + "\n".join(
                        [f"({key}) {value}" for key, value in entry["options"].items()]
                    )
                    + "\nOnly give the best option enclosed in parentheses, i.e. (A), (B), (C), or (D). "
                    + "You must always give an option, even if you are not sure.",
                ),
            ],
        ),
    ]

    return conversation


def make_model_output(
    entry: PhysGameBenchmarkEntry, decoded_output: str
) -> ModelOutput:
    return {
        "question_id": entry["question_id"],
        "answer": entry["answer"],
        "output": decoded_output,
    }


##### Per-Eval Code End #####


if __name__ == "__main__":
    asyncio.run(main())
