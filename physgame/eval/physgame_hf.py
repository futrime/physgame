import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict, cast

import loguru
import numpy as np
import torch
import tqdm
import transformers.image_transforms as image_transforms
import transformers.image_utils as image_utils
from accelerate import Accelerator
from numpy.typing import NDArray
from PIL.Image import Image
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.configuration_utils import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature
from transformers.generation.utils import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import (
    AutoModel,
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
)
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.processing_utils import ProcessorMixin
from transformers.utils.generic import PaddingStrategy

from physgame.datasets.physgame_benchmark import (
    PhysGameBenchmarkDataset,
    PhysGameBenchmarkEntry,
)

logger = loguru.logger


@dataclass
class EvalArgs:
    model: str
    output_base_dir: str

    batch_size: int
    num_frames: Optional[int]
    video_fps: Optional[int]

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


def load_model(model_name_or_path: str) -> PreTrainedModel:
    AUTO_CLASSES = [
        AutoModelForImageTextToText,
        AutoModelForVision2Seq,
        AutoModel,
    ]

    errors: List[Exception] = []

    for model_class in AUTO_CLASSES:
        try:
            model = model_class.from_pretrained(
                model_name_or_path,
                attn_implementation="flash_attention_2",
                device_map=f"cuda:{torch.cuda.current_device()}", # Should not be "auto".
                torch_dtype="bfloat16",
                trust_remote_code=True,
            )

            if not isinstance(model, GenerationMixin):
                continue

            return cast(PreTrainedModel, model)

        except Exception as e:
            errors.append(e)

    raise ExceptionGroup(f"Failed to load model {model_name_or_path}", errors)


def load_processor(model_name_or_path: str) -> ProcessorMixin:
    # First try to load the tokenizer from the model name or path.
    try:
        return AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            use_fast=True,
        )
    except:
        pass

    # Then try to load the tokenizer from the model config.
    model_config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    assert isinstance(model_config, PretrainedConfig)

    return AutoProcessor.from_pretrained(
        model_config._name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )


def parse_args() -> EvalArgs:
    parser = argparse.ArgumentParser()

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
    parser.add_argument(
        "--num-frames",
        type=int,
    )
    parser.add_argument(
        "--video-fps",
        type=int,
    )

    parsed_args, _ = parser.parse_known_args()

    eval_args = EvalArgs(
        model=parsed_args.model,
        output_base_dir=parsed_args.output_base_dir,
        batch_size=parsed_args.batch_size,
        num_frames=parsed_args.num_frames,
        video_fps=parsed_args.video_fps,
    )

    if eval_args.num_frames is None and eval_args.video_fps is None:
        # If both num_frames and video_fps are None, use the default of 8.
        eval_args.num_frames = 8

    return eval_args


def main() -> None:
    eval_args = parse_args()

    accelerator = Accelerator()

    if accelerator.is_main_process:
        logger.info(f"Running {eval_args.eval_name} evaluation with args: {eval_args}")
        logger.info(f"Results will be saved to {eval_args.output_dir}")

    if os.path.exists(os.path.join(eval_args.output_dir, "metrics.json")):
        if accelerator.is_main_process:
            logger.info(f"Evaluation already completed. Skipping...")
        return

    os.makedirs(eval_args.output_dir, exist_ok=True)

    # Load data.
    dataset = load_data()

    dataloader = DataLoader(
        dataset,
        batch_size=eval_args.batch_size,
        shuffle=False,
        collate_fn=lambda x: x,
    )
    dataloader = cast(DataLoader, accelerator.prepare(dataloader))

    # Prepare model.
    processor = load_processor(eval_args.model)

    model = load_model(eval_args.model)

    all_model_outputs: List[ModelOutput] = []

    # Generate outputs.
    for batch in tqdm(
        dataloader, desc="Generating outputs", disable=not accelerator.is_main_process
    ):
        batch: List[DatasetEntry]

        conversations = [
            make_conversation(
                entry, num_frames=eval_args.num_frames, video_fps=eval_args.video_fps
            )
            for entry in batch
        ]

        inputs = processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            padding=PaddingStrategy.LONGEST,
            padding_side="left",
            return_dict=True,
            return_tensors="pt",
            tokenize=True,
        )
        assert isinstance(inputs, BatchFeature)
        inputs = inputs.to(
            device=model.device,
            dtype=model.dtype,
        )

        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=16,
            temperature=None,
            top_p=None,
            top_k=None,
        )
        assert isinstance(outputs, Tensor)

        decoded_outputs: List[str] = processor.post_process_image_text_to_text(
            outputs[
                :, cast(Tensor, inputs.input_ids).shape[1] :
            ],  # Only decode the generated tokens.
        )
        assert len(decoded_outputs) == len(batch)

        all_model_outputs.extend(
            [
                make_model_output(entry, decoded_output)
                for entry, decoded_output in zip(batch, decoded_outputs)
            ],
        )

    # Gather outputs.
    all_model_outputs = cast(
        List[ModelOutput],
        accelerator.gather_for_metrics(all_model_outputs, use_gather_object=True),
    )

    accelerator.end_training()

    if not accelerator.is_main_process:
        # Only the main process should save the results.
        return

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

DATASET_DIR = ".dev/PhysGame/PhysGame-Benchmark"


type DatasetEntry = PhysGameBenchmarkEntry


class ModelOutput(TypedDict):
    question_id: str
    answer: str
    output: str


def check_answers(model_outputs: List[ModelOutput]) -> Dict[str, float]:
    correct = 0

    for model_output in tqdm(model_outputs, desc="Checking answers"):
        match = re.search(r"\(?([A-D])\)", model_output["output"])
        if match and match.group(1) == model_output["answer"]:
            correct += 1

    accuracy = correct / len(model_outputs)

    return {
        "accuracy": accuracy,
    }


def load_data() -> PhysGameBenchmarkDataset:
    return PhysGameBenchmarkDataset(DATASET_DIR)


def make_conversation(
    entry: PhysGameBenchmarkEntry,
    *,
    num_frames: Optional[int],
    video_fps: Optional[int],
) -> List[Dict[str, Any]]:
    video_path = os.path.join(
        DATASET_DIR, "PhysGame-Benchmark", entry["question_id"] + ".mp4"
    )

    video, _ = image_utils.load_video(video_path, num_frames=num_frames, fps=video_fps)
    video: NDArray[np.uint8]
    assert video.shape[0] == num_frames

    images: List[Image] = [
        image_transforms.to_pil_image(video[i]) for i in range(video.shape[0])
    ]

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Watch the video carefully and analyze the events and object movements, "
                    + "focusing on any inconsistencies with physical laws. "
                    + "Identify and highlight instances where the behavior deviates from expected real-world physics, "
                    + "and select the most accurate option to describe the detected glitch.\n",
                },
            ],
        },
        {
            "role": "user",
            "content": [
                *[
                    {
                        "type": "image",
                        "image": image,
                    }
                    for image in images
                ],
                {
                    "type": "text",
                    "text": entry["question"]
                    + "\n"
                    + "\n".join(
                        [f"({key}) {value}" for key, value in entry["options"].items()]
                    )
                    + "\nOnly give the best option enclosed in parentheses, i.e. (A), (B), (C), or (D). "
                    + "You must always give an option, even if you are not sure.",
                },
            ],
        },
    ]


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
    main()
