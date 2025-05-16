import datetime
import os
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, TypedDict, cast

import dotenv
import loguru
import torch
import transformers.trainer_utils as trainer_utils
import wandb
from accelerate import Accelerator
from datasets import Dataset
from transformers.feature_extraction_utils import BatchFeature
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import AutoModelForImageTextToText
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from trl import SFTConfig, SFTTrainer

from physgame.datasets.physinstruct import PhysInstructDataset
from physgame.datasets.physqa import PhysQADataset

logger = loguru.logger


@dataclass
class TrainArgs:
    model: str
    output_base_dir: str

    max_dataset_size: Optional[int]

    @property
    def model_name(self) -> str:
        model_name = os.path.basename(os.path.normpath(self.model))
        return model_name

    @property
    def output_dir(self) -> str:
        return os.path.join(
            self.output_base_dir,
            self.model_name,
            self.train_name,
        )

    @property
    def train_name(self) -> str:
        file_name = os.path.basename(__file__)
        file_name = file_name.replace(".py", "")
        return file_name


class PromptCompletionEntry(TypedDict):
    prompt: List[Dict[str, Any]]
    completion: List[Dict[str, Any]]


def prepare_dataset(max_dataset_size: Optional[int]) -> Dataset:
    physinstruct = PhysInstructDataset()
    physqa = PhysQADataset()

    if max_dataset_size is None or max_dataset_size >= len(physinstruct) + len(physqa):
        physinstruct_sample_size = len(physinstruct)
        physqa_sample_size = len(physqa)
    else:
        physinstruct_sample_size = int(
            len(physinstruct) * max_dataset_size / (len(physinstruct) + len(physqa))
        )
        physqa_sample_size = max_dataset_size - physinstruct_sample_size

    def gen() -> Generator[PromptCompletionEntry, None, None]:
        for idx in range(physinstruct_sample_size):
            entry = physinstruct[idx]

            prompt_completion = PromptCompletionEntry(
                prompt=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "path": entry["video_path"],
                            },
                            {
                                "type": "text",
                                "text": entry["question"],
                            },
                        ],
                    },
                ],
                completion=[
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": entry["answer"],
                            }
                        ],
                    }
                ],
            )

            yield prompt_completion

        for idx in range(physqa_sample_size):
            entry = physqa[idx]

            prompt_completion = PromptCompletionEntry(
                prompt=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "path": entry["video_path"],
                            },
                            {
                                "type": "text",
                                "text": "Watch the video carefully and analyze the events and object movements, "
                                + "focusing on any inconsistencies with physical laws. "
                                + "Identify and highlight instances where the behavior deviates from expected real-world physics, "
                                + "and select the most accurate option to describe the detected glitch.\n"
                                + entry["question"],
                            },
                        ],
                    },
                ],
                completion=[
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": entry["answer"],
                            }
                        ],
                    }
                ],
            )

            yield prompt_completion

    dataset = Dataset.from_generator(gen)
    assert isinstance(dataset, Dataset)

    return dataset


def parse_args() -> TrainArgs:
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output-base-dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--max-dataset-size",
        type=int,
    )

    args, _ = parser.parse_known_args()

    return TrainArgs(
        model=args.model,
        output_base_dir=args.output_base_dir,
        max_dataset_size=args.max_dataset_size,
    )


def main() -> None:
    dotenv.load_dotenv()

    accelerator = Accelerator()

    train_args = parse_args()

    if accelerator.is_main_process:
        logger.info(
            f"Running {train_args.train_name} evaluation with args: {train_args}"
        )
        logger.info(f"Results will be saved to {train_args.output_dir}")

    model_output_dir = os.path.join(train_args.output_dir, "model")

    os.makedirs(model_output_dir, exist_ok=True)

    trainer_config = SFTConfig(
        bf16=True,
        max_grad_norm=5.0,
        dataset_kwargs={
            "skip_prepare_dataset": True,
        },
        do_train=True,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=2e-5,
        logging_steps=1,
        lr_scheduler_type="cosine",
        num_train_epochs=1,
        optim="adamw_torch",
        output_dir=model_output_dir,
        per_device_train_batch_size=3,
        remove_unused_columns=False,
        report_to="wandb",
        run_name=f"{train_args.model_name}-{train_args.train_name}-{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        save_steps=100,
        save_strategy="steps",
        save_total_limit=2,
        # torch_empty_cache_steps=1,
        warmup_ratio=0.03,
        weight_decay=0.0,
    )

    if accelerator.is_main_process:
        wandb.init(
            dir=os.path.join(train_args.output_dir, "wandb"),
            name=trainer_config.run_name,
            group=f"{train_args.model_name}-{train_args.train_name}",
        )

    # 1. Load dataset.

    dataset = prepare_dataset(train_args.max_dataset_size)

    # 2. Load model.

    processor = AutoProcessor.from_pretrained(
        train_args.model,
        trust_remote_code=True,
        use_fast=True,
    )
    assert isinstance(processor, ProcessorMixin)

    tokenizer = AutoTokenizer.from_pretrained(
        train_args.model,
        trust_remote_code=True,
        use_fast=True,
    )
    assert isinstance(tokenizer, PreTrainedTokenizerBase)

    model = AutoModelForImageTextToText.from_pretrained(
        train_args.model,
        attn_implementation="flash_attention_2",
        torch_dtype="bfloat16",
        trust_remote_code=True,
    )
    assert isinstance(model, PreTrainedModel)

    if isinstance(model, Qwen2_5_VLForConditionalGeneration):
        model.visual.requires_grad_(False)

    model.train()

    # 3. Fine-tune model.

    def collate_fn(entries: List[PromptCompletionEntry]) -> BatchFeature:
        prompt_inputs = processor.apply_chat_template(
            [entry["prompt"] for entry in entries],
            num_frames=8,
            do_resize=True,
            size={
                "longest_edge": 1280 * 720,
                "shortest_edge": 0,
            },
            padding="longest",
            padding_side="right",
            return_dict=True,
            return_tensors="pt",
            tokenize=True,
            video_load_backend="opencv",
        )
        assert isinstance(prompt_inputs, BatchFeature)

        completion_inputs = processor.apply_chat_template(
            [entry["completion"] for entry in entries],
            padding="longest",
            padding_side="right",
            return_dict=True,
            return_tensors="pt",
            tokenize=True,
        )
        assert isinstance(completion_inputs, BatchFeature)

        prompt_valid_lens = prompt_inputs.attention_mask.sum(dim=1)
        completion_valid_lens = completion_inputs.attention_mask.sum(dim=1)

        batch_shape = (
            len(entries),
            (prompt_valid_lens + completion_valid_lens).max(),
        )

        input_ids = torch.full(
            batch_shape,
            fill_value=cast(int, tokenizer.pad_token_id),
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            batch_shape,
            dtype=torch.long,
        )
        labels = torch.full(
            batch_shape,
            fill_value=-100,
            dtype=torch.long,
        )

        for i in range(len(entries)):
            input_ids[i, -(prompt_valid_lens[i] + completion_valid_lens[i]) :] = (
                torch.cat(
                    (
                        prompt_inputs.input_ids[i, : prompt_valid_lens[i]],
                        completion_inputs.input_ids[i, : completion_valid_lens[i]],
                    )
                )
            )

            attention_mask[i, -(prompt_valid_lens[i] + completion_valid_lens[i]) :] = 1

            labels[
                i,
                -completion_valid_lens[i] :,
            ] = completion_inputs.input_ids[i, : completion_valid_lens[i]]

        inputs = prompt_inputs
        inputs["input_ids"] = input_ids
        inputs["attention_mask"] = attention_mask
        inputs["labels"] = labels

        return inputs

    trainer = SFTTrainer(
        model=model,
        args=trainer_config,
        data_collator=collate_fn,
        processing_class=tokenizer,
        train_dataset=dataset,
    )

    trainer.train(
        resume_from_checkpoint=trainer_utils.get_last_checkpoint(model_output_dir)
    )

    if accelerator.is_main_process:
        final_save_path = os.path.join(
            model_output_dir,
            f"{train_args.model_name}-{train_args.train_name}",
        )

        os.makedirs(final_save_path, exist_ok=True)

        model.save_pretrained(final_save_path)
        processor.save_pretrained(final_save_path)


if __name__ == "__main__":
    main()
