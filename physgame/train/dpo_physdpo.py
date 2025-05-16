import datetime
import os
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, TypedDict, Union, cast

import dotenv
import loguru
import torch
import torch.nn as nn
import torch.nn.functional
import transformers.trainer_utils as trainer_utils
import trl.trainer.utils
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
from trl import DPOConfig, DPOTrainer

import physgame.train.utils as utils
from physgame.datasets.physdpo import PhysDPODataset

logger = loguru.logger


@dataclass
class TrainArgs:
    model: str
    output_base_dir: str

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


class PreferenceEntry(TypedDict):
    video: str
    prompt: List[Dict[str, Any]]
    chosen: List[Dict[str, Any]]
    rejected: List[Dict[str, Any]]


def prepare_dataset() -> Dataset:
    physdpo = PhysDPODataset()

    n_max_gen = len(physdpo)

    def gen() -> Generator[PreferenceEntry, None, None]:
        for idx in range(len(physdpo)):
            if idx >= n_max_gen:
                break

            entry = physdpo[idx]

            preference = PreferenceEntry(
                video=entry["video"],
                prompt=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "path": entry["video"],
                            },
                            {
                                "type": "text",
                                "text": entry["prompt"],
                            },
                        ],
                    }
                ],
                chosen=[
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": entry["chosen"],
                            }
                        ],
                    }
                ],
                rejected=[
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": entry["rejected"],
                            }
                        ],
                    }
                ],
            )

            yield preference

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

    args, _ = parser.parse_known_args()

    return TrainArgs(
        model=args.model,
        output_base_dir=args.output_base_dir,
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

    if not os.path.exists(os.path.join(train_args.output_dir, "train_id.txt")):
        train_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        with open(os.path.join(train_args.output_dir, "train_id.txt"), "w") as f:
            f.write(train_id)
    else:
        with open(os.path.join(train_args.output_dir, "train_id.txt"), "r") as f:
            train_id = f.read().strip()

    trainer_config = DPOConfig(
        # auto_find_batch_size=False,
        bf16=True,
        max_grad_norm=5.0,
        do_train=True,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=2e-5,
        logging_steps=1,
        lr_scheduler_type="cosine",
        max_length=None,
        num_train_epochs=1,
        optim="adamw_torch",
        output_dir=model_output_dir,
        per_device_train_batch_size=2,
        remove_unused_columns=False,
        report_to="wandb",
        run_name=f"{train_args.model_name}-{train_args.train_name}-{train_id}",
        save_steps=10,
        save_strategy="steps",
        save_total_limit=2,
        torch_empty_cache_steps=1,
        warmup_ratio=0.03,
        weight_decay=0.0,
    )

    if accelerator.is_main_process:
        wandb.init(
            dir=os.path.join(train_args.output_dir, "wandb"),
            name=trainer_config.run_name,
        )

    # 1. Load dataset.

    dataset = prepare_dataset()

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

    ref_model = AutoModelForImageTextToText.from_pretrained(
        train_args.model,
        attn_implementation="flash_attention_2",
        torch_dtype="bfloat16",
        trust_remote_code=True,
    )
    assert isinstance(model, PreTrainedModel)

    if isinstance(model, Qwen2_5_VLForConditionalGeneration):
        model.visual.requires_grad_(False)

    model.train()

    # 3. Train model.

    def collate_fn(entries: List[PreferenceEntry]) -> BatchFeature:
        prompt_inputs = processor.apply_chat_template(
            [entry["prompt"] for entry in entries],
            num_frames=8,
            do_resize=True,
            size={
                "longest_edge": 1280 * 720,
                "shortest_edge": 0,
            },
            padding="longest",
            padding_side="left",
            return_dict=True,
            return_tensors="pt",
            tokenize=True,
            video_load_backend="opencv",
        )
        assert isinstance(prompt_inputs, BatchFeature)

        chosen_inputs = processor.apply_chat_template(
            [entry["chosen"] for entry in entries],
            padding="longest",
            padding_side="left",
            return_dict=True,
            return_tensors="pt",
            tokenize=True,
        )
        assert isinstance(chosen_inputs, BatchFeature)

        rejected_inputs = processor.apply_chat_template(
            [entry["rejected"] for entry in entries],
            padding="longest",
            padding_side="left",
            return_dict=True,
            return_tensors="pt",
            tokenize=True,
        )
        assert isinstance(rejected_inputs, BatchFeature)

        inputs = prompt_inputs
        inputs["prompt_input_ids"] = prompt_inputs.input_ids
        inputs["prompt_attention_mask"] = prompt_inputs.attention_mask
        inputs["chosen_input_ids"] = chosen_inputs.input_ids
        inputs["chosen_attention_mask"] = chosen_inputs.attention_mask
        inputs["rejected_input_ids"] = rejected_inputs.input_ids
        inputs["rejected_attention_mask"] = rejected_inputs.attention_mask

        del inputs["input_ids"]
        del inputs["attention_mask"]

        return inputs

    trainer = CustomDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=trainer_config,
        data_collator=collate_fn,
        processing_class=processor,
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


class CustomDPOTrainer(DPOTrainer):
    def concatenated_forward(
        self, model: nn.Module, batch: dict[str, Union[list, torch.LongTensor]]
    ):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        num_examples = batch["prompt_input_ids"].shape[0]  # type: ignore

        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.padding_value)  # type: ignore

        model_kwargs = {}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        # Copy all items whose key does not end with "_input_ids" or "_attention_mask"
        for key, value in concatenated_batch.items():
            if not key.endswith("_input_ids") and not key.endswith("_attention_mask"):
                model_kwargs[key] = value

        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]
        if self.is_encoder_decoder:
            labels = completion_input_ids
            labels[completion_attention_mask == 0] = self.label_pad_token_id
            outputs = model(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                labels=labels,  # we need the labels for the logits to be returned
                **model_kwargs,
            )
            logits = outputs.logits
            loss_mask = completion_attention_mask.bool()
        else:
            # Concatenate the prompt and completion inputs
            input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
            attention_mask = torch.cat(
                (prompt_attention_mask, completion_attention_mask), dim=1
            )
            # Mask the prompt but not the completion for the loss
            loss_mask = torch.cat(
                (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
                dim=1,
            )

            # Flush right to ensure Qwen2.5VL
            attention_mask, input_ids, loss_mask = utils.flush_right(
                attention_mask, input_ids, loss_mask
            )

            # Truncate right
            if self.max_length is not None:
                if self.truncation_mode == "keep_end":
                    input_ids = input_ids[:, -self.max_length :]
                    attention_mask = attention_mask[:, -self.max_length :]
                    loss_mask = loss_mask[:, -self.max_length :]
                elif self.truncation_mode == "keep_start":
                    input_ids = input_ids[:, : self.max_length]
                    attention_mask = attention_mask[:, : self.max_length]
                    loss_mask = loss_mask[:, : self.max_length]
                else:
                    raise ValueError(
                        f"Unknown truncation mode: '{self.truncation_mode}'. Should be one of ['keep_end', "
                        "'keep_start']."
                    )

            if self.use_logits_to_keep:
                # Compute logits_to_keep based on loss_mask pattern:
                # [[0, 0, 0, x, x, x, x],
                #  [0, 0, 0, x, x, x, 0]]
                #         ^ start computing logits from here ([:, -(7-3+1):])
                first_compute_index = loss_mask.nonzero(as_tuple=True)[1].min()
                logits_to_keep = (
                    loss_mask.shape[1] - first_compute_index
                ).item() + 1  # +1 for the first label
                model_kwargs["logits_to_keep"] = logits_to_keep

            if self.padding_free:
                # Flatten the input_ids, position_ids, and loss_mask
                # input_ids = [[a, b, c, 0], ->     input_ids = [[a, b, c, d, e, f, g]]
                #              [d, e, f, g]]     position_ids = [[0, 1, 2, 0, 1, 2, 3]]
                input_ids = input_ids[attention_mask.bool()].unsqueeze(0)
                loss_mask = loss_mask[attention_mask.bool()].unsqueeze(0)
                position_ids = (
                    attention_mask.cumsum(1)[attention_mask.bool()].unsqueeze(0) - 1
                )
                model_kwargs["position_ids"] = position_ids
            else:
                model_kwargs["attention_mask"] = attention_mask

            outputs = model(input_ids, **model_kwargs)
            logits = outputs.logits

            # Offset the logits by one to align with the labels
            labels = torch.roll(input_ids, shifts=-1, dims=1)
            loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()

            if self.use_logits_to_keep:
                # Align labels with logits
                # logits:    -,  -, [x2, x3, x4, x5, x6]
                #                     ^ --------- ^       after logits[:, :-1, :]
                # labels:   [y0, y1, y2, y3, y4, y5, y6]
                #                         ^ --------- ^   with logits_to_keep=4, [:, -4:]
                # loss_mask: [0,  0,  0,  1,  1,  1,  1]
                labels = labels[:, -logits_to_keep:]  # type: ignore
                loss_mask = loss_mask[:, -logits_to_keep:]  # type: ignore

        if logits.shape[:2] != labels.shape[:2]:
            # for llava, the returned logits include the image tokens (placed before the text tokens)
            seq_len = labels.shape[1]
            logits = logits[:, -seq_len:]

        # Compute the log probabilities of the labels
        labels[~loss_mask] = (
            0  # dummy token; we'll ignore the losses on these tokens later
        )
        per_token_logps = trl.trainer.utils.selective_log_softmax(logits, labels)
        per_token_logps[~loss_mask] = 0
        per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)

        if self.padding_free:
            # Unflatten the per_token_logps (shape: [1, sum_seq_len] -> [batch_size, seq_len])
            batch_size, seq_len = attention_mask.shape  # type: ignore
            per_token_logps_ = torch.zeros(
                batch_size,
                seq_len,
                device=outputs.logits.device,
                dtype=outputs.logits.dtype,
            )
            per_token_logps_[attention_mask.bool()] = per_token_logps  # type: ignore
            per_token_logps = per_token_logps_

        all_logps = per_token_logps.sum(-1)

        output = {}

        if self.use_weighting:
            with torch.no_grad():
                # Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827
                logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
                weights_adjustment_factor = torch.logsumexp(
                    2 * logprobs, dim=-1
                )  # same as sum(probs**2) in log space
                per_token_logps_adjusted = per_token_logps - weights_adjustment_factor
                all_weights = (per_token_logps_adjusted * loss_mask).sum(
                    -1
                ) / loss_mask.sum(-1)
                chosen_weights = all_weights[:num_examples]
                rejected_weights = all_weights[num_examples:]
                output["policy_weights"] = torch.clamp(
                    torch.exp(chosen_weights + rejected_weights), max=1
                )

        if self.args.rpo_alpha is not None:  # type: ignore
            # Only use the chosen logits for the RPO loss
            chosen_logits = logits[:num_examples]
            chosen_labels = labels[:num_examples]

            # Compute the log probabilities of the labels
            output["nll_loss"] = torch.nn.functional.cross_entropy(
                torch.flatten(chosen_logits, end_dim=1),
                torch.flatten(chosen_labels, end_dim=1),
                ignore_index=0,
            )

        if self.loss_type == "ipo":
            all_logps = all_logps / loss_mask.sum(-1)

        output["chosen_logps"] = all_logps[:num_examples]
        output["rejected_logps"] = all_logps[num_examples:]

        # Compute the mean logits
        if self.padding_free:
            # position_ids contains a sequence of range identifiers (e.g., [[0, 1, 2, 0, 1, 2, 3, ...]]).
            # There are 2*num_examples ranges in total: the first half corresponds to the chosen tokens,
            # and the second half to the rejected tokens.
            # To find the start of the rejected tokens, we look for the num_examples+1-th zero in pos_id.
            split_idx = (position_ids == 0).nonzero(as_tuple=True)[1][num_examples]  # type: ignore
            mean_chosen_logits = logits[0, :split_idx][loss_mask[0, :split_idx]].mean()
            mean_rejected_logits = logits[0, split_idx:][
                loss_mask[0, split_idx:]
            ].mean()
        else:
            mean_chosen_logits = logits[:num_examples][loss_mask[:num_examples]].mean()
            mean_rejected_logits = logits[num_examples:][
                loss_mask[num_examples:]
            ].mean()

        output["mean_chosen_logits"] = mean_chosen_logits
        output["mean_rejected_logits"] = mean_rejected_logits

        if self.aux_loss_enabled:
            output["aux_loss"] = outputs.aux_loss

        return output

    @staticmethod
    def concatenated_inputs(batch, padding_value: int) -> dict[str, torch.LongTensor]:
        """
        Concatenate the `chosen` and `rejected` inputs from the batch into a single tensor for both the prompt
        and completion sequences.

        Args:
            batch (`dict[str, Union[list, torch.LongTensor]]`):
                A batch of input data. The batch must contain the following keys:

                - `"prompt_input_ids"`: Tensor of shape `(batch_size, prompt_length)` representing the prompt input IDs.
                - `"chosen_input_ids"`: Tensor of shape `(batch_size, chosen_length)` representing the chosen completion input IDs.
                - `"rejected_input_ids"`: Tensor of shape `(batch_size, rejected_length)` representing the rejected completion input IDs.
                - `"prompt_pixel_values"` (optional): Tensor for pixel values, if available.
                - `"prompt_pixel_attention_mask"` (optional): Tensor for pixel attention masks, if available.

            padding_value (`int`):
                The padding value to use for the concatenated completion sequences (`chosen_input_ids` and
                `rejected_input_ids`).

        Returns:
            `dict[str, torch.LongTensor]`: A dictionary containing:

                - `"prompt_input_ids"`: Concatenated prompt input IDs of shape `(2 * batch_size, prompt_length)`.
                - `"completion_input_ids"`: Concatenated chosen and rejected completion input IDs of shape `(2 * batch_size, max_completion_length)`.
                - `"prompt_attention_mask"`: Concatenated prompt attention masks of shape `(2 * batch_size, prompt_length)`.
                - `"completion_attention_mask"`: Concatenated chosen and rejected attention masks of shape `(2 * batch_size, max_completion_length)`.
                - `"pixel_values"` (optional): Concatenated pixel values if `"prompt_pixel_values"` are present.
                - `"pixel_attention_mask"` (optional): Concatenated pixel attention masks if `"prompt_pixel_attention_mask"` are present.

        Notes:
            The completion input IDs and attention masks are padded to the maximum completion length of the chosen
            or rejected sequences.
        """
        output = {}

        batch = cast(dict[str, torch.Tensor], batch)

        # For the prompt, the input_ids are the same for both the chosen and rejected responses
        output["prompt_input_ids"] = torch.cat(
            [batch["prompt_input_ids"], batch["prompt_input_ids"]], dim=0
        )
        output["prompt_attention_mask"] = torch.cat(
            [batch["prompt_attention_mask"], batch["prompt_attention_mask"]], dim=0
        )

        # For all items whose key does not end with "_input_ids" or "_attention_mask"
        for key, value in batch.items():
            if not key.endswith("_input_ids") and not key.endswith("_attention_mask"):
                if isinstance(value, torch.Tensor):
                    output[key] = torch.cat([value, value], dim=0)
                elif isinstance(value, list):
                    output[key] = value + value

        # Concatenate the chosen and rejected completions
        max_completion_length = max(
            batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1]
        )
        output["completion_input_ids"] = torch.cat(
            (
                trl.trainer.utils.pad_to_length(
                    batch["chosen_input_ids"],
                    max_completion_length,
                    pad_value=padding_value,
                ),
                trl.trainer.utils.pad_to_length(
                    batch["rejected_input_ids"],
                    max_completion_length,
                    pad_value=padding_value,
                ),
            ),
        )
        output["completion_attention_mask"] = torch.cat(
            (
                trl.trainer.utils.pad_to_length(
                    batch["chosen_attention_mask"], max_completion_length, pad_value=0
                ),
                trl.trainer.utils.pad_to_length(
                    batch["rejected_attention_mask"], max_completion_length, pad_value=0
                ),
            ),
        )

        return output

    def _prepare_dataset(
        self,
        dataset,
        processing_class,
        args,
        dataset_name,
    ):
        return dataset


if __name__ == "__main__":
    main()
