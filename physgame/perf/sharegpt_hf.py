"""Benchmarking text generation performance using Hugging Face Transformers."""

import json
import os
import random
import timeit
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import List, Optional, Tuple, cast

import loguru
import numpy as np
import requests
import torch
from tqdm import tqdm
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.utils import GenerationMixin
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
)
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

logger = loguru.logger


@dataclass
class PerfArgs:
    model: str
    output_base_dir: str

    batch_size: int
    num_prompts: int

    @property
    def perf_name(self) -> str:
        file_name = os.path.basename(__file__)
        file_name = file_name.replace(".py", "")
        return file_name

    @property
    def output_dir(self) -> str:
        model_name = os.path.basename(os.path.normpath(self.model))

        return os.path.join(
            self.output_base_dir,
            f"{model_name}",
            f"{self.perf_name}",
        )

def load_model(model_name_or_path: str) -> PreTrainedModel:
    AUTO_CLASSES = [
        AutoModelForImageTextToText,
        AutoModelForVision2Seq,
        AutoModel,
        AutoModelForCausalLM,
    ]

    for model_class in AUTO_CLASSES:
        try:
            model = model_class.from_pretrained(
                model_name_or_path,
                attn_implementation="sdpa",
                device_map="auto",
                torch_dtype="bfloat16",
                trust_remote_code=True,
            )

            if not isinstance(model, GenerationMixin):
                continue

            return cast(PreTrainedModel, model)

        except:
            continue

    raise ValueError("Failed to load model with any of the possible classes.")

def load_tokenizer(model_name_or_path: str) -> PreTrainedTokenizerBase:
    # First try to load the tokenizer from the model name or path.
    try:
        return AutoTokenizer.from_pretrained(
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

    return AutoTokenizer.from_pretrained(
        model_config._name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )


def parse_args() -> PerfArgs:
    parser = ArgumentParser()
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
        "--num-prompts",
        type=int,
        default=20,
    )

    parsed_args, _ = parser.parse_known_args()

    return PerfArgs(
        model=parsed_args.model,
        batch_size=parsed_args.batch_size,
        num_prompts=parsed_args.num_prompts,
        output_base_dir=parsed_args.output_base_dir,
    )


def main() -> None:
    perf_args = parse_args()

    if os.path.exists(os.path.join(perf_args.output_dir, "metrics.json")):
        logger.info(f"Evaluation already completed. Skipping...")
        return

    os.makedirs(perf_args.output_dir, exist_ok=True)

    # Load model.
    tokenizer = load_tokenizer(perf_args.model)

    model = load_model(perf_args.model)

    # Load data.
    get_dataset_args = Namespace(
        dataset_name="sharegpt",
        dataset_path="",
        num_prompts=perf_args.num_prompts,
        sharegpt_output_len=None,
        sharegpt_context_len=None,
        prompt_suffix="",
        apply_chat_template=False,
    )

    input_requests = get_dataset(get_dataset_args, tokenizer)

    # Perform the benchmark.
    # Initialize metrics collection
    metrics = {
        "successful_requests": 0,
        "benchmark_start_time": timeit.default_timer(),
        "total_input_tokens": sum(x[1] for x in input_requests),
        "total_output_tokens": sum(x[2] for x in input_requests),
        "e2e_latencies": [],
        "ttfts": [],
        "itls": [],
    }

    # Process requests in batches
    for i in tqdm(range(0, len(input_requests), perf_args.batch_size)):
        batch = input_requests[i : i + perf_args.batch_size]
        prompts = [x[0] for x in batch]
        prompt_lens = [x[1] for x in batch]
        output_lens = [x[2] for x in batch]

        # Tokenize the inputs
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            truncation=True,
            max_length=max(prompt_lens),
        ).to(model.device)
        assert isinstance(inputs, BatchEncoding)

        # Generate tokens and measure timing

        with torch.inference_mode():
            # Measure time to first token
            ttft_start = timeit.default_timer()
            first_token = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=1,
                do_sample=False,
                temperature=None,
                pad_token_id=tokenizer.eos_token_id,
            )
            ttft = (timeit.default_timer() - ttft_start) * 1000  # convert to ms

            # Full generation
            e2e_start_time = timeit.default_timer()

            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max(output_lens),
                do_sample=False,
                temperature=None,
                pad_token_id=tokenizer.eos_token_id,
            )
            assert isinstance(outputs, torch.Tensor)

        e2e_latency = (timeit.default_timer() - e2e_start_time) * 1000  # convert to ms

        # Calculate inter-token latency (ITL) - we approximate this since we can't measure each token generation time
        generated_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
        itl = e2e_latency / generated_tokens if generated_tokens > 0 else 0

        # Record metrics for each request in the batch
        for j in range(len(batch)):
            metrics["successful_requests"] += 1
            metrics["e2e_latencies"].append(e2e_latency)
            metrics["ttfts"].append(ttft)
            metrics["itls"].append(itl)

    # Calculate final metrics
    benchmark_duration = timeit.default_timer() - metrics["benchmark_start_time"]
    total_tokens = metrics["total_input_tokens"] + metrics["total_output_tokens"]

    metrics_summary = {
        "backend": "HuggingFace",
        "batch_size": perf_args.batch_size,
        "successful_requests": metrics["successful_requests"],
        "benchmark_duration_s": round(benchmark_duration, 2),
        "total_input_tokens": metrics["total_input_tokens"],
        "total_generated_tokens": metrics["total_output_tokens"],
        "request_throughput": round(
            metrics["successful_requests"] / benchmark_duration, 2
        ),
        "input_token_throughput": round(
            metrics["total_input_tokens"] / benchmark_duration, 2
        ),
        "output_token_throughput": round(
            metrics["total_output_tokens"] / benchmark_duration, 2
        ),
        "total_token_throughput": round(total_tokens / benchmark_duration, 2),
        "concurrency": perf_args.batch_size,
        "latency": {
            "e2e": {
                "mean_ms": round(np.mean(metrics["e2e_latencies"]), 2),
                "median_ms": round(np.median(metrics["e2e_latencies"]), 2),
            },
            "ttft": {
                "mean_ms": round(np.mean(metrics["ttfts"]), 2),
                "median_ms": round(np.median(metrics["ttfts"]), 2),
                "p99_ms": round(np.percentile(metrics["ttfts"], 99), 2),
            },
            "itl": {
                "mean_ms": round(np.mean(metrics["itls"]), 2),
                "median_ms": round(np.median(metrics["itls"]), 2),
                "p95_ms": round(np.percentile(metrics["itls"], 95), 2),
                "p99_ms": round(np.percentile(metrics["itls"], 99), 2),
                "max_ms": round(np.max(metrics["itls"]), 2),
            },
        },
    }

    # Save results.
    logger.info(f"Saving results to {perf_args.output_dir}")

    # Write metrics to file
    with open(
        os.path.join(perf_args.output_dir, "metrics.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metrics_summary, f, indent=2)


##### Copied from https://github.com/sgl-project/sglang/blob/main/python/sglang/bench_serving.py

ASSISTANT_SUFFIX = "Assistant:"
SHAREGPT_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"


def download_and_cache_file(url: str, filename: Optional[str] = None):
    """Read and cache a file from a url."""
    if filename is None:
        filename = os.path.join("/tmp", url.split("/")[-1])

    # Check if the cache file already exists
    if os.path.exists(filename):
        return filename

    print(f"Downloading from {url} to {filename}")

    # Stream the response to show the progress bar
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check for request errors

    # Total size of the file in bytes
    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024  # Download in chunks of 1KB

    # Use tqdm to display the progress bar
    with (
        open(filename, "wb") as f,
        tqdm(
            desc=filename,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))

    return filename


def get_dataset(args, tokenizer):
    if args.dataset_name == "sharegpt":
        input_requests = sample_sharegpt_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_output_len=args.sharegpt_output_len,
            context_len=args.sharegpt_context_len,
            prompt_suffix=args.prompt_suffix,
            apply_chat_template=args.apply_chat_template,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")
    return input_requests


def remove_prefix(text: str, prefix: str) -> str:
    return text[len(prefix) :] if text.startswith(prefix) else text


def remove_suffix(text: str, suffix: str) -> str:
    return text[: -len(suffix)] if text.endswith(suffix) else text


def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
    context_len: Optional[int] = None,
    prompt_suffix: Optional[str] = "",
    apply_chat_template=False,
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Download sharegpt if necessary
    if not os.path.isfile(dataset_path) and dataset_path == "":
        dataset_path = download_and_cache_file(SHAREGPT_URL)

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)

    # Filter out the conversations with less than 2 turns.
    dataset = [
        data
        for data in dataset
        if len(data.get("conversations", data.get("conversation", []))) >= 2
    ]
    # Only keep the first two turns of each conversation.
    dataset = [
        (
            data.get("conversations", data.get("conversation", []))[0]["value"],
            data.get("conversations", data.get("conversation", []))[1]["value"],
        )
        for data in dataset
    ]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        if prompt_suffix:
            prompt = (
                remove_suffix(prompt, ASSISTANT_SUFFIX)
                + prompt_suffix
                + ASSISTANT_SUFFIX
            )

        if apply_chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            prompt = cast(str, prompt).replace(cast(str, tokenizer.bos_token), "")

        prompt_token_ids = tokenizer.encode(prompt)
        completion = dataset[i][1]
        completion_token_ids = tokenizer.encode(completion)
        prompt_len = len(prompt_token_ids)
        output_len = (
            len(completion_token_ids) if fixed_output_len is None else fixed_output_len
        )

        if prompt_len < 2 or output_len < 2:
            # Prune too short sequences.
            continue

        if context_len and prompt_len + output_len > context_len:
            # Prune too long sequences.
            continue

        filtered_dataset.append((prompt, prompt_len, output_len))

    logger.info(f"#Input tokens: {np.sum([x[1] for x in filtered_dataset])}")
    logger.info(f"#Output tokens: {np.sum([x[2] for x in filtered_dataset])}")
    return filtered_dataset


##### End of copied code.

if __name__ == "__main__":
    main()
