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
from transformers.feature_extraction_utils import BatchFeature
from transformers.generation.utils import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
)
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

from physgame.eval.physgame_hf import load_data, make_conversation

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
    ]

    errors: List[Exception] = []

    for model_class in AUTO_CLASSES:
        try:
            model = model_class.from_pretrained(
                model_name_or_path,
                # attn_implementation="sdpa",
                device_map="auto",
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
        default=10,
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
    processor = load_processor(perf_args.model)

    model = load_model(perf_args.model)

    # Load data.
    dataset = load_data()
    input_requests = [
        make_conversation(x, num_frames=8, video_fps=None)
        for x in tqdm([x for x in dataset][: perf_args.num_prompts])
    ]

    # Perform the benchmark.
    # Initialize metrics collection
    metrics = {
        "successful_requests": 0,
        "benchmark_start_time": timeit.default_timer(),
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "e2e_latencies": [],
        "ttfts": [],
        "itls": [],
    }

    # Process requests in batches
    for i in tqdm(range(0, len(input_requests), perf_args.batch_size)):
        conversations = input_requests[i : i + perf_args.batch_size]

        inputs = processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            padding="longest",
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

        # Generate tokens and measure timing
        gen_args = {
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
        }

        with torch.inference_mode():
            # Measure time to first token
            ttft_start = timeit.default_timer()
            _ = model.generate(
                **inputs,
                max_new_tokens=1,
                **gen_args,
            )
            ttft = (timeit.default_timer() - ttft_start) * 1000  # convert to ms

            # Full generation
            e2e_start_time = timeit.default_timer()

            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                **gen_args,
            )
            assert isinstance(outputs, torch.Tensor)

        e2e_latency = (timeit.default_timer() - e2e_start_time) * 1000  # convert to ms

        # Calculate inter-token latency (ITL) - we approximate this since we can't measure each token generation time
        generated_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
        itl = e2e_latency / generated_tokens if generated_tokens > 0 else 0

        metrics["total_input_tokens"] += (
            inputs.input_ids.shape[1] * perf_args.batch_size
        )
        metrics["total_output_tokens"] += generated_tokens * perf_args.batch_size

        # Record metrics for each request in the batch
        for j in range(len(conversations)):
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


if __name__ == "__main__":
    main()
