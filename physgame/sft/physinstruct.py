from typing import Any, Dict, Generator, List, Literal, TypedDict, cast

import numpy as np
import transformers.image_transforms as image_transforms
import transformers.image_utils as image_utils
from datasets import Dataset
from numpy.typing import NDArray
from PIL.Image import Image
from trl import SFTConfig, SFTTrainer

from physgame.datasets.physinstruct import PhysInstructDataset

N_FRAMES = 8


class PromptCompletionEntry(TypedDict):
    prompt: List[
        Dict[Literal["role", "content"], str | List[Dict[Literal["type"] | str, Any]]]
    ]
    completion: List[
        Dict[Literal["role", "content"], str | List[Dict[Literal["type"] | str, Any]]]
    ]


def build_dataset(dataset_dir: str) -> Dataset:
    torch_dataset = PhysInstructDataset(dataset_dir)

    def gen() -> Generator[PromptCompletionEntry, None, None]:
        for entry in torch_dataset:
            video, _ = image_utils.load_video(entry["video"], num_frames=N_FRAMES)
            video: NDArray[np.uint8]
            assert video.shape[0] == N_FRAMES

            images: List[Image] = [
                image_transforms.to_pil_image(video[i]) for i in range(N_FRAMES)
            ]

            yield PromptCompletionEntry(
                prompt=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": "Watch the video carefully and analyze the events and object movements, "
                                + "focusing on any inconsistencies with physical laws. "
                                + "Identify and highlight instances where the behavior deviates from expected real-world physics, "
                                + "and describe the detected glitch.",
                            }
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

    return cast(Dataset, Dataset.from_generator(gen))


def main() -> None:
    training_args = SFTConfig(
        optim="adamw_torch",
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        num_train_epochs=1,
        warmup_ratio=0.03,
        weight_decay=0,
        dataset_num_proc=64,
        per_device_train_batch_size=1,
    )

    trainer = SFTTrainer(
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        args=training_args,
        train_dataset=build_dataset(".dev/PhysGame/PhysInstruct-40k"),
    )

    trainer.train()


if __name__ == "__main__":
    main()
