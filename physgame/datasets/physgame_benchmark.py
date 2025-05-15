import json
import os
from typing import Dict, List, Literal, TypedDict

from torch.utils.data import Dataset

PHYSGAME_BENCHMARK_DIR = ".dev/datasets/PhysGame/PhysGame-Benchmark"


class PhysGameBenchmarkEntry(TypedDict):
    question_id: str
    question: str
    options: Dict[Literal["A", "B", "C", "D"], str]
    answer: str
    class_anno: str
    subclass_anno: str
    video: str


class PhysGameBenchmarkDataset(Dataset[PhysGameBenchmarkEntry]):
    _entries: List[PhysGameBenchmarkEntry]

    def __init__(self):
        anno_path = os.path.join(PHYSGAME_BENCHMARK_DIR, "PhysGame_880_annotation.json")

        with open(anno_path, "r") as f:
            self._entries: List[PhysGameBenchmarkEntry] = json.load(f)

            for i in range(len(self._entries)):
                self._entries[i]["video"] = os.path.join(
                    PHYSGAME_BENCHMARK_DIR,
                    "PhysGame-Benchmark",
                    self._entries[i]["question_id"] + ".mp4",
                )

    def __getitem__(self, index) -> PhysGameBenchmarkEntry:
        return self._entries[index]

    def __len__(self) -> int:
        return len(self._entries)
