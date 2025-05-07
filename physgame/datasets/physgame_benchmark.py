import json
import os
from typing import Dict, List, Literal, TypedDict

from torch.utils.data import Dataset


class PhysGameBenchmarkEntry(TypedDict):
    question_id: str
    question: str
    options: Dict[Literal["A", "B", "C", "D"], str]
    answer: str
    class_anno: str
    subclass_anno: str


class PhysGameBenchmarkDataset(Dataset[PhysGameBenchmarkEntry]):
    _entries: List[PhysGameBenchmarkEntry]

    def __init__(self, dataset_dir: str):
        anno_path = os.path.join(dataset_dir, "PhysGame_880_annotation.json")

        with open(anno_path, "r") as f:
            self._entries: List[PhysGameBenchmarkEntry] = json.load(f)

    def __getitem__(self, index) -> PhysGameBenchmarkEntry:
        return self._entries[index]

    def __len__(self) -> int:
        return len(self._entries)

if __name__ == "__main__":
    dataset = PhysGameBenchmarkDataset(".dev/PhysGame/PhysGame-Benchmark")
    print(len(dataset))
    print(dataset[0])
