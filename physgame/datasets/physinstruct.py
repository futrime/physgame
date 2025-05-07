import json
import os
from typing import Dict, List, Literal, TypedDict

from torch.utils.data import Dataset


class PhysInstructEntry(TypedDict):
    video: str
    question: str
    answer: str


class RawDatasetEntry(TypedDict):
    video: str
    QA: List[Dict[Literal["i", "q", "a"], str]]


class PhysInstructDataset(Dataset[PhysInstructEntry]):
    _entries: List[PhysInstructEntry]

    def __init__(self, dataset_dir: str):
        anno_path = os.path.join(dataset_dir, "PhysInstruct_anno_40k.json")

        with open(anno_path, "r") as f:
            raw_entries: List[RawDatasetEntry] = json.load(f)

        self._entries = [
            PhysInstructEntry(
                video=os.path.join(dataset_dir, "PhysInstruct", raw_entry["video"]),
                question=qa["q"],
                answer=qa["a"],
            )
            for raw_entry in raw_entries
            for qa in raw_entry["QA"]
        ]

    def __getitem__(self, index) -> PhysInstructEntry:
        return self._entries[index]

    def __len__(self) -> int:
        return len(self._entries)

if __name__ == "__main__":
    dataset = PhysInstructDataset(".dev/PhysGame/PhysInstruct-40k")
    print(len(dataset))
    print(dataset[0])
