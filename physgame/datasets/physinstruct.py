import json
import os
from typing import Dict, List, Literal, TypedDict

from torch.utils.data import Dataset

PHYSINSTRUCT_DIR = ".dev/datasets/PhysGame/PhysInstruct-40k"


class PhysInstructEntry(TypedDict):
    video_path: str
    question: str
    answer: str


class PhysInstructDataset(Dataset[PhysInstructEntry]):
    class RawEntry(TypedDict):
        video: str
        QA: List[Dict[Literal["i", "q", "a"], str]]

    _entries: List[PhysInstructEntry]

    def __init__(self):
        anno_path = os.path.join(PHYSINSTRUCT_DIR, "PhysInstruct_anno_40k.json")

        with open(anno_path, "r", encoding="utf-8") as f:
            raw_entries: List[PhysInstructDataset.RawEntry] = json.load(f)

        self._entries = [
            PhysInstructEntry(
                video_path=os.path.join(
                    PHYSINSTRUCT_DIR, "PhysInstruct", raw_entry["video"]
                ),
                question=qa["q"],
                answer=qa["a"],
            )
            for raw_entry in raw_entries
            for qa in raw_entry["QA"]
        ]

    def __getitem__(self, index: int) -> PhysInstructEntry:
        return self._entries[index]

    def __len__(self) -> int:
        return len(self._entries)
