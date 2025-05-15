import json
import os
from typing import List, TypedDict

from torch.utils.data import Dataset

PHYSDPO_DIR = ".dev/datasets/PhysGame/PhysDPO-10k"


class PhysDPOEntry(TypedDict):
    id: str
    video: str
    prompt: str
    chosen: str
    rejected: str
    chosen_score: float
    rejected_score: float


class PhysDPODataset(Dataset[PhysDPOEntry]):
    _entries: List[PhysDPOEntry]

    def __init__(self):
        anno_path = os.path.join(PHYSDPO_DIR, "PhysDPO_anno_10k.json")

        with open(anno_path, "r", encoding="utf-8") as f:
            self._entries: List[PhysDPOEntry] = json.load(f)

        for i in range(len(self._entries)):
            self._entries[i]["video"] = os.path.join(
                PHYSDPO_DIR, "PhysDPO", self._entries[i]["video"]
            )

    def __getitem__(self, index: int) -> PhysDPOEntry:
        return self._entries[index]

    def __len__(self) -> int:
        return len(self._entries)
