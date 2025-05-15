import json
import os
import random
from typing import Dict, List, Literal, TypedDict

from torch.utils.data import Dataset

from physgame.datasets.physinstruct import PhysInstructEntry

PHYINSTRUCT_DIR = ".dev/datasets/PhysGame/PhysInstruct-40k"
PHYSQA_DIR = ".dev/datasets/PhysQA"

type OptionId = Literal["A", "B", "C", "D"]


class RawQAItem(TypedDict):
    question: str
    A: str
    B: str
    C: str
    D: str
    answer: OptionId


class RawEntry(TypedDict):
    idx: int
    video: str
    QA: List[RawQAItem]


class PhysQADataset(Dataset[PhysInstructEntry]):
    _entries: List[PhysInstructEntry]

    def __init__(self):
        anno_path = os.path.join(PHYSQA_DIR, "physqa_anno.jsonl")

        with open(anno_path, "r", encoding="utf-8") as f:
            raw_entries: List[RawEntry] = [json.loads(line) for line in f.readlines()]

        self._entries = [
            PhysInstructEntry(
                video_path=os.path.join(
                    PHYINSTRUCT_DIR, "PhysInstruct", raw_entry["video"]
                ),
                question=shuffled_qa["question"]
                + f"\n(A) {shuffled_qa['A']}"
                + f"\n(B) {shuffled_qa['B']}"
                + f"\n(C) {shuffled_qa['C']}"
                + f"\n(D) {shuffled_qa['D']}"
                + "\nOnly give the best option enclosed in parentheses, i.e. (A), (B), (C), or (D). "
                + "You must always give an option, even if you are not sure.",
                answer=f"({shuffled_qa['answer']})",
            )
            for raw_entry in raw_entries
            for qa in raw_entry["QA"]
            for shuffled_qa in [self._shuffle_qa_options(qa)]
        ]

    def __getitem__(self, index: int) -> PhysInstructEntry:
        return self._entries[index]

    def __len__(self) -> int:
        return len(self._entries)

    @staticmethod
    def _shuffle_qa_options(
        qa: RawQAItem,
    ) -> RawQAItem:
        local_random = random.Random(0)

        options: List[OptionId] = ["A", "B", "C", "D"]
        local_random.shuffle(options)

        # Map from original positions to new positions
        option_mapping: Dict[OptionId, OptionId] = {
            "A": options[0],
            "B": options[1],
            "C": options[2],
            "D": options[3],
        }

        # Map from new positions back to original positions (for the answer)
        reverse_mapping: Dict[OptionId, OptionId] = {
            v: k for k, v in option_mapping.items()
        }

        return RawQAItem(
            question=qa["question"],
            A=qa[option_mapping["A"]],
            B=qa[option_mapping["B"]],
            C=qa[option_mapping["C"]],
            D=qa[option_mapping["D"]],
            answer=reverse_mapping[qa["answer"]],
        )
