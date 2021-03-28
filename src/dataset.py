from typing import List, Dict

import torch
from torch.utils.data import Dataset

from utils import Vocab


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
        mode: str,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len
        self.mode = mode

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def intent_collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        tokens = self.vocab.encode_batch([s["text"].split() for s in samples])
        if self.mode != "TEST":
            labels = [self.label_mapping[s["intent"]] for s in samples]

            return {
                "intents": torch.tensor(tokens, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }
        else:
            idx = [s["id"] for s in samples]

            return {
                "intents": torch.tensor(tokens, dtype=torch.long),
                "id": idx,
            }

    def slot_collate_fn(self, samples: List[Dict]) -> Dict:
        tokens = self.vocab.encode_batch([s["tokens"] for s in samples])
        padding_len = len(tokens[0])

        if self.mode != "TEST":
            labels = [
                [self.label_mapping[tag] for tag in s["tags"]]
                + [0 for _ in range(padding_len - len(s["tags"]))]
                for s in samples
            ]

            return {
                "slots": torch.tensor(tokens, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

        else:
            idx = [s["id"] for s in samples]

            return {
                "slots": torch.tensor(tokens, dtype=torch.long),
                "id": idx,
            }

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]
