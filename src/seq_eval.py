import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

from seqeval.metrics import classification_report
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, tag2idx, args.max_len, "EVAL")
    # TODO: crecate DataLoader for test dataset
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=dataset.slot_collate_fn
    )

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
        "slot",
    ).to(args.device)
    model.eval()

    ckpt = torch.load(args.ckpt_path)
    # load weights into model
    model.load_state_dict(ckpt)

    # TODO: predict dataset

    # TODO: write prediction to file (args.pred_file)

    result = []
    ground_truth = []
    size = len(dataset)
    current = 0
    cnt = 0
    joint_acc = 0
    token_acc = 0
    token_num = 0

    for seq in dataloader:
        slots = seq["slots"].to(args.device)

        current += len(slots)
        print(f"[{current:>5d}/{size:>5d}]", end="\r")

        pred = model(slots)["labels"]
        for GT, label in zip(seq["labels"], pred.argmax(dim=-1).type(torch.long)):
            result.append(
                [
                    dataset.idx2label(tagId.item())
                    for tagId in label[1 : len(data[cnt]["tokens"]) + 1]
                ]
            )
            ground_truth.append(
                [
                    dataset.idx2label(i.item())
                    for i in GT[1 : len(data[cnt]["tokens"]) + 1]
                ]
            )
            joint_acc += np.all(np.array(result[-1]) == np.array(ground_truth[-1]))
            token_acc += np.sum(np.array(result[-1]) == np.array(ground_truth[-1]))
            token_num += len(result[-1])

            cnt += 1

    print(classification_report(ground_truth, result))
    print(f"joint: {joint_acc / cnt}")
    print(f"token: {token_acc / token_num}")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file", type=Path, help="Path to the test file.", required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path", type=Path, help="Path to model checkpoint.", required=True
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
