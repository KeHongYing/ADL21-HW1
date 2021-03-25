import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import trange

from dataset import SeqClsDataset
from utils import Vocab
from model import SeqClassifier

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    total_correct, total_loss = 0, 0
    current = 0
    cnt = 0

    model.train()
    for batch, data in enumerate(dataloader):
        correct = 0
        cnt += 1

        intents = data["intents"].to(device)
        labels = data["labels"].to(device)

        pred = model(intents)["labels"]
        loss = loss_fn(pred, labels)

        correct += (pred.argmax(dim=1) == labels).type(torch.float).sum().item()
        total_correct += correct
        total_loss += loss
        correct /= pred.shape[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current += len(intents)
        loss = loss.item()
        print(
            f"Acc: {(100 * correct):>4.1f}%, loss: {loss:>7f}, [{current:>6d}/{size:>6d}]",
            end="\r",
        )

    total_correct /= current
    total_loss /= cnt
    print(
        f"Acc: {(100 * total_correct):>4.1f}%, loss: {total_loss:>7f}, [{current:>6d}/{size:>6d}]",
    )


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)

    total_correct, total_loss = 0, 0
    cnt = 0

    model.eval()
    with torch.no_grad():
        for data in dataloader:
            intents = data["intents"].to(device)
            labels = data["labels"].to(device)

            cnt += 1

            pred = model(intents)["labels"]
            total_loss += loss_fn(pred, labels).item()
            total_correct += (
                (pred.argmax(dim=1) == labels).type(torch.float).sum().item()
            )

    total_loss /= cnt
    total_correct /= size

    print(f"Val Acc: {(100 * total_correct):>4.1f}%, Val loss: {total_loss:>7f}")

    return total_correct, total_loss


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len, mode=split)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    dataloader = {
        split: DataLoader(
            datasets[split],
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=datasets[split].collate_fn,
        )
        for split in SPLITS
    }

    torch.manual_seed(args.seed)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(
        embeddings=embeddings,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        num_class=datasets[TRAIN].num_classes,
    ).to(args.device)

    # TODO: init optimizer
    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-7, patience=5)

    loss_fn = torch.torch.nn.CrossEntropyLoss()
    max_acc, min_loss = 0, 100
    early_stop = 0
    for epoch in range(args.num_epoch):
        print(f"Epoch: {epoch + 1}")
        # TODO: Training loop - iterate over train dataloader and update model weights
        # TODO: Evaluation loop - calculate accuracy and save model weights
        train(dataloader[TRAIN], model, loss_fn, optimizer, args.device)
        acc, loss = test(dataloader[DEV], model, loss_fn, args.device)

        scheduler.step(loss)

        if acc > max_acc:
            max_acc = acc
            torch.save(model.state_dict(), args.ckpt_dir / f"{args.model}_best.pt")
            print(f"model is better than before, save model to {args.model}_best.pt")
            
        if loss > min_loss:
            early_stop += 1
        else:
            early_stop = 0
            min_loss = loss
        
        if early_stop == 10:
            print("Early stop...")
            break
    
    print(f"Done! Best model Acc: {(100 * max_acc):>4.1f}%")
    torch.save(model.state_dict(), args.ckpt_dir / f"{args.model}.pt")

    with open("result.txt", "a") as f:
        f.write(f"{args.model}, {max_acc:>5f}\n")

    # TODO: Inference on test set


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="model name.",
        default="model",
    )
    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    # misc
    parser.add_argument("--seed", type=int, default=0xb06902074)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
