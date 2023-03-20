import os
import argparse
import logging
import csv
import torch
from model import TransformerModel
from tqdm import tqdm
from utils import Tokenizer, Vocabulary


class Dataset (torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class Dataloader (torch.utils.data.DataLoader):
    def __init__(self, dataset, vocab, batch_size, shuffle, device):
        super().__init__(dataset, batch_size, shuffle, collate_fn=self.collate_fn)
        self.vocab = vocab
        self.device = device

    def collate_fn(self, batch):
        x, y = zip(*batch)
        max_len = max([len(s) for s in x])
        x = [s + [self.vocab.get_pad_index()] * (max_len - len(s)) for s in x]
        x = torch.LongTensor(x).to(device=self.device)
        y = torch.FloatTensor(y).to(device=self.device)
        return x, y
    

def build_vocab(data_file, tokenizer, max_size):
    vocab = Vocabulary()

    with open(data_file, "r") as f:
        reader = csv.reader(f, delimiter=",", quotechar='"')
        rows_count = sum(1 for _ in reader) - 1
        f.seek(0)

        next(reader)

        for row in tqdm(reader, desc="Building vocab", total=rows_count):
            vocab.update(tokenizer.tokenize(row[0]))

    vocab.truncate(max_size)
    return vocab


def load_datasets(data_file, tokenizer, vocab, val_size=0.1, test_size=0.1):
    x = []
    y = []
    with open(data_file, "r") as f:
        reader = csv.reader(f, delimiter=",", quotechar='"')
        rows_count = sum(1 for _ in reader) - 1
        f.seek(0)

        next(reader)

        for row in tqdm(reader, desc="Loading data  ", total=rows_count):
            x.append([vocab.get_index(t) for t in tokenizer.tokenize(row[0])])
            y.append(int(row[1]))

    val_idx = int(len(x) * val_size)
    test_idx = int(len(x) * val_size + test_size)
    
    val_x = x[:val_idx]
    val_y = y[:val_idx]
    test_x = x[val_idx:test_idx]
    test_y = y[val_idx:test_idx]
    train_x = x[test_idx:]
    train_y = y[test_idx:]

    return Dataset(train_x, train_y), Dataset(val_x, val_y), Dataset(test_x, test_y)


def train(model, dataloader, criterion, optimizer, grad_norm_clip=None):
    epoch_loss = 0
    correct = 0

    model.train()
    for batch in tqdm(dataloader, desc="Training Pass"):
        x, y = batch

        optimizer.zero_grad()
        y_out = model(x)
        y_out = y_out.squeeze()
        loss = criterion(y_out, y)
        loss.backward()

        if grad_norm_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)

        optimizer.step()
        
        epoch_loss += loss.item()
        with torch.no_grad():
            correct += (torch.sigmoid(y_out).round() == y).float().sum().item()

    return epoch_loss / len(dataloader), correct / len(dataloader.dataset)


def validate(model, dataloader, criterion, device):
    epoch_loss = 0
    correct = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation Pass"):
            x, y = batch
            y_out = model(x)
            y_out = y_out.squeeze()
            loss = criterion(y_out, y)

            epoch_loss += loss.item()
            correct += (torch.sigmoid(y_out).round() == y).float().sum().item()

    return epoch_loss / len(dataloader), correct / len(dataloader.dataset)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        torch.nn.init.kaiming_uniform(m.weight.data)


def run_training(args):
    logging.info("Training started")
    logging.info(f"* Epochs: {args.epochs}")
    logging.info(f"* Batch size: {args.batch_size}")
    logging.info(f"* Learning rate: {args.lr}")
    logging.info(f"* Data: {args.data}")
    logging.info(f"* Seed: {args.seed}")

    if not os.path.exists(args.model):
        os.makedirs(args.model)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer()
    vocab = build_vocab(args.data, tokenizer, max_size=20000)
    train_set, val_set, _ = load_datasets(args.data, tokenizer, vocab)
    train_loader = Dataloader(train_set, vocab=vocab, batch_size=args.batch_size, shuffle=True, device=device)
    val_loader = Dataloader(val_set, vocab=vocab, batch_size=args.batch_size, shuffle=False, device=device)
    
    model = TransformerModel(vocab_size=len(vocab), max_len=200, embed_dim=128, hidden_dim=128*4, num_layers=4, num_heads=4, unk_index=vocab.get_unk_index(), pad_index=vocab.get_pad_index(), dropout=0.1)
    model.apply(initialize_weights)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    model.to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Mode parameters: {param_count:,}")
    
    for epoch in range(args.epochs):
        logging.info(f"Epoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc = train(model, train_loader, criterion, optimizer, args.grad_norm_clip)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        logging.info(f"  Train loss: {train_loss:.4f}")
        logging.info(f"  Train acc : {train_acc*100:.2f}%")
        logging.info(f"  Val loss  : {val_loss:.4f}")
        logging.info(f"  Val acc   : {val_acc*100:.2f}%")

    torch.save(model.state_dict(), os.path.join(args.model, "model.pt"))
    vocab.save(os.path.join(args.model, "vocab.txt"))        
    logging.info("Training finished")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--grad_norm_clip", type=float, default=0.1)
    parser.add_argument("--data", type=str, default="data.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="model.pt")

    run_training(parser.parse_args())  