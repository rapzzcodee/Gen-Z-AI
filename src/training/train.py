import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup
import sentencepiece as spm
import json, sys, random
from pathlib import Path
from src.model.genz_transformer import GenZTransformer

DATA_PATH = "data/processed/genz_dataset.jsonl"
VOCAB_PATH = "data/vocab/genz.model"

# Hyperparameters
MAX_LEN = 128
BATCH = 16
LR = 3e-4
EPOCHS = 3
SPLIT = 0.9  # 90% train, 10% valid

# Load tokenizer
sp = spm.SentencePieceProcessor()
sp.load(VOCAB_PATH)
VOCAB = sp.get_piece_size()

# Load dataset
with open(DATA_PATH, encoding="utf-8") as f:
    lines = [json.loads(l)["text"] for l in f]

random.shuffle(lines)
split_idx = int(len(lines) * SPLIT)
train_data = lines[:split_idx]
valid_data = lines[split_idx:]

class TextDS(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        t = sp.encode(self.data[i], out_type=int)[:MAX_LEN]
        if len(t) < MAX_LEN:
            t += [0] * (MAX_LEN - len(t))
        return torch.tensor(t)

train_ds = TextDS(train_data)
valid_ds = TextDS(valid_data)
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=BATCH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GenZTransformer(vocab_size=VOCAB).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
sched = get_cosine_schedule_with_warmup(
    opt, num_warmup_steps=100, num_training_steps=len(train_dl)*EPOCHS
)
loss_fn = torch.nn.CrossEntropyLoss()

def evaluate():
    model.eval()
    losses = []
    with torch.no_grad():
        for x in valid_dl:
            x = x.to(device)
            out = model(x)
            loss = loss_fn(out.view(-1, VOCAB), x.view(-1))
            losses.append(loss.item())
    return sum(losses) / len(losses)

print("Training started ... 😎🔥")

for epoch in range(EPOCHS):
    model.train()
    for i, x in enumerate(train_dl):
        x = x.to(device)
        out = model(x)
        loss = loss_fn(out.view(-1, VOCAB), x.view(-1))
        loss.backward()
        opt.step()
        sched.step()
        opt.zero_grad()

        if i % 100 == 0:
            print(f"Epoch {epoch+1} Step {i}/{len(train_dl)} Loss {loss.item():.4f}")

    val_loss = evaluate()
    print(f"✅ Epoch {epoch+1} DONE | Valid Loss: {val_loss:.4f}")

torch.save(model.state_dict(), "gen.pt")
print("✅ Training Selesai! GEN lahir! 🤖🔥")
