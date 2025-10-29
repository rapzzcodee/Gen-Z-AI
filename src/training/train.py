import torch
from torch.utils.data import DataLoader, Dataset
from src.model.genz_transformer import GenZTransformer
from transformers import get_cosine_schedule_with_warmup
import sentencepiece as spm
import json, random

MAX_LEN = 128
BATCH = 16
LR = 3e-4
EPOCHS = 3

sp = spm.SentencePieceProcessor()
sp.load("data/vocab/genz.model")

with open("data/processed/genz_dataset.jsonl") as f:
    data = [json.loads(l)["text"] for l in f]

class TextDS(Dataset):
    def __len__(self): return len(data)
    def __getitem__(self, i):
        t = sp.encode(data[i], out_type=int)[:MAX_LEN]
        if len(t)<MAX_LEN: t += [0]*(MAX_LEN-len(t))
        return torch.tensor(t)

ds = TextDS()
dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

model = GenZTransformer(vocab_size=sp.get_piece_size())
opt = torch.optim.AdamW(model.parameters(), lr=LR)
sched = get_cosine_schedule_with_warmup(opt, 100, len(dl)*EPOCHS)
loss_fn = torch.nn.CrossEntropyLoss()

for e in range(EPOCHS):
    for x in dl:
        out = model(x)
        loss = loss_fn(out.view(-1, sp.get_piece_size()), x.view(-1))
        loss.backward()
        opt.step()
        sched.step()
        opt.zero_grad()
    print(f"Epoch {e} Loss: {loss.item()}")

torch.save(model.state_dict(), "genz_model.pt")
print("Training DONE 😎🔥")
