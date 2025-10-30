import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import sentencepiece as spm
from src.model.genz_transformer import GenZTransformer

sp = spm.SentencePieceProcessor()
sp.load("data/vocab/genz.model")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GenZTransformer(vocab_size=sp.get_piece_size())
model.load_state_dict(torch.load("gen.pt", map_location=device))
model.to(device)
model.eval()

def generate_reply(prompt, max_len=30):
    ids = sp.encode(prompt, out_type=int)
    x = torch.tensor([ids], device=device)
    for _ in range(max_len):
        out = model(x)
        next_id = torch.argmax(out[0, -1]).item()
        ids.append(next_id)
        if next_id == 0:
            break
        x = torch.tensor([ids], device=device)
    return sp.decode(ids)

while True:
    q = input("lu: ")
    if q.lower() in ["exit", "quit"]:
        break
    print("GEN:", generate_reply(q))
