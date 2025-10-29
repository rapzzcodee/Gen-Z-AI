import torch
import sentencepiece as spm
from PIL import Image
import io
from transformers import CLIPProcessor, CLIPModel
from src.model.genz_transformer import GenZTransformer

sp = spm.SentencePieceProcessor()
sp.load("data/vocab/genz.model")
device = "cpu"

model = GenZTransformer(vocab_size=sp.get_piece_size())
model.load_state_dict(torch.load("genz_model.pt", map_location=device))
model.eval()

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

def decode(tokens):
    return sp.decode(tokens)

def generate_reply(q):
    ids = sp.encode(q, out_type=int)
    x = torch.tensor([ids + [0]*max(0,50-len(ids))], device=device)
    out = model(x)
    idx = torch.argmax(out[0][-1]).item()
    return decode(ids+[idx])

def generate_from_image(b):
    img = Image.open(io.BytesIO(b)).convert("RGB")
    inputs = clip_proc(images=img, return_tensors="pt")
    feat = clip_model.get_image_features(**inputs).detach().numpy()
    desc = "gambar vibes asik bro" if feat.mean()>0 else "gambar misterius"
    return generate_reply(desc)
