from fastapi import FastAPI, UploadFile, File
from src.utils.infer import generate_reply, generate_from_image

app = FastAPI()

@app.get("/chat")
async def chat(q: str):
    return {"response": generate_reply(q)}

@app.post("/vision")
async def vision(image: UploadFile = File(...)):
    content = await image.read()
    return {"response": generate_from_image(content)}
