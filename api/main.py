import sys
sys.path.append("../src")

import suggestions
from fastapi import FastAPI

app = FastAPI()


@app.get("/get_suggestions")
async def get_suggestions(text: str):
    generated = suggestions.get(text)
    return {"suggestions": generated}


@app.get("/retrain/")
async def retrain(notes_path: str):

    print(f"Retraining with new notes : {notes_path}")
    suggestions.retrain(notes_path)
