import sys
sys.path.append("../src")

import suggestions
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/items/{item_id}")
async def read_item(item_id):
    return {"item_id": item_id}



@app.get("/files/{file_path:path}")
async def read_file(file_path: str):
    return {"file_path": file_path}


@app.get("/suggestions/{text:str}")
async def get_suggestions(text: str):
    generated = suggestions.get(text)
    return {"suggestions": generated}


