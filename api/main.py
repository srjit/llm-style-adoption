import sys
sys.path.append("../src")

import suggestions
from fastapi import FastAPI

app = FastAPI()


@app.get("/suggestions/{text:str}")
async def get_suggestions(text: str):
    generated = suggestions.get(text)
    return {"suggestions": generated}


