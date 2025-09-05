from typing import Optional
from fastapi import FastAPI
from Agent import run
import uvicorn


app = FastAPI()

SERVER_IP = '0.0.0.0'
SERVER_PORT = 8000


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}


if __name__ == '__main__':
    uvicorn.run(app, host=SERVER_IP, port=SERVER_PORT)
