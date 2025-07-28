# app/openai_proxy.py

import os
from fastapi import APIRouter, FastAPI, Request, HTTPException
import openai

# Read your key from env (make sure it’s set in your Docker/hosting env)
openai.api_key = os.getenv("OPENAI_API_KEY")

router = APIRouter(prefix="/openai")

@router.post("/chat/completions")
async def proxy_chat_completions(request: Request):
    """
    Proxy for OpenAI Chat Completion endpoint.
    Expects the same JSON body as openai.ChatCompletion.create().
    """
    body = await request.json()
    try:
        resp = openai.ChatCompletion.create(**body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return resp

@router.post("/completions")
async def proxy_completions(request: Request):
    """
    Proxy for the classic Completion endpoint.
    """
    body = await request.json()
    try:
        resp = openai.Completion.create(**body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return resp

@router.post("/embeddings")
async def proxy_embeddings(request: Request):
    """
    Proxy for Embeddings.
    Expects JSON like: { "model": "...", "input": "your text or [texts]" }
    """
    body = await request.json()
    try:
        resp = openai.Embedding.create(**body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return resp

@router.post("/images/generations")
async def proxy_image_generations(request: Request):
    """
    Proxy for DALL·E image generation.
    Expects JSON like: { "prompt": "...", "n": 1, "size": "512x512" }
    """
    body = await request.json()
    try:
        resp = openai.Image.create(**body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return resp
# (… your other routes, including /predict etc. …)
