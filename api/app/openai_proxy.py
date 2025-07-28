# app/openai_proxy.py

import os
from fastapi import APIRouter, Request, HTTPException
from openai import OpenAI
from slowapi import Limiter
from slowapi.util import get_remote_address

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
limiter = Limiter(key_func=get_remote_address)

# Read your key from env (make sure it’s set in your Docker/hosting env)

router = APIRouter(prefix="/openai")

@router.post("/chat/completions")
@limiter.limit("5/minute")
async def proxy_chat_completions(request: Request):
    """
    Proxy for OpenAI Chat Completion endpoint.
    Expects the same JSON body as openai.ChatCompletion.create().
    """
    body = await request.json()
    try:
        resp = client.chat.completions.create(**body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return resp

@router.post("/completions")
@limiter.limit("5/minute")
async def proxy_completions(request: Request):
    """
    Proxy for the classic Completion endpoint.
    """
    body = await request.json()
    try:
        resp = client.completions.create(**body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return resp

@router.post("/embeddings")
@limiter.limit("5/minute")
async def proxy_embeddings(request: Request):
    """
    Proxy for Embeddings.
    Expects JSON like: { "model": "...", "input": "your text or [texts]" }
    """
    body = await request.json()
    try:
        resp = client.embeddings.create(**body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return resp

@router.post("/images/generations")
@limiter.limit("5/minute")
async def proxy_image_generations(request: Request):
    """
    Proxy for DALL·E image generation.
    Expects JSON like: { "prompt": "...", "n": 1, "size": "512x512" }
    """
    body = await request.json()
    try:
        resp = client.images.generate(**body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return resp
# (… your other routes, including /predict etc. …)
