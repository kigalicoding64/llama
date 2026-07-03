# Serve a custom Llama model online

This repository now includes a small FastAPI server for turning a downloaded or Hugging Face-hosted Llama checkpoint into an HTTP chat service.

## Install

```bash
pip install -e '.[server]'
```

## Run in safe mock mode

Mock mode starts without downloading weights, which is useful for validating deployment, routing, and API clients.

```bash
LLAMA_ONLINE_BACKEND=mock llama-online-server
```

## Deploy on Vercel

Vercel can run the FastAPI app as a Python function through the root `app.py` entrypoint, with `api/index.py` kept as a secondary compatible entrypoint. Because Vercel functions have bundle and runtime limits, use the OpenAI-compatible remote backend there instead of trying to package multi-GB model weights into the function. Point the backend at any OpenAI-compatible Llama host, such as a vLLM server, llama.cpp server, Hugging Face endpoint with an OpenAI-compatible route, or Vercel AI Gateway.

1. Push this repository to GitHub and confirm the deployed commit includes `app.py`, `api/index.py`, `vercel.json`, and the `[tool.vercel]` section in `pyproject.toml`.
2. Import it in Vercel, or redeploy the project if it already exists.
3. In Vercel project settings, leave the build command empty/default and do not set a frontend framework preset.
4. Add these environment variables in the Vercel project settings:

```bash
LLAMA_ONLINE_BACKEND=openai
LLAMA_OPENAI_BASE_URL=https://your-openai-compatible-host.example/v1
LLAMA_MODEL_ID=your-llama-model-name
LLAMA_API_KEY=your-secret-api-key
LLAMA_SYSTEM_PROMPT='You are a helpful custom assistant for my product.'
```

After deployment, call `https://your-project.vercel.app/v1/chat/completions` or `https://your-project.vercel.app/generate`. If Vercel reports `No python entrypoint found`, redeploy a commit that includes root `app.py`; Vercel scans `app.py` automatically and the `pyproject.toml` entrypoint also points to `app:app`.

## Run with local real model weights

After you have accepted the model license and have access to the model weights, choose a model that fits your hardware and start the server with the `transformers` backend.

```bash
export LLAMA_ONLINE_BACKEND=transformers
export LLAMA_MODEL_ID=meta-llama/Llama-3.2-1B-Instruct
export LLAMA_SYSTEM_PROMPT='You are a helpful custom assistant for my product.'
llama-online-server
```

The following environment variables customize the server:

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLAMA_ONLINE_BACKEND` | `mock` | Use `mock` for deployment checks, `openai` for Vercel/remote inference, or `transformers` for local GPU inference. |
| `LLAMA_MODEL_ID` | `meta-llama/Llama-3.2-1B-Instruct` | Hugging Face model id, local checkpoint path, or remote OpenAI-compatible model name. |
| `LLAMA_OPENAI_BASE_URL` | `https://api.openai.com/v1` | Base URL for `LLAMA_ONLINE_BACKEND=openai`. |
| `LLAMA_API_KEY` | unset | Bearer token for the remote OpenAI-compatible backend. |
| `LLAMA_SYSTEM_PROMPT` | `You are a helpful, concise assistant.` | Custom behavior injected when the caller does not provide a system message. |
| `LLAMA_DEVICE_MAP` | `auto` | Device placement passed to Transformers. |
| `LLAMA_TORCH_DTYPE` | `auto` | Torch dtype passed to Transformers. |
| `HOST` | `0.0.0.0` | HTTP bind address. |
| `PORT` | `8000` | HTTP bind port. |

## Call the API

```bash
curl http://localhost:8000/health
```

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role": "user", "content": "Write a friendly welcome message."}
    ],
    "max_new_tokens": 128
  }'
```

A simpler `/generate` endpoint is also available and returns `{ "text": "..." }`.

## Deployment notes

- Put the service behind HTTPS before exposing it publicly.
- Add authentication at your gateway or reverse proxy.
- Start in mock mode first, then switch to `openai` on Vercel or `transformers` on GPU hosts once routing works.
- Do not deploy raw Llama weights to Vercel functions; use a remote inference host for full model inference.
- Use a model size that fits your GPU memory when running the `transformers` backend; larger Llama models require multiple high-memory GPUs.
