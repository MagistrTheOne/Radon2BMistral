# RADON — Hugging Face Space (Docker)

## Quick Start

1. Create Space → Runtime: **Docker**.
2. Connect repository or push via `git push` to Space.
3. In Settings add Secrets (if needed): `HF_TOKEN`.
4. Space will automatically build the image from Dockerfile and expose API on `$PORT`.

## Testing

Once deployed, the Space will be available at:
```
https://huggingface.co/spaces/<username>/radon
```

Test endpoints:
- `GET /health` - health check
- `POST /api/v1/generate` - text generation
- `POST /vk/webhook` - VK webhook (if configured)

## Environment Variables

Configure in Space Settings → Variables:
- `RADON_ARCH` - model architecture (gpt2/t5)
- `RADON_CONFIG` - config path
- `LOG_LEVEL` - logging level

