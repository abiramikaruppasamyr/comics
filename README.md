# Comics Phase 1 Scaffold

Offline-first image generation app scaffold with:

- `frontend/`: Vite + React + TypeScript + Tailwind CSS
- `backend/`: FastAPI API with a CPU-first Stable Diffusion service
- `output/`: generated images

The backend is wired for a local Stable Diffusion v1.5 `.safetensors` model plus the matching YAML config file.

## Manual setup

### Backend

```bash
cd backend
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Frontend

```bash
cd frontend
npm install
```

## Run

```bash
./run_backend.sh
./run_frontend.sh
```

## Notes

- The backend forces CPU execution and disables GPU visibility.
- Images are written to `output/` in the repo root.
- The model is loaded on demand and explicitly unloaded after generation to release RAM.
