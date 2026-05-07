# Funky Fashion Finder

Full-stack outfit recommender with a React frontend and FastAPI backend.

## Requirements

- Python
- Node.js
- npm

## Run Locally

### Backend

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
uvicorn backend.main:app --reload
```

### Frontend

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

## Environment

```env
VITE_API_BASE_URL=http://localhost:8000
```

## API

### `POST /predict`

Required fields:

- `shoulder`
- `bust`
- `waist`
- `hips`
- `belly`
- `height`
