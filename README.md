# Funky Fashion Finder

Funky Fashion Finder is a full-stack outfit recommender project with a React frontend and a FastAPI backend. It combines a body-type quiz, style suggestion flows, curated fashion news, and trending product links in a single fashion assistant experience.

The current app mixes real and demo functionality:

- The body-type quiz calls a backend `POST /predict` API.
- The quiz falls back to client-side body-type logic if the backend is unavailable.
- Outfit recommendations are currently generated in the frontend with simulated logic.
- Fashion news and product listings are currently static curated content.

## Features

- Multi-step body measurement quiz for shoulder, bust, waist, hips, belly, and height
- Backend-powered body-type prediction with confidence score
- Client-side fallback prediction for better local development resilience
- Outfit recommendation form based on body type, color, weather, occasion, and clothing preferences
- Fashion news page with curated external reading links
- Trending products page with external product links

## Tech Stack

### Frontend

- React 19
- TypeScript
- Vite
- React Router
- Lucide React

### Backend

- FastAPI
- Pydantic
- pandas
- NumPy
- scikit-learn
- imbalanced-learn
- TensorFlow
- joblib

### ML

- Rule-based body-type heuristics
- Saved scaler and model artifacts under `backend/models`
- Ensemble-style prediction flow using classical ML models and a neural network when available

## Project Structure

```text
outfit-recommender/
|-- backend/
|   |-- main.py
|   |-- body_type_model.py
|   |-- requirements.txt
|   |-- models/
|   `-- Body Measurements _ original_CSV.csv
|-- frontend/
|   |-- package.json
|   |-- vite.config.ts
|   `-- src/
|       |-- App.tsx
|       `-- pages/
|           |-- BodyTypeQuiz.tsx
|           |-- OutfitRecommendationsPage.tsx
|           |-- NewsPage.tsx
|           `-- ProductsPage.tsx
`-- README.md
```

Key entrypoints:

- Backend app: `backend/main.py`
- Backend API creation: `backend/body_type_model.py`
- Frontend body-type API consumer: `frontend/src/pages/BodyTypeQuiz.tsx`

## Local Setup

### 1. Clone the project

```powershell
git clone <your-repo-url>
cd outfit-recommender
```

### 2. Start the backend

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

Run the FastAPI server:

```powershell
uvicorn backend.main:app --reload
```

The backend will be available at `http://localhost:8000`.

### 3. Start the frontend

Move into the frontend app and install dependencies:

```powershell
cd frontend
npm.cmd install
```

Start the Vite development server:

```powershell
npm.cmd run dev
```

If PowerShell blocks `npm.ps1`, keep using `npm.cmd` instead.

## Environment Variables

The frontend reads the backend base URL from `VITE_API_BASE_URL`.

Default behavior:

```env
VITE_API_BASE_URL=http://localhost:8000
```

PowerShell example:

```powershell
$env:VITE_API_BASE_URL="http://localhost:8000"
npm.cmd run dev
```

If the variable is not set, the frontend already defaults to `http://localhost:8000`.

## API

### `POST /predict`

Predicts the user's body type from six body measurements.

Request body:

```json
{
  "shoulder": 38,
  "bust": 90,
  "waist": 72,
  "hips": 94,
  "belly": 80,
  "height": 165
}
```

Response body:

```json
{
  "body_type": "Hourglass",
  "confidence": 0.84
}
```

Possible body types used in the app:

- Apple
- Pear
- Inverted Triangle
- Hourglass
- Rectangle

## Development Notes

- The backend adds permissive CORS with `allow_origins=["*"]`. Restrict this before production use.
- `backend/main.py` only creates the app and loads the prediction pipeline. It does not retrain models on startup.
- The training dataset is included as `backend/Body Measurements _ original_CSV.csv`.
- `backend/train_model.py` contains encoder-saving code but is not wired into the running app.
- `backend/train_models.py` is currently empty.
- The frontend includes static/demo pages for recommendations, news, and products, so those sections do not currently depend on backend APIs.

## Available Frontend Scripts

From `frontend/`:

```powershell
npm.cmd run dev
npm.cmd run build
npm.cmd run lint
npm.cmd run preview
```

## Future Improvements

- Replace simulated outfit recommendations with a backend recommendation service
- Add persistence for user profiles, wardrobes, and saved looks
- Add tests for the API and frontend pages
- Add deployment configuration for frontend and backend environments
