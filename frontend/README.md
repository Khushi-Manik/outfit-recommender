# Outfit Recommender Frontend

## Development

Use `npm.cmd` in PowerShell if `npm.ps1` is blocked by execution policy.

```powershell
npm.cmd install
npm.cmd run dev
```

To point the frontend at a different backend URL, set `VITE_API_BASE_URL`.

```powershell
$env:VITE_API_BASE_URL="http://localhost:8000"
npm.cmd run dev
```

## Build

```powershell
npm.cmd run build
```

The body-type quiz expects the backend `POST /predict` endpoint to be available.
