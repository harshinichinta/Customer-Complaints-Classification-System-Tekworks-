# Customer Complaints and Classification System

This project trains a classifier to label customer complaints (from `data/complaints.csv`) and exposes a FastAPI backend and Streamlit frontend.

Setup (Windows):

1. Create virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Train model (this will create `models/` with saved artifacts):

```powershell
python ml/train.py
```

4. Start backend (run in one terminal):

```powershell
uvicorn backend.main:app --reload
```

5. Start frontend (run in a second terminal):

```powershell
streamlit run frontend/app.py
```

Notes:
- Set `BACKEND_URL` environment variable if the backend is hosted elsewhere.
- The backend loads saved model artifacts from `models/` by default.
