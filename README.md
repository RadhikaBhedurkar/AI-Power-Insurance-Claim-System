# Vehicle Insurance Claim Assessment (AI-assisted, human-reviewed)

## Step 1 - Environment and data
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
Install Tesseract OCR (Linux: apt install tesseract-ocr; Windows: UB-Mannheim installer).
Kaggle (kaggle.com/datasets, search these names and check each licence):
- "CarDD" - damage detection boxes/masks -> Step 2
- "Car Damage Severity Dataset" (minor/moderate/severe folders) -> Step 3
- CORD receipts (also on Hugging Face: naver-clova-ix/cord-v2) -> optional OCR/field extraction tuning
Download with: kaggle datasets download -d <owner/slug> -p data --unzip
No Kaggle dataset has verified repair costs: collect real invoices into data/repair_costs.csv
(severity, n_parts, glass, lamp, vehicle_age, cost).

## Step 2-4 - Train (mkdir models first)
python training/train_yolo.py      # detection; copy best.pt to models/yolo_damage.pt
python training/train_severity.py  # models/severity.pt
python training/train_cost.py      # models/cost.joblib (SYNTHETIC until you swap in real data)

## Step 5 - Run locally
uvicorn app.api:app --reload                 # API docs at :8000/docs
streamlit run app/dashboard.py               # dashboard at :8501

## Step 6 - Deploy
docker compose up --build   # Postgres + API + dashboard
Put on a VPS/cloud VM behind HTTPS (Caddy/Nginx), add auth (e.g. OAuth2/JWT) before real use.
Mobile: the Streamlit UI works in a phone browser (add to home screen). For a native app later,
build a Flutter/React Native client against the same FastAPI endpoints.

## Before production
Evaluate on held-out data (mAP, severity F1, cost MAE), add auth, audit logs, encryption of uploads/PII,
and keep the human officer as final decision-maker.
