# SvaraAI — Reply Classifier (Scaffold)

This is a ready-to-run scaffold for the SvaraAI internship assignment.

## What's included
- notebook.ipynb (outline placeholder)
- README.md (this file)
- answers.md (short answers template)
- requirements.txt
- Dockerfile
- .gitignore
- app.py (FastAPI)
- serve_model.py (helper to run uvicorn)
- src/ (python modules)
- artifacts/ (place to put your trained models)
- tests/ (basic test)

## Quick setup (local)
1. Create and activate a virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate    # mac/linux
.\.venv\Scripts\activate   # windows
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Place the dataset `reply_classification_dataset.csv` in the project root.

4. Train baseline:
```bash
python -m src.train --data_path ./reply_classification_dataset.csv --out artifacts/baseline_model.pkl
```

5. (Optional) Fine-tune transformer (GPU recommended):
```bash
python -m src.finetune_trainer --data_path ./reply_classification_dataset.csv --output_dir artifacts/hf_model --epochs 3
```

6. Run API:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## What you must run locally
- Running training scripts to produce artifacts/
- (Optional) Fine-tuning on GPU/Colab
