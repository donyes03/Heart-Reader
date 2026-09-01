import os
import sys
import glob
import random
import io
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ─── Configuration des chemins ────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).resolve().parent
ROOT = FRONTEND_DIR.parent
sys.path.insert(0, str(ROOT))

# ─── Déclaration manuelle ─────────────────────────────────────────────────
SUPERCLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

CLASS_DESCRIPTIONS = {
    "NORM": "Normal Sinus Rhythm",
    "MI":   "Myocardial Infarction",
    "STTC": "ST/T Change",
    "CD":   "Conduction Disturbance",
    "HYP":  "Hypertrophy",
}

app = FastAPI(title="Heart Reader Edge AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Chargement du modèle ONNX ───────────────────────────────────────────
try:
    import onnxruntime as ort
    MODEL_PATH = str(ROOT / "results" / "fusion_model.onnx")
    if os.path.exists(MODEL_PATH):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        session = ort.InferenceSession(MODEL_PATH, sess_options=opts)
        print("✅ Modèle ONNX chargé avec succès.")
    else:
        session = None
        print("⚠️ Fichier ONNX introuvable. Inférence simulée.")
except ImportError:
    session = None
    print("⚠️ onnxruntime n'est pas installé. Inférence simulée.")

# ─── Fonction de parsing CSV ────────────────────────────────────────────
def parse_ecg_csv(text_content):
    first_line = text_content.split("\n")[0].strip()
    has_header = False
    try:
        [float(x.strip()) for x in first_line.split(",")]
    except ValueError:
        has_header = True
        
    if has_header:
        df = pd.read_csv(io.StringIO(text_content))
    else:
        df = pd.read_csv(io.StringIO(text_content), header=None)
        
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    
    features = None
    if df.shape[1] > 12:
        signal = df.iloc[:, :12].values.astype(np.float32)
        features = df.iloc[0, 12:].values.astype(np.float32)
        features = np.nan_to_num(features, nan=0.0)
    else:
        signal = df.values.astype(np.float32)
        
    if signal.shape[1] != 12 and signal.shape[0] == 12:
        signal = signal.T
        
    if signal.shape[0] < 1000:
        pad = np.zeros((1000 - signal.shape[0], 12), dtype=np.float32)
        signal = np.vstack([signal, pad])
    elif signal.shape[0] > 1000:
        signal = signal[:1000]
        
    return signal, features

# ─── Routes API ──────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": session is not None,
        "test_samples_available": 100,
        "classes": SUPERCLASSES,
    }

@app.get("/api/model-info")
def model_info():
    return {
        "classes": SUPERCLASSES,
        "class_descriptions": CLASS_DESCRIPTIONS,
        "num_parameters": 18540000, 
        "num_features": 1313,
        "input_shape": {"leads": 12, "samples": 1000, "duration_sec": 10, "sampling_rate": 100},
        "lead_names": LEAD_NAMES,
        "performance": {
            # Tes vrais résultats de Kaggle !
            "test_macro_auc": 0.9286, 
            "test_f1": 0.73, 
            # Remplacement par tes F1-Scores par classe
            "per_class_f1": {"NORM": 0.86, "MI": 0.72, "STTC": 0.76, "CD": 0.76, "HYP": 0.54}
        }
    }

@app.get("/api/random-sample")
def random_sample():
    # ROOT pointe vers le dossier 'heart_reader'. On cherche test_files à l'intérieur.
    test_dir = ROOT / "test_files"
    
    if not test_dir.exists():
        raise HTTPException(status_code=404, detail=f"Dossier introuvable : {test_dir}")
        
    csv_files = glob.glob(str(test_dir / "*.csv"))
    if not csv_files:
        raise HTTPException(status_code=404, detail=f"Aucun fichier CSV trouvé dans : {test_dir}")
        
    random_file = random.choice(csv_files)
    filename = os.path.basename(random_file)
    
    with open(random_file, "r", encoding="utf-8") as f:
        contents = f.read()
        
    signal, features = parse_ecg_csv(contents)
    pred = _predict_onnx(signal, features)

    return {
        "sample_index": filename,
        "signal": signal.tolist(),
        "ground_truth": ["Inconnu"],
        "predictions": pred,
        "lead_names": LEAD_NAMES,
    }

@app.post("/api/predict")
async def predict_upload(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        text = contents.decode("utf-8").strip()
        
        signal, features = parse_ecg_csv(text)
        pred = _predict_onnx(signal, features)

        return {
            "filename": file.filename,
            "signal_shape": list(signal.shape),
            "signal": signal.tolist(),
            "predictions": pred,
            "lead_names": LEAD_NAMES,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def _predict_onnx(signal: np.ndarray, features: np.ndarray) -> dict:
    signal_input = np.expand_dims(signal, axis=0).astype(np.float32)
    
    if features is None:
        features = np.zeros(1313, dtype=np.float32)
    features_input = np.expand_dims(features, axis=0).astype(np.float32)

    if session is not None:
        input_names = [inp.name for inp in session.get_inputs()]
        
        inputs = {input_names[0]: signal_input}
        if len(input_names) > 1:
            inputs[input_names[1]] = features_input
            
        outputs = session.run(None, inputs)
        logits = outputs[0][0]
    else:
        logits = np.random.uniform(-1.5, 1.5, size=(5,))

    probs = 1.0 / (1.0 + np.exp(-logits))

    result = {}
    for i, cls_name in enumerate(SUPERCLASSES):
        p = float(probs[i])
        result[cls_name] = {
            "probability": round(p, 4),
            "predicted": p > 0.5,
            "description": CLASS_DESCRIPTIONS[cls_name],
        }
    return result