"""
FraudShield Analytics - API de Detección de Fraude
UPQ | Dr. Isaza | Full-Stack Data Engineer

Vinculación académica:
  - Ecuaciones Diferenciales: la variable 'Time' se interpreta como
    parámetro de un sistema dinámico; el flujo de transacciones sigue
    dV/dt donde V representa el vector de features PCA en el tiempo.
  - Minería de Datos: el modelo Random Forest usa medidas de
    disimilaridad (Gini impurity) en cada nodo para separar clases,
    análogo a la distancia de Minkowski en clasificación supervisada.
  - Seguridad de BD: el audit_hash SHA-256 garantiza integridad;
    cualquier modificación posterior al registro rompe el hash.
"""

import os
import hashlib
import joblib
import numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

load_dotenv()

# ── Configuración ──────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/fraudshield"
)
MODEL_PATH = os.getenv("MODEL_PATH", "modelo_fraude_semana5.pkl")

# ── Carga del modelo ───────────────────────────────────────────────────────────
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Modelo cargado: {MODEL_PATH}")
except FileNotFoundError:
    raise RuntimeError(
        f"❌ No se encontró el modelo en '{MODEL_PATH}'. "
        "Asegúrate de que el archivo .pkl esté en la misma carpeta."
    )

# ── Conexión a PostgreSQL ──────────────────────────────────────────────────────
engine = create_engine(DATABASE_URL)

# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FraudShield Analytics API",
    description="Sistema de detección de fraude en tarjetas de crédito - UPQ",
    version="1.0.0",
)


# ── Esquema de entrada ─────────────────────────────────────────────────────────
class Transaction(BaseModel):
    Time:   float = Field(..., description="Segundos desde la primera transacción")
    Amount: float = Field(..., ge=0, description="Monto de la transacción en USD")
    V1:  float; V2:  float; V3:  float; V4:  float
    V5:  float; V6:  float; V7:  float; V8:  float
    V9:  float; V10: float; V11: float; V12: float
    V13: float; V14: float; V15: float; V16: float
    V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float
    V25: float; V26: float; V27: float; V28: float


# ── Utilidades ─────────────────────────────────────────────────────────────────
def compute_audit_hash(tx: Transaction, prediction: int, ts: datetime) -> str:
    """SHA-256 sobre todos los campos — Seguridad de BD."""
    raw = (
        f"{ts.isoformat()}{tx.Time}{tx.Amount}"
        f"{tx.V1}{tx.V2}{tx.V3}{tx.V4}{tx.V5}{tx.V6}{tx.V7}"
        f"{tx.V8}{tx.V9}{tx.V10}{tx.V11}{tx.V12}{tx.V13}{tx.V14}"
        f"{tx.V15}{tx.V16}{tx.V17}{tx.V18}{tx.V19}{tx.V20}{tx.V21}"
        f"{tx.V22}{tx.V23}{tx.V24}{tx.V25}{tx.V26}{tx.V27}{tx.V28}"
        f"{prediction}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def tx_to_array(tx: Transaction) -> np.ndarray:
    """
    Orden EXACTO verificado en modelo_fraude_semana5.pkl:
    [Time, V1..V28, Amount] — sin pandas, numpy puro.
    """
    return np.array([[
        tx.Time,
        tx.V1,  tx.V2,  tx.V3,  tx.V4,  tx.V5,  tx.V6,  tx.V7,
        tx.V8,  tx.V9,  tx.V10, tx.V11, tx.V12, tx.V13, tx.V14,
        tx.V15, tx.V16, tx.V17, tx.V18, tx.V19, tx.V20, tx.V21,
        tx.V22, tx.V23, tx.V24, tx.V25, tx.V26, tx.V27, tx.V28,
        tx.Amount
    ]])


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "sistema": "FraudShield Analytics", "version": "1.0.0"}


@app.post("/predict", tags=["Predicción"])
def predict(tx: Transaction):
    """Recibe transacción, predice fraude, persiste con audit_hash."""
    features   = tx_to_array(tx)
    prediction = int(model.predict(features)[0])
    proba      = float(model.predict_proba(features)[0][1])
    ts         = datetime.now(timezone.utc)
    audit_hash = compute_audit_hash(tx, prediction, ts)

    try:
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO fraud_logs
                    (timestamp, amount, time_delta, prediction, audit_hash)
                VALUES
                    (:ts, :amount, :time_delta, :prediction, :audit_hash)
            """), {
                "ts":         ts,
                "amount":     tx.Amount,
                "time_delta": tx.Time,
                "prediction": prediction,
                "audit_hash": audit_hash,
            })
            conn.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al guardar en BD: {e}")

    return {
        "prediction":        prediction,
        "label":             "FRAUDE" if prediction == 1 else "LEGÍTIMA",
        "fraud_probability": round(proba, 4),
        "audit_hash":        audit_hash,
        "timestamp":         ts.isoformat(),
    }


@app.get("/stats", tags=["Monitoreo"])
def stats():
    """Resumen para monitoreo."""
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT COUNT(*), SUM(prediction), ROUND(AVG(amount)::NUMERIC, 2)
            FROM fraud_logs
        """)).fetchone()
    return {
        "total_transacciones": row[0],
        "total_fraudes":       row[1],
        "monto_promedio_usd":  float(row[2]) if row[2] else 0.0,
    }
