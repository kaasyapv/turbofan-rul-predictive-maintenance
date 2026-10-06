"""FastAPI service. Run with: uvicorn src.serve:app"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .predict import load, predict

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
models = {name: load(name) for name in SUBSETS}
app = FastAPI(title="turbofan rul")


class Reading(BaseModel):
    cycle: float
    settings: list[float] = Field(min_length=3, max_length=3)
    sensors: list[float] = Field(min_length=21, max_length=21)


class Request(BaseModel):
    subset: str
    readings: list[Reading] = Field(min_length=1)


@app.get("/health")
def health():
    return {"status": "ok", "subsets": SUBSETS}


@app.post("/predict")
def rul(req: Request):
    if req.subset not in models:
        raise HTTPException(422, f"unknown subset {req.subset}")
    rows = [[r.cycle, *r.settings, *r.sensors] for r in req.readings]
    mean, spread = predict(*models[req.subset], rows)
    return {"rul_cycles": mean, "model_spread": spread, "maintenance_flag": mean <= 30,
            "cycles_seen": len(rows)}
