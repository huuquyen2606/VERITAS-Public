import uvicorn
import argparse
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Any

# --- IMPORT SHARED LOGIC ---
from baseline_detectors import MODEL_REGISTRY
from baseline_detectors.ensemble import EnsembleManager

# Initialize API
app = FastAPI(
    title="AI Malware Detection Server", description="API interface for ai_pkg"
)


# ==========================================
# 0. HELPER: SANITIZE BINARY DATA
# ==========================================
def clean_output(data):
    """
    Recursively walk through the dictionary/list.
    If we find raw 'bytes', convert them to a Hex string.
    This prevents UnicodeDecodeError when sending JSON.
    """
    if isinstance(data, dict):
        return {k: clean_output(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_output(i) for i in data]
    elif isinstance(data, bytes):
        return data.hex()
    else:
        return data


# ==========================================
# 1. GLOBAL STATE
# ==========================================
server_state = {
    "manager": None,
    "single_model": None,
    "mode": "empty",
}


# ==========================================
# 2. DATA MODELS
# ==========================================
class LoadConfig(BaseModel):
    models: Optional[List[str]] = None
    model: Optional[str] = None
    weights: Optional[Dict[str, str]] = None
    single_weights: Optional[str] = None


class PredictRequest(BaseModel):
    input: str


class EvadeRequest(BaseModel):
    adv_dirs: List[str]
    map: Dict[str, Any]


# ==========================================
# 3. API ENDPOINTS
# ==========================================


@app.get("/status")
def get_status():
    if server_state["mode"] == "ensemble":
        names = server_state["manager"].model_names
        return {"status": "active", "mode": "ensemble", "models": names}
    elif server_state["mode"] == "single":
        name = server_state["single_model"].__class__.__name__
        return {"status": "active", "mode": "single", "model": name}
    else:
        return {"status": "idle", "message": "No models loaded. POST to /load"}


@app.post("/load")
def load_engine(config: LoadConfig):
    print(f"\n[*] API Request: Loading configuration...")
    try:
        if config.models:
            manager = EnsembleManager(config.models, config.weights or {})
            server_state["manager"] = manager
            server_state["single_model"] = None
            server_state["mode"] = "ensemble"
            return {"message": f"Ensemble loaded with {len(config.models)} models."}

        elif config.model:
            if config.model not in MODEL_REGISTRY:
                raise HTTPException(
                    status_code=404, detail=f"Model '{config.model}' not found."
                )

            ModelClass = MODEL_REGISTRY[config.model]
            model_instance = ModelClass()
            if config.single_weights:
                model_instance.load_weights(config.single_weights)

            server_state["single_model"] = model_instance
            server_state["manager"] = None
            server_state["mode"] = "single"
            return {"message": f"Single model '{config.model}' loaded."}
        else:
            raise HTTPException(
                status_code=400, detail="Config must provide 'model' or 'models'."
            )

    except Exception as e:
        print(f"[!] Server Load Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def run_prediction(req: PredictRequest):
    if server_state["mode"] == "empty":
        raise HTTPException(status_code=400, detail="Server is idle. Call /load first.")

    input_path = req.input
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail=f"Path not found: {input_path}")

    # Safety Check for Empty Files (Bypass for .npz as they have specific structures)
    if (
        os.path.isfile(input_path)
        and os.path.getsize(input_path) == 0
        and not input_path.endswith(".npz")
    ):
        return {
            "status": "error",
            "results": {
                input_path: {
                    "SingleModel": {
                        "label": "ERROR",
                        "Virus": 0.0,
                        "details": "Empty File",
                    }
                }
            },
        }

    try:
        results = {}
        # 1. Ensemble Logic
        if server_state["mode"] == "ensemble":
            results = server_state["manager"].predict(input_path)

        # 2. Single Logic
        else:
            model = server_state["single_model"]
            try:
                raw_data = model.predict(input_path)
                # Normalize structure
                model_name = "SingleModel"
                for fpath, data in raw_data.items():
                    results[fpath] = {model_name: data}
            except Exception as inner_e:
                print(f"[!] Model Prediction Crash: {inner_e}")
                results[input_path] = {
                    "SingleModel": {
                        "label": "ERROR",
                        "Virus": 1.0,
                        "details": str(inner_e),
                    }
                }

        # --- FIX APPLIED HERE ---
        # Clean the results to ensure no raw bytes exist
        clean_results = clean_output(results)

        return {"status": "success", "results": clean_results}

    except Exception as e:
        # Catch generic server errors
        print(f"[!] Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evade")
def run_evasion_test(req: EvadeRequest):
    if server_state["mode"] == "empty":
        raise HTTPException(status_code=400, detail="Server is idle. Call /load first.")

    try:
        label_map = {k: int(v) for k, v in req.map.items()}
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Label mapping values must be integers."
        )

    try:
        report = {}
        if server_state["mode"] == "ensemble":
            report = server_state["manager"].evade(req.adv_dirs, label_map)
        else:
            model = server_state["single_model"]
            raw_report = model.evade(req.adv_dirs, label_map)
            for tech, stats in raw_report.items():
                report[tech] = {"SingleModel": stats}

        # --- FIX APPLIED HERE TOO ---
        return {"status": "success", "report": clean_output(report)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the AI Malware Analysis Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    args = parser.parse_args()

    print(f"[*] Starting AI Server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
