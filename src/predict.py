"""
Predict aqueous solubility (logS) for new SMILES strings.
"""

import joblib
import numpy as np
from src.features import smiles_to_features

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

MODEL_PATH = "data/model.pkl"


def load_model(model_path: str = None):
    return joblib.load(model_path or MODEL_PATH)


def predict(smiles: str | list[str], model=None) -> dict | list[dict]:
    if model is None:
        model = load_model()

    single = isinstance(smiles, str)
    smiles_list = [smiles] if single else smiles

    results = []
    for smi in smiles_list:
        try:
            X = smiles_to_features(smi).reshape(1, -1)
            logS = float(model.predict(X)[0])
            # Convert logS (log10 mol/L) to mg/mL for context
            # logS = log10(mol/L), MW needed for mg/mL — report mol/L
            sol_mol_L = 10 ** logS
            results.append({
                "smiles": smi,
                "logS": round(logS, 3),
                "solubility_mol_L": f"{sol_mol_L:.2e}",
                "category": _categorize(logS),
                "error": None,
            })
        except ValueError as e:
            results.append({"smiles": smi, "logS": None, "error": str(e)})

    return results[0] if single else results


def _categorize(logS: float) -> str:
    if logS > -1:
        return "highly soluble"
    elif logS > -3:
        return "soluble"
    elif logS > -5:
        return "moderately soluble"
    else:
        return "poorly soluble"
