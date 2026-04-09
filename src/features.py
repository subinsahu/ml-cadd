"""
Extract molecular features from SMILES strings using RDKit.
Features: physicochemical descriptors + Morgan fingerprints (ECFP4).
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


DESCRIPTOR_NAMES = [
    "MolWt",
    "LogP",
    "NumHDonors",
    "NumHAcceptors",
    "TPSA",
    "NumRotatableBonds",
    "RingCount",
    "NumAromaticRings",
    "FractionCSP3",
    "NumHeavyAtoms",
]

MORGAN_RADIUS = 2
MORGAN_NBITS = 1024


def smiles_to_mol(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return mol


def compute_descriptors(mol) -> np.ndarray:
    return np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        Descriptors.TPSA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        mol.GetNumHeavyAtoms(),
    ], dtype=float)


def compute_morgan_fp(mol) -> np.ndarray:
    gen = GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_NBITS)
    fp = gen.GetFingerprint(mol)
    return np.array(fp, dtype=float)


def smiles_to_features(smiles: str) -> np.ndarray:
    mol = smiles_to_mol(smiles)
    descriptors = compute_descriptors(mol)
    morgan = compute_morgan_fp(mol)
    return np.concatenate([descriptors, morgan])


def featurize_dataframe(df: pd.DataFrame, smiles_col: str = "smiles") -> tuple:
    """
    Returns (X, valid_mask) where X is the feature matrix and
    valid_mask is a boolean array indicating valid SMILES rows.
    """
    features = []
    valid_mask = []

    for smi in df[smiles_col]:
        try:
            features.append(smiles_to_features(smi))
            valid_mask.append(True)
        except ValueError:
            features.append(None)
            valid_mask.append(False)

    valid_mask = np.array(valid_mask)
    X = np.array([f for f in features if f is not None])
    return X, valid_mask


def feature_names() -> list[str]:
    morgan_names = [f"morgan_{i}" for i in range(MORGAN_NBITS)]
    return DESCRIPTOR_NAMES + morgan_names
