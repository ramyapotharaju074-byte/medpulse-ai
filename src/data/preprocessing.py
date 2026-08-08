import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from pathlib import Path
import joblib
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

def build_preprocessor_pipeline() -> ColumnTransformer:
    """
    Creates a scikit-learn ColumnTransformer pipeline with median imputer + scaler for numericals,
    and most-frequent imputer + one-hot encoder for categoricals.
    """
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, config.NUMERICAL_FEATURES),
            ('cat', cat_pipeline, config.CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor

def get_transformed_feature_names(preprocessor: ColumnTransformer) -> list:
    """
    Extracts explicit feature names after OneHotEncoding transformation.
    """
    output_features = list(config.NUMERICAL_FEATURES)
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    
    if hasattr(cat_encoder, 'get_feature_names_out'):
        encoded_cats = list(cat_encoder.get_feature_names_out(config.CATEGORICAL_FEATURES))
        output_features.extend(encoded_cats)
    return output_features

def prepare_data(df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42) -> Dict[str, Any]:
    """
    Cleans, splits, and transforms raw medical DataFrame.
    Returns dictionary with raw split, transformed split, and fitted preprocessor.
    """
    X = df[config.FEATURE_COLUMNS].copy()
    y = df[config.TARGET_COL].copy()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = build_preprocessor_pipeline()
    X_train_transformed = preprocessor.fit_transform(X_train_raw)
    X_test_transformed = preprocessor.transform(X_test_raw)

    transformed_feature_names = get_transformed_feature_names(preprocessor)

    # Save preprocessor artifact
    preprocessor_path = config.MODELS_DIR / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)
    print(f"[+] Saved preprocessing pipeline to {preprocessor_path}")

    return {
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "X_train": X_train_transformed,
        "X_test": X_test_transformed,
        "y_train": y_train.values,
        "y_test": y_test.values,
        "feature_names": transformed_feature_names,
        "preprocessor": preprocessor
    }

def load_preprocessor(model_dir: Path = config.MODELS_DIR) -> ColumnTransformer:
    path = model_dir / "preprocessor.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found at {path}. Run data pipeline first.")
    return joblib.load(path)
