"""
FetalGuard AI — Source Package
==========================================
Modular implementation of the complete Phase 01–8 pipeline.

Modules
-------
data_loader   : Raw data loading and Arrow-safe dtype conversion
preprocessor  : Winsorization, feature engineering, target mapping
pipeline      : End-to-end train/test split, scaling, resampling
model         : Model training, cost-sensitive evaluation
explainer     : SHAP, LIME, PDP/ICE wrappers
fairness      : Proxy-group fairness metrics and AIF360 integration
predictor     : Inference-time prediction and explanation
"""

__version__ = "1.0.0"
__all__ = [
    "data_loader",
    "preprocessor",
    "pipeline",
    "model",
    "explainer",
    "fairness",
    "predictor",
]
