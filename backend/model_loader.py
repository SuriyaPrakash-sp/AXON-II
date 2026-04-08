"""
model_loader.py — load model once at Flask startup and expose globally

Usage in routes.py:
    from model_loader import get_model_bundle
    bundle = get_model_bundle()
    result = bundle.predict(raw_input)
"""

import sys
from pathlib import Path

# Allow importing from ml/ directory
ML_DIR = Path(__file__).parent.parent / "ml"
sys.path.insert(0, str(ML_DIR))

from predict import load_model, run_inference

_bundle = None   # module-level singleton


class ModelBundle:
    """Wraps the model, adj_norm, norm_params, and device in one object."""

    def __init__(self):
        print("[model_loader] Loading model …")
        self.model, self.adj_norm, self.norm_params, self.device = load_model()
        print("[model_loader] Model ready.")

    def predict(self, raw_input):
        """
        Run inference and return color dict.

        Args:
            raw_input: dict (single snapshot) or list of dicts (sequence)

        Returns:
            dict  {"N1": "GREEN", ...}
        """
        return run_inference(
            raw_input,
            self.model,
            self.adj_norm,
            self.norm_params,
            self.device,
        )


def get_model_bundle() -> ModelBundle:
    """Return the singleton ModelBundle, initialising it on first call."""
    global _bundle
    if _bundle is None:
        _bundle = ModelBundle()
    return _bundle