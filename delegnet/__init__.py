# OpenMP load order on macOS: torch must initialise before scikit-learn in the main process,
# while XGBoost (which runs in its own spawned worker, see xgb_worker.py) must never share a
# process with torch. The worker sets DELEGNET_NO_TORCH so importing this package skips torch.
import os

if not os.environ.get("DELEGNET_NO_TORCH"):
    import torch  # noqa: F401
