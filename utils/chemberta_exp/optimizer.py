#
############################# Not Necessary ###############################
###########################################################################
#
# # optimizer.py
# """
# Utility for creating (and optionally re‑loading) the optimizer.

# Typical usage
# -------------
# from chemberta_model import ChemBERTaRegressor
# from optimizer       import build_optimizer

# model = ChemBERTaRegressor()
# optim = build_optimizer(model, lr=1e-5, weight_decay=0.01)

# # ── later, if you saved an optimizer state ──
# optim = build_optimizer(model, lr=1e-5, weight_decay=0.01,
#                         ckpt_path="optim_state.pt", device="cuda")
# """
# from pathlib import Path
# from typing import Optional, Union

# import torch
# from torch.optim import AdamW


# def build_optimizer(
#     model: torch.nn.Module,
#     *,
#     lr: float = 1e-5,
#     weight_decay: float = 0.01,
#     eps: float = 1e-8,
#     ckpt_path: Optional[Union[str, Path]] = None,
#     device: Optional[Union[str, torch.device]] = None,
# ) -> AdamW:
#     """
#     Create an AdamW optimizer with separate weight‑decay groups and
#     (optionally) load its state‑dict from `ckpt_path`.

#     Parameters
#     ----------
#     model        : the network whose parameters you want to optimise
#     lr           : learning‑rate
#     weight_decay : wd applied to *most* weights (bias/LayerNorm excluded)
#     eps          : AdamW ε term
#     ckpt_path    : .pt file with previously saved `optimizer.state_dict()`
#     device       : where to map the checkpoint (defaults to CPU / current)
#     """
#     no_decay = {"bias", "LayerNorm.weight"}
#     grouped_params = [
#         {
#             "params": [p for n, p in model.named_parameters()
#                        if not any(nd in n for nd in no_decay)],
#             "weight_decay": weight_decay,
#         },
#         {
#             "params": [p for n, p in model.named_parameters()
#                        if any(nd in n for nd in no_decay)],
#             "weight_decay": 0.0,
#         },
#     ]

#     optimizer = AdamW(grouped_params, lr=lr, eps=eps)

#     # ── optional restore───────────────────────────────────────────────
#     if ckpt_path is not None and Path(ckpt_path).is_file():
#         map_loc = device or "cpu"
#         state = torch.load(ckpt_path, map_location=map_loc)
#         optimizer.load_state_dict(state)
#         print(f"[optimizer] state restored from {ckpt_path}")

#     return optimizer
