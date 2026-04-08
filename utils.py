"""Shared utilities for algorithm subclasses."""

from custom_policies import AdaptiveGradientScaler


def sync_scaler_buffers(source_actor, target_actor):
    """Copy AdaptiveGradientScaler buffers from source to target actor.

    polyak_update only syncs parameters, not buffers. The scaler's running
    stats (g_mean, g_var) are registered as buffers, so we must copy them
    explicitly after each polyak update.
    """
    source_scalers = [m for m in source_actor.modules()
                      if isinstance(m, AdaptiveGradientScaler)]
    target_scalers = [m for m in target_actor.modules()
                      if isinstance(m, AdaptiveGradientScaler)]
    for src, tgt in zip(source_scalers, target_scalers):
        tgt.g_mean.copy_(src.g_mean)
        tgt.g_var.copy_(src.g_var)
