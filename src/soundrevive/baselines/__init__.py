from soundrevive.baselines.declick import median_declick
from soundrevive.baselines.passthrough import passthrough
from soundrevive.baselines.spectral_gating import spectral_gate
from soundrevive.baselines.wiener import wiener_filter

BASELINES = {
    "passthrough": passthrough,
    "spectral_gate": spectral_gate,
    "wiener": wiener_filter,
    "median_declick": median_declick,
}

__all__ = ["BASELINES", "median_declick", "passthrough", "spectral_gate", "wiener_filter"]
