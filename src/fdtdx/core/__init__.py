from .physics.losses import metric_efficiency
from .physics.metrics import compute_energy, poynting_flux
from .wavelength import WaveCharacter
from .switch import OnOffSwitch

__all__ = ["WaveCharacter", "metric_efficiency", "compute_energy", "poynting_flux", "OnOffSwitch"]
