from enum import Enum
from pydantic import BaseSettings

class WhiteningMethod(Enum):
    Basic='Basic'
    Gaussian='Gaussian'


class GaussianWhiteningParams(BaseSettings):
    smoothing_kernel_size: float = 5
    water_level: float = 1e-3
