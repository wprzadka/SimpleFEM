from enum import Enum


class MaterialProperty(Enum):
    TestMaterial = (1, 0.3)
    CarbonSteel = (200e9, 0.28)
    Gold = (78e9, 0.44)
    Diamond = (1035e9, 0.29)  # Poisson's ratio = 0.1-0.29
    Polystyrene = (2.8e9, 0.38)  # Young's modulus = 2.8-3.5 GPa
    Silicon = (107e9, 0.27)
    Mica = (34.5e9, 0.205)
