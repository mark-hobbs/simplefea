import numpy as np


class ConstitutiveModel:
    """
    Base class for constitutive models
    """

    def __init__(self, material, **kwargs):
        self.material = material


class LinearElasticModel(ConstitutiveModel):
    """
    Linear elastic material

    stress tensor = stiffness tensor x strain tensor
    """

    def __init__(self, material, **kwargs):
        super().__init__(material, **kwargs)
        self.C = self._compute_C(material.E, material.v)

    def _compute_C(self, E, v):
        """
        Compute the stiffness tensor C : plane strain
        """
        factor = E / ((1 + v) * (1 - 2 * v))
        return factor * np.array(
            [[1 - v, v, 0], [v, 1 - v, 0], [0, 0, (1 - 2 * v) / 2]]
        )

    def _compute_C(self, E, v):
        """
        Compute the stiffness tensor C : plane stress
        """
        factor = E / (1 - v**2)
        return factor * np.array([[1, v, 0], [v, 1, 0], [0, 0, (1 - v) / 2]])
