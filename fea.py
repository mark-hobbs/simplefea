import numpy as np
import scipy
import matplotlib.pyplot as plt

class Model:
    def __init__(self, mesh, boundary_conditions):
        self.mesh = mesh
        self.boundary_conditions = boundary_conditions

        self.K = GlobalStiffnessMatrix(self.mesh)

    def solve(self):
        return np.linalg.solve(K, f)

    def plot(self):
        """
        Plot the solution
        """
        pass


class Geometry:
    pass


class Mesh:
    """
    Triangular elements
    """
    def __init__(self, nodes):
        self.mesh = scipy.spatial.Delaunay(nodes)


class LocalStiffnessMatrix:
    pass


class GlobalStiffnessMatrix:
    pass


class Material:
    """
    Attributes
    ----------
    E : float
    Young's modulus

    v : float
        Poisson's ratio
    """

    def __init__(self, E, v):
        self.E = E
        self.v = v


class ConstitutiveModel:
    """
    Linear elastic material

    stress tensor = stiffness tensor x strain tensor
    """

    def __init__(self, material):
        self.C = self._compute_C(material.E, material.C)

    def _compute_C(E, v):
        """
        Compute the stiffness tensor C : plane strain
        """
        return (
            E
            / ((1 + v) * (1 - 2 * v))
            * np.array([[1 - v, v, 0], [v, 1 - v, 0], [0, 0, .5 - v]])
        )


class ShapeFunction:
    pass


class BoundaryConditions:
    pass