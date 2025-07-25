import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
from typing import Literal, Optional

from .stiffness import GlobalStiffnessMatrix


class Model:
    """
    Finite Element Model

    Attributes
    ----------
    mesh : Mesh
        Mesh object containing the nodes and elements of the model

    bc : BoundaryConditions
        Boundary conditions applied to the model

    constraints : np.ndarray
        Constraints applied to the model

    K : np.ndarray
        Global stiffness matrix

    u : np.ndarray, optional
        Nodal displacements, computed after solving F = Ku

    _solved : bool
        Flag indicating whether the model has been solved
    """

    def __init__(self, mesh, boundary_conditions, constraints):
        self.mesh = mesh
        self.bc = boundary_conditions
        self.constraints = constraints

        self.K = GlobalStiffnessMatrix(self.mesh, constraints).K
        self.u: Optional[np.ndarray] = None
        self.strain: Optional[np.ndarray] = None
        self.stress: Optional[np.ndarray] = None
        self._solved: bool = False

    def solve(self) -> np.ndarray:
        """
        F = Ku (b = ax)
        """
        self.u = scipy.sparse.linalg.spsolve(
            scipy.sparse.csr_matrix(self.K), self.bc.f
        ).reshape(-1, 2)
        self._solved = True
        return self.u

    def plot_boundary_conditions(self):
        """
        TODO: add option to plot constraints in x and y direction
        """
        ax = self.mesh.plot(nodes=False, simplices=False)
        ax.plot(
            self.mesh.points[:, 0][self.bc.flag[:, 0] == 1],
            self.mesh.points[:, 1][self.bc.flag[:, 0] == 1],
            "o",
            markersize=2.5,
            c="blue",
        )

    def plot_solution(self, dsf=1):
        """
        Plot the solution (deformed mesh)

        dsf : float
            Displacement scale factor
        """
        displaced_points = self.mesh.points + (self.u * dsf)
        _, ax = plt.subplots(figsize=(8, 8))
        ax.triplot(
            displaced_points[:, 0],
            displaced_points[:, 1],
            self.mesh.triangles.simplices,
            linewidth=0.5,
            color="gray",
        )
        ax.set_aspect("equal")
        ax.set_title("Deformed mesh")

    def plot_u(self, component):
        """
        Plot the specified component of displacement u at nodes

        Parameters
        ----------
        component : int
            0 for x displacement, 1 for y displacement
        """
        if component not in (0, 1):
            raise ValueError(
                "Component must be 0 for x displacement or 1 for y displacement."
            )

        _, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(
            self.mesh.points[:, 0],
            self.mesh.points[:, 1],
            c=self.u[:, component],
            cmap="viridis",
        )
        ax.set_aspect("equal")
        ax.set_title(f"u$_{{{ 'x' if component == 0 else 'y' }}}$")

    def plot_stress(self):
        element_stresses = np.array(
            [
                element.compute_stress(element.compute_strain(self.u))
                for element in self.mesh.elements
            ]
        )[:, 0]

        nodal_stresses = np.zeros(len(self.mesh.points))
        counts = np.zeros(len(self.mesh.points))
        for stress, tri in zip(element_stresses, self.mesh.triangles.simplices):
            nodal_stresses[tri] += stress
            counts[tri] += 1
        nodal_stresses /= counts

        _, ax = plt.subplots(figsize=(8, 8))
        tpc = ax.tripcolor(
            self.mesh.points[:, 0],
            self.mesh.points[:, 1],
            self.mesh.triangles.simplices,
            nodal_stresses,
            cmap="jet",
            shading="gouraud",
        )
        ax.set_aspect("equal")

    def _plot_nodal_field(self):
        """
        Generic helper to plot a nodal scalar field
        """
        pass
