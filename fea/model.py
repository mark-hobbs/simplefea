import numpy as np
import matplotlib.pyplot as plt

from .stiffness import GlobalStiffnessMatrix


class Model:
    def __init__(self, mesh, boundary_conditions, constraints):
        self.mesh = mesh
        self.bc = boundary_conditions
        self.constraints = constraints

        self.K = GlobalStiffnessMatrix(self.mesh, constraints).K

    def solve(self):
        """
        F = Ku (b = ax)
        """
        self.u = np.linalg.solve(self.K, self.bc.f).reshape(-1, 2)
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
        stresses = np.array(
            [
                element.compute_stress(element.compute_strain(self.u))
                for element in self.mesh.elements
            ]
        )

        _, ax = plt.subplots(figsize=(8, 8))
        ax.tripcolor(
            self.mesh.points[:, 0],
            self.mesh.points[:, 1],
            self.mesh.triangles.simplices,
            facecolors=stresses[:, 0],
            cmap="jet",
        )
        ax.set_aspect("equal")
