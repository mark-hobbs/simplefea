import scipy
import numpy as np
import matplotlib.pyplot as plt

from .elements import TriangularElement


class Mesh:
    def __init__(self, points, material, t):
        self.points = points
        self.material = material
        self.t = t
        self.n_nodes = len(self.points)
        self.triangles = scipy.spatial.Delaunay(self.points)
        self.elements = self._build_elements()

    def _build_elements(self):
        """
        Built a list of triangular elements
        """
        return [
            TriangularElement(nodes, self.points[nodes], self.t, self.material)
            for nodes in self.triangles.simplices
        ]

    def info(self):
        """
        Print the number of nodes and number of elements
        """
        print(f"Number of nodes: {self.n_nodes}")
        print(f"Number of elements: {len(self.elements)}")

    def plot(self, nodes=False, simplices=False):
        """
        Plot the undeformed mesh

        Parameters
        ----------
        nodes : bool, optional
            If True, plot node indices next to the nodes. Defaults to False.

        simplices : bool, optional
            If True, plot simplex indices next to the centroids of each
            simplex. Defaults to False.
        """
        _, ax = plt.subplots(figsize=(8, 8))
        ax.triplot(
            self.points[:, 0],
            self.points[:, 1],
            self.triangles.simplices,
            linewidth=0.5,
            color="gray",
        )
        ax.plot(
            self.points[:, 0],
            self.points[:, 1],
            "o",
            markersize=2.5,
            markeredgecolor="black",
        )

        if nodes:
            for i, p in enumerate(self.points):
                ax.text(p[0], p[1], f"{i}", fontsize=10, ha="right", va="bottom")

        if simplices:
            for j, t in enumerate(self.triangles.simplices):
                p = np.mean(self.points[t], axis=0)
                ax.text(p[0], p[1], f"{j}", color="gray", fontsize=8)

        ax.set_aspect("equal")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Mesh")
        return ax
