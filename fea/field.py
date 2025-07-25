import numpy as np
import matplotlib.pyplot as plt


class Field:
    """
    Represents a scalar or vector field on a finite element mesh

    Attributes
    ----------
    values : np.ndarray
        Field values. Shape can be (n_nodes,) for scalar nodal fields
        or (n_nodes, dim) for vector nodal fields.

    location : Literal["nodal", "elemental"]
        Location of the field values

    name : str
        Descriptive name for the field (e.g., "displacement", "stress")
    """

    def __init__(
        self,
        values: np.ndarray,
        location: Literal["nodal", "elemental"],
        name: str = "",
    ):
        self.values = np.array(values)
        self.location = location
        self.name = name

    def magnitude(self) -> np.ndarray:
        """
        Return the magnitude if the field is vector-valued
        """
        if self.values.ndim == 1:
            return self.values
        return np.linalg.norm(self.values, axis=1)

    def plot(
        self, mesh, component: Optional[int] = None, cmap="viridis", shading="gouraud"
    ):
        """
        Plot the field using the mesh.
        - If the field is scalar, plot directly.
        - If the field is vector, specify a component (0=x, 1=y).
        """
        if self.location != "nodal":
            raise NotImplementedError("Elemental plotting not implemented yet.")

        data = self.values if self.values.ndim == 1 else self.values[:, component]
        _, ax = plt.subplots(figsize=(8, 8))
        tpc = ax.tripcolor(
            mesh.points[:, 0],
            mesh.points[:, 1],
            mesh.triangles.simplices,
            data,
            cmap=cmap,
            shading=shading,
        )
        ax.set_aspect("equal")
        ax.set_title(self.name or "Field")
        plt.colorbar(tpc, ax=ax)
        return ax
