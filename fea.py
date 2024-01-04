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


class BoundaryConditions:
    """
    Attributes
    ----------
    flag : ndarray
        0 - no boundary condition
        1 - the node is subject to a boundary condition
    """
    def __init__(self, flag, unit_vector, magnitude):
        self.flag = flag
        self.unit_vector
        self.magnitude


class Mesh:
    def __init__(self, points):
        self.points = points
        self.triangles = scipy.spatial.Delaunay(self.points)
        self.elements = self._build_elements()

    def _build_elements(self):
        """
        Built a list of triangular elements
        """
        elements = []
        tri = scipy.spatial.Delaunay(self.points)
        for nodes in tri.simplices:
            elements.append(TriangularElement(nodes, self.points[nodes]))
        return elements

    def plot(self):
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
        ax.set_aspect("equal")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Mesh")


class TriangularElement:
    """
    Linear triangular element class

    Attributes
    ----------
    nodes : list
        List of nodes corresponding to the element

    vertices : np.ndarray
        Array containing the (x, y) coordinates of the element's vertices

    area : float
        Area of the triangular element

    Methods
    -------
    _compute_area():
        Calculate the area of the triangular element using the cross product

    """
    def __init__(self, nodes, vertices):
        """
        Initialise a linear triangular element.

        Parameters
        ----------
        nodes : list
            List of nodes corresponding to the element

        vertices : np.ndarray
            Array containing the (x, y) coordinates of the element's vertices
        """
        self.nodes = nodes
        self.vertices = vertices
        self.area = self._compute_area()
        self.shape_functions = self._generate_shape_functions()

    def _compute_area(self):
        """
        Calculate the area of the triangular element using the cross product

        Returns
        -------
        area : float
            Area of the triangular element
        """
        return 0.5 * abs(np.cross(self.vertices[1] - self.vertices[0],
                                  self.vertices[2] - self.vertices[0]))
    
    @staticmethod
    def _generate_shape_functions():
        """
        Generate shape functions for linear triangular element

        Returns
        -------
        list of functions
            List of shape functions for the element
        """
        N1 = lambda xi, eta: 1 - xi - eta
        N2 = lambda xi, eta: xi
        N3 = lambda xi, eta: eta
        return [N1, N2, N3]
    
    def _compute_shape_function_derivatives():
        pass

    def _compute_B_matrix(self):
        """
        Calculate the strain-displacement matrix
        """
        pass

    def _compute_element_stiffness_matrix(self):
        """
        B : np.ndarray

        C : np.ndarray
            Stiffness tensor

        k_e = t_e * A_e * B^T * C * B
        """
        pass



class ShapeFunction:
    """
    This class is probably not needed as the TriangularElement class captures
    the relevant functionality
    """
    pass


class LocalStiffnessMatrix:
    """
    This class is probably not needed as the TriangularElement class captures
    the relevant functionality
    """
    pass


class GlobalStiffnessMatrix:
    """
    Global Stiffness Matrix class

    - Assemble the global stiffness matrix by summing contributions from 
      individual elements.
    - Account for boundary conditions during assembly.
    """
    def __init__(self, mesh):
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
        factor =  E / ((1 + v) * (1 - 2 * v))
        return factor * np.array([[1 - v, v, 0], [v, 1 - v, 0], [0, 0, 0.5 - v]])
