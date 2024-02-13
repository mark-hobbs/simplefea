import numpy as np
import scipy
import matplotlib.pyplot as plt


class Model:
    def __init__(self, mesh, boundary_conditions):
        self.mesh = mesh
        self.boundary_conditions = boundary_conditions

        self.K = GlobalStiffnessMatrix(self.mesh)

    def solve(self):
        return np.linalg.solve(self.K.K, self.boundary_conditions.f)

    def plot(self):
        """
        Plot the solution
        """
        pass


class Mesh:
    def __init__(self, points, material, t):
        self.points = points
        self.material = material
        self.t = t
        self.n_nodes = len(self.points)
        self.triangles = scipy.spatial.Delaunay(self.points)
        self.elements = self._build_elements(material.constitutive_model)

    def _build_elements(self, constitutive_model):
        """
        Built a list of triangular elements
        """
        return [
            TriangularElement(nodes, self.points[nodes], self.t, constitutive_model)
            for nodes in self.triangles.simplices
        ]

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
        self.unit_vector = unit_vector
        self.magnitude = magnitude
        self.f = (unit_vector * magnitude).reshape(-1, 1)


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

    def __init__(self, nodes, vertices, t, constitutive_model):
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
        self.B = self._compute_B_matrix()
        self.k = self._compute_element_stiffness_matrix(t, constitutive_model)

    def _compute_area(self):
        """
        Calculate the area of the triangular element using the cross product

        Returns
        -------
        area : float
            Area of the triangular element
        """
        return 0.5 * abs(
            np.cross(
                self.vertices[1] - self.vertices[0], self.vertices[2] - self.vertices[0]
            )
        )

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

    @staticmethod
    def _compute_B_matrix():
        """
        Calculate the strain-displacement matrix
        """
        return np.array(
            [[-1, 0, 1, 0, 0, 0], [0, -1, 0, 0, 0, 1], [-1, -1, 0, 1, 1, 0]]
        )

    def _compute_element_stiffness_matrix(self, t, constitutive_model):
        """
        B : np.ndarray

        C : np.ndarray
            Stiffness tensor

        k_e = t_e * A_e * B^T * C * B
        """
        return t * self.area * np.transpose(self.B) @ constitutive_model.C @ self.B


class GlobalStiffnessMatrix:
    """
    Global Stiffness Matrix class

    - Assemble the global stiffness matrix by summing contributions from
      individual elements.
    - Account for boundary conditions during assembly.
    """

    def __init__(self, mesh):
        self.K = self._assemble_K(mesh)

    def _assemble_K(self, mesh):
        """
        Assemble the global stiffness matrix by summing contribution from
        individual elements

        Returns
        -------
        K : np.ndarray
            Global stiffness matrix
        """
        K = np.zeros((2 * mesh.n_nodes, 2 * mesh.n_nodes))

        for element in mesh.elements:
            k_e = element.k
            for i in range(len(element.nodes)):
                for j in range(len(element.nodes)):
                    I = element.nodes[i]
                    J = element.nodes[j]
                    K[2 * I : 2 * I + 2, 2 * J : 2 * J + 2] += k_e[
                        2 * i : 2 * i + 2, 2 * j : 2 * j + 2
                    ]
        return K


class Material:
    """
    Attributes
    ----------
    E : float
        Young's modulus

    v : float
        Poisson's ratio

    constitutive_model : ConstitutiveModel
        Material constitutive model
    """

    def __init__(self, E, v, constitutive_model=None, **kwargs):
        self.E = E
        self.v = v
        if constitutive_model:
            self.constitutive_model = constitutive_model(self, **kwargs)
        else:
            self.constitutive_model = None

    def set_constitutive_model(self, constitutive_model_cls, **kwargs):
        self.constitutive_model = constitutive_model_cls(self, **kwargs)


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
        return factor * np.array([[1 - v, v, 0], [v, 1 - v, 0], [0, 0, 0.5 - v]])
