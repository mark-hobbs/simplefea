import numpy as np


class FiniteElement:
    """
    Base class for finite element representation

    Attributes
    ----------
    nodes : list
        List of nodes corresponding to the element

    vertices : np.ndarray
        Array containing the (x, y) coordinates of the element's vertices

    area : float
        Area of the element

    Methods
    -------
    _compute_area():
        Calculate the area of the element
    """

    pass


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

    def __init__(self, nodes, vertices, t, material):
        """
        Initialise a linear triangular element.

        Parameters
        ----------
        nodes : list
            List of nodes corresponding to the element

        vertices : np.ndarray
            Array containing the (x, y) coordinates of the element's vertices

        t : float
            Thickness of the element

        material : Material
            Material assigned to the element
        """
        self.nodes = nodes
        self.vertices = vertices
        self.t = t
        self.material = material
        self.area = self._compute_area()
        self.shape_functions = self._generate_shape_functions()
        self.B = self._compute_B_matrix()
        self.k = self._compute_element_stiffness_matrix()

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

        TODO: not used. Remove?
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
        x_i, y_i = self.vertices[0]
        x_j, y_j = self.vertices[1]
        x_m, y_m = self.vertices[2]

        gamma_i = x_m - x_j
        gamma_j = x_i - x_m
        gamma_m = x_j - x_i

        beta_i = y_j - y_m
        beta_j = y_m - y_i
        beta_m = y_i - y_j

        alpha = np.array(
            [
                [beta_i, 0, beta_j, 0, beta_m, 0],
                [0, gamma_i, 0, gamma_j, 0, gamma_m],
                [gamma_i, beta_i, gamma_j, beta_j, gamma_m, beta_m],
            ]
        )
        return (1 / (2 * self.area)) * alpha

    def _compute_element_stiffness_matrix(self):
        """
        B : np.ndarray
            Strain-displacement matrix

        C : np.ndarray
            Stiffness tensor

        k_e = t_e * A_e * B^T * C * B
        """
        return (
            self.t
            * self.area
            * np.transpose(self.B)
            @ self.material.constitutive_model.C
            @ self.B
        )

    def compute_strain(self, u):
        self.strain = self.B @ u[self.nodes].flatten()
        return self.strain

    def compute_stress(self, strains):
        self.stress = self.material.constitutive_model.C @ strains
        return self.stress
