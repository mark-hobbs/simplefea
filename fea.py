import numpy as np
import scipy
import matplotlib.pyplot as plt


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
        stresses = np.zeros((len(self.mesh.elements), 3))

        for i, element in enumerate(self.mesh.elements):
            strain = element.compute_strain(self.u)
            stresses[i] = element.compute_stress(strain)

        _, ax = plt.subplots(figsize=(8, 8))
        ax.tripcolor(
            self.mesh.points[:, 0],
            self.mesh.points[:, 1],
            self.mesh.triangles.simplices,
            facecolors=stresses[:, 0],
            cmap="jet",
        )
        ax.set_aspect("equal")


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


class BoundaryConditions:
    """
    TODO - AppliedForce(BoundaryConditions)
           AppliedDisplacement(BoundaryConditions)
           Constraint(BoundaryConditions)

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
        self.f = self._build_f()

    def _build_f(self):
        return (self.unit_vector * self.magnitude).flatten()

    def plot(self):
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


class GlobalStiffnessMatrix:
    """
    Global Stiffness Matrix class

    - Assemble the global stiffness matrix by summing contributions from
      individual elements.
    - Account for boundary conditions during assembly.
    """

    def __init__(self, mesh, constraints):
        self.K = self._assemble_K(mesh, constraints)

    def _assemble_K(self, mesh, constraints):
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
            for i in range(len(element.nodes)):
                for j in range(len(element.nodes)):
                    I = element.nodes[i]
                    J = element.nodes[j]
                    K[(2 * I) : (2 * I) + 2, (2 * J) : (2 * J) + 2] += element.k[
                        (2 * i) : (2 * i) + 2, (2 * j) : (2 * j) + 2
                    ]

        for i, node in enumerate(constraints.flatten()):
            if node == 1:
                K[i, :] = 0
                K[:, i] = 0
                K[i, i] = 1

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
        return factor * np.array(
            [[1 - v, v, 0], [v, 1 - v, 0], [0, 0, (1 - 2 * v) / 2]]
        )

    def _compute_C(self, E, v):
        """
        Compute the stiffness tensor C : plane stress
        """
        factor = E / (1 - v**2)
        return factor * np.array([[1, v, 0], [v, 1, 0], [0, 0, (1 - v) / 2]])