import numpy as np


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

        for i, constraint in enumerate(constraints.flatten()):
            if constraint == 1:
                K[i, :] = 0
                K[:, i] = 0
                K[i, i] = 1

        return K
