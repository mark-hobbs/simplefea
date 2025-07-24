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
