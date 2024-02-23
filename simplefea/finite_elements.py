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