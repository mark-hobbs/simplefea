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
