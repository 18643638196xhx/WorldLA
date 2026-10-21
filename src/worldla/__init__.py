"""WorldLA; importing this package does not initialize JAX/CUDA."""
__version__ = "0.2.0"


def __getattr__(name):
    if name == "WorldLAConfig":
        from worldla.models.configuration import WorldLAConfig
        return WorldLAConfig
    if name == "WorldLAModel":
        from worldla.models.worldla import WorldLAModel
        return WorldLAModel
    if name == "WorldLAPolicy":
        from worldla.policy import WorldLAPolicy
        return WorldLAPolicy
    raise AttributeError(name)
