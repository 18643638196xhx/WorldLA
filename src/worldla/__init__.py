"""WorldLA; importing this package does not initialize JAX/CUDA."""
__version__ = "0.1.0"


def __getattr__(name):
    if name == "WorldLAPolicy":
        from worldla.policy import WorldLAPolicy
        return WorldLAPolicy
    raise AttributeError(name)
