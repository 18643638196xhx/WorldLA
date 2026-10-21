"""WorldLA models; public classes are imported lazily."""


def __getattr__(name):
    if name == 'WorldLAModel':
        from worldla.models.worldla import WorldLAModel
        return WorldLAModel
    if name == 'WorldLAConfig':
        from worldla.models.configuration import WorldLAConfig
        return WorldLAConfig
    if name == 'WorldLABackbone':
        from worldla.models.backbone import WorldLABackbone
        return WorldLABackbone
    raise AttributeError(name)
