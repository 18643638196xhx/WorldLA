"""Bind SAPIEN 3 rendering to the first visible CUDA device in this process."""


def bind_visible_renderer():
    import sapien
    import sapien.core as legacy
    if not hasattr(sapien, 'Device'):
        print('[worldla] Native renderer selection: SAPIEN has no Device API.', flush=True)
        return None
    device = sapien.Device('cuda:0')
    renderer_base = sapien.pysapien.render.SapienRenderer
    legacy_renderer = legacy.SapienRenderer

    class BoundRenderer(legacy_renderer):
        def __init__(self, **unused):
            renderer_base.__init__(self, device)

    def create_scene(engine, config=None):
        if config is not None:
            sapien.physx.set_scene_config(config)
        return sapien.Scene([sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem(device)])

    legacy.SapienRenderer = BoundRenderer
    legacy.Engine.create_scene = create_scene
    print(f'[worldla] Renderer bound to visible cuda:0, PCI={device.pci_string}', flush=True)
    return device.pci_string
