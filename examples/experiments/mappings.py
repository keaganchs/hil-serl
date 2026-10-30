# Lazy: each config imports its robot stack (franka_env, pynput, ...), which the P0 machines don't have.
def _lazy(module, name="TrainConfig"):
    def make():
        import importlib

        return getattr(importlib.import_module(module), name)()

    return make


CONFIG_MAPPING = {
    "ram_insertion": _lazy("experiments.ram_insertion.config"),
    "usb_pickup_insertion": _lazy("experiments.usb_pickup_insertion.config"),
    "object_handover": _lazy("experiments.object_handover.config"),
    "egg_flip": _lazy("experiments.egg_flip.config"),
    "p0_ball": _lazy("experiments.p0_ball.config"),
}
