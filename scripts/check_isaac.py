"""Bounded GPU startup check; no robot parameters or training are assumed."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
try:
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab_physx.physics import PhysxCfg
    sim = SimulationContext(SimulationCfg(dt=0.002, physics=PhysxCfg(), device="cuda:0"))
    sim.reset()
    for _ in range(10):
        sim.step(render=False)
    print("PAWWEAVER_PHYSX_STARTUP_OK", flush=True)
finally:
    launcher.app.close()
