"""Bounded end-to-end PhysX PPO test on the synthetic box fixture, never a hardware policy."""
import argparse
import json
from pathlib import Path
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--randomize",action="store_true")
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
launcher=AppLauncher(args)
try:
    import torch
    from rsl_rl.models import MLPModel
    from rsl_rl.storage import RolloutStorage
    from pawweaver.isaac_env import WholeBodyEnv
    from pawweaver.learning import WholeBodyActor,AuxiliaryPPO
    from pawweaver.bundle import export_bundle
    config=json.loads(Path("configs/training.json").read_text())
    config.update(episode_seconds=.24,trajectory_prediction=True,domain_randomization=args.randomize)
    env=WholeBodyEnv(Path("artifacts/software-fixture"),config,4,args.device,7)
    obs=env.get_observations()
    actor=WholeBodyActor(obs,{"actor":["policy"]},"actor",18,hidden_dims=[64,32],prediction=True,
        distribution_cfg={"class_name":"rsl_rl.modules.distribution:GaussianDistribution","init_std":.2})
    critic=MLPModel(obs,{"critic":["critic"]},"critic",1,hidden_dims=[32])
    storage=RolloutStorage("rl",4,16,obs,[18],args.device)
    algorithm=AuxiliaryPPO(actor,critic,storage,num_learning_epochs=2,num_mini_batches=2,device=args.device)
    with torch.inference_mode():
        for _ in range(16):
            action=algorithm.act(obs)
            obs,reward,done,extras=env.step(action)
            algorithm.process_env_step(obs,reward,done,extras)
        algorithm.compute_returns(obs)
    losses=algorithm.update()
    export_bundle(actor,env.spec,env.manifest,config,Path("artifacts/software-fixture/bundle"),trained=False)
    import numpy as np
    with torch.no_grad():
        np.savez("artifacts/software-fixture/bundle/golden.npz",observation=obs["policy"].cpu().numpy(),
                 action=actor(obs).cpu().numpy())
    result={"kind":"synthetic_software_test","environments":4,"rollout_steps":16,"losses":losses,"randomization":args.randomize,
            "hardware_validation":False,"passed":all(torch.isfinite(torch.tensor(v)) for v in losses.values())}
    Path("artifacts/software-fixture/ppo-report.json").write_text(json.dumps(result,indent=2)+"\n")
    print("PAWWEAVER_PPO_PIPELINE_OK",json.dumps(result),flush=True)
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    import sys
    launcher.app.close(exit_code=int(sys.exc_info()[0] is not None))
