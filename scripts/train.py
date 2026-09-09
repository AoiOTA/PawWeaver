"""Train one whole-body actor in PhysX. Formal runs require a verified hardware asset."""
import argparse
import copy
import json
import time
import subprocess
import importlib.metadata
from pathlib import Path
from isaaclab.app import AppLauncher
from pawweaver.assets.build import verify_asset

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--asset",type=Path,default=Path("assets/generated/verified"))
parser.add_argument("--config",type=Path,default=Path("configs/training.json"))
parser.add_argument("--output",type=Path,required=True)
parser.add_argument("--num-envs",type=int,default=1024)
parser.add_argument("--iterations",type=int,default=1000)
parser.add_argument("--seed",type=int,default=0)
parser.add_argument("--resume",type=Path)
parser.add_argument("--initialize-from",type=Path,help="Transfer compatible weights into the next curriculum stage")
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
if args.resume and args.initialize_from:
    parser.error("Choose resume or initialize-from")
manifest=verify_asset(args.asset)  # Fail before starting the GPU application.
if manifest["robot"]!="as2_edu_piper_h_rgbd":
    raise ValueError("Formal training requires the verified AS2 EDU + Piper-H + selected RGB-D asset")
config=json.loads(args.config.read_text())
launcher=AppLauncher(args)
try:
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from rsl_rl.models import MLPModel
    from rsl_rl.storage import RolloutStorage
    from pawweaver.isaac_env import WholeBodyEnv
    from pawweaver.learning import WholeBodyActor,AuxiliaryPPO
    from pawweaver.bundle import export_bundle
    env=WholeBodyEnv(args.asset,config,args.num_envs,args.device,args.seed)
    obs=env.get_observations()
    groups={"actor":["policy"],"critic":["critic"]}
    actor=WholeBodyActor(obs,groups,"actor",18,hidden_dims=config["actor_hidden_dims"],
        obs_normalization=True,prediction=config["trajectory_prediction"],velocity=config["velocity_estimation"],
        distribution_cfg={"class_name":"rsl_rl.modules.distribution:GaussianDistribution","init_std":.5})
    critic=MLPModel(obs,groups,"critic",1,hidden_dims=[512,256,128],obs_normalization=True)
    storage=RolloutStorage("rl",args.num_envs,config["rollout_steps"],obs,[18],args.device)
    algorithm=AuxiliaryPPO(actor,critic,storage,device=args.device,num_learning_epochs=config["learning_epochs"],
        num_mini_batches=config["mini_batches"],gamma=config["gamma"],lam=config["lam"],
        learning_rate=config["learning_rate"],desired_kl=config["desired_kl"],entropy_coef=config["entropy_coef"],
        clip_param=config["clip_param"])
    args.output.mkdir(parents=True,exist_ok=True)
    metadata={"config":config,"seed":args.seed,"asset_hash":manifest["asset_hash"],"num_envs":args.num_envs,
              "engine":"PhysX","torch":torch.__version__,
              "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
              "git_dirty":bool(subprocess.check_output(["git","status","--porcelain"],text=True).strip()),
              "versions":{name:importlib.metadata.version(name) for name in ("isaaclab","rsl-rl-lib","mujoco")},
              "initialize_from":str(args.initialize_from) if args.initialize_from else None}
    (args.output/"run.json").write_text(json.dumps(metadata,indent=2)+"\n")
    start_iteration=0
    if args.initialize_from:
        checkpoint=torch.load(args.initialize_from,map_location=args.device,weights_only=False)
        if checkpoint["metadata"]["asset_hash"]!=metadata["asset_hash"]:
            raise ValueError("Curriculum transfer requires identical hardware asset")
        for key in ("trajectory_prediction","velocity_estimation","actor_hidden_dims"):
            if checkpoint["metadata"]["config"][key]!=config[key]:
                raise ValueError(f"Curriculum transfer architecture differs: {key}")
        algorithm.load(checkpoint["algorithm"],{"actor":True,"critic":True,"optimizer":False},True)
    if args.resume:
        checkpoint=torch.load(args.resume,map_location=args.device,weights_only=False)
        if checkpoint["metadata"]["asset_hash"]!=metadata["asset_hash"] or checkpoint["metadata"]["config"]!=config:
            raise ValueError("Resume asset/config differs from checkpoint")
        algorithm.load(checkpoint["algorithm"],None,True)
        algorithm.learning_rate=algorithm.optimizer.param_groups[0]["lr"]
        env.reference.sampler.load_state_dict(checkpoint["sampler"])
        env.reference.rng.bit_generator.state=checkpoint["reference_rng"]
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if args.device.startswith("cuda"):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        start_iteration=checkpoint["iteration"]+1
    writer=SummaryWriter(str(args.output/"tensorboard"))
    for iteration in range(start_iteration,start_iteration+args.iterations):
        begin=time.perf_counter()
        with torch.inference_mode():
            for _ in range(config["rollout_steps"]):
                actions=algorithm.act(obs)
                obs,reward,done,extras=env.step(actions)
                algorithm.process_env_step(obs,reward,done,extras)
            algorithm.compute_returns(obs)
        losses=algorithm.update()
        seconds=time.perf_counter()-begin
        stats=dict(losses,iteration=iteration,steps_per_second=args.num_envs*config["rollout_steps"]/seconds,
            tracking_error_m=extras["tracking_error_m"],fall_fraction=extras["fall_fraction"])
        with (args.output/"metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(stats)+"\n")
        for key,value in stats.items():
            writer.add_scalar(key,value,iteration)
        print(json.dumps(stats),flush=True)
        if iteration%50==0 or iteration==start_iteration+args.iterations-1:
            checkpoint={"algorithm":algorithm.save(),"iteration":iteration,"metadata":metadata,
                        "sampler":env.reference.sampler.state_dict(),"reference_rng":env.reference.rng.bit_generator.state,
                        "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),
                        "resume_semantics":"Optimizer/RNG/sampler restored; simulator episodes restart."}
            torch.save(checkpoint,args.output/f"checkpoint_{iteration:06d}.pt")
            export_bundle(actor,env.spec,manifest,config,args.output/"bundle",trained=True,metadata=metadata)
    writer.close()
    env.close()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    import sys
    launcher.app.close(exit_code=int(sys.exc_info()[0] is not None))
