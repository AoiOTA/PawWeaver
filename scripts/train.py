"""Train one whole-body actor in PhysX. Formal runs require a verified hardware asset."""
import argparse
import math
import hashlib
import json
import time
import subprocess
import importlib.metadata
from pathlib import Path
from isaaclab.app import AppLauncher
from pawweaver.training_inputs import training_inputs, check_training_identity

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--asset",type=Path,default=Path("assets/generated/verified"))
parser.add_argument("--config",type=Path,default=Path("configs/training.json"))
parser.add_argument("--output",type=Path,required=True)
parser.add_argument("--num-envs",type=int,default=1024)
parser.add_argument("--iterations",type=int,default=1000)
parser.add_argument("--seed",type=int,default=0)
parser.add_argument("--diagnostic",action="store_true",help="Engineering probe only; never a trained acceptance bundle")
parser.add_argument("--provisional-spec",type=Path,help="Required explicit provisional spec with sources for diagnostic mode")
parser.add_argument("--resume",type=Path)
parser.add_argument("--initialize-from",type=Path,help="Transfer compatible weights into the next curriculum stage")
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
if args.resume and args.initialize_from:
    parser.error("Choose resume or initialize-from")
manifest,spec,provisional=training_inputs(args.asset,diagnostic=args.diagnostic,provisional_spec=args.provisional_spec)  # Fail before GPU startup.
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
    env=WholeBodyEnv(args.asset,config,args.num_envs,args.device,args.seed,
        diagnostic=args.diagnostic,provisional_spec=args.provisional_spec)
    def finite_observations(observations):
        if any(not torch.isfinite(value).all() for value in observations.values()):
            raise FloatingPointError("Non-finite observations")
    obs=env.get_observations()
    if args.diagnostic:
        finite_observations(obs)
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
    metadata.update(diagnostic=args.diagnostic,provisional_spec=provisional)
    if args.diagnostic:
        metadata.update(asset_manifest=manifest,
            usd_conversion=json.loads((args.asset/"usd/conversion.json").read_text()),
            source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                (Path(__file__),Path("src/pawweaver/isaac_env.py"),Path("src/pawweaver/training_inputs.py"),
                 Path("src/pawweaver/learning.py"),Path("src/pawweaver/isaac_robot.py"))},
            actual_armature_kg_m2={name:env.loaded_armature[:,i].tolist() for i,name in enumerate(env.spec.joint_names)},
            contact_body_mapping={name:sensor.body_names for name,sensor in env.scene.sensors.items()},
            initial_foot_link_height_m=(env.robot.data.body_link_pose_w.torch[:,env.foot_ids,2]-env.scene.env_origins[:,None,2]).tolist(),
            initial_base_height_m=(env.state().base_pos_w[:,2]-env.scene.env_origins[:,2]).tolist(),
            evidence_limit="Provisional real-geometry integration only; no hardware validity or task acceptance.")
    (args.output/"run.json").write_text(json.dumps(metadata,indent=2)+"\n")
    start_iteration=0
    if args.initialize_from:
        checkpoint=torch.load(args.initialize_from,map_location=args.device,weights_only=False)
        check_training_identity(checkpoint["metadata"],metadata)
        for key in ("trajectory_prediction","velocity_estimation","actor_hidden_dims"):
            if checkpoint["metadata"]["config"][key]!=config[key]:
                raise ValueError(f"Curriculum transfer architecture differs: {key}")
        algorithm.load(checkpoint["algorithm"],{"actor":True,"critic":True,"optimizer":False},True)
    if args.resume:
        checkpoint=torch.load(args.resume,map_location=args.device,weights_only=False)
        check_training_identity(checkpoint["metadata"],metadata)
        if checkpoint["metadata"]["config"]!=config:
            raise ValueError("Resume asset/config differs from checkpoint")
        algorithm.load(checkpoint["algorithm"],None,True)
        algorithm.learning_rate=algorithm.optimizer.param_groups[0]["lr"]
        env.reference.sampler.load_state_dict(checkpoint["sampler"])
        env.reference.rng.bit_generator.state=checkpoint["reference_rng"]
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if args.device.startswith("cuda"):
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
        start_iteration=checkpoint["iteration"]+1
    optimizer_steps=[]
    if args.diagnostic:
        def check_gradients(optimizer, *unused):
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError("Non-finite optimizer gradient")
        def record_optimizer_step(optimizer, *unused):
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if not torch.isfinite(parameter).all():
                        raise FloatingPointError("Non-finite optimized parameter")
            optimizer_steps.append(1)
        algorithm.optimizer.register_step_pre_hook(check_gradients)
        algorithm.optimizer.register_step_post_hook(record_optimizer_step)
    writer=SummaryWriter(str(args.output/"tensorboard"))
    for iteration in range(start_iteration,start_iteration+args.iterations):
        begin=time.perf_counter()
        if args.diagnostic:
            before=[p.detach().clone() for p in actor.parameters()]
            steps_before=len(optimizer_steps)
            probe={"falls":0,"resets":0,"torque_saturated":0,"torque_samples":0,
                   "tracking_error_sum_m":0.,"tracking_error_max_m":0.,"base_height_min_m":float("inf"),
                   "reward_min":float("inf"),"reward_max":float("-inf"),"action_abs_max":0.}
        with torch.inference_mode():
            for _ in range(config["rollout_steps"]):
                actions=algorithm.act(obs)
                obs,reward,done,extras=env.step(actions)
                if args.diagnostic:
                    finite_observations(obs)
                    if not torch.isfinite(actions).all() or not torch.isfinite(reward).all():
                        raise FloatingPointError("Non-finite actions/rewards")
                    for key in ("falls","resets","torque_saturated","torque_samples"):
                        probe[key]+=extras[key]
                    probe["tracking_error_sum_m"]+=extras["tracking_error_m"]
                    for key in ("tracking_error_max_m",):
                        probe[key]=max(probe[key],extras[key])
                    probe["base_height_min_m"]=min(probe["base_height_min_m"],extras["base_height_min_m"])
                    probe["reward_min"]=min(probe["reward_min"],float(reward.min()))
                    probe["reward_max"]=max(probe["reward_max"],float(reward.max()))
                    probe["action_abs_max"]=max(probe["action_abs_max"],float(actions.abs().max()))
                algorithm.process_env_step(obs,reward,done,extras)
            algorithm.compute_returns(obs)
        losses=algorithm.update()
        if args.diagnostic:
            if not all(math.isfinite(value) for value in losses.values()):
                raise FloatingPointError("Non-finite reported loss")
            delta=max(float((p.detach()-old).abs().max()) for p,old in zip(actor.parameters(),before))
            completed=len(optimizer_steps)-steps_before
            if completed!=config["learning_epochs"]*config["mini_batches"] or delta<=0:
                raise RuntimeError("Diagnostic PPO did not complete expected parameter updates")
            probe.update(actor_parameter_max_abs_change=delta,optimizer_steps=completed,
                tracking_error_mean_m=probe.pop("tracking_error_sum_m")/config["rollout_steps"],
                saturation_fraction=probe["torque_saturated"]/probe["torque_samples"],finite_checks_passed=True)
        seconds=time.perf_counter()-begin
        stats=dict(losses,iteration=iteration,steps_per_second=args.num_envs*config["rollout_steps"]/seconds,
            tracking_error_m=extras["tracking_error_m"],fall_fraction=extras["fall_fraction"])
        if args.diagnostic:
            stats.update(probe)
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
            export_bundle(actor,env.spec,manifest,config,args.output/"bundle",trained=not args.diagnostic,metadata=metadata)
    writer.close()
    env.close()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    import sys
    launcher.app.close(exit_code=int(sys.exc_info()[0] is not None))
