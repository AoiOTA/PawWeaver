"""CPU-only export of an E3 diagnostic 279-input, single-18 Actor checkpoint."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"]=""
import torch
from tensordict import TensorDict
from pawweaver.bundle import export_bundle
from pawweaver.contracts import ActuatorSpec
from pawweaver.learning import WholeBodyActor
from pawweaver.observations import observation_spec


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    torch.set_num_threads(1)
    contents=args.checkpoint.read_bytes()
    checkpoint=torch.load(io.BytesIO(contents),map_location="cpu",weights_only=False)
    metadata=checkpoint["metadata"];config=metadata["config"]
    if not metadata["diagnostic"] or config.get("task_mode")!="velocity_ee_pose":
        raise ValueError("E3 exporter requires a diagnostic velocity_ee_pose checkpoint")
    spec=observation_spec(config)
    if spec.size!=279 or metadata["observation"]!=spec.to_dict():
        raise ValueError("Checkpoint observation contract must be the explicit 279-input E3 contract")
    asset=metadata["asset_manifest"];provisional=metadata["provisional_spec"]
    if not metadata["asset_hash"]==asset["asset_hash"]==provisional["asset_hash"]:
        raise ValueError("Checkpoint asset/provisional identities differ")
    if "seed" not in metadata:
        raise ValueError("Checkpoint must retain its training seed")
    actuators=ActuatorSpec.from_dict(provisional["actuators"])
    obs=TensorDict({"policy":torch.zeros(1,spec.size)},batch_size=[1])
    actor=WholeBodyActor(obs,{"actor":["policy"]},"actor",18,
        hidden_dims=config["actor_hidden_dims"],obs_normalization=True,observation_dim=spec.size,
        prediction=config["trajectory_prediction"],velocity=config["velocity_estimation"],
        leg_mean_transform=config.get("leg_mean_transform","identity"),
        distribution_cfg={"class_name":"rsl_rl.modules.distribution:GaussianDistribution","init_std":.5})
    actor.load_state_dict(checkpoint["algorithm"]["actor_state_dict"],strict=True)
    actor.eval()
    if any(not torch.isfinite(value).all() for value in actor.state_dict().values()):
        raise FloatingPointError("Checkpoint Actor contains non-finite values")
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=export_bundle(actor,actuators,asset,config,args.output,trained=False,metadata=metadata)
    provenance=dict(checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(contents).hexdigest(),
        iteration=checkpoint["iteration"],completed_iterations_in_checkpoint=checkpoint["iteration"]+1,
        training_progress=metadata.get("training_progress"),policy_sha256=manifest["policy_sha256"],
        device="cpu",observation_size=spec.size,seed=metadata["seed"],trained=False,
        note="Snapshot index is zero-based:250/500/999 contain251/501/1000 iterations in this fresh run. JIT hashes can differ across exports; compare deterministic actions. Source checkpoint was only read.")
    (args.output/"checkpoint_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    print(json.dumps(provenance),flush=True)


if __name__=="__main__":
    main()
