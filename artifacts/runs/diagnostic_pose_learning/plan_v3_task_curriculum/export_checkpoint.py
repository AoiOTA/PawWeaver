"""CPU-only snapshot export for this E2 diagnostic learning run."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
from tensordict import TensorDict
from pawweaver.bundle import export_bundle
from pawweaver.contracts import ActuatorSpec
from pawweaver.learning import WholeBodyActor
from pawweaver.observations import ObservationSpec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    torch.set_num_threads(1)
    contents = args.checkpoint.read_bytes()
    checkpoint = torch.load(io.BytesIO(contents), map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    config = metadata["config"]
    if not metadata["diagnostic"]:
        raise ValueError("This E2 exporter only accepts diagnostic checkpoints")
    if metadata["observation"] != ObservationSpec().to_dict():
        raise ValueError("Checkpoint observation contract differs")
    asset = metadata["asset_manifest"]
    provisional = metadata["provisional_spec"]
    if not metadata["asset_hash"] == asset["asset_hash"] == provisional["asset_hash"]:
        raise ValueError("Checkpoint asset/provisional identities differ")
    actuators = ActuatorSpec.from_dict(provisional["actuators"])
    obs = TensorDict({"policy": torch.zeros(1, ObservationSpec().size)}, batch_size=[1])
    actor = WholeBodyActor(obs, {"actor": ["policy"]}, "actor", 18,
        hidden_dims=config["actor_hidden_dims"], obs_normalization=True,
        prediction=config["trajectory_prediction"], velocity=config["velocity_estimation"],
        leg_mean_transform=config.get("leg_mean_transform", "identity"),
        distribution_cfg={"class_name": "rsl_rl.modules.distribution:GaussianDistribution", "init_std": .5})
    actor.load_state_dict(checkpoint["algorithm"]["actor_state_dict"], strict=True)
    actor.eval()
    if any(not torch.isfinite(value).all() for value in actor.state_dict().values()):
        raise FloatingPointError("Checkpoint Actor contains non-finite values")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = export_bundle(actor, actuators, asset, config, args.output, trained=False, metadata=metadata)
    provenance = dict(checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=hashlib.sha256(contents).hexdigest(), iteration=checkpoint["iteration"],
        policy_sha256=manifest["policy_sha256"], device="cpu",
        note="JIT serialization hash can differ from another export; compare deterministic actions.")
    (args.output / "checkpoint_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance), flush=True)


if __name__ == "__main__":
    main()
