"""Checksummed deployment bundle. The runtime imports no training framework."""
import hashlib
import json
from pathlib import Path
import torch
from .contracts import ActuatorSpec, canonical_hash
from .observations import ObservationSpec,observation_spec as configured_observation_spec

def export_bundle(actor,actuators:ActuatorSpec,asset_manifest:dict,config:dict,output:Path,*,trained:bool,metadata=None):
    observation=configured_observation_spec(config)
    if actor.obs_dim!=observation.size or getattr(actor,"observation_dim",actor.obs_dim)!=observation.size:
        raise ValueError("Bundle actor differs from the configured observation contract")
    if trained and not asset_manifest["ready_for_training"]:
        raise ValueError("Cannot mark a policy trained for unverified hardware")
    output.mkdir(parents=True,exist_ok=True)
    policy = torch.jit.script(actor.as_jit().cpu().eval())
    policy.save(str(output/"policy.pt"))
    # Deployment consumers need identity fields, not a duplicate of the full run record.
    training_metadata = {key:metadata[key] for key in ("seed","diagnostic","provisional_spec")
                         if metadata is not None and key in metadata}
    manifest = {"schema_version":2,"asset_hash":asset_manifest["asset_hash"],"trained":trained,
                "actuators":actuators.to_dict(),"observation":observation.to_dict(),
                "training_config":config,"training_metadata":training_metadata,
                "policy_sha256":hashlib.sha256((output/"policy.pt").read_bytes()).hexdigest()}
    manifest["bundle_hash"] = canonical_hash(manifest)
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest

def load_bundle(path:Path,asset_hash:str,require_trained=True,observation_spec=None):
    manifest = json.loads((path/"manifest.json").read_text())
    checksum = manifest.pop("bundle_hash")
    if canonical_hash(manifest)!=checksum:
        raise ValueError("Bundle manifest checksum mismatch")
    if manifest["schema_version"]!=2:
        raise ValueError("Unsupported policy bundle schema; 276 pose observations are required and position-only bundles are incompatible")
    if manifest["asset_hash"]!=asset_hash:
        raise ValueError("Policy bundle and robot asset versions differ")
    if require_trained and not manifest["trained"]:
        raise ValueError("Untrained smoke policies cannot be used for acceptance evaluation")
    if hashlib.sha256((path/"policy.pt").read_bytes()).hexdigest()!=manifest["policy_sha256"]:
        raise ValueError("Policy file checksum mismatch")
    observation=observation_spec or ObservationSpec()
    if manifest["observation"]!=observation.to_dict():
        raise ValueError("Unsupported observation contract")
    if configured_observation_spec(manifest["training_config"]).to_dict()!=manifest["observation"]:
        raise ValueError("Bundle training config and observation contract differ")
    spec = ActuatorSpec.from_dict(manifest["actuators"])
    return torch.jit.load(str(path/"policy.pt"),map_location="cpu").eval(),spec,manifest
