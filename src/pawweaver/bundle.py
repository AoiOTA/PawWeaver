"""Checksummed deployment bundle. The runtime imports no training framework."""
import hashlib
import json
from pathlib import Path
import torch
from .contracts import ActuatorSpec, canonical_hash
from .observations import ObservationSpec

def export_bundle(actor,actuators:ActuatorSpec,asset_manifest:dict,config:dict,output:Path,*,trained:bool,metadata=None):
    if trained and not asset_manifest["ready_for_training"]:
        raise ValueError("Cannot mark a policy trained for unverified hardware")
    output.mkdir(parents=True,exist_ok=True)
    policy = torch.jit.script(actor.as_jit().cpu().eval())
    policy.save(str(output/"policy.pt"))
    manifest = {"schema_version":1,"asset_hash":asset_manifest["asset_hash"],"trained":trained,
                "actuators":actuators.to_dict(),"observation":ObservationSpec().to_dict(),
                "training_config":config,"training_metadata":metadata or {},
                "policy_sha256":hashlib.sha256((output/"policy.pt").read_bytes()).hexdigest()}
    manifest["bundle_hash"] = canonical_hash(manifest)
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest

def load_bundle(path:Path,asset_hash:str,require_trained=True):
    manifest = json.loads((path/"manifest.json").read_text())
    checksum = manifest.pop("bundle_hash")
    if canonical_hash(manifest)!=checksum:
        raise ValueError("Bundle manifest checksum mismatch")
    if manifest["schema_version"]!=1 or manifest["asset_hash"]!=asset_hash:
        raise ValueError("Policy bundle and robot asset versions differ")
    if require_trained and not manifest["trained"]:
        raise ValueError("Untrained smoke policies cannot be used for acceptance evaluation")
    if hashlib.sha256((path/"policy.pt").read_bytes()).hexdigest()!=manifest["policy_sha256"]:
        raise ValueError("Policy file checksum mismatch")
    if manifest["observation"]!=ObservationSpec().to_dict():
        raise ValueError("Unsupported observation contract")
    spec = ActuatorSpec.from_dict(manifest["actuators"])
    return torch.jit.load(str(path/"policy.pt"),map_location="cpu").eval(),spec,manifest
