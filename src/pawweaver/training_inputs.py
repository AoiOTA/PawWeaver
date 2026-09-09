"""Explicit provisional input for real-geometry engineering probes; no hardware promotion."""
import json
from pathlib import Path
from .assets.build import verify_asset
from .assets.model import RobotTree
from .contracts import ActuatorSpec, JOINT_NAMES


def training_inputs(asset: Path, *, diagnostic=False, provisional_spec: Path | None = None):
    if diagnostic != (provisional_spec is not None):
        raise ValueError("--diagnostic requires --provisional-spec, which is diagnostic-only")
    manifest = verify_asset(asset, require_ready=not diagnostic)
    if diagnostic and manifest["robot"] != "as2_edu_piper_h_rgbd":
        raise ValueError("Diagnostic training requires real AS2 EDU + Piper-H + RGB-D identity")
    if not diagnostic:
        return manifest, ActuatorSpec.load(asset / "actuators.json"), None
    supplied = json.loads(provisional_spec.read_text())
    if supplied.get("status") != "provisional_engineering_only" or not supplied.get("sources"):
        raise ValueError("Provisional spec requires explicit status and sources")
    if supplied.get("asset_hash") != manifest["asset_hash"]:
        raise ValueError("Provisional spec asset identity differs")
    spec = ActuatorSpec.from_dict(supplied["actuators"])
    tree = RobotTree.load(asset / "robot.urdf")
    for field in ("lower", "upper", "velocity"):
        expected = tuple(float(tree.joints[name].find("limit").get(field)) for name in JOINT_NAMES)
        if getattr(spec, field) != expected:
            raise ValueError(f"Provisional {field} must match canonical URDF by joint name")
    return manifest, spec, supplied


def check_training_identity(previous: dict, current: dict):
    if previous["asset_hash"] != current["asset_hash"]:
        raise ValueError("Checkpoint hardware asset differs")
    if previous.get("diagnostic", False) != current.get("diagnostic", False):
        raise ValueError("Checkpoint diagnostic/formal mode differs")
    if previous.get("provisional_spec") != current.get("provisional_spec"):
        raise ValueError("Checkpoint provisional spec or provenance differs")
