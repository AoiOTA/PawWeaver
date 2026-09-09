"""Convert the canonical URDF using the pinned Isaac Lab/Sim importer."""
import argparse
import hashlib
import json
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("asset", type=Path)
parser.add_argument("--diagnostic", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
from pawweaver.assets.build import verify_asset
manifest = verify_asset(args.asset, require_ready=not args.diagnostic)
launcher = AppLauncher(args)
try:
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    directory = args.asset.resolve()/"usd"
    cfg = UrdfConverterCfg(asset_path=str(args.asset.resolve()/"robot.urdf"),
        usd_dir=str(directory), usd_file_name="robot.usd", force_usd_conversion=True,
        fix_base=False, merge_fixed_joints=False, self_collision=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(target_type="none"))
    converter = UrdfConverter(cfg)
    files = {str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "conversion.json"}
    (directory/"conversion.json").write_text(json.dumps({"source_asset_hash":manifest["asset_hash"],
        "ready_for_training":manifest["ready_for_training"], "files":files},indent=2)+"\n")
    print("PAWWEAVER_USD_OK", converter.usd_path, flush=True)
finally:
    launcher.app.close()
