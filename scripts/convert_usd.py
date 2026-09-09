"""Convert the canonical URDF using the pinned Isaac Lab/Sim importer."""
import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("asset", type=Path)
parser.add_argument("--diagnostic", action="store_true")
parser.add_argument("--output-subdir",type=Path,help="Fresh directory beneath usd/ (or usd-fixed/); keep prior conversion files intact")
parser.add_argument("--fixed-base", action="store_true",help="Separate fixed-base USD for actuator response checks")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
from pawweaver.assets.build import verify_asset
manifest = verify_asset(args.asset, require_ready=not args.diagnostic)
usd_root = args.asset.resolve()/("usd-fixed" if args.fixed_base else "usd")
if args.output_subdir is not None:
    if args.output_subdir.is_absolute() or ".." in args.output_subdir.parts or str(args.output_subdir) == ".":
        parser.error("--output-subdir must name a fresh relative child directory")
    directory = usd_root/args.output_subdir
    if directory.exists():
        raise ValueError(f"Conversion output already exists: {directory}")
else:
    directory = usd_root
launcher = AppLauncher(args)
try:
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    from pxr import Usd, UsdPhysics
    from pawweaver.assets.model import RobotTree
    cfg = UrdfConverterCfg(asset_path=str(args.asset.resolve()/"robot.urdf"),
        usd_dir=str(directory), usd_file_name="robot.usd", force_usd_conversion=True,
        fix_base=args.fixed_base, merge_fixed_joints=False, self_collision=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(target_type="none"))
    converter = UrdfConverter(cfg)
    stage = Usd.Stage.Open(converter.usd_path)
    if stage is None:
        raise ValueError("Could not open imported USD")
    rigid_paths = {}
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_paths.setdefault(prim.GetName(), []).append(prim.GetPath())
    pairs = RobotTree.load(args.asset/"robot.urdf").fixed_collision_pairs()
    resolved = []
    for names in pairs:
        for name in names:
            if len(rigid_paths.get(name, [])) != 1:
                raise ValueError(f"Fixed collision link must resolve to one USD rigid body: {name}: {rigid_paths.get(name, [])}")
        source, target = (rigid_paths[name][0] for name in names)
        resolved.append({"links":list(names),"source_prim":str(source),"target_prim":str(target)})
    # Add to existing relations; do not merge fixed bodies or alter any movable-joint collision.
    for pair in resolved:
        api = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(pair["source_prim"]))
        api.CreateFilteredPairsRel().AddTarget(pair["target_prim"])
    stage.GetRootLayer().Save()
    stage = Usd.Stage.Open(converter.usd_path)
    for pair in resolved:
        targets = UsdPhysics.FilteredPairsAPI(stage.GetPrimAtPath(pair["source_prim"])).GetFilteredPairsRel().GetTargets()
        if pair["target_prim"] not in [str(target) for target in targets]:
            raise ValueError(f"USD fixed-pair relationship did not persist: {pair}")
    files = {str(p.relative_to(usd_root)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "conversion.json"}
    metadata = {"source_asset_hash":manifest["asset_hash"],
        "entrypoint":str(Path(converter.usd_path).relative_to(usd_root)),
        "ready_for_training":manifest["ready_for_training"], "files":files,
        "producer_sha256":{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
            (Path(__file__).resolve(),Path(__file__).resolve().parents[1]/"src/pawweaver/assets/model.py")},
        "fixed_collision_pairs":resolved,
        "collision_filter_rule":"All collision-bearing links in each canonical fixed-connected component; never cross movable joints."}
    payload = json.dumps(metadata,indent=2)+"\n"
    if directory != usd_root:
        (directory/"conversion.json").write_text(payload)
    # Publish the existing locator only after import, relationship verification and hashes succeed.
    with tempfile.NamedTemporaryFile(mode="w",dir=usd_root,prefix=".conversion-",delete=False) as stream:
        stream.write(payload)
        pending = Path(stream.name)
    pending.replace(usd_root/"conversion.json")
    print("PAWWEAVER_USD_OK", converter.usd_path, flush=True)
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    import sys
    launcher.app.close(exit_code=int(sys.exc_info()[0] is not None))
