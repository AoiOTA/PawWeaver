"""Isaac Lab 3 PhysX scene creation. Import only after AppLauncher."""
import json
import hashlib
from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_physx.physics import PhysxCfg
from .contracts import JOINT_NAMES
from .assets.model import RobotTree

def create_scene(asset:Path,num_envs:int,device="cuda:0",spec=None,contacts=False,fixed_base=False,initial_base_height_m=.5):
    usd_dir=asset/("usd-fixed" if fixed_base else "usd")
    conversion = json.loads((usd_dir/"conversion.json").read_text())
    usd = usd_dir/conversion.get("entrypoint","robot/robot.usda")
    if not usd.is_file():
        raise ValueError("Run scripts/convert_usd.py first")
    manifest=json.loads((asset/"manifest.json").read_text())
    if conversion["source_asset_hash"]!=manifest["asset_hash"]:
        raise ValueError("USD conversion is stale; reconvert the current canonical asset")
    for name,digest in conversion["files"].items():
        if name.endswith("fk-report.json"):
            continue
        path=usd_dir/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError(f"USD file checksum mismatch: {name}")
    tree=RobotTree.load(asset/"robot.urdf")
    midpoint={name:(float(tree.joints[name].find("limit").get("lower"))+
                    float(tree.joints[name].find("limit").get("upper")))/2 for name in JOINT_NAMES}
    def vector(name,default):
        return dict(zip(JOINT_NAMES,getattr(spec,name))) if spec else default
    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/Ground",spawn=sim_utils.CuboidCfg(
            size=(200.,200.,.1),collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.8,dynamic_friction=.8,restitution=0.)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.,0.,-.05)))
        robot = ArticulationCfg(prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=str(usd.resolve()),activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False,max_depenetration_velocity=1.),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True,
                    solver_position_iteration_count=8,solver_velocity_iteration_count=2)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.,0.,2. if fixed_base else initial_base_height_m),joint_pos=vector("default_pos",midpoint)),
            actuators={"effort":IdealPDActuatorCfg(joint_names_expr=[".*"],stiffness=0.,damping=0.,
                effort_limit=1.e9,effort_limit_sim=1.e9,velocity_limit_sim=1.e6,
                armature=vector("armature",0.),friction=vector("frictionloss",0.),
                dynamic_friction=vector("frictionloss",0.),viscous_friction=vector("damping",0.))})
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.002,device=device,physics=PhysxCfg()))
    scene_cfg=SceneCfg(num_envs=num_envs,env_spacing=8.)
    if contacts:
        from isaaclab.sensors import ContactSensorCfg
        paths={tree.root_name:"{ENV_REGEX_NS}/Robot/Geometry/"+tree.root_name}
        pending=[tree.root_name]
        while pending:
            name=pending.pop()
            if tree.links[name].find("inertial") is not None:
                setattr(scene_cfg,"contact_"+name,ContactSensorCfg(prim_path=paths[name],update_period=0.))
            for joint_name in tree.children[name]:
                child=tree.joints[joint_name].find("child").get("link")
                paths[child]=paths[name]+"/"+child
                pending.append(child)
    scene = InteractiveScene(scene_cfg)
    if contacts:
        # URDF importer 3 keeps nested rigid bodies. The stock activation traversal
        # stops at the first rigid body, so explicitly cover every articulated link.
        from pxr import UsdPhysics,PhysxSchema
        for prim in sim.stage.Traverse():
            if prim.GetPath().pathString.startswith("/World/envs/") and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
    sim.reset()
    scene.update(.002)
    return sim,scene
