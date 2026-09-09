"""Static FK and source mass parity for the converted USD, independent of actuator tuning."""
import argparse
import json
from pathlib import Path
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("asset",type=Path)
parser.add_argument("--samples",type=int,default=25)
parser.add_argument("--contacts",action="store_true",help="Also initialize every named contact sensor")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
try:
    import numpy as np
    import torch
    from pawweaver.isaac_robot import create_scene
    import mujoco
    from pawweaver.assets.model import RobotTree,numbers
    from pawweaver.math import rpy_quat,rpy_matrix
    from pawweaver.contracts import JOINT_NAMES,FOOT_NAMES,named_indices
    sim,scene = create_scene(args.asset,1,args.device,contacts=args.contacts)
    robot = scene["robot"]
    if args.contacts:
        for name,sensor in scene.sensors.items():
            if name.startswith("contact_"):
                if sensor.num_sensors!=1 or sensor.body_names!=[name.removeprefix("contact_")]:
                    raise ValueError(f"Contact sensor maps unexpected bodies: {name} {sensor.body_names}")
    tree = RobotTree.load(args.asset/"robot.urdf")
    indices = named_indices(robot.joint_names,JOINT_NAMES)
    print("BODY_NAMES",robot.body_names,flush=True)
    names = [name for name in (*FOOT_NAMES,"arm_gripper_base") if name in robot.body_names]
    if len(names)!=5:
        raise ValueError("USD lost foot or gripper frames")
    rng = np.random.default_rng(731)
    mjmodel=mujoco.MjModel.from_xml_path(str(args.asset/"robot.xml"))
    mjdata=mujoco.MjData(mjmodel)
    worst = 0.;worst_angle=0.;worst_gravity=0.;gravity_passed=True
    for _ in range(args.samples):
        pose = robot.data.default_root_pose.torch.clone()
        xyz=rng.uniform(-1,1,3);xyz[2]+=2
        rpy=rng.uniform(-.5,.5,3)
        quat=rpy_quat(rpy)
        pose[:,:3] = torch.tensor(xyz[None],device=args.device)
        pose[:,3:]=torch.tensor(quat[[1,2,3,0]][None],device=args.device)
        robot.write_root_pose_to_sim_index(root_pose=pose)
        qs = {name:rng.uniform(float(tree.joints[name].find("limit").get("lower")),
                              float(tree.joints[name].find("limit").get("upper"))) for name in JOINT_NAMES}
        joints = torch.zeros_like(robot.data.joint_pos.torch)
        joints[:,indices] = torch.tensor([list(qs.values())],device=args.device)
        robot.write_joint_position_to_sim_index(position=joints)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joints))
        sim.forward()
        scene.update(.002)
        root = np.eye(4); root[:3,3]=xyz;root[:3,:3]=rpy_matrix(rpy)
        fk = tree.forward(qs,root)
        for name in names:
            actual = robot.data.body_link_pos_w.torch[0,robot.body_names.index(name)].cpu().numpy()
            worst = max(worst,float(np.linalg.norm(actual-fk[name][:3,3])))
            qxyzw=robot.data.body_link_quat_w.torch[0,robot.body_names.index(name)].cpu().numpy()
            normalized=qxyzw[[3,0,1,2]].astype(float);normalized/=np.linalg.norm(normalized)
            rotation=np.zeros(9);mujoco.mju_quat2Mat(rotation,normalized)
            cosine=(np.trace(rotation.reshape(3,3).T@fk[name][:3,:3])-1)/2
            worst_angle=max(worst_angle,float(np.arccos(np.clip(cosine,-1,1))))
        tcp_offset=numbers(tree.joints["tcp_mount"].find("origin").get("xyz"))
        gripper_index=robot.body_names.index("arm_gripper_base")
        gripper=robot.data.body_link_pose_w.torch[0,gripper_index].cpu().numpy()
        rotation=np.zeros(9);mujoco.mju_quat2Mat(rotation,gripper[[6,3,4,5]].astype(float))
        tcp=gripper[:3]+rotation.reshape(3,3)@tcp_offset
        worst=max(worst,float(np.linalg.norm(tcp-fk["tcp"][:3,3])))
        mjdata.qpos[:3]=xyz;mjdata.qpos[3:7]=quat
        for name,q in qs.items():
            mjdata.qpos[mjmodel.joint(name).qposadr[0]]=q
        mujoco.mj_forward(mjmodel,mjdata)
        mjgravity=np.array([mjdata.qfrc_bias[mjmodel.joint(name).dofadr[0]] for name in JOINT_NAMES])
        # PhysX generalized forces put six floating-base DOFs before named joints.
        pxgravity=robot.data.gravity_compensation_forces.torch[0].cpu().numpy()[np.asarray(indices)+6]
        difference=np.abs(pxgravity-mjgravity)
        worst_gravity=max(worst_gravity,float(difference.max()))
        gravity_passed=gravity_passed and bool(np.all(difference<.005+.02*np.abs(mjgravity)))
    mass = float(robot.data.body_mass.torch.sum())
    manifest=json.loads((args.asset/"manifest.json").read_text())
    result = {"asset_hash":manifest["asset_hash"],"hardware_validation":manifest["ready_for_training"],
              "samples":args.samples,"max_position_error_m":worst,"source_mass_kg":tree.mass,"physx_mass_kg":mass,
              "max_orientation_error_deg":float(np.degrees(worst_angle)),"gravity_max_absolute_difference_nm":worst_gravity,
              "gravity_tolerance":"0.005 Nm + 2% of MuJoCo gravity torque, per joint", "gravity_passed":gravity_passed,
              "passed":worst<.001 and worst_angle<np.deg2rad(.1) and abs(mass-tree.mass)<.001 and gravity_passed}
    (args.asset/"usd/fk-report.json").write_text(json.dumps(result,indent=2)+"\n")
    print("PAWWEAVER_ISAAC_FK",json.dumps(result),flush=True)
    if not result["passed"]:
        raise RuntimeError("USD model validation failed")
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    import sys
    launcher.app.close(exit_code=int(sys.exc_info()[0] is not None))
