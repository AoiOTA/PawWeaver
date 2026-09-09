"""Compare fixed-base, single-joint steps through the shared explicit 2 ms PD in both engines."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("asset",type=Path)
parser.add_argument("--output",type=Path,required=True)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
from pawweaver.assets.build import verify_asset
manifest=verify_asset(args.asset)
launcher=AppLauncher(args)
try:
    import numpy as np
    import torch
    import mujoco
    from pawweaver.contracts import ActuatorSpec,JOINT_NAMES,named_indices
    from pawweaver.control import JointPD
    from pawweaver.isaac_robot import create_scene
    spec=ActuatorSpec.load(args.asset/"actuators.json")
    sim,scene=create_scene(args.asset,18,args.device,spec,fixed_base=True)
    robot=scene["robot"];indices=named_indices(robot.joint_names,JOINT_NAMES)
    root=ET.parse(args.asset/"robot.xml").getroot()
    root.find("compiler").set("meshdir",str((args.asset/"meshes").resolve()))
    base=root.find("worldbody/body[@name='base_link']")
    for joint in list(base):
        if joint.tag=="freejoint" or (joint.tag=="joint" and joint.get("type")=="free"):
            base.remove(joint)
    base.set("pos","0 0 2")
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding="unicode"))
    data=[mujoco.MjData(model) for _ in range(18)]
    qids=[model.joint(n).qposadr[0] for n in JOINT_NAMES]
    vids=[model.joint(n).dofadr[0] for n in JOINT_NAMES]
    motors=[model.actuator(n+"_motor").id for n in JOINT_NAMES]
    for d in data:
        d.qpos[qids]=spec.default_pos;mujoco.mj_forward(model,d)
    pxpd=JointPD(spec,18,args.device);mjpd=JointPD(spec,18)
    action=torch.eye(18)*.1
    records=[]
    for step in range(250):
        if step%spec.decimation==0:
            command=action if step>=50 else torch.zeros_like(action)
            pxpd.command(command.to(args.device));mjpd.command(command)
        pxq=robot.data.joint_pos.torch[:,indices]
        pxv=robot.data.joint_vel.torch[:,indices]
        mjq=torch.tensor(np.array([d.qpos[qids] for d in data]),dtype=torch.float32)
        mjv=torch.tensor(np.array([d.qvel[vids] for d in data]),dtype=torch.float32)
        pt=pxpd.torque(pxq,pxv);mt=mjpd.torque(mjq,mjv)
        robot.set_joint_effort_target_index(target=pt.contiguous(),joint_ids=indices)
        scene.write_data_to_sim();sim.step(render=False);scene.update(.002)
        for i,d in enumerate(data):
            d.ctrl[motors]=mt[i].numpy();mujoco.mj_step(model,d)
        records.append((robot.data.joint_pos.torch[:,indices].cpu().numpy().copy(),np.array([d.qpos[qids] for d in data])))
    trace=np.asarray(records)
    difference=np.abs(trace[:,0]-trace[:,1])
    tolerance=.002+.1*np.abs(trace[:,1]-np.asarray(spec.default_pos))
    passed=bool(np.isfinite(trace).all() and np.all(difference<=tolerance))
    result={"asset_hash":manifest["asset_hash"],"hardware_validation":manifest["robot"]!="synthetic_software_fixture",
        "joint_step_cases":18,"duration_per_case_s":.5,"step_onset_s":.1,"action_step":.1,
        "max_position_difference_rad":float(difference.max()),"tolerance":"0.002 rad + 10% of displacement from initial joint position",
        "passed":passed}
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/"traces.npz",positions=trace,joint_names=JOINT_NAMES,physics_dt=.002)
    (args.output/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print("PAWWEAVER_PD_RESPONSE",json.dumps(result),flush=True)
    if not passed:
        raise RuntimeError("Fixed-base actuator responses disagree")
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    import sys
    launcher.app.close(exit_code=int(sys.exc_info()[0] is not None))
