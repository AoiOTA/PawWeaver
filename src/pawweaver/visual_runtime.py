"""Rendered wrist RGB-D at 30 Hz drives the same 50 Hz whole-body policy."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import mujoco
import numpy as np
from PIL import Image
from .contracts import GoalSample
from .mujoco_runtime import MujocoRunner
from .trajectories import Trajectory
from .vision import Intrinsics,MarkerEstimator,DelayedMeasurements,quaternion_rotation,rotation_quaternion

def orientation_error_rad(actual,target):
    a=np.asarray(actual,float);b=np.asarray(target,float)
    return float(2*np.arccos(np.clip(abs(np.dot(a,b))/(np.linalg.norm(a)*np.linalg.norm(b)),0.,1.)))

class RenderedVision:
    def __init__(self,runner,calibration,scenario,trajectory,output):
        self.runner,self.scenario,self.trajectory=runner,scenario,trajectory
        self.color=Intrinsics(**calibration["color"])
        self.depth=Intrinsics(**calibration["depth"])
        self.estimator=MarkerEstimator(self.color,self.depth,calibration["depth_to_color"],
            scenario["marker_id"],scenario["marker_size_m"],scenario["marker_to_goal"],
            scenario["marker_to_goal_quat_wxyz"])
        self.queue=DelayedMeasurements(scenario.get("max_age_s",.5))
        self.rng=np.random.default_rng(scenario.get("seed",731))
        self.color_renderer=mujoco.Renderer(runner.model,self.color.height,self.color.width)
        self.depth_renderer=mujoco.Renderer(runner.model,self.depth.height,self.depth.width)
        self.depth_renderer.enable_depth_rendering()
        self.marker_id=runner.model.body("vision_marker").mocapid[0]
        self.offset=np.asarray(scenario["marker_to_goal"],float)
        self.marker_to_goal_rotation=self.estimator.marker_to_goal_rotation
        self.next_capture=0.
        self.reference_time=float(trajectory.timestamps[0])
        self.hold=True
        self.samples=[]
        self.controls=[]
        self.last_rgb=None
        self.output=output

    def tick(self,runner):
        now=float(runner.data.time)
        if not self.hold:
            self.reference_time+=runner.spec.physics_dt
        goal=self.trajectory.sample(self.reference_time)
        goal_quat=self.trajectory.sample_orientation(self.reference_time)
        marker_rotation=quaternion_rotation(goal_quat)@self.marker_to_goal_rotation.T
        runner.data.mocap_pos[self.marker_id]=goal-marker_rotation@self.offset
        runner.data.mocap_quat[self.marker_id]=rotation_quaternion(marker_rotation)
        mujoco.mj_forward(runner.model,runner.data)
        if now+1e-9<self.next_capture:
            return
        self.next_capture+=1/30.
        self.color_renderer.update_scene(runner.data,camera="rgbd_color")
        self.depth_renderer.update_scene(runner.data,camera="rgbd_depth")
        rgb=self.color_renderer.render().copy()
        depth=self.depth_renderer.render().copy()
        occluded=any(start<=now<end for start,end in self.scenario.get("occlusion_intervals_s",[]))
        if occluded:
            rgb[:]=0
        missing=self.scenario.get("depth_missing_probability",0.)
        depth[self.rng.random(depth.shape)<missing]=np.nan
        camera=runner.data.camera("rgbd_color")
        world_from_color=np.eye(4)
        world_from_color[:3,:3]=camera.xmat.reshape(3,3)@np.diag([1.,-1.,-1.])
        world_from_color[:3,3]=camera.xpos
        measurement=self.estimator.measure(rgb,depth,now,world_from_color)
        error=None
        orientation_error=None
        if measurement is not None:
            noisy=np.asarray(measurement.position)+self.rng.normal(0,self.scenario.get("position_noise_m",0.),3)
            measurement=GoalSample(now,tuple(noisy),measurement.orientation_wxyz,measurement.valid,measurement.confidence)
            # Truth is used only below for scoring, never passed to the estimator/controller.
            error=float(np.linalg.norm(noisy-goal))
            orientation_error=orientation_error_rad(measurement.orientation_wxyz,goal_quat)
            self.queue.enqueue(measurement,now+self.scenario.get("latency_s",0.))
        self.samples.append({"capture_s":now,"measured":measurement is not None,"measurement_error_m":error,
            "measurement_orientation_error_rad":orientation_error,"occluded":occluded})
        self.last_rgb=rgb

    def close(self):
        self.color_renderer.close();self.depth_renderer.close()

def attach_marker(runner,asset,scenario,output):
    marker=cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),scenario["marker_id"],160)
    image=np.full((200,200),255,np.uint8);image[20:180,20:180]=marker
    texture=output/"marker.png";Image.fromarray(image).save(texture)
    root=ET.parse(asset/"robot.xml").getroot()
    root.find("compiler").set("meshdir",str((asset/"meshes").resolve()))
    assets=root.find("asset")
    ET.SubElement(assets,"texture",name="vision_marker_tex",type="2d",file=str(texture.resolve()))
    ET.SubElement(assets,"material",name="vision_marker_mat",texture="vision_marker_tex",texrepeat="1 1",specular="0")
    body=ET.SubElement(root.find("worldbody"),"body",name="vision_marker",mocap="true")
    half=scenario["marker_size_m"]/.8/2
    ET.SubElement(body,"geom",type="box",size=f"{half} {half} .0005",material="vision_marker_mat",contype="0",conaffinity="0")
    runner.model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding="unicode"))
    runner.data=mujoco.MjData(runner.model)
    runner.reset()

def run_visual(runner,asset,trajectory,scenario,output,duration=60.,record=False):
    calibration=json.loads((asset/"camera.json").read_text())
    output.mkdir(parents=True,exist_ok=True)
    attach_marker(runner,asset,scenario,output)
    # Reset's local TCP placeholder is not an accepted camera measurement at t=0.
    runner.observations.stamp.fill_(-float("inf"))
    runner.observations.valid.zero_()
    runner.observations.confidence.zero_()
    vision=RenderedVision(runner,calibration,scenario,trajectory,output)
    runner.physics_callback=vision.tick
    writer=None
    if record:
        import imageio.v2 as imageio
        writer=imageio.get_writer(output/"wrist-view.mp4",fps=50,macro_block_size=1)
    try:
        vision.tick(runner)
        for _ in range(round(duration/.02)):
            now=float(runner.data.time)
            sample=vision.queue.update(now)
            vision.hold=vision.queue.hold_required(now)
            runner.step(measurement=sample,hold=vision.hold)
            state=runner.state()
            truth=trajectory.sample(vision.reference_time)
            base=runner.data.body("base_link")
            fallen=bool(base.xpos[2]<.2 or base.xmat.reshape(3,3)[2,2]<.35)
            vision.controls.append({"time_s":float(runner.data.time),"hold":vision.hold,
                "tcp_error_m":float(np.linalg.norm(state.tcp_pos_w.numpy()[0]-truth)),
                "tcp_orientation_error_rad":orientation_error_rad(state.tcp_quat_w.numpy()[0],trajectory.sample_orientation(vision.reference_time)),
                "reference_time_s":vision.reference_time,"fallen":fallen})
            if writer is not None and vision.last_rgb is not None:
                writer.append_data(vision.last_rgb)
            if fallen:
                break
        errors=[r["measurement_error_m"] for r in vision.samples if r["measured"]]
        orientation_errors=[r["measurement_orientation_error_rad"] for r in vision.samples if r["measured"]]
        recoveries=[]
        for _,end in scenario.get("occlusion_intervals_s",[]):
            recovered=next((r["time_s"] for r in vision.controls if r["time_s"]>=end and not r["hold"]),None)
            recoveries.append(None if recovered is None else recovered-end)
        report={"mode":"rendered_rgbd_with_simulated_base_odometry","policy_sha256":runner.bundle["policy_sha256"],
            "asset_hash":runner.manifest["asset_hash"],"camera_hz":30,"control_hz":50,
            "measurement_rmse_m":float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
            "measurement_orientation_rmse_rad":float(np.sqrt(np.mean(np.square(orientation_errors)))) if orientation_errors else None,
            "missed_frame_fraction":float(np.mean([not r["measured"] for r in vision.samples])),
            "hold_fraction":float(np.mean([r["hold"] for r in vision.controls])),
            "fallen":any(r["fallen"] for r in vision.controls),
            "recovery_seconds":recoveries,"scenario":scenario,"captures":vision.samples,"controls":vision.controls}
        (output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
        return report
    finally:
        runner.physics_callback=None
        vision.close()
        if writer is not None:
            writer.close()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("asset","bundle","trajectory","scenario","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--duration",type=float,default=60.)
    parser.add_argument("--record",action="store_true")
    args=parser.parse_args()
    runner=MujocoRunner(args.asset,args.bundle)
    report=run_visual(runner,args.asset,Trajectory.load(args.trajectory),json.loads(args.scenario.read_text()),args.output,args.duration,args.record)
    print(json.dumps({k:v for k,v in report.items() if k not in ("captures","controls")},indent=2))

if __name__=="__main__":
    main()
