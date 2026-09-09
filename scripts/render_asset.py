"""Render a source-geometry preview, explicitly labelled as unverified when appropriate."""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image,ImageDraw
import imageio.v2 as imageio
from pawweaver.assets.build import verify_asset
from pawweaver.contracts import JOINT_NAMES,FOOT_NAMES
from pawweaver.training_inputs import training_inputs

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("asset",type=Path)
parser.add_argument("--output",type=Path,default=Path("artifacts/preview"))
parser.add_argument("--video",action="store_true")
args=parser.parse_args()
manifest=verify_asset(args.asset,require_ready=False)
args.output.mkdir(parents=True,exist_ok=True)
model=mujoco.MjModel.from_xml_path(str(args.asset/"robot.xml"))
model.vis.headlight.ambient[:]=.5
model.vis.headlight.diffuse[:]=.8
model.vis.headlight.specular[:]=.2
data=mujoco.MjData(model)
diagnostic=not manifest["ready_for_training"]
_,spec,_=training_inputs(args.asset,diagnostic=diagnostic,
    provisional_spec=Path(__file__).resolve().parents[1]/"configs/diagnostic_actuators.json" if diagnostic else None)
pose=spec.default_pos
for name,q in zip(JOINT_NAMES,pose):
    data.qpos[model.joint(name).qposadr[0]]=q
mujoco.mj_forward(model,data)
data.qpos[2]+=.025-min(data.body(name).xpos[2] for name in FOOT_NAMES)
mujoco.mj_forward(model,data)
renderer=mujoco.Renderer(model,height=720,width=1280)
camera=mujoco.MjvCamera()
camera.lookat[:]=[.1,0,.5];camera.distance=2.1;camera.elevation=-18;camera.azimuth=135
options=mujoco.MjvOption();options.geomgroup[3]=0
label="PawWeaver | AS2 + Piper-H + D435 | SDK zero reference, enabled-control target | provisional; not settled/hardware-verified"
def frame(angle):
    camera.azimuth=angle
    renderer.update_scene(data,camera=camera,scene_option=options)
    result=Image.fromarray(renderer.render())
    draw=ImageDraw.Draw(result)
    draw.rectangle((0,0,1280,44),fill=(20,28,35))
    draw.text((20,15),label if not manifest["ready_for_training"] else "PawWeaver | Verified asset preview",fill=(240,240,240))
    return np.asarray(result)
Image.fromarray(frame(135)).save(args.output/"robot.png")
if args.video:
    with imageio.get_writer(args.output/"asset-orbit.mp4",fps=30,codec="libx264") as writer:
        for angle in np.linspace(135,495,120):
            writer.append_data(frame(float(angle)))
renderer.close()
print(args.output.resolve())
