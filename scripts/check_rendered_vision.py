"""Measure an actual rendered RGB-D marker using a SYNTHETIC camera calibration.

This validates perception geometry with a synthetic, explicitly defined calibration.
"""
import json
from pathlib import Path
import cv2
import mujoco
import numpy as np
from PIL import Image
from pawweaver.cameras import mujoco_intrinsics
from pawweaver.vision import Intrinsics,MarkerEstimator

root=Path("artifacts/vision-fixture")
root.mkdir(parents=True,exist_ok=True)
marker=cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),7,160)
texture=np.full((200,200),255,np.uint8);texture[20:180,20:180]=marker
Image.fromarray(texture).save(root/"marker.png")
intr=Intrinsics(640,480,400,400,321,238,.1,5.)
camera=" ".join(f'{key}="{value}"' for key,value in mujoco_intrinsics(intr).items())
xml=f'''<mujoco><visual><global offwidth="640" offheight="480"/></visual>
<asset><texture name="marker" type="2d" file="{(root/'marker.png').resolve()}"/>
<material name="marker" texture="marker" texrepeat="1 1" texuniform="false" specular="0"/></asset>
<worldbody><camera name="test" {camera}/><body name="target" pos=".07 -.03 -1">
<geom type="box" size=".1 .1 .001" material="marker"/></body></worldbody></mujoco>'''
model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model);mujoco.mj_forward(model,data)
renderer=mujoco.Renderer(model,480,640)
try:
    renderer.update_scene(data,camera="test")
    rgb=renderer.render().copy()
    renderer.enable_depth_rendering()
    depth=renderer.render().copy()
    Image.fromarray(rgb).save(root/"rendered.png")
    estimator=MarkerEstimator(intr,intr,np.eye(4),7,.16,[0,0,0],[1,0,0,0])
    world_from_color=np.diag([1.,-1.,-1.,1.])
    sample=estimator.measure(rgb,depth,0.,world_from_color)
    if sample is None:
        raise RuntimeError("Rendered marker was not measured; inspect artifacts/vision-fixture/rendered.png")
    truth=data.body("target").xpos.copy();truth[2]+=.001
    error=float(np.linalg.norm(np.asarray(sample.position)-truth))
    result={"kind":"synthetic_camera_geometry_test","rgbd_validation":False,"position_error_m":error,
            "measurement":sample.position,"passed":error<.005}
    (root/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    assert result["passed"]
finally:
    renderer.close()
