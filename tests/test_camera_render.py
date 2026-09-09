"""Actual MuJoCo RGB pixels validate off-center intrinsics, not just projection algebra."""
import os
import xml.etree.ElementTree as ET
import numpy as np
import pytest
import mujoco
from pawweaver.vision import Intrinsics
from pawweaver.cameras import mujoco_intrinsics

@pytest.mark.skipif(os.environ.get("MUJOCO_GL")!="egl",reason="Set MUJOCO_GL=egl for offscreen render checks")
def test_calibrated_pixel_projection():
    intr=Intrinsics(320,240,200,210,171,126,.1,5.)
    root=ET.Element("mujoco")
    world=ET.SubElement(root,"worldbody")
    ET.SubElement(world,"camera",name="test",**mujoco_intrinsics(intr))
    ET.SubElement(world,"geom",type="sphere",pos="0.1 -0.05 -1",size=".015",rgba="1 0 0 1")
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding="unicode"))
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    renderer=mujoco.Renderer(model,240,320)
    try:
        renderer.update_scene(data,camera="test")
        rgb=renderer.render()
        v,u=np.where((rgb[:,:,0]>100)&(rgb[:,:,1]<70))
        assert len(u)>10
        assert abs(u.mean()-(intr.cx+intr.fx*.1))<.6
        assert abs(v.mean()-(intr.cy+intr.fy*.05))<.6
    finally:
        renderer.close()
