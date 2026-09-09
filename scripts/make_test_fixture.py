"""Generate a tiny 18-joint synthetic robot ONLY for software integration tests.

This is deliberately box geometry with invented, explicitly synthetic inertias.
It contains no AS2/Piper asset and is not a hardware model or acceptance result.
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from pawweaver.assets.build import fixed_joint,set_inertial,generate_mjcf
from pawweaver.assets.model import RobotTree
from pawweaver.contracts import ActuatorSpec,JOINT_NAMES,canonical_hash

output=Path("artifacts/software-fixture")
output.mkdir(parents=True,exist_ok=True)
root=ET.Element("robot",name="synthetic_software_fixture")
def link(name,mass,size):
    node=ET.SubElement(root,"link",name=name)
    x,y,z=size
    set_inertial(node,{"mass_kg":mass,"com_m":[0,0,0],"inertia_kg_m2":[
        [mass*(y*y+z*z)/12,0,0],[0,mass*(x*x+z*z)/12,0],[0,0,mass*(x*x+y*y)/12]]})
    for kind in ("visual","collision"):
        ET.SubElement(ET.SubElement(ET.SubElement(node,kind),"geometry"),"box",size=" ".join(map(str,size)))
def joint(name,parent,child,xyz):
    node=ET.SubElement(root,"joint",name=name,type="revolute")
    ET.SubElement(node,"parent",link=parent);ET.SubElement(node,"child",link=child)
    ET.SubElement(node,"origin",xyz=" ".join(map(str,xyz)),rpy="0 0 0")
    ET.SubElement(node,"axis",xyz="0 1 0")
    ET.SubElement(node,"limit",lower="-2",upper="2",effort="5",velocity="4")
link("base_link",5.,[.4,.3,.1])
for leg,x,y in (("FR",.2,-.15),("FL",.2,.15),("RR",-.2,-.15),("RL",-.2,.15)):
    parent="base_link"
    for number,part in enumerate(("hip","thigh","calf")):
        name=f"{leg}_{part}"
        link(name,.15,[.03,.03,.1])
        joint(name+"_joint",parent,name,[x,y,-.05] if number==0 else [0,0,-.13])
        parent=name
    link(leg+"_foot",.05,[.04,.04,.03])
    root.append(fixed_joint(leg+"_foot_fixed",parent,leg+"_foot",[0,0,-.08]))
parent="base_link"
for number in range(1,7):
    name=f"arm_link{number}"
    link(name,.05,[.02,.02,.07])
    joint(f"arm_joint{number}",parent,name,[0,0,.08])
    parent=name
link("arm_gripper_base",.05,[.04,.04,.02])
root.append(fixed_joint("gripper_mount",parent,"arm_gripper_base",[0,0,.06]))
ET.SubElement(root,"link",name="tcp")
root.append(fixed_joint("tcp_mount","arm_gripper_base","tcp",[0,0,.04]))
ET.SubElement(root,"link",name="rgbd_optical")
root.append(fixed_joint("fixture_camera_mount","arm_gripper_base","rgbd_optical",[.08,0,0],[0,1.5707963267948966,0]))
spec=ActuatorSpec(JOINT_NAMES,(0.,)*18,(.2,)*18,(.2,)*18,(.005,)*18,(-2.,)*18,(2.,)*18,
                  (5.,)*18,(4.,)*18,(0.,)*18,(0.,)*18,(0.,)*18,(0,)*18)
tree=RobotTree(root)
ET.indent(root)
ET.ElementTree(root).write(output/"robot.urdf",encoding="utf-8",xml_declaration=True)
calibration={"color":dict(width=640,height=480,fx=400,fy=400,cx=321,cy=238,min_depth=.1,max_depth=5.),
             "depth":dict(width=640,height=400,fx=400,fy=400,cx=318,cy=201,min_depth=.1,max_depth=5.),
             "depth_to_color":[[1,0,0,.02],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
             "provenance":"Synthetic calibration for software test, not a physical camera measurement"}
generate_mjcf(tree,output,spec,calibration)
(output/"camera.json").write_text(json.dumps(calibration,indent=2)+"\n")
(output/"actuators.json").write_text(json.dumps(spec.to_dict(),indent=2)+"\n")
files={name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in ("robot.urdf","robot.xml","actuators.json","camera.json")}
manifest={"schema_version":1,"robot":"synthetic_software_fixture","ready_for_training":True,
          "camera_calibrated":False,"mass_kg":tree.mass,"files":files,"joint_names":JOINT_NAMES,
          "notes":"Synthetic boxes for software tests. No AS2 or Piper data; not hardware validation."}
manifest["asset_hash"]=canonical_hash(manifest)
(output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(output.resolve())
