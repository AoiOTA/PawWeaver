"""D435 source geometry with explicitly provisional dynamics for diagnostic previews."""
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from .model import format_vec

def add_d435_reference(root,upstream:Path,output:Path,profile:dict):
    from .build import fixed_joint,set_inertial
    directory=upstream/"agilex_d435"
    source=ET.parse(directory/"agx_arm_description/urdf/piper_h_gripper_d435.urdf").getroot()
    names={"camera_stand_link":"rgbd_stand","d435_camera_link":"rgbd_mount",
           "camera_bottom_screw_frame":"rgbd_bottom_screw","camera_link":"rgbd_body"}
    for original in source.findall("link"):
        if original.get("name") not in names:
            continue
        node=copy.deepcopy(original);node.set("name",names[original.get("name")])
        for mesh in node.findall(".//mesh"):
            filename=mesh.get("filename").removeprefix("package://")
            destination=output/"meshes"/(Path(filename).stem+".obj")
            if not destination.exists():
                scene=trimesh.load(directory/filename,force="scene")
                scene.to_geometry().export(destination)
            mesh.set("filename",str(destination.relative_to(output)))
        if node.get("name")=="rgbd_body":
            # The vendor URDF explicitly labels its inertia unreliable. Do not copy it.
            # A transparent uniform-box approximation is used ONLY in the diagnostic asset.
            x,y,z=profile["body_size_xyz_m"]
            mass=profile["nominal_mass_kg"]
            set_inertial(node,{"mass_kg":mass,"com_m":[0,-.0175,0],"inertia_kg_m2":np.diag(
                [mass*(y*y+z*z)/12,mass*(x*x+z*z)/12,mass*(x*x+y*y)/12]).tolist()})
        if node.get("name")=="rgbd_stand":
            # Official preview mesh rotations differed for visual and collision.
            node.find("collision/origin").attrib.update(node.find("visual/origin").attrib)
        root.append(node)
    for original in source.findall("joint"):
        child=original.find("child").get("link")
        if child not in names:
            continue
        node=copy.deepcopy(original);node.set("name","reference_"+node.get("name"))
        parent=node.find("parent").get("link")
        node.find("parent").set("link","arm_gripper_base" if parent=="gripper_base" else names[parent])
        node.find("child").set("link",names[child])
        root.append(node)
    ET.SubElement(root,"link",name="rgbd_optical")
    root.append(fixed_joint("rgbd_optical_mount","rgbd_body","rgbd_optical",[0,.015,0],[-np.pi/2,0,-np.pi/2]))
    return {"model":"RealSense D435","geometry":"AgileX Piper-H D435 reference assembly",
            "camera_mass_kg":profile["nominal_mass_kg"],"camera_inertia":"Provisional uniform box using datasheet mass and size",
            "bracket_mass_kg":.2,"bracket_inertia":"Unverified values from AgileX reference URDF",
            "calibration":"Virtual nominal pinhole model, not a measured device calibration",
            "source":profile["mount_source"],"hardware_validation":False}
