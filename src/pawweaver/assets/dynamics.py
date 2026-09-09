"""Resolve named inertials from Unitree's official AS2 MuJoCo reference."""
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from .model import numbers,origin

def as2_dynamics(root,upstream):
    from .build import set_inertial
    source=upstream/"unitree_as2_dynamics/unitree_robots/as2/as2.xml"
    model=ET.parse(source).getroot()
    links={n.get("name"):n for n in root.findall("link")}
    by_child={j.find("child").get("link"):j for j in root.findall("joint")}
    changes=[]
    for body in model.findall(".//body"):
        name=body.get("name");inertial=body.find("inertial")
        if inertial is None:
            continue
        if name not in links:
            raise ValueError(f"Reference AS2 body absent from URDF: {name}")
        if name!="base_link":
            transform=origin(by_child[name])
            if not np.allclose(transform[:3,3],numbers(body.get("pos")),atol=1e-6) or not np.allclose(transform[:3,:3],np.eye(3),atol=1e-7):
                raise ValueError(f"AS2 reference joint frame differs: {name}")
            if body.get("quat") or body.get("euler"):
                raise ValueError("Review newly rotated AS2 body frame")
            joint=body.find("joint")
            if joint is not None and not np.allclose(numbers(joint.get("axis")),numbers(by_child[name].find("axis").get("xyz"))):
                raise ValueError(f"AS2 reference joint axis differs: {name}")
        rotation=np.zeros(9)
        mujoco.mju_quat2Mat(rotation,numbers(inertial.get("quat"),"1 0 0 0"))
        rotation=rotation.reshape(3,3)
        inertia=rotation@np.diag(numbers(inertial.get("diaginertia")))@rotation.T
        previous=float(links[name].find("inertial/mass").get("value"))
        value={"mass_kg":float(inertial.get("mass")),"com_m":numbers(inertial.get("pos")).tolist(),"inertia_kg_m2":inertia.tolist()}
        set_inertial(links[name],value)
        changes.append({"link":name,"urdf_mass_kg":previous,"reference":value})
    return {"source":"https://github.com/unitreerobotics/unitree_mujoco/blob/1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d/unitree_robots/as2/as2.xml",
        "reference_mass_kg":sum(x["reference"]["mass_kg"] for x in changes),
        "original_urdf_mass_kg":sum(x["urdf_mass_kg"] for x in changes),"bodies":changes,
        "interpretation":"Official simulation reference; body frames and axes matched before replacing named inertials. EDU physical configuration remains to be confirmed."}
