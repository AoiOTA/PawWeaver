"""Generate URDF and MJCF from one parameter tree. Diagnostic output is never training-ready."""
from __future__ import annotations
import hashlib
import json
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from pawweaver.contracts import ActuatorSpec, JOINT_NAMES, canonical_hash
from pawweaver.math import rpy_matrix, rpy_quat
from .audit import audit_hardware, require_hardware_ready, source_paths, write_audit
from .model import RobotTree, copy_source, format_vec, numbers


def fixed_joint(name: str, parent: str, child: str, xyz, rpy=(0.,0.,0.)) -> ET.Element:
    joint = ET.Element("joint", name=name, type="fixed")
    ET.SubElement(joint,"parent",link=parent)
    ET.SubElement(joint,"child",link=child)
    ET.SubElement(joint,"origin",xyz=format_vec(xyz),rpy=format_vec(rpy))
    return joint


def set_inertial(link: ET.Element, value: dict):
    mass = float(value["mass_kg"])
    com = np.asarray(value["com_m"], dtype=float)
    inertia = np.asarray(value["inertia_kg_m2"], dtype=float)
    if com.shape != (3,) or inertia.shape != (3,3) or not np.isfinite(inertia).all():
        raise ValueError("Invalid rigid body mass properties")
    eig = np.linalg.eigvalsh(inertia)
    if mass <= 0 or not np.isfinite(com).all() or not np.allclose(inertia,inertia.T) or eig.min() <= 0:
        raise ValueError("Mass must be positive and inertia symmetric positive definite")
    if eig.max() > eig.sum()-eig.max()+1e-9:
        raise ValueError("Principal inertia violates triangle inequality")
    old = link.find("inertial")
    if old is not None:
        link.remove(old)
    node = ET.SubElement(link,"inertial")
    ET.SubElement(node,"mass",value=str(mass))
    ET.SubElement(node,"origin",xyz=format_vec(com),rpy="0 0 0")
    ET.SubElement(node,"inertia",ixx=str(inertia[0,0]),iyy=str(inertia[1,1]),izz=str(inertia[2,2]),
                  ixy=str(inertia[0,1]),ixz=str(inertia[0,2]),iyz=str(inertia[1,2]))


def append_rigid_body(root: ET.Element, value: dict):
    if value["name"] in {node.get("name") for node in root.findall("link")}:
        raise ValueError(f"Duplicate body {value['name']}")
    size = np.asarray(value["collision_size_m"], dtype=float)
    if size.shape != (3,) or not np.isfinite(size).all() or (size <= 0).any():
        raise ValueError("Collision box needs three positive dimensions")
    link = ET.SubElement(root,"link",name=value["name"])
    set_inertial(link,value)
    for kind in ("visual","collision"):
        node = ET.SubElement(link,kind)
        ET.SubElement(node,"origin",xyz=format_vec(value.get("geometry_center_m",[0,0,0])),rpy="0 0 0")
        ET.SubElement(ET.SubElement(node,"geometry"),"box",size=format_vec(size))
        if kind == "visual":
            mat = ET.SubElement(node,"material",name=value["name"]+"_material")
            ET.SubElement(mat,"color",rgba="0.22 0.25 0.3 1")
    root.append(fixed_joint(value["name"]+"_mount",value["parent"],value["name"],
                            value["xyz_m"],value.get("rpy_rad",[0,0,0])))


def compose_urdf(hardware: dict, upstream: Path, output: Path, diagnostic: bool) -> tuple[RobotTree, dict]:
    as2, arm, gripper = source_paths(upstream)
    root = ET.Element("robot",name="pawweaver_as2_edu_piper_h")
    root.extend(copy_source(ET.parse(as2).getroot(),""))
    root.extend(copy_source(ET.parse(arm).getroot(),"arm_"))
    root.extend(copy_source(ET.parse(gripper).getroot(),"arm_"))

    # Remove the massless mimic driver and freeze both physical fingers at the requested width.
    width = float(hardware["gripper_width_m"])
    if not 0 <= width <= 0.1:
        raise ValueError("Gripper opening must be in [0,0.1] m")
    for node in list(root):
        if node.get("name") in ("arm_gripper", "arm_gripper_link"):
            root.remove(node)
        elif node.tag == "joint" and node.find("mimic") is not None:
            q = width*float(node.find("mimic").get("multiplier","1"))
            node_origin = node.find("origin")
            xyz = numbers(node_origin.get("xyz"))
            xyz += rpy_matrix(numbers(node_origin.get("rpy"))) @ numbers(node.find("axis").get("xyz"))*q
            node_origin.set("xyz",format_vec(xyz))
            node.set("type","fixed")
            for tag in ("axis","limit","mimic"):
                child = node.find(tag)
                if child is not None:
                    node.remove(child)

    # Resolve STL equivalents for the vendor's DAE visuals. Preserve scale and origin.
    mesh_dir = output/"meshes"
    mesh_dir.mkdir(parents=True,exist_ok=True)
    for link in root.findall("link"):
        is_arm = link.get("name").startswith("arm_")
        base = arm.parent.parent/"meshes" if is_arm else as2.parent.parent/"meshes"
        for mesh in link.findall(".//mesh"):
            basename = Path(mesh.get("filename")).name
            if basename.endswith(".dae"):
                basename = basename[:-4]+".stl"
            source = base/basename
            if not source.is_file():
                raise FileNotFoundError(source)
            target = mesh_dir/(("arm_" if is_arm else "as2_")+basename)
            if not target.exists() or hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(source.read_bytes()).digest():
                shutil.copyfile(source,target)
            mesh.set("filename",str(target.relative_to(output)))

    notes = {}
    overrides = hardware["verified_overrides"]
    if diagnostic:
        # A source-only assembly preview: no guessed DC1 mass or actuator calibration.
        # Mount is deliberately above the full source base mesh; NOT a verified rail transform.
        import trimesh
        bounds = trimesh.load(as2.parent.parent/"meshes/base_link.STL").bounds
        mount = [0.,0.,float(bounds[1,2])+0.01]
        root.append(fixed_joint("arm_mount","base_link","arm_base_link",mount))
        tcp = [0.,0.,0.138]  # terminal finger plane in the official fixed-width gripper geometry
        notes = {"mode":"source_geometry_diagnostic", "omitted_unverified_components":["adapter","DC1","camera bracket","cables"],
                 "mount_xyz_m":mount,"mount_rule":"source base mesh upper bound + 10 mm clearance; visualization only",
                 "tcp_rule":"official gripper finger terminal plane z=0.138 m in arm_gripper_base"}
    else:
        assembly = overrides["assembly"]["values"]
        root.append(fixed_joint("arm_mount","base_link","arm_base_link",assembly["arm_mount_xyz_m"],assembly["arm_mount_rpy_rad"]))
        append_rigid_body(root,assembly["adapter"])
        tcp = assembly["tcp_xyz_m"]
        append_rigid_body(root,overrides["camera_rigid_body"]["values"])
        if overrides["camera_rigid_body"]["values"]["name"] != "dc1_body":
            raise ValueError("Camera rigid body must be named dc1_body")
        for body in overrides["cable_rigid_bodies"]["values"]:
            append_rigid_body(root,body)
        reconciled = overrides["mass_reconciliation"]["values"]
        for body in reconciled.get("additional_bodies",[]):
            append_rigid_body(root,body)
        links = {link.get("name"):link for link in root.findall("link")}
        for name, value in reconciled.get("link_inertials",{}).items():
            set_inertial(links[name],value)
        joints = {joint.get("name"):joint for joint in root.findall("joint")}
        mapping = overrides["arm_joint_mapping"]["values"]
        if set(mapping) != {f"arm_joint{i}" for i in range(1,7)}:
            raise ValueError("Resolve all six arm joints explicitly")
        for name, value in mapping.items():
            joint = joints[name]
            joint.find("limit").set("lower",str(value["lower_rad"]))
            joint.find("limit").set("upper",str(value["upper_rad"]))
            if "origin_xyz_m" in value:
                joint.find("origin").set("xyz",format_vec(value["origin_xyz_m"]))
            if "origin_rpy_rad" in value:
                joint.find("origin").set("rpy",format_vec(value["origin_rpy_rad"]))
        spec = ActuatorSpec.from_dict(overrides["actuators"]["values"])
        for i, name in enumerate(spec.joint_names):
            limit = joints[name].find("limit")
            if not np.allclose([float(limit.get("lower")),float(limit.get("upper"))],[spec.lower[i],spec.upper[i]],atol=1e-6):
                raise ValueError(f"Actuator and geometric limit mismatch: {name}")
            limit.set("effort",str(spec.effort[i]))
            limit.set("velocity",str(spec.velocity[i]))
            old = joints[name].find("dynamics")
            if old is not None:
                joints[name].remove(old)
            ET.SubElement(joints[name],"dynamics",damping=str(spec.damping[i]),friction=str(spec.frictionloss[i]))
        (output/"actuators.json").write_text(json.dumps(spec.to_dict(),indent=2)+"\n")
        ET.SubElement(root,"link",name="dc1_optical")
        root.append(fixed_joint("dc1_optical_mount","dc1_body","dc1_optical",assembly["optical_xyz_m"],assembly["optical_rpy_rad"]))
    ET.SubElement(root,"link",name="tcp")
    root.append(fixed_joint("tcp_mount","arm_gripper_base","tcp",tcp))
    tree = RobotTree(root)
    actual = {name for name, joint in tree.joints.items() if joint.get("type") != "fixed"}
    if actual != set(JOINT_NAMES):
        raise ValueError(f"Expected exactly 18 actuated joints; got {actual}")
    ET.indent(root)
    ET.ElementTree(root).write(output/"robot.urdf",encoding="utf-8",xml_declaration=True)
    return tree, notes


def generate_mjcf(tree: RobotTree, output: Path, spec: ActuatorSpec | None, calibration: dict | None = None) -> Path:
    """Translate canonical geometry and full COM inertias, without MuJoCo URDF importer defaults."""
    root = ET.Element("mujoco",model="pawweaver")
    ET.SubElement(root,"compiler",angle="radian",autolimits="true",meshdir="meshes",inertiafromgeom="false",fusestatic="false")
    ET.SubElement(root,"option",timestep="0.002",gravity="0 0 -9.81",integrator="implicitfast",cone="elliptic",iterations="100")
    visual = ET.SubElement(root,"visual")
    ET.SubElement(visual,"global",offwidth="1280",offheight="720")
    defaults = ET.SubElement(root,"default")
    ET.SubElement(defaults,"geom",friction="0.8 0.01 0.001",condim="3",solref="0.01 1",solimp="0.9 0.95 0.001")
    assets, world = ET.SubElement(root,"asset"), ET.SubElement(root,"worldbody")
    ET.SubElement(world,"light",pos="0 -2 4",dir="0 0 -1",diffuse="0.8 0.8 0.8")
    ET.SubElement(world,"geom",name="floor",type="plane",size="200 200 0.1",rgba="0.18 0.22 0.24 1")
    contact = ET.SubElement(root,"contact")
    mesh_names = {}
    actuators = ET.SubElement(root,"actuator")
    limits = {}
    indices = {name:index for index,name in enumerate(JOINT_NAMES)}

    def body(parent: ET.Element, name: str, joint: ET.Element | None = None):
        link = tree.links[name]
        xyz, rpy = np.zeros(3), np.zeros(3)
        if joint is not None and joint.find("origin") is not None:
            xyz = numbers(joint.find("origin").get("xyz"))
            rpy = numbers(joint.find("origin").get("rpy"))
        if joint is None:
            xyz[2] = 0.50
        node = ET.SubElement(parent,"body",name=name,pos=format_vec(xyz),quat=format_vec(rpy_quat(rpy)))
        if joint is None:
            ET.SubElement(node,"freejoint",name="floating_base")
        elif joint.get("type") != "fixed":
            i = indices[joint.get("name")]
            limit = joint.find("limit")
            axis = joint.find("axis")
            attrs = {"name":joint.get("name"),"type":"hinge" if joint.get("type") == "revolute" else "slide",
                     "axis":axis.get("xyz","1 0 0"),"range":f"{limit.get('lower')} {limit.get('upper')}"}
            attrs.update(damping=str(spec.damping[i] if spec else 0.),
                         frictionloss=str(spec.frictionloss[i] if spec else 0.),
                         armature=str(spec.armature[i] if spec else 0.))
            ET.SubElement(node,"joint",**attrs)
            limits[joint.get("name")] = spec.effort[i] if spec else float(limit.get("effort"))
        inertial = link.find("inertial")
        if inertial is not None:
            item = inertial.find("inertia")
            matrix = np.array([[float(item.get("ixx")),float(item.get("ixy")),float(item.get("ixz"))],
                               [float(item.get("ixy")),float(item.get("iyy")),float(item.get("iyz"))],
                               [float(item.get("ixz")),float(item.get("iyz")),float(item.get("izz"))]])
            io = inertial.find("origin")
            rotation = rpy_matrix(numbers(io.get("rpy")))
            matrix = rotation @ matrix @ rotation.T
            ET.SubElement(node,"inertial",mass=inertial.find("mass").get("value"),pos=io.get("xyz","0 0 0"),
                          fullinertia=format_vec([matrix[0,0],matrix[1,1],matrix[2,2],matrix[0,1],matrix[0,2],matrix[1,2]]))
        for kind in ("visual","collision"):
            for index, geometry in enumerate(link.findall(kind)):
                go, shape = geometry.find("origin"), list(geometry.find("geometry"))[0]
                attrs = {"name":f"{name}_{kind}_{index}","group":"1" if kind == "visual" else "3",
                         "contype":"0" if kind == "visual" else "1", "conaffinity":"0" if kind == "visual" else "1"}
                if go is not None:
                    attrs.update(pos=go.get("xyz","0 0 0"),quat=format_vec(rpy_quat(numbers(go.get("rpy")))))
                if shape.tag == "mesh":
                    key = (shape.get("filename"),shape.get("scale","1 1 1"))
                    if key not in mesh_names:
                        mesh_name = f"mesh_{len(mesh_names)}"
                        mesh_names[key] = mesh_name
                        ET.SubElement(assets,"mesh",name=mesh_name,file=Path(key[0]).name,scale=key[1])
                    attrs.update(type="mesh",mesh=mesh_names[key])
                elif shape.tag == "box":
                    attrs.update(type="box",size=format_vec(numbers(shape.get("size"))*.5))
                elif shape.tag == "sphere":
                    attrs.update(type="sphere",size=shape.get("radius"))
                elif shape.tag == "cylinder":
                    attrs.update(type="cylinder",size=f"{shape.get('radius')} {float(shape.get('length'))*.5}")
                else:
                    raise ValueError(f"Unsupported URDF geometry {shape.tag}")
                color = geometry.find("material/color")
                attrs["rgba"] = color.get("rgba") if color is not None else ("0.65 0.7 0.78 1" if not name.startswith("arm_") else "0.25 0.28 0.33 1")
                ET.SubElement(node,"geom",**attrs)
        if name == "tcp":
            ET.SubElement(node,"site",name="tcp_site",type="sphere",size="0.008",rgba="0 1 0 1")
        if name == "dc1_optical" and calibration is not None:
            # OpenCV +Z forward,+Y down -> MuJoCo -Z forward,+Y up.
            fy, height = float(calibration["fy"]), int(calibration["height"])
            if fy <= 0 or height <= 0:
                raise ValueError("Invalid calibrated camera focal length or image height")
            ET.SubElement(node,"camera",name="dc1",quat="0 1 0 0",fovy=str(np.degrees(2*np.arctan(height/(2*fy)))))
        for child_joint in tree.children[name]:
            cj = tree.joints[child_joint]
            child_name = cj.find("child").get("link")
            body(node,child_name,cj)
            ET.SubElement(contact,"exclude",body1=name,body2=child_name)

    body(world,tree.root_name)
    for name in JOINT_NAMES:
        limit = limits[name]
        ET.SubElement(actuators,"motor",name=name+"_motor",joint=name,gear="1",ctrllimited="true",ctrlrange=f"{-limit} {limit}")
    ET.indent(root)
    path = output/"robot.xml"
    ET.ElementTree(root).write(path,encoding="utf-8",xml_declaration=True)
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(path))
    if model.nu != 18 or model.nv != 24:
        raise ValueError(f"Unexpected DOFs: nu={model.nu}, nv={model.nv}")
    if abs(model.body_mass.sum()-tree.mass) > 1e-7:
        raise ValueError("MJCF conversion changed total mass")
    return path


def build_assets(hardware_path: Path, upstream: Path, output: Path, diagnostic=False) -> dict:
    report = audit_hardware(hardware_path,upstream)
    if not diagnostic:
        require_hardware_ready(report)
    output.mkdir(parents=True,exist_ok=True)
    write_audit(report,output)
    hardware = json.loads(hardware_path.read_text())
    tree, notes = compose_urdf(hardware,upstream,output,diagnostic)
    spec = None if diagnostic else ActuatorSpec.from_dict(hardware["verified_overrides"]["actuators"]["values"])
    generate_mjcf(tree,output,spec,None if diagnostic else hardware["camera"].get("calibration"))
    files = {str(path.relative_to(output)):hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(output.rglob('*')) if path.is_file() and path.name != "manifest.json"}
    manifest = {"schema_version":1,"robot":hardware["robot"],"ready_for_training":not diagnostic and report["ready_for_training"],
                "camera_calibrated":not diagnostic and report["camera_calibrated"],"mass_kg":tree.mass,
                "hardware_hash":canonical_hash(hardware),"joint_names":list(JOINT_NAMES),"files":files,"notes":notes}
    manifest["asset_hash"] = canonical_hash(manifest)
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


def verify_asset(directory: Path, require_ready=True) -> dict:
    manifest = json.loads((directory/"manifest.json").read_text())
    expected_hash = manifest.pop("asset_hash")
    if canonical_hash(manifest) != expected_hash:
        raise ValueError("Asset manifest checksum mismatch")
    manifest["asset_hash"] = expected_hash
    if require_ready and not manifest["ready_for_training"]:
        raise ValueError("Diagnostic/source-only assets cannot be used for formal training or acceptance evaluation")
    for name, checksum in manifest["files"].items():
        path = directory/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Asset checksum mismatch: {name}")
    return manifest
