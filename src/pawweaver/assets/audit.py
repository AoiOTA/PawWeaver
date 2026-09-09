"""Fail-closed hardware readiness checks; unknown physical values never become zeros."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from pawweaver.contracts import ActuatorSpec


def source_paths(upstream: Path) -> tuple[Path, Path, Path]:
    as2 = upstream / "unitree_as2/robots/as2_description/urdf/as2.urdf"
    piper = upstream / "agilex_piper_h/piper_h/urdf/piper_h_description.urdf"
    gripper = piper.with_name("piper_h_with_gripper_description.xacro")
    return as2, piper, gripper


def inspect_urdf(path: Path) -> dict:
    root = ET.parse(path).getroot()
    links, joints, issues = [], [], []
    for link in root.findall("link"):
        inertia = link.find("inertial")
        if inertia is None:
            continue
        mass = float(inertia.find("mass").get("value"))
        i = inertia.find("inertia")
        matrix = np.array([[float(i.get("ixx")), float(i.get("ixy")), float(i.get("ixz"))],
                           [float(i.get("ixy")), float(i.get("iyy")), float(i.get("iyz"))],
                           [float(i.get("ixz")), float(i.get("iyz")), float(i.get("izz"))]])
        eigenvalues = np.linalg.eigvalsh(matrix)
        if mass <= 0 or eigenvalues.min() <= 0 or eigenvalues.max() > eigenvalues.sum()-eigenvalues.max()+1e-8:
            issues.append(f"Invalid mass or inertia in {link.get('name')}")
        links.append({"name": link.get("name"), "mass_kg": mass,
                      "inertia_kg_m2": matrix.tolist(),
                      "com_m": [float(v) for v in inertia.find("origin").get("xyz", "0 0 0").split()]})
    for joint in root.findall("joint"):
        if joint.get("type") not in ("revolute", "prismatic", "continuous"):
            continue
        limit = joint.find("limit")
        joints.append({"name": joint.get("name"), "type": joint.get("type"),
                       "limit": dict(limit.attrib) if limit is not None else None})
    return {"path": str(path), "mass_kg": sum(link["mass_kg"] for link in links),
            "links": links, "joints": joints, "issues": issues}


REQUIRED_EVIDENCE = {
    "assembly": "AS2 rail-to-base transform, adapter/support geometry and mass properties, wrist/TCP transforms",
    "camera_rigid_body": "DC1 and camera bracket mass, centre of mass, inertia and collision dimensions",
    "cable_rigid_bodies": "Cable mass allocation with source; explicitly document absence if none is fitted",
    "mass_reconciliation": "Explain AS2 17.64 vs 20 kg and Piper-H 4.167 vs 4.5 kg; identify omitted components or correct model revision",
    "arm_joint_mapping": "Resolve manual/URDF limits and zero offsets by FK/CAD comparison",
    "actuators": "18-joint PD, continuous/peak effort interpretation, damping, armature, friction and delay evidence",
}


def audit_hardware(hardware: Path, upstream: Path) -> dict:
    config = json.loads(hardware.read_text())
    issues = []
    reports = {}
    provenance_path = upstream / "provenance.json"
    if not provenance_path.exists():
        issues.append({"field": "upstream", "message": "Run fetch-assets first; provenance missing"})
    else:
        provenance = json.loads(provenance_path.read_text())
        for source in provenance["sources"]:
            for item in source["files"]:
                path = upstream / item["path"]
                if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                    issues.append({"field": "upstream", "message": f"Source checksum mismatch: {item['path']}"})
    for name, path in zip(("as2", "piper_h", "gripper_with_flange"), source_paths(upstream)):
        if path.exists():
            reports[name] = inspect_urdf(path)
            issues.extend({"field": name, "message": value} for value in reports[name]["issues"])
        else:
            issues.append({"field": name, "message": f"Missing source: {path}"})
    overrides = config.get("verified_overrides", {})
    for field, requirement in REQUIRED_EVIDENCE.items():
        value = overrides.get(field)
        if not isinstance(value, dict) or not value.get("source") or not value.get("notes"):
            issues.append({"field": field, "message": requirement})
    actuators = overrides.get("actuators")
    if isinstance(actuators, dict):
        try:
            ActuatorSpec.from_dict(actuators["values"])
        except (KeyError, TypeError, ValueError) as error:
            issues.append({"field": "actuators", "message": f"Invalid actuator contract: {error}"})
    # Geometry and override schemas are checked by the compiler before marking a built asset ready.
    return {"schema_version": 1, "robot": config["robot"], "ready_for_training": not issues,
            "issues": issues, "sources": reports,
            "manual_claims": config["manual_claims"],
            "camera_calibrated": config["camera"].get("calibration") is not None}


def require_hardware_ready(report: dict):
    if not report["ready_for_training"]:
        details = "\n".join(f"  - {issue['field']}: {issue['message']}" for issue in report["issues"])
        raise ValueError("M0 hardware readiness gate is not satisfied:\n" + details)


def write_audit(report: dict, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    (output / "hardware-audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    lines = ["# Hardware readiness audit", "", f"Ready for training: **{report['ready_for_training']}**", "",
             "The public models contain useful inertial and geometric data. Differences below remain unresolved;",
             "this report does not replace missing values with guessed dynamics.", "", "## Source mass totals", ""]
    for name, source in report["sources"].items():
        lines.append(f"- {name}: {source['mass_kg']:.6f} kg")
    lines += ["", "## Required evidence", ""]
    lines += [f"- **{issue['field']}**: {issue['message']}" for issue in report["issues"]]
    (output / "hardware-audit.md").write_text("\n".join(lines)+"\n")

