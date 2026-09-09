"""A canonical URDF tree and independent NumPy forward kinematics."""
from __future__ import annotations
from copy import deepcopy
from itertools import combinations
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from pawweaver.math import axis_angle_matrix, transform


def numbers(text: str | None, default="0 0 0") -> np.ndarray:
    return np.fromstring(default if text is None else text, sep=" ", dtype=float)


def format_vec(values) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def origin(element: ET.Element) -> np.ndarray:
    node = element.find("origin")
    return np.eye(4) if node is None else transform(numbers(node.get("xyz")), numbers(node.get("rpy")))


class RobotTree:
    def __init__(self, root: ET.Element):
        self.xml = root
        self.links = {link.get("name"): link for link in root.findall("link")}
        self.joints = {joint.get("name"): joint for joint in root.findall("joint")}
        self.children = {name: [] for name in self.links}
        child_names = set()
        for name, joint in self.joints.items():
            parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
            if parent not in self.links or child not in self.links or child in child_names:
                raise ValueError(f"Invalid tree edge: {name}")
            self.children[parent].append(name)
            child_names.add(child)
        roots = set(self.links)-child_names
        if len(roots) != 1:
            raise ValueError(f"Expected one robot root, got {roots}")
        self.root_name = roots.pop()
        if len(self.forward({})) != len(self.links):
            raise ValueError("Robot contains disconnected links or a cycle")

    @classmethod
    def load(cls, path: Path) -> "RobotTree":
        return cls(ET.parse(path).getroot())

    def forward(self, positions: dict[str, float], root_transform: np.ndarray | None = None) -> dict[str, np.ndarray]:
        poses = {self.root_name: np.eye(4) if root_transform is None else root_transform.copy()}
        pending = [self.root_name]
        while pending:
            parent = pending.pop()
            for name in self.children[parent]:
                joint = self.joints[name]
                child = joint.find("child").get("link")
                motion = np.eye(4)
                axis = joint.find("axis")
                axis_vector = numbers(axis.get("xyz"), "1 0 0") if axis is not None else np.array([1.,0.,0.])
                kind, q = joint.get("type"), positions.get(name, 0.0)
                if kind in ("revolute", "continuous"):
                    motion[:3,:3] = axis_angle_matrix(axis_vector, q)
                elif kind == "prismatic":
                    motion[:3,3] = axis_vector*q
                elif kind != "fixed":
                    raise ValueError(f"Unsupported joint type: {kind}")
                if child in poses:
                    raise ValueError("Robot joint cycle")
                poses[child] = poses[parent] @ origin(joint) @ motion
                pending.append(child)
        return poses

    def fixed_collision_pairs(self) -> list[tuple[str, str]]:
        """Collision-bearing links welded through any number of fixed, possibly massless frames."""
        neighbors = {name: [] for name in self.links}
        for joint in self.joints.values():
            if joint.get("type") == "fixed":
                parent = joint.find("parent").get("link")
                child = joint.find("child").get("link")
                neighbors[parent].append(child)
                neighbors[child].append(parent)
        remaining = set(self.links)
        pairs = []
        while remaining:
            pending = [min(remaining)]
            component = set()
            while pending:
                name = pending.pop()
                if name in component:
                    continue
                component.add(name)
                pending.extend(neighbors[name])
            remaining.difference_update(component)
            collision_links = sorted(name for name in component if self.links[name].find("collision") is not None)
            pairs.extend(combinations(collision_links, 2))
        return sorted(pairs)

    @property
    def mass(self) -> float:
        return sum(float(link.find("inertial/mass").get("value")) for link in self.links.values()
                   if link.find("inertial/mass") is not None)


def copy_source(source: ET.Element, prefix: str) -> list[ET.Element]:
    """Strip world anchors and prefix named links/joints/materials without xacro evaluation."""
    result = []
    for node in source:
        if node.tag not in ("link", "joint"):
            continue
        if node.tag == "link" and node.get("name") == "world":
            continue
        if node.tag == "joint" and node.find("parent").get("link") == "world":
            continue
        node = deepcopy(node)
        node.set("name", prefix+node.get("name"))
        for element in node.iter():
            if element.tag in ("parent", "child"):
                element.set("link", prefix+element.get("link"))
            elif element.tag == "mimic":
                element.set("joint", prefix+element.get("joint"))
            elif element.tag == "material" and element.get("name"):
                element.set("name", prefix+element.get("name"))
        result.append(node)
    return result

