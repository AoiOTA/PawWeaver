import xml.etree.ElementTree as ET
import pytest
from pawweaver.assets.model import RobotTree


def tree(kind='fixed', reverse=False):
    links=[ET.fromstring(f'<link name="{name}">{"<collision/>" if collision else ""}</link>')
           for name,collision in [('a',True),('massless_frame',False),('b',True),('c',True),('d',True)]]
    joints=[ET.fromstring(f'<joint name="{parent}_{child}" type="{joint_type}"><parent link="{parent}"/><child link="{child}"/></joint>')
            for parent,child,joint_type in [('a','massless_frame','fixed'),('massless_frame','b','fixed'),('b','c',kind),('c','d','fixed')]]
    root=ET.Element('robot')
    root.extend(list(reversed(links+joints)) if reverse else links+joints)
    return RobotTree(root)


def test_fixed_pairs_traverse_massless_frames_and_are_deterministic():
    expected=[('a','b'),('a','c'),('a','d'),('b','c'),('b','d'),('c','d')]
    assert tree().fixed_collision_pairs()==expected
    assert tree(reverse=True).fixed_collision_pairs()==expected


@pytest.mark.parametrize('kind',['revolute','continuous','prismatic'])
def test_fixed_pairs_never_cross_movable_joints(kind):
    assert tree(kind).fixed_collision_pairs()==[('a','b'),('c','d')]
