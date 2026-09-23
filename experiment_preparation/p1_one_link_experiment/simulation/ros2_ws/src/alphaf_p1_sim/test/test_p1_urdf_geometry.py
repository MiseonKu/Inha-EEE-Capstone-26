from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
XACRO_PATH = ROOT / "urdf" / "so101_p1_one_link.xacro"
URDF_PATH = ROOT / "urdf" / "so101_p1_one_link.urdf"


def _expanded_root() -> ET.Element:
    assert XACRO_PATH.exists()
    assert URDF_PATH.exists()
    result = subprocess.run(["xacro", str(XACRO_PATH)], check=True, text=True, capture_output=True)
    return ET.fromstring(result.stdout)


def _xyz(element: ET.Element) -> list[float]:
    return [float(value) for value in element.attrib["xyz"].split()]


def test_urdf_xacro_exists_and_required_links_are_present():
    root = _expanded_root()
    links = {link.attrib["name"] for link in root.findall("link")}
    assert {"base_link", "yaw_link", "pitch_link", "link_1_custom", "link1_tip", "tool0"}.issubset(links)


def test_only_p1_yaw_and_pitch_are_active_joints():
    root = _expanded_root()
    active = {joint.attrib["name"] for joint in root.findall("joint") if joint.attrib["type"] in {"revolute", "continuous"}}
    assert active == {"joint_1_yaw", "joint_2_pitch"}
    assert not any(any(word in name for word in ["elbow", "wrist", "gripper"]) for name in active)


def test_link1_tip_and_tool0_geometry():
    root = _expanded_root()
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    tip_origin = _xyz(joints["link1_tip_joint"].find("origin"))
    tool_origin = _xyz(joints["tool0_joint"].find("origin"))
    assert tip_origin == [0.274123, 0.0, 0.0]
    assert tool_origin == [0.0, 0.0, 0.0]


def test_collision_box_matches_p1_bar_geometry():
    root = _expanded_root()
    link = next(link for link in root.findall("link") if link.attrib["name"] == "link_1_custom")
    collision = link.find("collision")
    origin = _xyz(collision.find("origin"))
    box_size = [float(value) for value in collision.find("geometry").find("box").attrib["size"].split()]
    assert box_size == [0.233, 0.040, 0.012]
    assert origin == [0.157623, 0.0, 0.0]
