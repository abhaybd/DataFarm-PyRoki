"""DROID-style Panda + Robotiq scene, downloaded from pinned MuJoCo Menagerie assets."""

from __future__ import annotations

import copy
import hashlib
import json
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

MENAGERIE_COMMIT = "cda5236f98383f6fbaa4820c6b18dba892cd087f"
HOME = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.7854])
CUBE_SIZE = 0.04
STARTS = np.array([[0.42, -0.23, 0.022], [0.51, -0.25, 0.022], [0.55, -0.12, 0.022]])
DISC_CENTER = np.array([0.48, 0.22])
DISC_RADIUS = 0.145
DISC_TOP = 0.016
TARGETS = np.array([[0.425, 0.19, 0.039], [0.525, 0.19, 0.039], [0.475, 0.275, 0.039]])
COLORS = [(0.92, 0.22, 0.24), (0.18, 0.55, 0.95), (1.0, 0.68, 0.12)]


def fetch_assets(directory: Path) -> Path:
    """Download only the two robot folders, keeping binaries in an ignored cache."""
    root = directory / MENAGERIE_COMMIT
    marker = root / "download_complete.json"
    if marker.exists():
        return root
    url = f"https://api.github.com/repos/google-deepmind/mujoco_menagerie/git/trees/{MENAGERIE_COMMIT}?recursive=1"
    request = urllib.request.Request(url, headers={"User-Agent": "DataFarm-PyRoki-example"})
    with urllib.request.urlopen(request, timeout=60) as stream:
        entries = json.load(stream)["tree"]
    selected = [
        entry
        for entry in entries
        if entry["type"] == "blob"
        and entry["path"].startswith(("franka_emika_panda/", "robotiq_2f85/"))
        and not entry["path"].endswith((".png", ".jpg", ".mp4"))
    ]

    def download(entry):
        destination = root / entry["path"]
        if destination.exists():
            data = destination.read_bytes()
        else:
            remote = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{MENAGERIE_COMMIT}/{entry['path']}"
            with urllib.request.urlopen(remote, timeout=60) as response:
                data = response.read()
        digest = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if digest != entry["sha"]:
            raise ValueError(f"Menagerie blob checksum mismatch: {entry['path']}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(download, selected))
    marker.write_text(json.dumps({"commit": MENAGERIE_COMMIT, "files": selected}, indent=2))
    return root


def _absolute_meshes(root, folder):
    for mesh in root.findall("asset/mesh"):
        if mesh.get("name") is None:
            mesh.set("name", Path(mesh.get("file")).stem)
        mesh.set("file", str((folder / "assets" / mesh.get("file")).resolve()))


def build_scene(asset_root: Path, output: Path):
    panda = ET.parse(asset_root / "franka_emika_panda/panda_nohand.xml").getroot()
    gripper = ET.parse(asset_root / "robotiq_2f85/2f85.xml").getroot()
    _absolute_meshes(panda, asset_root / "franka_emika_panda")
    _absolute_meshes(gripper, asset_root / "robotiq_2f85")
    # Prefix all gripper identifiers, including default classes and references.
    identifiers = {element.get("name") for element in gripper.iter() if element.get("name")}
    identifiers.update(
        element.get("class") for element in gripper.iter("default") if element.get("class")
    )
    for element in gripper.iter():
        for key, value in list(element.attrib.items()):
            if (
                key
                in (
                    "name",
                    "class",
                    "childclass",
                    "mesh",
                    "material",
                    "joint",
                    "joint1",
                    "joint2",
                    "tendon",
                    "body1",
                    "body2",
                )
                and value in identifiers
            ):
                element.set(key, "rq_" + value)
    for default in gripper.findall(".//default"):
        if default.get("class", "").startswith("rq_pad_box"):
            default.find("geom").set("friction", "1.4 0.01 0.001")
    scene = ET.Element("mujoco", model="DROID Franka — three cubes onto a disc")
    ET.SubElement(scene, "compiler", angle="radian", autolimits="true")
    ET.SubElement(
        scene,
        "option",
        timestep="0.002",
        integrator="implicitfast",
        cone="elliptic",
        impratio="10",
        iterations="60",
    )
    ET.SubElement(scene, "size", njmax="3000", nconmax="1000")
    visual = ET.SubElement(scene, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="960")
    ET.SubElement(visual, "quality", shadowsize="2048", offsamples="4")
    ET.SubElement(
        visual, "headlight", ambient="0.35 0.35 0.35", diffuse="0.7 0.7 0.7", specular="0.2 0.2 0.2"
    )
    ET.SubElement(visual, "rgba", haze="0.12 0.15 0.2 1")
    defaults = ET.SubElement(scene, "default")
    ET.SubElement(defaults, "geom", solref="0.006 1", friction="1 0.01 0.001")
    for source in (panda, gripper):
        for child in source.find("default"):
            defaults.append(copy.deepcopy(child))
    assets = ET.SubElement(scene, "asset")
    for source in (panda, gripper):
        for child in source.find("asset"):
            assets.append(copy.deepcopy(child))
    ET.SubElement(
        assets,
        "texture",
        name="sky",
        type="skybox",
        builtin="gradient",
        rgb1="0.13 0.17 0.23",
        rgb2="0.025 0.04 0.065",
        width="512",
        height="512",
    )
    ET.SubElement(
        assets,
        "material",
        name="table_material",
        rgba="0.16 0.20 0.25 1",
        specular="0.25",
        shininess="0.2",
    )
    ET.SubElement(
        assets,
        "material",
        name="disc_material",
        rgba="0.73 0.81 0.83 1",
        specular="0.6",
        shininess="0.5",
    )
    world = ET.SubElement(scene, "worldbody")
    ET.SubElement(
        world,
        "light",
        pos="0.2 -0.6 2.5",
        dir="0 0 -1",
        diffuse="0.85 0.85 0.85",
        castshadow="true",
    )
    ET.SubElement(
        world,
        "light",
        pos="0.6 1.5 1.8",
        dir="0 -0.5 -1",
        diffuse="0.5 0.6 0.7",
        castshadow="false",
    )
    ET.SubElement(
        world,
        "geom",
        name="table",
        type="box",
        pos="0.35 0 -0.035",
        size="0.9 0.65 0.035",
        material="table_material",
    )
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        pos="0 0 -0.08",
        size="3 3 0.01",
        rgba="0.06 0.08 0.11 1",
    )
    ET.SubElement(
        world,
        "geom",
        name="disc",
        type="cylinder",
        pos=f"{DISC_CENTER[0]} {DISC_CENTER[1]} {DISC_TOP / 2}",
        size=f"{DISC_RADIUS} {DISC_TOP / 2}",
        material="disc_material",
    )
    for i, target in enumerate(TARGETS):
        ET.SubElement(
            world,
            "site",
            name=f"target{i}",
            type="cylinder",
            pos=f"{target[0]} {target[1]} {DISC_TOP + 0.0003}",
            size="0.028 0.0002",
            rgba="0.45 0.55 0.6 0.35",
        )
    arm = copy.deepcopy(panda.find("worldbody/body"))
    attachment = arm.find(".//body[@name='attachment']")
    attachment.append(copy.deepcopy(gripper.find("worldbody/body")))
    for body in arm.iter("body"):
        body.set("gravcomp", "1")
    world.append(arm)
    for i, (position, color) in enumerate(zip(STARTS, COLORS, strict=True)):
        body = ET.SubElement(world, "body", name=f"cube{i}", pos=" ".join(map(str, position)))
        ET.SubElement(body, "freejoint", name=f"cube{i}_free")
        ET.SubElement(
            body,
            "geom",
            name=f"cube{i}_geom",
            type="box",
            size="0.02 0.02 0.02",
            mass="0.04",
            rgba=" ".join(map(str, (*color, 1))),
            friction="1.2 0.02 0.002",
            condim="4",
        )
    for tag in ("actuator", "contact", "tendon", "equality"):
        parent = ET.SubElement(scene, tag)
        for source in (panda, gripper):
            if source.find(tag) is not None:
                parent.extend(copy.deepcopy(list(source.find(tag))))
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(scene)
    ET.ElementTree(scene).write(output, encoding="unicode")
    return mujoco.MjModel.from_xml_path(str(output))


def make_urdf(model, data, output):
    """Match PyRoki kinematics to the compiled MuJoCo arm/TCP, without a second robot model."""
    root = ET.Element("robot", name="droid_franka")
    ET.SubElement(root, "link", name="link0")
    velocities = [2.175] * 4 + [2.61] * 3
    for i in range(1, 8):
        body = model.body(f"link{i}").id
        joint_id = model.joint(f"joint{i}").id
        ET.SubElement(root, "link", name=f"link{i}")
        joint = ET.SubElement(root, "joint", name=f"joint{i}", type="revolute")
        ET.SubElement(joint, "parent", link=f"link{i - 1}")
        ET.SubElement(joint, "child", link=f"link{i}")
        q = model.body_quat[body]
        rpy = Rotation.from_quat(q[[1, 2, 3, 0]]).as_euler("xyz")
        ET.SubElement(
            joint,
            "origin",
            xyz=" ".join(map(str, model.body_pos[body])),
            rpy=" ".join(map(str, rpy)),
        )
        ET.SubElement(joint, "axis", xyz=" ".join(map(str, model.jnt_axis[joint_id])))
        ET.SubElement(
            joint,
            "limit",
            lower=str(model.jnt_range[joint_id, 0]),
            upper=str(model.jnt_range[joint_id, 1]),
            velocity=str(velocities[i - 1]),
            effort="87",
        )
    flange = model.body("link7").id
    site = model.site("rq_pinch").id
    mujoco.mj_forward(model, data)
    rotation = data.xmat[flange].reshape(3, 3)
    offset = rotation.T @ (data.site_xpos[site] - data.xpos[flange])
    rpy = Rotation.from_matrix(rotation.T @ data.site_xmat[site].reshape(3, 3)).as_euler("xyz")
    ET.SubElement(root, "link", name="tcp")
    joint = ET.SubElement(root, "joint", name="tcp_fixed", type="fixed")
    ET.SubElement(joint, "parent", link="link7")
    ET.SubElement(joint, "child", link="tcp")
    ET.SubElement(joint, "origin", xyz=" ".join(map(str, offset)), rpy=" ".join(map(str, rpy)))
    ET.indent(root)
    ET.ElementTree(root).write(output, encoding="unicode")


def initial_state(model):
    data = mujoco.MjData(model)
    for i, q in enumerate(HOME, 1):
        data.qpos[model.joint(f"joint{i}").qposadr] = q
    data.ctrl[:7] = HOME
    data.ctrl[7] = 0.0  # Robotiq: zero=open, 255=close
    mujoco.mj_forward(model, data)
    for _ in range(300):
        mujoco.mj_step(model, data)
    data.time = 0.0
    return data
