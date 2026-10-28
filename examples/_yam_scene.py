"""Two native-coordinate YAM arms with physical linear grippers and four free cubes."""

from __future__ import annotations

import copy
import hashlib
import json
import urllib.request
import warnings
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
from _pick_place_scene import MENAGERIE_COMMIT
from scipy.spatial.transform import Rotation

ARMS = ("left", "right")
HOME = np.array([0.0, 1.047, 1.047, 0.0, 0.0, 0.0])
ARM_NAMES = {arm: tuple(f"{arm}_joint{i}" for i in range(1, 7)) for arm in ARMS}
JOINT_NAMES = ARM_NAMES["left"] + ARM_NAMES["right"]
STARTS = np.array(
    [[0.27, 0.22, 0.021], [0.32, 0.40, 0.021], [0.27, -0.22, 0.021], [0.32, -0.40, 0.021]]
)
TARGETS = np.array(
    [[0.40, 0.18, 0.037], [0.40, 0.32, 0.037], [0.40, -0.18, 0.037], [0.40, -0.32, 0.037]]
)
TRAY_TOP = 0.016
COLORS = ((0.98, 0.35, 0.28), (1.0, 0.72, 0.18), (0.22, 0.66, 0.98), (0.61, 0.43, 0.96))


def fetch_assets(directory: Path):
    root = directory / MENAGERIE_COMMIT
    marker = root / "yam_complete.json"
    if marker.exists():
        return root / "i2rt_yam"
    url = f"https://api.github.com/repos/google-deepmind/mujoco_menagerie/git/trees/{MENAGERIE_COMMIT}?recursive=1"
    with urllib.request.urlopen(url, timeout=60) as f:
        entries = json.load(f)["tree"]
    entries = [
        e
        for e in entries
        if e["type"] == "blob"
        and e["path"].startswith("i2rt_yam/")
        and not e["path"].endswith((".png", ".md"))
    ]

    def fetch(entry):
        path = root / entry["path"]
        if path.exists():
            data = path.read_bytes()
        else:
            remote = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{MENAGERIE_COMMIT}/{entry['path']}"
            with urllib.request.urlopen(remote, timeout=60) as f:
                data = f.read()
        if hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() != entry["sha"]:
            raise ValueError(f"YAM asset checksum mismatch: {entry['path']}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, entries))
    marker.write_text(json.dumps({"commit": MENAGERIE_COMMIT, "files": entries}, indent=2))
    return root / "i2rt_yam"


def build_scene(
    assets,
    output,
    *,
    starts=STARTS,
    targets=TARGETS,
    tray_center=(0.41, 0),
    tray_halfsize=(0.06, 0.43),
):
    template = ET.parse(assets / "yam_linear_4310.xml").getroot()
    root = ET.Element("mujoco", model="Bimanual YAM — marginal motion priors")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(
        root,
        "option",
        timestep="0.002",
        integrator="implicitfast",
        cone="elliptic",
        iterations="80",
        impratio="10",
    )
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="960")
    ET.SubElement(visual, "quality", shadowsize="2048", offsamples="4")
    ET.SubElement(
        visual, "headlight", ambient="0.4 0.4 0.4", diffuse="0.6 0.6 0.6", specular="0.2 0.2 0.2"
    )
    defaults = copy.deepcopy(template.find("default"))
    root.append(defaults)
    # Both modes share these simulation gains and contact settings.
    for cls in defaults.findall(".//default"):
        name = cls.get("class")
        if name in ("dm4340", "dm4310"):
            cls.find("position").set("kp", "180" if name == "dm4340" else "90")
            cls.find("position").set("kv", "16" if name == "dm4340" else "8")
        if name == "finger_collision":
            cls.find("geom").set("friction", "2 0.01 0.001")
            cls.find("geom").set("solref", "0.005 1")
    asset = copy.deepcopy(template.find("asset"))
    for mesh in asset.findall("mesh"):
        if mesh.get("name") is None:
            mesh.set("name", Path(mesh.get("file")).stem)
        mesh.set("file", str((assets / "assets" / mesh.get("file")).resolve()))
    ET.SubElement(
        asset,
        "texture",
        name="sky",
        type="skybox",
        builtin="gradient",
        rgb1="0.15 0.20 0.29",
        rgb2="0.035 0.05 0.08",
        width="512",
        height="512",
    )
    root.append(asset)
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(
        world, "light", pos="0.3 -0.7 2", dir="0 0 -1", diffuse="0.9 0.9 0.9", castshadow="true"
    )
    ET.SubElement(world, "light", pos="0.7 0.8 1.6", dir="-0.3 -0.2 -1", diffuse="0.4 0.4 0.4")
    ET.SubElement(
        world,
        "geom",
        name="table",
        type="box",
        pos="0.32 0 -0.035",
        size="0.65 0.68 0.035",
        rgba="0.22 0.27 0.34 1",
        friction="1 .01 .001",
    )
    ET.SubElement(
        world,
        "geom",
        name="tray",
        type="box",
        pos=f"{tray_center[0]} {tray_center[1]} 0.008",
        size=f"{tray_halfsize[0]} {tray_halfsize[1]} 0.008",
        rgba="0.76 0.83 0.86 1",
        friction="1 .01 .001",
    )
    for i, xyz in enumerate(targets):
        ET.SubElement(
            world,
            "geom",
            type="cylinder",
            pos=f"{xyz[0]} {xyz[1]} {TRAY_TOP + 0.0003}",
            size=".027 .0002",
            contype="0",
            conaffinity="0",
            rgba=".38 .47 .54 .45",
        )
    contact = ET.SubElement(root, "contact")
    equality = ET.SubElement(root, "equality")
    actuators = ET.SubElement(root, "actuator")
    for arm, y in zip(ARMS, (0.31, -0.31), strict=True):
        body = copy.deepcopy(template.find("worldbody/body"))
        body.set("pos", f"0 {y} .02")
        # Share mesh/material/default definitions; namespace physical identifiers.
        identifiers = {e.get("name") for e in body.iter() if e.get("name")}
        identifiers.update(e.get("name") for e in template.find("actuator"))

        def prefix(element):
            for e in element.iter():
                for key in ("name", "joint", "joint1", "joint2", "body1", "body2"):
                    if e.get(key) in identifiers:
                        e.set(key, arm + "_" + e.get(key))

        prefix(body)
        for b in body.iter("body"):
            b.set("gravcomp", "1")
        for actuator in template.find("actuator"):
            a = copy.deepcopy(actuator)
            prefix(a)
            actuators.append(a)
        for tag, parent in (("contact", contact), ("equality", equality)):
            for element in template.find(tag):
                e = copy.deepcopy(element)
                prefix(e)
                parent.append(e)
        world.append(body)
    for i, (xyz, color) in enumerate(zip(starts, COLORS[: len(starts)], strict=True)):
        b = ET.SubElement(world, "body", name=f"cube{i}", pos=" ".join(map(str, xyz)))
        ET.SubElement(b, "freejoint", name=f"cube{i}_free")
        ET.SubElement(
            b,
            "geom",
            name=f"cube{i}_geom",
            type="box",
            size=".02 .02 .02",
            mass=".035",
            rgba=" ".join(map(str, (*color, 1))),
            friction="1.5 .01 .001",
            condim="4",
            solref=".006 1",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(output, encoding="unicode")
    return mujoco.MjModel.from_xml_path(str(output))


def initial_state(model, *, settle=True):
    data = mujoco.MjData(model)
    for arm in ARMS:
        for name, value in zip(ARM_NAMES[arm], HOME, strict=True):
            data.qpos[model.joint(name).qposadr] = value
            data.ctrl[model.actuator(name).id] = value
        for finger in ("left", "right"):
            data.qpos[model.joint(f"{arm}_{finger}_finger").qposadr] = 0.0475
        data.ctrl[model.actuator(f"{arm}_gripper").id] = 0.0475
    mujoco.mj_forward(model, data)
    if settle:
        for _ in range(300):
            mujoco.mj_step(model, data)
    data.time = 0
    return data


def make_urdf(model, data, arm, output):
    """Export one 6-DOF arm in world coordinates from the compiled MuJoCo kinematics."""
    root = ET.Element("robot", name=f"yam_{arm}")
    ET.SubElement(root, "link", name="world")
    base = arm + "_arm"
    ET.SubElement(root, "link", name=base)
    fixed = ET.SubElement(root, "joint", name=arm + "_mount", type="fixed")
    ET.SubElement(fixed, "parent", link="world")
    ET.SubElement(fixed, "child", link=base)
    base_id = model.body(base).id
    ET.SubElement(fixed, "origin", xyz=" ".join(map(str, model.body_pos[base_id])))
    previous = base
    for i, name in enumerate(ARM_NAMES[arm], 1):
        body = model.body(f"{arm}_link_{i}").id
        jid = model.joint(name).id
        child = f"{arm}_link_{i}"
        ET.SubElement(root, "link", name=child)
        joint = ET.SubElement(root, "joint", name=name, type="revolute")
        ET.SubElement(joint, "parent", link=previous)
        ET.SubElement(joint, "child", link=child)
        # At a wrist's Euler singularity SciPy chooses a valid zero third angle;
        # the simulator/PyRoki FK check below the exporter validates the rotation.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Gimbal lock detected.*", category=UserWarning
            )
            rpy = Rotation.from_quat(model.body_quat[body][[1, 2, 3, 0]]).as_euler("xyz")
        ET.SubElement(
            joint,
            "origin",
            xyz=" ".join(map(str, model.body_pos[body])),
            rpy=" ".join(map(str, rpy)),
        )
        ET.SubElement(joint, "axis", xyz=" ".join(map(str, model.jnt_axis[jid])))
        ET.SubElement(
            joint,
            "limit",
            lower=str(model.jnt_range[jid, 0]),
            upper=str(model.jnt_range[jid, 1]),
            velocity="2",
            effort="28",
        )
        previous = child
    mujoco.mj_forward(model, data)
    flange = model.body(previous).id
    site = model.site(f"{arm}_grasp_site").id
    rot = data.xmat[flange].reshape(3, 3)
    offset = rot.T @ (data.site_xpos[site] - data.xpos[flange])
    rpy = Rotation.from_matrix(rot.T @ data.site_xmat[site].reshape(3, 3)).as_euler("xyz")
    ET.SubElement(root, "link", name="tcp")
    joint = ET.SubElement(root, "joint", name=arm + "_tcp", type="fixed")
    ET.SubElement(joint, "parent", link=previous)
    ET.SubElement(joint, "child", link="tcp")
    ET.SubElement(joint, "origin", xyz=" ".join(map(str, offset)), rpy=" ".join(map(str, rpy)))
    ET.indent(root)
    ET.ElementTree(root).write(output, encoding="unicode")
