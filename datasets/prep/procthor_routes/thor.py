"""AI2-THOR renderer, camera and coordinates exactly as the procthor-walks training data.

Port of Rupert Tawiah-Quashie's `prj_habitat/scripts/generate_procthor_data.py` (the generator of procthor-walks) and
tjepa-dev `analyses/stimuli/pt.py`: ai2thor 5.0.0 CloudRendering, houses from his ProcTHOR-10K export, a free
third-party camera at 1.454 m, pitch 0, 90 deg horizontal FOV at 4:3 (vertical 73.74 deg), 160 x 120, physics paused,
agent body invisible (checked per house), ProcTHOR's own lighting. The walkable set is the FURNISHED house's reachable
grid (0.25 m), shared by the empty cell, so both cells use identical camera paths.

Coordinates: Unity (x_u, z_u, yaw_u in degrees, clockwise from +z). Stored poses use the training data's habitat
convention: x_h = -x_u, z_h = z_u, heading = pi - yaw_u (wrapped to (-pi, pi]); forward -> dY_ego = -1, right -> +1.
"""
from __future__ import annotations

import copy
import gzip
import json
import math
from pathlib import Path

import numpy as np

W, H = 160, 120
CAM_H = 1.454
VFOV = math.degrees(2 * math.atan(math.tan(math.radians(90) / 2) * H / W))      # 73.74
SNAP_TOL = 0.13                     # the generator's tolerance to the 0.25 m reachable grid


def to_habitat(x_u, z_u, yaw_u_deg):
    """Unity camera pose -> stored (x_h, z_h, heading). Verbatim from the generator."""
    return -x_u, z_u, math.atan2(math.sin(math.pi - math.radians(yaw_u_deg)), math.cos(math.pi - math.radians(yaw_u_deg)))


def from_habitat(x_h, z_h, heading):
    """Stored pose -> Unity camera pose (tjepa-dev pt.py)."""
    return -x_h, z_h, math.degrees(math.pi - heading) % 360.0


def _selftest():
    """The generator's check (verbatim maths): forward -> dY_ego -1, right -> dX_ego +1."""
    for yaw in (0.0, 37.0, 90.0, 200.0, 315.0):
        th = math.radians(yaw); fwd_u = (math.sin(th), math.cos(th)); right_u = (math.cos(th), -math.sin(th))
        for vec, want in ((fwd_u, (0.0, -1.0)), (right_u, (1.0, 0.0))):
            x0, z0, h = to_habitat(0.0, 0.0, yaw); x1, z1, _ = to_habitat(vec[0], vec[1], yaw)
            dX, dY = x1 - x0, z1 - z0; c, s = math.cos(h), math.sin(h)
            got = (dX * c - dY * s, dX * s + dY * c)
            assert abs(got[0] - want[0]) < 1e-9 and abs(got[1] - want[1]) < 1e-9, (yaw, vec, got, want)


_selftest()


def load_split(root: Path, split: str) -> list:
    with gzip.open(Path(root) / f"{split}.json.gz", "rt") as fh:
        return json.load(fh)


def empty_house(house: dict) -> dict:
    return {**copy.deepcopy(house), "objects": []}


def time_of_day(house):
    d = [L for L in house["proceduralParameters"]["lights"] if L["type"] == "directional"][0]
    return {66: "Midday", 6: "GoldenHour", 82: "BlueHour"}.get(d["rotation"]["x"], f"rotx{d['rotation']['x']}")


def controller(scene):
    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering
    c = Controller(scene=scene, platform=CloudRendering, width=W, height=H, gridSize=0.25, snapToGrid=False,
                   renderDepthImage=False, renderImage=False, makeAgentsVisible=False)
    c.step(action="PausePhysicsAutoSim")
    return c


def reset(c, scene):
    c.reset(scene=scene)
    c.step(action="PausePhysicsAutoSim")


def reachable_positions(c) -> np.ndarray:
    ev = c.step(action="GetReachablePositions")
    return np.array([[p["x"], p["z"]] for p in ev.metadata["actionReturn"]])


def assert_agent_invisible(ctrl, house_idx, cell):
    """Verbatim from the generator: a free camera facing the parked agent must render pixel-identical frames with the
    agent present and teleported away. Raises if the agent body renders."""
    ev = ctrl.step(action="GetReachablePositions"); reach = np.array([[p["x"], p["z"]] for p in ev.metadata["actionReturn"]])
    ag, ayaw = ev.metadata["agent"]["position"], ev.metadata["agent"]["rotation"]["y"]
    fwd = np.array([math.sin(math.radians(ayaw)), math.cos(math.radians(ayaw))]); cam = np.array([ag["x"], ag["z"]]) + 1.5 * fwd
    look = (ayaw + 180.0) % 360.0
    e1 = ctrl.step(action="AddThirdPartyCamera", position=dict(x=float(cam[0]), y=CAM_H, z=float(cam[1])), rotation=dict(x=0, y=look, z=0), fieldOfView=VFOV)
    f1 = e1.third_party_camera_frames[0][..., :3].astype(int)
    far = reach[np.argmax(np.linalg.norm(reach - [ag["x"], ag["z"]], axis=1))]
    ctrl.step(action="TeleportFull", position=dict(x=float(far[0]), y=0.9, z=float(far[1])), rotation=dict(x=0, y=ayaw, z=0), horizon=0, standing=True, forceAction=True)
    e2 = ctrl.step(action="UpdateThirdPartyCamera", thirdPartyCameraId=0, position=dict(x=float(cam[0]), y=CAM_H, z=float(cam[1])), rotation=dict(x=0, y=look, z=0))
    f2 = e2.third_party_camera_frames[0][..., :3].astype(int)
    diff = int(np.abs(f1 - f2).max())
    ctrl.step(action="TeleportFull", position=ag, rotation=dict(x=0, y=ayaw, z=0), horizon=0, standing=True, forceAction=True)
    if diff > 0:
        raise RuntimeError(f"AGENT BODY VISIBLE in house {house_idx} {cell}: max pixel diff {diff}")
    return diff


class Camera:
    """The walk camera: camera id 0 is the probe camera added by assert_agent_invisible; ours is the last added."""

    def __init__(self, c):
        self.c = c
        ev = c.step(action="AddThirdPartyCamera", position=dict(x=0.0, y=CAM_H, z=0.0), rotation=dict(x=0, y=0, z=0),
                    fieldOfView=VFOV)
        self.id = len(ev.third_party_camera_frames) - 1

    def shot(self, x_u, z_u, yaw_u_deg) -> np.ndarray:
        ev = self.c.step(action="UpdateThirdPartyCamera", thirdPartyCameraId=self.id,
                         position=dict(x=float(x_u), y=CAM_H, z=float(z_u)), rotation=dict(x=0, y=float(yaw_u_deg), z=0))
        return ev.third_party_camera_frames[self.id][..., :3].copy()
