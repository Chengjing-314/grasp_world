import os
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pytorch_kinematics as pk
import torch
import torch.nn.functional as F
from einops import rearrange

import trimesh

from grasp_world.constants import ASSETS_ROOT
from grasp_world.utils.sphere_model_utils import load_sphere_database
from grasp_world.utils.hand_utils import debug_self_penetration

from collections import defaultdict
from urdfpy import URDF

import fpsample

from grasp_world.utils.hand_utils import (
    obtain_links_collision_geometry,
    sample_hand_link_contact_points,
)
from grasp_world.utils.utils import (
    rotation_from_facing,
    rotation_to_tf,
)


@dataclass
class DexHandConfig:
    urdf_path: str = ""
    """urdf path of the hand"""
    sphere_rep_json: str = ""
    """json path of the sphere representation"""
    self_penetration_groups: dict[str, list[str]] = field(default_factory=dict)
    """self penetration groups, self penetration is disabled if empty, links in the same group are assumed to be penetration free"""
    self_penetration_ignore_pairs: list[tuple[str, str]] = field(default_factory=list)
    """self penetration ignore pairs, self penetration is disabled for these pairs"""
    palm_facing_direction: list[float] = field(default_factory=list)
    """palm facing direction to obtain hand contact points, please manually inspect the urdf to find the palm facing direction"""
    contact_link: list[tuple[str, int]] = field(default_factory=list)
    """contact link and number of contact points to be sampled from the link"""


class DexHand:
    def __init__(self, config: DexHandConfig):
        self.config = config
        try:
            self.robot = pk.build_chain_from_urdf(open(self.config.urdf_path, "rb").read())
            self.link_names = self.robot.get_link_names()[6:]
            self._zero_q = torch.zeros((1, self.robot.n_joints), dtype=torch.float32)
        except Exception as e:
            raise ValueError(f"Failed to load URDF: {e}")
        self.sphere_rep = load_sphere_database(self.config.sphere_rep_json, branch=7, depth=2)

        self.build_collision_groups()

    def build_collision_groups(self):
        link_names = [ln for ln in self.link_names if ln in self.sphere_rep]
        self.collision_link_names = link_names
        link_to_idx = {ln: i for i, ln in enumerate(link_names)}
        group_id = {
            ln: gid
            for gid, (grp, lst) in enumerate(self.config.self_penetration_groups.items())
            for ln in lst
            if ln in link_to_idx
        }

        self.sphere_link_id = []
        link_to_slice = {}
        offset = 0
        for ln in link_names:
            spheres = self.sphere_rep.get(ln, [])
            if not spheres:
                continue
            n = len(spheres)
            link_to_slice[ln] = slice(offset, offset + n)
            self.sphere_link_id.extend([link_to_idx[ln]] * n)
            offset += n
        self.sphere_link_id = torch.tensor(self.sphere_link_id, dtype=torch.long)
        S = len(self.sphere_link_id)

        link_group = torch.tensor([group_id.get(ln, -1) for ln in link_names], dtype=torch.long)
        same_group = link_group[:, None] == link_group[None, :]
        allowed_link_pair = ~same_group
        allowed = allowed_link_pair[self.sphere_link_id][:, self.sphere_link_id]
        palm_slice = link_to_slice.get("palm", None)
        if palm_slice is not None:
            allowed[palm_slice, :] = False
            allowed[:, palm_slice] = False

        mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)

        allowed &= mask
        self.pair_i, self.pair_j = allowed.nonzero(as_tuple=True)

    def forward_kinematics(self, q: torch.Tensor) -> dict[str, pk.Frame]:
        fk = self.robot.forward_kinematics(q)
        return fk

    def compute_world_spheres(
        self,
        q: torch.Tensor,
        return_type: Literal["np", "pt"] = "pt",
    ) -> tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
        fk = self.forward_kinematics(q)
        world_centers, world_radii = [], []

        for link in self.link_names:
            link_pose = fk[link].get_matrix()  # (1,4,4)
            R, t = link_pose[:, :3, :3], link_pose[:, :3, 3]
            local_c = torch.stack([s.local_center for s in self.sphere_rep[link]]).unsqueeze(0)
            local_r = torch.stack([s.radius for s in self.sphere_rep[link]]).unsqueeze(0)
            world_c = rearrange((R @ rearrange(local_c, "b n c -> b c n")), "b c n -> b n c") + t
            world_centers.append(world_c)
            world_radii.append(local_r)

        world_centers = torch.cat(world_centers, dim=1)
        world_radii = torch.cat(world_radii, dim=1)

        if return_type == "pt":
            return world_centers, world_radii

        if return_type == "np":

            def to_np(x):
                return x.detach().cpu().numpy() if torch.is_tensor(x) else x

            return to_np(world_centers), to_np(world_radii)

        raise ValueError(f"Invalid return_type '{return_type}'. Expected 'pt' or 'np'.")

    def calculate_self_penetration(self, q: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """Compute differentiable sphere–sphere self-collision loss."""
        C, r = self.compute_world_spheres(q, "pt")  # (B,S,3), (B,S)
        I_idx, J_idx = self.pair_i.to(C.device), self.pair_j.to(C.device)
        diff = C[:, I_idx] - C[:, J_idx]  # (B,M,3)
        dist = diff.norm(dim=-1)  # (B,M)
        overlap = F.relu(r[:, I_idx] + r[:, J_idx] - dist)  # (B,M)
        if debug:
            debug_self_penetration(
                overlap, I_idx, J_idx, self.collision_link_names, self.sphere_link_id
            )
        return overlap.sum(dim=-1)  # (B,)

    @property
    def zero_q(self, device: torch.device = None) -> torch.Tensor:
        if device is None:
            return self._zero_q
        return self._zero_q.to(device)

    def sample_hand_link_contact_points(
        self, num_samples_per_link=10, cosine_thresh=0.2, visualize=False
    ):
        """
        Sample contact points on each hand link facing approximately in the palm-facing direction.
        """
        facing_dir = np.array(self.config.palm_facing_direction, dtype=np.float32)
        facing_dir /= np.linalg.norm(facing_dir)

        fk = self.forward_kinematics(self.zero_q)
        link_geom_dict = obtain_links_collision_geometry(self.config.urdf_path, self.link_names)

        points, normals = sample_hand_link_contact_points(
            fk,
            facing_dir,
            link_geom_dict,
            self.link_names,
            num_samples_per_link,
            cosine_thresh,
        )

        if visualize:
            scene = trimesh.Scene()
            scene.add_geometry(trimesh.PointCloud(points, colors=[200, 100, 100, 255]))
            frame = trimesh.creation.axis()
            scene.add_geometry(frame)
            scene.show()

        return points, normals


HANDS = {
    "shadow_hand": DexHandConfig(
        urdf_path=os.path.join(ASSETS_ROOT, "hand/shadow_hand/shadow_hand.urdf"),
        sphere_rep_json=os.path.join(
            ASSETS_ROOT, "hand/shadow_hand/shadowhand_sphere_database.json"
        ),
        self_penetration_groups={
            "thumb": ["thbase", "thhub", "thproximal", "thmiddle", "thdistal"],
            "index": ["ffknuckle", "ffproximal", "ffmiddle", "ffdistal"],
            "middle": ["mfknuckle", "mfproximal", "mfmiddle", "mfdistal"],
            "ring": ["rfknuckle", "rfproximal", "rfmiddle", "rfdistal"],
            "pinky": ["lfmetacarpal", "lfknuckle", "lfproximal", "lfmiddle", "lfdistal"],
            "palm": ["palm"],
        },
        palm_facing_direction=[0, -1, 0],
        contact_link=[("ffproximal", 1), ("ffmiddle", 1), ("ffdistal", 1), ("ffknuckle", 1)],
    ),
}

# hand_model = DexHand(HANDS["shadow_hand"])
# q = torch.zeros((1, hand_model.robot.n_joints), dtype=torch.float32, requires_grad=True)
# hand_model.sample_hand_link_contact_points(visualize=True)
# hand_model.calculate_self_penetration(q)
