import torch
import pytorch_kinematics as pk
from einops import rearrange
import numpy as np
import viser
from grasp_world.utils.sphere_model_utils import load_sphere_database
from pathlib import Path

# --- Config ---
urdf_path = "/home/chengjing/Chengjing/grasp_world/assets/hand/shadow_hand/shadow_hand.urdf"
sphere_db_path = "/home/chengjing/Chengjing/grasp_world/assets/hand/shadow_hand/shadow_hand_sphere_database.json"
N_SCENES = 10

# --- Load chain and sphere database ---
chain = pk.build_chain_from_urdf(open(urdf_path, "rb").read())
sphere_db = load_sphere_database(sphere_db_path)
all_links = list(sphere_db.keys())
n_joints = chain.n_joints

# --- Sample random qpos (fix first 6 DOF to zero) ---
qs = []
for _ in range(N_SCENES):
    q = torch.zeros((1, n_joints), dtype=torch.float32)
    q[:, 6:] = torch.rand(1, n_joints - 6) * 0.8 - 0.4  # uniform random [-0.4, 0.4]
    qs.append(q)


# --- Forward kinematics function ---
def compute_world_spheres(q):
    fk = chain.forward_kinematics(q)
    world_centers, world_radii = [], []

    for link in all_links:
        link_pose = fk[link].get_matrix()  # (1,4,4)
        R, t = link_pose[:, :3, :3], link_pose[:, :3, 3]
        local_c = torch.stack([s.local_center for s in sphere_db[link]]).unsqueeze(0)
        local_r = torch.stack([s.radius for s in sphere_db[link]]).unsqueeze(0)
        world_c = rearrange((R @ rearrange(local_c, "b n c -> b c n")), "b c n -> b n c") + t
        world_centers.append(world_c)
        world_radii.append(local_r)
    world_centers = torch.cat(world_centers, dim=1).squeeze(0)
    world_radii = torch.cat(world_radii, dim=1).squeeze(0)
    return world_centers.detach().cpu().numpy(), world_radii.detach().cpu().numpy()


# --- Initialize Viser client ---
vis = viser.ViserServer()
vis.scene.reset()

# --- Load original hand mesh (for reference) ---
# vis.load_urdf(urdf_path, show_links=True, show_collisions=False, name="hand_mesh")

# --- Precompute sphere scenes ---
scenes = []
for q in qs:
    c, r = compute_world_spheres(q)
    scenes.append((c, r))

from grasp_world.utils.viz_ui import VizUI

viz = VizUI()
viz.visualize_hand_sphere_state(scenes)

print(f"[viser] Running visualization with {N_SCENES} random Shadow Hand poses...")

print("👉 Open http://localhost:8080 to view")

vis.sleep_forever()
