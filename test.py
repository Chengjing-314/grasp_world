import pytorch_kinematics as pk
import torch
from grasp_world.utils.sphere_model_utils import load_sphere_database

from einops import rearrange


# --- Load chain from URDF ---
urdf_path = "/home/chengjing/Chengjing/grasp_world/assets/hand/shadow_hand/shadow_hand.urdf"
chain = pk.build_chain_from_urdf(open(urdf_path, "rb").read())
chain.print_tree()

sphere_db = load_sphere_database(
    "/home/chengjing/Chengjing/grasp_world/assets/hand/shadow_hand/shadow_hand_sphere_database.json"
)

target_link = "palm"  # name from the URDF (change if different)
trans_joints = ["shadow_rootlink_1", "shadow_rootlink_2", "shadow_rootlink_3"]

q = torch.zeros((1, chain.n_joints), dtype=torch.float32, requires_grad=True)

fk = chain.forward_kinematics(q)

__import__("IPython").embed(header="test.py:24")

all_links = list(sphere_db.keys())

world_centers = []
world_radii = []
for link in all_links:
    link_pose = fk[link].get_matrix()
    rot, trans = link_pose[:, :3, :3], link_pose[:, :3, 3]
    local_c = torch.stack([s.local_center for s in sphere_db[link]]).unsqueeze(0)
    local_r = torch.stack([s.radius for s in sphere_db[link]]).unsqueeze(0)
    world_c = rearrange((rot @ rearrange(local_c, " b n c -> b c n")), "b c n -> b n c") + trans
    world_radii.append(local_r.expand(q.shape[0], -1))
    world_centers.append(world_c)

world_radii = torch.cat(world_radii, dim=1)
world_centers = torch.cat(world_centers, dim=1)
