import torch
import trimesh
import urdfpy
import os
from collections import defaultdict
import logging
from rich.logging import RichHandler
import pytorch_kinematics
import numpy as np
import trimesh
import fpsample

logger = logging.getLogger(__name__)
logger.addHandler(RichHandler())
logger.setLevel("INFO")


def debug_self_penetration(
    overlap: torch.Tensor,
    I_idx: torch.Tensor,
    J_idx: torch.Tensor,
    collision_link_names: list[str],
    sphere_link_id: torch.Tensor,
):
    max_ov = overlap.max().item()
    min_ov = overlap.min().item()
    total_ov = overlap.sum().item()
    logger.info(f"overlap stats | max: {max_ov:.3e} | min: {min_ov:.3e} | total: {total_ov:.3e}")
    threshold = 2e-3
    contact_mask = overlap > threshold
    if contact_mask.any():
        batch_idx, pair_idx = contact_mask.nonzero(as_tuple=True)
        header = "overlap contacts (>{:.1e})".format(threshold)
        rows = []
        for b, p in zip(batch_idx.tolist(), pair_idx.tolist()):
            si = I_idx[p].item()
            sj = J_idx[p].item()
            li = collision_link_names[sphere_link_id[si]]
            lj = collision_link_names[sphere_link_id[sj]]
            ov = overlap[b, p].item()
            rows.append(f"  [batch {b}] {li} (sphere {si}) vs {lj} (sphere {sj}): {ov:.6e}")
        logger.info("\n".join([header, *rows]))


def obtain_links_collision_geometry(
    urdf_path: str, link_names: list[str]
) -> dict[str, list[tuple[str, str, np.ndarray]]]:
    robot = urdfpy.URDF.load(urdf_path)
    urdf_dir = os.path.dirname(urdf_path)

    link_geom_dict = defaultdict(list)

    for link in robot.links:
        for col in link.collisions:
            geom, origin = col.geometry, col.origin
            if geom.mesh is not None:
                link_geom_dict[link.name].append(
                    ("mesh", os.path.join(urdf_dir, geom.mesh.filename), origin)
                )
            elif geom.box is not None:
                link_geom_dict[link.name].append(("box", geom.box.size, origin))
            elif geom.sphere is not None:
                link_geom_dict[link.name].append(("sphere", geom.sphere.radius, origin))
            elif geom.cylinder is not None:
                link_geom_dict[link.name].append(
                    ("cylinder", (geom.cylinder.radius, geom.cylinder.length), origin)
                )
            else:
                logger.warning("Unsupported geometry on link {}".format(link.name))

    return link_geom_dict


def sample_hand_link_contact_points(
    fk: dict[str, pytorch_kinematics.Transform3d],
    hand_facing_dir: np.ndarray,
    link_geom_dict: dict[str, list[tuple[str, str, np.ndarray]]],
    link_names: list[str],
    num_samples_per_link: int = 10,
    cosine_thresh: float = 0.2,
):
    hand_points, hand_normals = [], []
    for link_name, geoms in link_geom_dict.items():
        link_pose = fk[link_name].get_matrix().squeeze(0).cpu().numpy()
        link_pts, link_nrm = [], []

        for geom_type, data, origin in geoms:
            col_T = origin
            T = link_pose @ col_T

            # Create geometry
            if geom_type == "mesh":
                m = trimesh.load_mesh(data, process=False)
            elif geom_type == "box":
                m = trimesh.creation.box(extents=data)
            elif geom_type == "sphere":
                m = trimesh.creation.icosphere(subdivisions=2, radius=data)
            elif geom_type == "cylinder":
                r, h = data
                m = trimesh.creation.cylinder(radius=r, height=h, sections=32)
            else:
                continue

            m.apply_transform(T)

            dense_pts, face_idx = trimesh.sample.sample_surface_even(m, 100 * num_samples_per_link)
            dense_nrm = m.face_normals[face_idx]
            dense_nrm /= np.linalg.norm(dense_nrm, axis=1, keepdims=True) + 1e-8

            mask = (dense_nrm @ hand_facing_dir) > cosine_thresh
            if not np.any(mask):
                continue

            dense_pts, dense_nrm = dense_pts[mask], dense_nrm[mask]

            if len(dense_pts) > num_samples_per_link:
                selected_idx = fpsample.bucket_fps_kdline_sampling(
                    dense_pts, num_samples_per_link, h=3
                )
                dense_pts, dense_nrm = dense_pts[selected_idx], dense_nrm[selected_idx]

            link_pts.append(dense_pts)
            link_nrm.append(dense_nrm)

        if link_pts:
            hand_points.append(np.concatenate(link_pts, axis=0))
            hand_normals.append(np.concatenate(link_nrm, axis=0))

    points = np.concatenate(hand_points, axis=0)
    normals = np.concatenate(hand_normals, axis=0)

    return points, normals


def visualize_hand_trimesh_with_coordinates(hand_mesh: trimesh.Trimesh):
    scene = trimesh.scene.Scene()
    scene.add_geometry(hand_mesh)
    coord = trimesh.creation.axis()
    scene.add_geometry(coord)
    scene.show()
