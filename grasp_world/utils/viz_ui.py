import viser
import numpy as np
from jaxtyping import Array
import pytorch_kinematics as pk
import urdfpy
import torch
from typing import Optional
import trimesh
import logging
from rich.logging import RichHandler
import os

logger = logging.getLogger(__name__)
logger.addHandler(RichHandler())
logger.setLevel("INFO")
import time


class VizUI:
    def __init__(self):
        self.vis = viser.ViserServer()
        self.vis.scene.reset()

    def visualize_robot_state(
        self,
        robot_urdf_path: str,
        q: torch.Tensor,
        root_pose: Optional[torch.Tensor] = None,
    ):
        """Visualize robot configuration(s) from URDF and joint states using viser."""
        try:
            urdf = urdfpy.URDF.load(robot_urdf_path)
            robot = pk.build_chain_from_urdf(open(robot_urdf_path, "rb").read())
            logger.info(f"Loaded URDF successfully: {robot_urdf_path}")
        except Exception as e:
            logger.error(f"Failed to load URDF: {e}")
            raise ValueError(f"Failed to load URDF: {e}")

        # Ensure q is batched
        if q.ndim == 1:
            q = q.unsqueeze(0)
        batch_size = q.shape[0]
        logger.info(f"Visualizing {batch_size} configuration(s).")

        fk = robot.forward_kinematics(q)

        def apply_root_pose(mat: np.ndarray, root_pose: Optional[torch.Tensor]):
            """Apply optional root pose to the transformation matrix."""
            if root_pose is None:
                return mat
            if root_pose.shape == (7,):  # (pos, quat)
                pos = root_pose[:3].cpu().numpy()
                quat = root_pose[3:].cpu().numpy()
                R = pk.matrix_from_quaternion(torch.tensor(quat)).numpy()
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = pos
                return T @ mat
            elif root_pose.shape == (4, 4):
                return root_pose.cpu().numpy() @ mat
            else:
                logger.warning(f"Invalid root_pose shape: {root_pose.shape}")
                raise ValueError("root_pose must be (4,4) or (7,)")

        link_objs = {}
        for link in urdf.links:
            if not link.visuals:
                continue
            link_objs[link.name] = []

            for vi, visual in enumerate(link.visuals):
                geom = visual.geometry
                origin = visual.origin if visual.origin is not None else np.eye(4)
                local_R = origin[:3, :3]
                local_t = origin[:3, 3]
                local_quat = pk.matrix_to_quaternion(torch.tensor(local_R)).numpy()

                trimesh_obj = None
                try:
                    # Mesh
                    if geom.mesh is not None:
                        mesh_path = os.path.join(
                            os.path.dirname(robot_urdf_path), geom.mesh.filename
                        )
                        trimesh_obj = trimesh.load_mesh(mesh_path, force="mesh")

                    # Box
                    elif geom.box is not None:
                        trimesh_obj = trimesh.creation.box(extents=geom.box.size)

                    # Sphere
                    elif geom.sphere is not None:
                        trimesh_obj = trimesh.creation.icosphere(
                            radius=float(geom.sphere.radius), subdivisions=3
                        )

                    # Cylinder
                    elif geom.cylinder is not None:
                        trimesh_obj = trimesh.creation.cylinder(
                            radius=float(geom.cylinder.radius),
                            height=float(geom.cylinder.length),
                            sections=32,
                        )

                    if trimesh_obj is None:
                        continue

                    obj = self.vis.scene.add_mesh_trimesh(
                        name=f"{link.name}_{vi}",
                        mesh=trimesh_obj,
                        scale=(1.0, 1.0, 1.0),
                        wxyz=tuple(local_quat),
                        position=tuple(local_t),
                        cast_shadow=True,
                        receive_shadow=True,
                    )
                    link_objs[link.name].append((obj, local_R, local_t))
                    logger.debug(f"Added visual for {link.name}")

                except Exception as e:
                    logger.warning(f"Failed to load visual for {link.name}: {e}")

        def update_scene(idx: int):
            logger.debug(f"Updating scene to configuration index {idx}")
            for link_name, visuals in link_objs.items():
                if link_name not in fk:
                    continue
                link_tf = fk[link_name].get_matrix()[idx].cpu().numpy()
                link_tf = apply_root_pose(link_tf, root_pose)

                for obj, local_R, local_t in visuals:
                    local_tf = np.eye(4)
                    local_tf[:3, :3] = local_R
                    local_tf[:3, 3] = local_t
                    world_tf = link_tf @ local_tf
                    obj.position = world_tf[:3, 3]
                    obj.wxyz = pk.matrix_to_quaternion(torch.tensor(world_tf[:3, :3])).numpy()

        # Add slider for batched visualization
        if batch_size > 1:
            slider = self.vis.gui.add_slider(
                label="Scene index",
                min=0,
                max=batch_size - 1,
                step=1,
                initial_value=0,
            )

            @slider.on_update
            def _on_slider(event: viser.GuiEvent):
                idx = int(event.target.value)
                update_scene(idx)

            update_scene(0)
        else:
            update_scene(0)

        logger.info("Visualization setup complete.")
        while True:
            time.sleep(10)

    def visualize_hand_sphere_state(
        self,
        scenes: list[tuple[np.ndarray, np.ndarray]],
        sphere_color: tuple[float, float, float] = [(0.2, 0.7, 1.0)],
    ):
        assert len(scenes) > 0, "scenes must contain at least one (centers, radii) pair"
        n_spheres = len(scenes[0][0])  # number of spheres per scene

        self.hand_sphere = []
        c0, r0 = scenes[0]
        for j in range(n_spheres):
            h = self.vis.scene.add_icosphere(
                position=tuple(c0[j]),
                radius=float(r0[j]),
                color=sphere_color,
                opacity=0.5,
                name=f"sphere_{j}",
            )
            self.hand_sphere.append(h)

        slider = self.vis.gui.add_slider(
            label="Scene index",
            min=0,
            max=len(scenes) - 1,
            step=1,
            initial_value=0,
        )

        @slider.on_update
        def _update_scene(event: viser.GuiEvent):
            idx = int(event.target.value)
            c, r = scenes[idx]
            for j, sphere in enumerate(self.hand_sphere):
                sphere.position = tuple(c[j])
                sphere.radius = float(r[j])


from grasp_world.robo.handmodel import DexHand, HANDS


shadow_config = HANDS["shadow_hand"]


VizUI().visualize_robot_state(shadow_config.urdf_path, torch.zeros((5, 28)))
