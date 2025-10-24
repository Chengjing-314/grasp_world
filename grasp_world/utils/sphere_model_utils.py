import torch
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np


@dataclass
class SphereSpec:
    local_center: torch.Tensor
    radius: torch.Tensor


def load_sphere_database(
    path: Path | str, branch: int = 8, depth: int = 1
) -> dict[str, list[SphereSpec]]:
    with Path(path).open("r") as file:
        raw_db = json.load(file)

    processed: dict[str, list[SphereSpec]] = {}
    for mesh_key, branch_dict in raw_db.items():
        cleand_mesh_key = mesh_key.split("::")[0]
        if not branch_dict:
            continue

        branch_map: dict[int, dict] = {int(k): v for k, v in branch_dict.items()}
        selected_branch = (
            branch if branch is not None and branch in branch_map else max(branch_map)
        )

        depth_map: dict[int, dict] = {int(k): v for k, v in branch_map[selected_branch].items()}
        if not depth_map:
            continue

        depth_keys = sorted(depth_map.keys())
        selected_depth = depth if depth is not None and depth in depth_map else depth_keys[-1]
        candidate_depths = [d for d in reversed(depth_keys) if d <= selected_depth]
        if not candidate_depths:
            candidate_depths = list(reversed(depth_keys))

        sphere_entries = []
        for depth_candidate in candidate_depths:
            sphere_entries = depth_map[depth_candidate].get("spheres", [])
            if sphere_entries:
                selected_depth = depth_candidate
                break

        if not sphere_entries:
            continue

        if cleand_mesh_key in processed:
            processed[cleand_mesh_key].extend(
                [
                    SphereSpec(
                        local_center=torch.tensor(entry["origin"]),
                        radius=torch.tensor(entry["radius"]),
                    )
                    for entry in sphere_entries
                ]
            )
        else:
            processed[cleand_mesh_key] = [
                SphereSpec(
                    local_center=torch.tensor(entry["origin"]),
                    radius=torch.tensor(entry["radius"]),
                )
                for entry in sphere_entries
            ]

    return processed
