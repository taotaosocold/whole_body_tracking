"""Custom height-field terrain generators."""

import random

import numpy as np

from isaaclab.terrains.height_field.utils import height_field_to_mesh


@height_field_to_mesh
def concentric_gap_terrain(difficulty, cfg):
    """Generate alternating concentric gap and ground rings around a flat platform."""

    gap_depth = int(abs(cfg.gap_depth) / cfg.vertical_scale)
    gap_width = cfg.gap_width_range[0] + difficulty * (
        cfg.gap_width_range[1] - cfg.gap_width_range[0]
    )
    ground_width = cfg.ground_width_range[0] + (1.0 - difficulty) * (
        cfg.ground_width_range[1] - cfg.ground_width_range[0]
    )

    gap_width_px = max(1, int(gap_width / cfg.horizontal_scale))
    ground_width_px = max(1, int(ground_width / cfg.horizontal_scale))
    ground_height_max = int(cfg.ground_height_max / cfg.vertical_scale)
    width_px = int(cfg.size[0] / cfg.horizontal_scale)
    length_px = int(cfg.size[1] / cfg.horizontal_scale)
    platform_width_px = int(cfg.platform_width / cfg.horizontal_scale)

    height_field = np.zeros((width_px, length_px))
    start_x, start_y = 0, 0
    stop_x, stop_y = width_px, length_px
    is_gap = True

    while (stop_x - start_x) > platform_width_px and (stop_y - start_y) > platform_width_px:
        if is_gap:
            height_field[start_x:stop_x, start_y:stop_y] = -gap_depth
            increment = gap_width_px
        else:
            height_field[start_x:stop_x, start_y:stop_y] = random.randint(
                -ground_height_max, ground_height_max
            )
            increment = ground_width_px
        start_x += increment
        stop_x -= increment
        start_y += increment
        stop_y -= increment
        is_gap = not is_gap

    x1 = (width_px - platform_width_px) // 2
    x2 = (width_px + platform_width_px) // 2
    y1 = (length_px - platform_width_px) // 2
    y2 = (length_px + platform_width_px) // 2
    height_field[x1:x2, y1:y2] = 0

    return np.rint(height_field).astype(np.int16)
