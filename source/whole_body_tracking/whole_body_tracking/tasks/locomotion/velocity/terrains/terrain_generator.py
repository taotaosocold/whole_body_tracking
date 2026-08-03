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


@height_field_to_mesh
def stones_bridge_terrain(difficulty, cfg):
    """Generate AME's narrow stepping-stone bridge terrain."""
    stone_width = cfg.stone_width_range[1] - difficulty * (
        cfg.stone_width_range[1] - cfg.stone_width_range[0]
    )
    stone_length = cfg.stone_length_range[1] - difficulty * (
        cfg.stone_length_range[1] - cfg.stone_length_range[0]
    )
    stone_distance = cfg.stone_distance_range[0] + difficulty * (
        cfg.stone_distance_range[1] - cfg.stone_distance_range[0]
    )
    lateral_distance = cfg.stone_lateral_distance_range[0] + difficulty * (
        cfg.stone_lateral_distance_range[1] - cfg.stone_lateral_distance_range[0]
    )

    width = int(cfg.size[0] / cfg.horizontal_scale)
    length = int(cfg.size[1] / cfg.horizontal_scale)
    stone_width = max(1, int(stone_width / cfg.horizontal_scale))
    stone_length = max(1, int(stone_length / cfg.horizontal_scale))
    stone_distance = max(1, int(stone_distance / cfg.horizontal_scale))
    lateral_distance = int(lateral_distance / cfg.horizontal_scale)
    stone_height = int(cfg.stone_height_max / cfg.vertical_scale)
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)
    heights = np.arange(-stone_height - 1, stone_height) if stone_height > 0 else np.array([0])
    height_field = np.full((width, length), int(cfg.holes_depth / cfg.vertical_scale))

    start_x = stone_distance
    while start_x < width:
        stop_x = min(width, start_x + stone_width)
        start_y = (length - stone_length) // 2 + np.random.choice(
            [-lateral_distance, lateral_distance]
        )
        height_field[start_x:stop_x, start_y : start_y + stone_length] = np.random.choice(heights)
        start_x = stop_x + stone_distance

    start_y = stone_distance
    while start_y < length:
        stop_y = min(length, start_y + stone_width)
        start_x = (width - stone_length) // 2 + np.random.choice(
            [-lateral_distance, lateral_distance]
        )
        height_field[start_x : start_x + stone_length, start_y:stop_y] = np.random.choice(heights)
        start_y = stop_y + stone_distance

    x1 = (width - platform_width) // 2
    x2 = (width + platform_width) // 2
    y1 = (length - platform_width) // 2
    y2 = (length + platform_width) // 2
    height_field[x1:x2, y1:y2] = 0
    return np.rint(height_field).astype(np.int16)


def _stakes_parameters(difficulty, cfg):
    stake_side = cfg.stake_side_range[1] - difficulty * (
        cfg.stake_side_range[1] - cfg.stake_side_range[0]
    )
    stake_gap = cfg.stake_gap_range[0] + difficulty * (
        cfg.stake_gap_range[1] - cfg.stake_gap_range[0]
    )
    return (
        max(1, int(stake_side / cfg.horizontal_scale)),
        max(0, int(stake_gap / cfg.horizontal_scale)),
        int(cfg.size[0] / cfg.horizontal_scale),
        int(cfg.size[1] / cfg.horizontal_scale),
    )


@height_field_to_mesh
def double_column_stakes_terrain(difficulty, cfg):
    """Generate AME's paired stake columns along both terrain axes."""
    side, gap, width, length = _stakes_parameters(difficulty, cfg)
    column_gap = cfg.column_gap_range[0] + difficulty * (
        cfg.column_gap_range[1] - cfg.column_gap_range[0]
    )
    column_gap = max(0, int(column_gap / cfg.horizontal_scale))
    jitter = max(0, int(cfg.column_jitter / cfg.horizontal_scale))
    height_max = max(0, int(cfg.stake_height_max / cfg.vertical_scale))
    heights = np.arange(-height_max, height_max + 1) if height_max else np.array([0])
    field = np.full((width, length), int(cfg.holes_depth / cfg.vertical_scale), dtype=float)
    rng = np.random.default_rng()
    lower, upper = side // 2, side - side // 2

    def paint(cx, cy):
        value = int(rng.choice(heights))
        field[max(0, cx-lower):min(width, cx+upper), max(0, cy-lower):min(length, cy+upper)] = value

    offset = max((side + column_gap) // 2, lower)
    for x in range(0, width, side + gap):
        for sign in (-1, 1):
            delta = int(rng.integers(-jitter, jitter + 1)) if jitter else 0
            paint(x, int(np.clip(length // 2 + sign * offset + delta, lower, length - upper)))
    for y in range(0, length, side + gap):
        for sign in (-1, 1):
            delta = int(rng.integers(-jitter, jitter + 1)) if jitter else 0
            paint(int(np.clip(width // 2 + sign * offset + delta, lower, width - upper)), y)

    platform = int(cfg.platform_width / cfg.horizontal_scale)
    x1, x2 = (width - platform) // 2, (width + platform) // 2
    y1, y2 = (length - platform) // 2, (length + platform) // 2
    field[x1:x2, y1:y2] = 0
    return np.rint(field).astype(np.int16)


@height_field_to_mesh
def alternate_column_stakes_terrain(difficulty, cfg):
    """Generate AME's alternating single stake columns along both axes."""
    side, gap, width, length = _stakes_parameters(difficulty, cfg)
    column_gap = cfg.column_gap_range[1] - difficulty * (
        cfg.column_gap_range[1] - cfg.column_gap_range[0]
    )
    offset = max(0, int(column_gap / cfg.horizontal_scale)) // 2
    jitter = max(0, int(cfg.column_jitter / cfg.horizontal_scale))
    height_max = max(0, int(cfg.stake_height_max / cfg.vertical_scale))
    heights = np.arange(-height_max, height_max + 1) if height_max else np.array([0])
    field = np.full((width, length), int(cfg.holes_depth / cfg.vertical_scale), dtype=float)
    rng = np.random.default_rng()
    lower, upper = side // 2, side - side // 2

    def paint(cx, cy):
        value = int(rng.choice(heights))
        field[max(0, cx-lower):min(width, cx+upper), max(0, cy-lower):min(length, cy+upper)] = value

    sign = 1
    for x in range(0, width, side + gap):
        delta = int(rng.integers(-jitter, jitter + 1)) if jitter else 0
        paint(x, length // 2 + sign * offset + delta)
        sign *= -1
    sign = 1
    for y in range(0, length, side + gap):
        delta = int(rng.integers(-jitter, jitter + 1)) if jitter else 0
        paint(width // 2 + sign * offset + delta, y)
        sign *= -1

    platform = int(cfg.platform_width / cfg.horizontal_scale)
    x1, x2 = (width - platform) // 2, (width + platform) // 2
    y1, y2 = (length - platform) // 2, (length + platform) // 2
    field[x1:x2, y1:y2] = 0
    return np.rint(field).astype(np.int16)
