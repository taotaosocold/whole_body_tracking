"""Rough-terrain curriculum migrated from AME_Locomotion."""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from .terrain_cfg import HfConcentricGapTerrainCfg

# 地形生成配置
ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    # 每一块地形8m*8m，每个地形之间直接拼接（虽然可视化看的时候好像有距离，但其实是算在8m内的）
    # 然后rows和cols表示一行有10个地形，一列有20个地形，所以地形占据8*8*10*20=12800^2。
    # border_width表示地形外围还会围着50m的平坦地形
    size=(8.0, 8.0),
    border_width=50.0,
    num_rows=10,
    num_cols=20,
    # horizontal_scale和vertical_scale设置就是反正是网格间距，这个就这个配置就行
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    # use_cache为false就是每次启动时重新生成地形，true则会用缓存
    use_cache=False,
    sub_terrains={
        # MeshPyramidStairsTerrainCfg是上楼梯的地形，占比0.1即10*20个地形中有20个地形是上楼梯的
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.2),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # 下楼梯的地形
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.2),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # 随机方块，每个方块宽0.45m，每个方块的高度变化为5cm～20cm随机生成
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.1,
            grid_width=0.45,
            grid_height_range=(0.05, 0.2),
            platform_width=2.0,
        ),
        # 随机粗糙面，就是地面存在起伏，就地面是类似于这种的──╱╲_╱╲__╱╲╱╲_╱╲──
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1,
            noise_range=(0.02, 0.10),
            noise_step=0.02,
            downsampled_scale=0.1,
            border_width=0.25,
        ),
        # 斜的上坡面
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
        # 斜的下坡面
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
        # 踏石，就是脚底有一块石头，然后可能前面还有快石头，石头和石头之间有个很深的沟或坑
        "hf_steppingstones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.2,
            stone_height_max=0.05,
            stone_width_range=(0.25, 0.5),
            stone_distance_range=(0.05, 0.25),
            platform_width=2.0,
            holes_depth=-2.0,
            border_width=0.25,
        ),
        # 多圈同心沟
        "hf_gaps": HfConcentricGapTerrainCfg(
            proportion=0.2,
            gap_width_range=(0.1, 0.5),
            platform_width=2.0,
            border_width=0.25,
            gap_depth=-2.0,
            ground_width_range=(0.5, 0.5),
            ground_height_max=0.025,
        ),
    },
)
"""Ten curriculum levels and twenty terrain columns, each tile 8 m x 8 m."""
