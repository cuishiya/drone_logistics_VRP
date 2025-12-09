"""
多仓库配置模块
定义所有仓库站点的位置和属性
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass


# =============================================================================
# 仓库站点原始数据（顺丰门店）
# =============================================================================
DEPOT_STATIONS_RAW = [
    {"id": 0, "name": "冠城高新科技店", "lon": 113.939042, "lat": 22.798145},
    {"id": 1, "name": "新健兴科技店", "lon": 113.949883, "lat": 22.772652},
    {"id": 2, "name": "碧眼新村店", "lon": 113.951536, "lat": 22.7567584},
    {"id": 3, "name": "南庄店", "lon": 113.910375, "lat": 22.7740880},
    {"id": 4, "name": "西田店", "lon": 113.900603, "lat": 22.8009850},
    {"id": 5, "name": "东发集配店", "lon": 113.922249, "lat": 22.7509074},
    {"id": 6, "name": "中央山店", "lon": 113.898593, "lat": 22.7765857},
    {"id": 7, "name": "大鸿科技店", "lon": 113.989764, "lat": 22.7398432},
    {"id": 8, "name": "汇业科技园店", "lon": 113.942272, "lat": 22.736281}
]


def lonlat_to_meters(lon: float, lat: float, ref_lon: float, ref_lat: float) -> Tuple[float, float]:
    """
    将经纬度转换为以参考点为原点的米制坐标
    使用简化的平面投影（适用于小范围区域）
    
    Args:
        lon: 经度
        lat: 纬度
        ref_lon: 参考点经度
        ref_lat: 参考点纬度
        
    Returns:
        (x, y) 米制坐标
    """
    # 地球半径（米）
    R = 6371000
    
    # 纬度差转换为米
    y = (lat - ref_lat) * np.pi / 180 * R
    
    # 经度差转换为米（考虑纬度影响）
    x = (lon - ref_lon) * np.pi / 180 * R * np.cos(ref_lat * np.pi / 180)
    
    return x, y


@dataclass
class DepotInfo:
    """
    仓库站点信息
    
    Attributes:
        id: 仓库唯一标识
        name: 仓库名称
        location: 米制坐标 (x, y)
        lon: 原始经度
        lat: 原始纬度
        num_drones: 该仓库的无人机数量
        num_batteries: 该仓库的电池数量
        num_charging_stations: 充电站数量
    """
    id: int
    name: str
    location: np.ndarray  # 米制坐标
    lon: float
    lat: float
    num_drones: int = 1  # 默认每个仓库1架无人机
    num_batteries: int = 2  # 默认每个仓库2块电池
    num_charging_stations: int = 2  # 默认2个充电站


def create_depot_infos(
    num_drones_per_depot: int = 1,
    num_batteries_per_depot: int = 2,
    num_charging_stations_per_depot: int = 2
) -> List[DepotInfo]:
    """
    创建所有仓库站点信息
    
    Args:
        num_drones_per_depot: 每个仓库的无人机数量
        num_batteries_per_depot: 每个仓库的电池数量
        num_charging_stations_per_depot: 每个仓库的充电站数量
        
    Returns:
        仓库信息列表
    """
    # 使用第一个站点作为参考点
    ref_lon = DEPOT_STATIONS_RAW[0]["lon"]
    ref_lat = DEPOT_STATIONS_RAW[0]["lat"]
    
    depot_infos = []
    for station in DEPOT_STATIONS_RAW:
        x, y = lonlat_to_meters(station["lon"], station["lat"], ref_lon, ref_lat)
        
        depot_info = DepotInfo(
            id=station["id"],
            name=station["name"],
            location=np.array([x, y]),
            lon=station["lon"],
            lat=station["lat"],
            num_drones=num_drones_per_depot,
            num_batteries=num_batteries_per_depot,
            num_charging_stations=num_charging_stations_per_depot
        )
        depot_infos.append(depot_info)
    
    return depot_infos


def get_depot_locations() -> List[np.ndarray]:
    """
    获取所有仓库的米制坐标列表
    
    Returns:
        仓库坐标列表
    """
    depot_infos = create_depot_infos()
    return [depot.location for depot in depot_infos]


def get_depot_count() -> int:
    """
    获取仓库数量
    
    Returns:
        仓库数量
    """
    return len(DEPOT_STATIONS_RAW)


def get_service_area_bounds() -> Tuple[float, float, float, float]:
    """
    获取服务区域边界
    
    Returns:
        (min_x, max_x, min_y, max_y) 米制坐标边界
    """
    locations = get_depot_locations()
    xs = [loc[0] for loc in locations]
    ys = [loc[1] for loc in locations]
    
    # 扩展边界以覆盖仓库周围区域
    margin = 2000  # 2公里边距
    return (
        min(xs) - margin,
        max(xs) + margin,
        min(ys) - margin,
        max(ys) + margin
    )


def find_nearest_depot(location: np.ndarray, depot_infos: List[DepotInfo]) -> DepotInfo:
    """
    找到距离给定位置最近的仓库
    
    Args:
        location: 目标位置坐标
        depot_infos: 仓库信息列表
        
    Returns:
        最近的仓库信息
    """
    min_dist = float('inf')
    nearest_depot = depot_infos[0]
    
    for depot in depot_infos:
        dist = np.linalg.norm(location - depot.location)
        if dist < min_dist:
            min_dist = dist
            nearest_depot = depot
    
    return nearest_depot


def plot_depots(save_path: str = 'sf_stations_map.png'):
    """
    绘制仓库分布图
    
    Args:
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    
    lons = [s["lon"] for s in DEPOT_STATIONS_RAW]
    lats = [s["lat"] for s in DEPOT_STATIONS_RAW]
    names = [s["name"] for s in DEPOT_STATIONS_RAW]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(lons, lats, c='blue', s=100, alpha=0.7, label='SF Stations (Depots)')
    
    # 添加标签
    for i, txt in enumerate(names):
        plt.annotate(i, (lons[i], lats[i]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=12, weight='bold')
    
    plt.title('Spatial Distribution of SF Express Depots (Multi-Depot)', fontsize=14)
    plt.xlabel('Longitude (E)', fontsize=12)
    plt.ylabel('Latitude (N)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.savefig(save_path)
    plt.show()


# 默认仓库配置（用于向后兼容）
DEFAULT_DEPOT_INFOS = create_depot_infos()
DEPOT_LOCATIONS = get_depot_locations()
NUM_DEPOTS = get_depot_count()


if __name__ == "__main__":
    # 测试代码
    print(f"总仓库数量: {NUM_DEPOTS}")
    print("\n仓库信息:")
    for depot in DEFAULT_DEPOT_INFOS:
        print(f"  {depot.id}: {depot.name} -> 坐标: ({depot.location[0]:.1f}, {depot.location[1]:.1f}) 米")
    
    bounds = get_service_area_bounds()
    print(f"\n服务区域边界: X=[{bounds[0]:.1f}, {bounds[1]:.1f}], Y=[{bounds[2]:.1f}, {bounds[3]:.1f}] 米")
    
    # 绘制仓库分布图
    plot_depots()