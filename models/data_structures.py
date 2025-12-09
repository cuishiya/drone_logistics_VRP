"""
DRPUDEC核心数据结构
定义: Order(订单), Drone(无人机), Battery(电池), Depot(仓库), Trip(行程), State(状态)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple
from enum import Enum
import numpy as np


class DroneStatus(Enum):
    """无人机运行状态"""
    IDLE = "idle"  # 在仓库，可用
    EN_ROUTE = "en_route"  # 飞往客户或返航中
    SERVING = "serving"  # 正在服务客户
    CHARGING = "charging"  # 等待电池更换/充电


class BatteryStatus(Enum):
    """电池状态"""
    IN_USE = "in_use"  # 安装在无人机上
    CHARGING = "charging"  # 充电中
    AVAILABLE = "available"  # 已充满，可用


@dataclass
class Order:
    """
    客户订单/请求
    
    属性:
        id: 订单唯一标识
        location: 客户位置坐标(x, y)
        weight: 包裹重量(kg)，记为q_r
        arrival_time: 订单到达时间（分钟）
        deadline: 软截止时间（分钟），记为l_r
        service_time: 服务客户所需时间（分钟），记为η
        is_served: 订单是否已服务
        assigned_trip_id: 分配的行程ID（如果有）
        assigned_depot_id: 分配的仓库ID（多仓库场景）
    """
    id: int
    location: np.ndarray
    weight: float
    arrival_time: float
    deadline: float
    service_time: float = 2.0
    is_served: bool = False
    assigned_trip_id: Optional[int] = None
    assigned_depot_id: Optional[int] = None  # 多仓库：分配的仓库ID
    
    def urgency(self, current_time: float) -> float:
        """
        计算紧急度分数（越低越紧急）
        紧急度 = 截止时间 - 当前时间
        """
        return self.deadline - current_time
    
    def is_urgent(self, current_time: float, threshold: float = 30.0) -> bool:
        """检查订单是否紧急（距离截止时间不足threshold分钟）"""
        return self.urgency(current_time) < threshold
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if isinstance(other, Order):
            return self.id == other.id
        return False


@dataclass
class Battery:
    """
    无人机电池
    
    属性:
        id: 电池唯一标识
        current_charge: 当前电量(0到E_max)
        max_charge: 最大容量
        cycle_count: 充放电循环次数（用于老化计算）
        status: 当前电池状态
        charge_start_time: 开始充电时间（如果正在充电）
    """
    id: int
    current_charge: float
    max_charge: float
    cycle_count: int = 0
    status: BatteryStatus = BatteryStatus.AVAILABLE
    charge_start_time: Optional[float] = None
    
    def time_to_full_charge(self, charge_rate: float) -> float:
        """计算充满所需时间"""
        if self.status == BatteryStatus.AVAILABLE:
            return 0.0
        remaining = self.max_charge - self.current_charge
        return remaining / charge_rate
    
    def update_charge(self, current_time: float, charge_rate: float):
        """根据经过时间更新电量"""
        if self.status == BatteryStatus.CHARGING and self.charge_start_time is not None:
            elapsed = current_time - self.charge_start_time
            self.current_charge = min(
                self.max_charge,
                self.current_charge + elapsed * charge_rate
            )
            if self.current_charge >= self.max_charge:
                self.current_charge = self.max_charge
                self.status = BatteryStatus.AVAILABLE
                self.charge_start_time = None


@dataclass
class Drone:
    """
    无人机
    
    属性:
        id: 无人机唯一标识
        location: 当前位置坐标(x, y)
        status: 当前运行状态
        battery: 当前安装的电池
        weight: 无人机自重(kg)，记为ν
        max_payload: 最大载重(kg)，记为Q
        current_trip: 当前执行的行程（如果有）
        available_time: 无人机可用时间（用于分配）
        total_distance: 总飞行距离（用于统计）
        home_depot_id: 所属仓库ID（多仓库场景）
        home_depot_location: 所属仓库位置（多仓库场景）
    """
    id: int
    location: np.ndarray
    status: DroneStatus
    battery: Optional[Battery]
    weight: float
    max_payload: float
    current_trip: Optional['Trip'] = None
    available_time: float = 0.0
    total_distance: float = 0.0
    home_depot_id: Optional[int] = None  # 多仓库：所属仓库ID
    home_depot_location: Optional[np.ndarray] = None  # 多仓库：所属仓库位置
    
    def remaining_energy(self) -> float:
        """获取剩余电池能量"""
        if self.battery is None:
            return 0.0
        return self.battery.current_charge
    
    def is_available(self, current_time: float) -> bool:
        """检查无人机是否可用于新分配"""
        return (self.status == DroneStatus.IDLE and 
                self.available_time <= current_time)


@dataclass
class Trip:
    """
    无人机行程（路线）
    
    属性:
        id: 行程唯一标识
        orders: 要服务的订单列表（按访问顺序）
        drone_id: 分配的无人机ID（如果已分配）
        planned_start_time: 计划出发时间
        expected_energy: 预期能耗
        energy_std: 能耗标准差
        total_distance: 总行程距离
        total_weight: 总载重
        is_executed: 行程是否已执行
        is_cancelled: 行程是否已取消
        depot_id: 出发仓库ID（多仓库场景）
        depot_location: 出发仓库位置（多仓库场景）
    """
    id: int
    orders: List[Order] = field(default_factory=list)
    drone_id: Optional[int] = None
    planned_start_time: Optional[float] = None
    expected_energy: float = 0.0
    energy_std: float = 0.0
    total_distance: float = 0.0
    total_weight: float = 0.0
    is_executed: bool = False
    is_cancelled: bool = False
    depot_id: Optional[int] = None  # 多仓库：出发仓库ID
    depot_location: Optional[np.ndarray] = None  # 多仓库：出发仓库位置
    
    def get_customer_ids(self) -> Set[int]:
        """获取此行程中的客户ID集合"""
        return {order.id for order in self.orders}
    
    def get_route(self, depot_location: np.ndarray = None) -> List[np.ndarray]:
        """
        Get full route including depot at start and end
        
        Args:
            depot_location: 仓库位置，如果为None则使用trip自身的depot_location
        """
        # 多仓库支持：优先使用传入的depot_location，否则使用trip自身的depot_location
        actual_depot = depot_location if depot_location is not None else self.depot_location
        if actual_depot is None:
            actual_depot = np.array([0.0, 0.0])  # 默认原点
        
        route = [actual_depot]
        for order in self.orders:
            route.append(order.location)
        route.append(actual_depot)
        return route
    
    def expected_duration(self, speed: float) -> float:
        """计算预期行程时长"""
        travel_time = self.total_distance / speed / 60.0  # 转换为分钟
        service_time = sum(o.service_time for o in self.orders)
        return travel_time + service_time
    
    def max_lateness(self, start_time: float, speed: float) -> float:
        """计算所有订单中的最大延迟"""
        if not self.orders:
            return 0.0
        
        current_time = start_time
        max_late = 0.0
        
        # 简化处理：假设顺序访问
        for i, order in enumerate(self.orders):
            if i == 0:
                # 从仓库出发
                dist = np.linalg.norm(order.location)
            else:
                dist = np.linalg.norm(order.location - self.orders[i-1].location)
            
            travel_time = dist / speed / 60.0
            current_time += travel_time
            
            lateness = max(0, current_time - order.deadline)
            max_late = max(max_late, lateness)
            
            current_time += order.service_time
        
        return max_late


@dataclass
class Depot:
    """
    Represents a depot/warehouse (支持多仓库场景)
    
    Attributes:
        id: 仓库唯一标识
        name: 仓库名称
        location: (x, y) coordinates
        available_batteries: List of available batteries
        charging_batteries: List of batteries currently charging
        num_charging_stations: Number of charging stations
        drones: 该仓库的无人机列表
    """
    location: np.ndarray
    available_batteries: List[Battery] = field(default_factory=list)
    charging_batteries: List[Battery] = field(default_factory=list)
    num_charging_stations: int = 5
    id: int = 0  # 多仓库：仓库ID
    name: str = "Depot"  # 多仓库：仓库名称
    drones: List['Drone'] = field(default_factory=list)  # 多仓库：该仓库的无人机
    
    def get_best_battery(self) -> Optional[Battery]:
        """
        获取最佳可用电池（使用次数最少，用于均衡老化）
        如果没有可用电池则返回None
        """
        if not self.available_batteries:
            return None
        # 按循环次数升序排序以均衡老化
        self.available_batteries.sort(key=lambda b: b.cycle_count)
        return self.available_batteries[0]
    
    def return_battery(self, battery: Battery, current_time: float):
        """将使用过的电池返回仓库充电"""
        battery.status = BatteryStatus.CHARGING
        battery.charge_start_time = current_time
        battery.cycle_count += 1
        self.charging_batteries.append(battery)
    
    def update_charging(self, current_time: float, charge_rate: float):
        """更新所有充电中的电池"""
        still_charging = []
        for battery in self.charging_batteries:
            battery.update_charge(current_time, charge_rate)
            if battery.status == BatteryStatus.AVAILABLE:
                self.available_batteries.append(battery)
            else:
                still_charging.append(battery)
        self.charging_batteries = still_charging


@dataclass
class State:
    """
    MDP State at decision epoch k (支持多仓库场景)
    
    Attributes:
        time: Current time t_k
        pending_orders: Orders waiting to be assigned
        active_trips: Trips that are planned but not yet started
        executing_trips: Trips currently being executed
        drones: Current state of all drones
        depot: Current depot state (向后兼容，单仓库场景)
        depots: 所有仓库状态列表（多仓库场景）
    """
    time: float
    pending_orders: List[Order] = field(default_factory=list)
    active_trips: List[Trip] = field(default_factory=list)
    executing_trips: List[Trip] = field(default_factory=list)
    drones: List[Drone] = field(default_factory=list)
    depot: Optional[Depot] = None  # 向后兼容：单仓库场景
    depots: List[Depot] = field(default_factory=list)  # 多仓库：所有仓库列表
    
    def get_unassigned_orders(self) -> List[Order]:
        """获取尚未分配到任何行程的订单"""
        return [o for o in self.pending_orders if o.assigned_trip_id is None]
    
    def get_available_drones(self) -> List[Drone]:
        """获取可用于新分配的无人机"""
        return [d for d in self.drones if d.is_available(self.time)]
    
    def get_urgent_orders(self, threshold: float = 30.0) -> List[Order]:
        """获取紧急订单"""
        return [o for o in self.pending_orders 
                if not o.is_served and o.is_urgent(self.time, threshold)]
