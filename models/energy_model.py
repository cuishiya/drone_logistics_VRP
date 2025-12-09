"""
带机会约束的能耗模型
基于论文: Chagas et al. (2025) 中的随机能耗模型部分

能耗公式：
e_{i_r, i_p}(q) = (d_{i_r, i_p} / ξ) * ((ν + q)^{3/2} * sqrt(g^3 / (2ρζn)))

其中：
- d: 节点之间的距离
- ξ: 随机速度（受风影响）
- ν: 无人机自重
- q: 载荷重量
- g: 重力加速度
- ρ: 空气密度
- ζ: 旋翼盘面积参数
- n: 旋翼数量
"""

import numpy as np
from scipy import stats
from typing import List, Tuple, Optional
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    GRAVITY, AIR_DENSITY, NUM_ROTORS, ROTOR_DISC_AREA,
    DRONE_WEIGHT, DRONE_SPEED_MEAN, DRONE_SPEED_STD,
    E_MAX, E_MIN, ALPHA, ALPHA_QUANTILE
)


@dataclass
class EnergyResult:
    """能耗计算结果"""
    expected_energy: float  # E[CE]
    energy_std: float  # SDV(CE)
    is_feasible: bool  # 是否满足机会约束
    safety_margin: float  # E_max - E_min - (E[CE] + Φ^{-1}(α) * SDV)


class EnergyModel:
    """
    无人机路径规划中的随机能耗模型
    
    实现论文中的机会约束能耗模型。
    """
    
    def __init__(
        self,
        drone_weight: float = DRONE_WEIGHT,
        gravity: float = GRAVITY,
        air_density: float = AIR_DENSITY,
        num_rotors: int = NUM_ROTORS,
        rotor_disc_area: float = ROTOR_DISC_AREA,
        speed_mean: float = DRONE_SPEED_MEAN,
        speed_std: float = DRONE_SPEED_STD,
        e_max: float = E_MAX,
        e_min: float = E_MIN,
        alpha: float = ALPHA
    ):
        self.drone_weight = drone_weight  # ν
        self.gravity = gravity  # g
        self.air_density = air_density  # ρ
        self.num_rotors = num_rotors  # n
        self.rotor_disc_area = rotor_disc_area  # ζ
        self.speed_mean = speed_mean
        self.speed_std = speed_std
        self.e_max = e_max
        self.e_min = e_min
        self.alpha = alpha
        self.alpha_quantile = stats.norm.ppf(alpha)  # Φ^{-1}(α)
        
        # 预计算常数因子: sqrt(g^3 / (2ρζn))
        self.power_factor = np.sqrt(
            (gravity ** 3) / (2 * air_density * rotor_disc_area * num_rotors)
        )
    
    def compute_edge_energy(
        self,
        distance: float,
        payload: float,
        speed: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        计算单条边上的能耗
        
        Args:
            distance: 节点间距离（米）
            payload: 当前载荷重量（kg）
            speed: 飞行速度（m/s），为None时使用平均速度
            
        Returns:
            (expected_energy, energy_variance) 二元组
        """
        if speed is None:
            speed = self.speed_mean
        
        # 总重量 = 无人机自重 + 载荷
        total_weight = self.drone_weight + payload
        
        # 功率消耗: (ν + q)^{3/2} * power_factor
        power = (total_weight ** 1.5) * self.power_factor
        
        # 能量 = 功率 * 时间 = 功率 * (距离 / 速度)
        # E[e] = d * power * E[1/ξ]
        # 对随机速度 ξ ~ N(μ, σ²)，近似 E[1/ξ] ≈ 1/μ
        # 且 Var(1/ξ) ≈ σ²/μ⁴
        
        expected_energy = distance * power / self.speed_mean
        
        # 使用Delta方法近似方差
        # Var(e) ≈ (d * power)² * Var(1/ξ) ≈ (d * power)² * σ²/μ⁴
        energy_variance = ((distance * power) ** 2) * (self.speed_std ** 2) / (self.speed_mean ** 4)
        
        return expected_energy, energy_variance
    
    def compute_trip_energy(
        self,
        route: List[np.ndarray],
        payloads: List[float]
    ) -> EnergyResult:
        """
        计算一个行程的总能耗
        
        Args:
            route: 位置列表 [仓库, 客户1, 客户2, ..., 仓库]
            payloads: 每一段上的载荷重量
                     payloads[i] 表示从 route[i] 到 route[i+1] 时的重量
                     
        Returns:
            包含期望能耗、标准差和可行性的 EnergyResult
        """
        if len(route) < 2:
            return EnergyResult(0.0, 0.0, True, self.e_max - self.e_min)
        
        total_expected = 0.0
        total_variance = 0.0
        
        for i in range(len(route) - 1):
            distance = np.linalg.norm(route[i+1] - route[i])
            payload = payloads[i] if i < len(payloads) else 0.0
            
            exp_e, var_e = self.compute_edge_energy(distance, payload)
            total_expected += exp_e
            # 假设各段独立，方差可相加
            total_variance += var_e
        
        total_std = np.sqrt(total_variance)
        
        # 检查机会约束：
        # E[CE] + Φ^{-1}(α) * SDV(CE) ≤ E_max - E_min
        constraint_value = total_expected + self.alpha_quantile * total_std
        available_energy = self.e_max - self.e_min
        is_feasible = constraint_value <= available_energy
        safety_margin = available_energy - constraint_value
        
        return EnergyResult(
            expected_energy=total_expected,
            energy_std=total_std,
            is_feasible=is_feasible,
            safety_margin=safety_margin
        )
    
    def compute_trip_energy_from_orders(
        self,
        depot_location: np.ndarray,
        orders: List['Order'],
        order_weights: Optional[List[float]] = None
    ) -> EnergyResult:
        """
        根据订单列表计算行程能耗
        
        Args:
            depot_location: 仓库位置
            orders: 按访问顺序排列的订单列表
            order_weights: 可选的订单重量列表（为None时使用order.weight）
            
        Returns:
            EnergyResult
        """
        if not orders:
            return EnergyResult(0.0, 0.0, True, self.e_max - self.e_min)
        
        # 构建路线
        route = [depot_location]
        for order in orders:
            route.append(order.location)
        route.append(depot_location)
        
        # 构建每段载重（随投递逐渐减小）
        if order_weights is None:
            order_weights = [o.weight for o in orders]
        
        # 每一段的载重：
        # 第0段: 仓库 -> 第1个客户，携带所有包裹
        # 第1段: 客户1 -> 客户2，携带除第1个外的所有包裹
        # ...
        # 最后一段: 最后一个客户 -> 仓库，不再携带包裹
        payloads = []
        remaining_weight = sum(order_weights)
        for i, weight in enumerate(order_weights):
            payloads.append(remaining_weight)
            remaining_weight -= weight
        payloads.append(0.0)  # Return to depot with no payload
        
        return self.compute_trip_energy(route, payloads)
    
    def check_insertion_feasibility(
        self,
        depot_location: np.ndarray,
        current_orders: List['Order'],
        new_order: 'Order',
        insert_position: int
    ) -> Tuple[bool, EnergyResult]:
        """
        检查在给定位置插入新订单是否满足能耗可行性
        
        Args:
            depot_location: 仓库位置
            current_orders: 当前行程中的订单
            new_order: 待插入的订单
            insert_position: 插入位置（0 = 第一个客户）
            
        Returns:
            (is_feasible, energy_result) 二元组
        """
        # Create new order list with insertion
        new_orders = current_orders[:insert_position] + [new_order] + current_orders[insert_position:]
        
        result = self.compute_trip_energy_from_orders(depot_location, new_orders)
        return result.is_feasible, result
    
    def get_max_feasible_distance(
        self,
        current_energy_used: float,
        current_energy_std: float,
        payload: float
    ) -> float:
        """
        计算在满足机会约束的前提下还能飞行的最大附加距离
        
        Args:
            current_energy_used: 已消耗的期望能量
            current_energy_std: 当前能量标准差
            payload: 当前载荷重量
            
        Returns:
            最大可追加距离（米）
        """
        available = self.e_max - self.e_min
        
        # 需要满足: (current + new_exp) + α_q * sqrt(current_var + new_var) ≤ available
        # 精确求解较复杂，这里使用保守估计
        
        # 保守估计：假设新的标准差线性相加（高估波动）
        remaining_budget = available - current_energy_used - self.alpha_quantile * current_energy_std
        
        if remaining_budget <= 0:
            return 0.0
        
        # 当前载重下的单位距离能量消耗
        total_weight = self.drone_weight + payload
        power = (total_weight ** 1.5) * self.power_factor
        energy_per_meter = power / self.speed_mean
        
        # 为方差加入安全系数
        safety_factor = 0.8
        
        return remaining_budget * safety_factor / energy_per_meter


# 距离计算的工具函数
def euclidean_distance(loc1: np.ndarray, loc2: np.ndarray) -> float:
    """计算两点之间的欧氏距离"""
    return np.linalg.norm(loc1 - loc2)


def compute_route_distance(route: List[np.ndarray]) -> float:
    """计算一条路径的总距离"""
    if len(route) < 2:
        return 0.0
    total = 0.0
    for i in range(len(route) - 1):
        total += euclidean_distance(route[i], route[i+1])
    return total
