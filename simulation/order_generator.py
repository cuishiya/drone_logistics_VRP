"""
Order Generation Module
Generates dynamic orders using Poisson process
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.data_structures import Order
from config import (
    LAMBDA_ORDERS, AREA_SIZE,
    SERVICE_TIME, MAX_PAYLOAD, T_HORIZON,
    AREA_BOUNDS, DEFAULT_DEPOT_INFOS, NUM_DEPOTS
)
from depot import find_nearest_depot, DepotInfo


class OrderGenerator:
    """
    Generates customer orders dynamically using Poisson process
    支持多仓库场景：订单生成后自动分配到最近的仓库
    
    Orders arrive according to Poisson process with rate λ.
    Locations are uniformly distributed in the service area.
    Deadlines are set based on arrival time + random service window.
    """
    
    def __init__(
        self,
        arrival_rate: float = LAMBDA_ORDERS,
        area_size: float = AREA_SIZE,
        service_time: float = SERVICE_TIME,
        max_weight: float = MAX_PAYLOAD,
        min_deadline_offset: float = 30.0,  # 最小截止时间偏移30分钟
        max_deadline_offset: float = 120.0,  # 最大截止时间偏移2小时
        seed: Optional[int] = None,
        depot_infos: List[DepotInfo] = None,
        area_bounds: tuple = None
    ):
        """
        初始化订单生成器
        
        Args:
            arrival_rate: 订单到达率（每分钟）
            area_size: 服务区域大小
            service_time: 每个客户的服务时间
            max_weight: 最大订单重量
            min_deadline_offset: 最小截止时间偏移
            max_deadline_offset: 最大截止时间偏移
            seed: 随机种子
            depot_infos: 仓库信息列表
            area_bounds: 服务区域边界
        """
        self.arrival_rate = arrival_rate
        self.area_size = area_size
        self.service_time = service_time
        self.max_weight = max_weight
        self.min_deadline_offset = min_deadline_offset
        self.max_deadline_offset = max_deadline_offset
        
        # 多仓库配置
        self.depot_infos = depot_infos if depot_infos is not None else DEFAULT_DEPOT_INFOS
        self.area_bounds = area_bounds if area_bounds is not None else AREA_BOUNDS
        
        self.rng = np.random.default_rng(seed)
        self._order_counter = 0
    
    def generate_orders_in_interval(
        self,
        start_time: float,
        end_time: float
    ) -> List[Order]:
        """
        Generate orders arriving in time interval [start_time, end_time)
        
        Uses Poisson process: number of arrivals ~ Poisson(λ * Δt)
        
        Args:
            start_time: Start of interval (minutes)
            end_time: End of interval (minutes)
            
        Returns:
            List of new Order objects
        """
        duration = end_time - start_time
        if duration <= 0:
            return []
        
        # Number of arrivals follows Poisson distribution
        expected_arrivals = self.arrival_rate * duration
        num_arrivals = self.rng.poisson(expected_arrivals)
        
        orders = []
        for _ in range(num_arrivals):
            order = self._generate_single_order(start_time, end_time)
            orders.append(order)
        
        return orders
    
    def _generate_single_order(
        self,
        start_time: float,
        end_time: float
    ) -> Order:
        """生成单个随机订单，自动分配到最近仓库"""
        self._order_counter += 1
        
        # 随机到达时间
        arrival_time = self.rng.uniform(start_time, end_time)
        
        # 基于服务区域边界生成位置
        min_x, max_x, min_y, max_y = self.area_bounds
        x = self.rng.uniform(min_x, max_x)
        y = self.rng.uniform(min_y, max_y)
        location = np.array([x, y])
        
        # 随机重量
        weight = self.rng.uniform(0.5, self.max_weight)
        
        # 随机截止时间
        deadline_offset = self.rng.uniform(
            self.min_deadline_offset,
            self.max_deadline_offset
        )
        deadline = arrival_time + deadline_offset
        
        # 分配到最近的仓库
        nearest_depot = find_nearest_depot(location, self.depot_infos)
        assigned_depot_id = nearest_depot.id
        
        return Order(
            id=self._order_counter,
            location=location,
            weight=weight,
            arrival_time=arrival_time,
            deadline=deadline,
            service_time=self.service_time,
            assigned_depot_id=assigned_depot_id
        )
    
    def generate_initial_orders(
        self,
        num_orders: int,
        time_horizon: float = T_HORIZON
    ) -> List[Order]:
        """
        生成初始订单集合，自动分配到最近仓库
        
        Args:
            num_orders: 订单数量
            time_horizon: 截止时间范围
            
        Returns:
            订单列表
        """
        orders = []
        for _ in range(num_orders):
            self._order_counter += 1
            
            # 所有订单在时间 0 到达
            arrival_time = 0.0
            
            # 基于服务区域边界生成位置
            min_x, max_x, min_y, max_y = self.area_bounds
            x = self.rng.uniform(min_x, max_x)
            y = self.rng.uniform(min_y, max_y)
            location = np.array([x, y])
            
            # 随机重量
            weight = self.rng.uniform(0.5, self.max_weight)
            
            # 随机截止时间
            deadline = self.rng.uniform(
                self.min_deadline_offset,
                min(self.max_deadline_offset * 2, time_horizon)
            )
            
            # 分配到最近的仓库
            nearest_depot = find_nearest_depot(location, self.depot_infos)
            assigned_depot_id = nearest_depot.id
            
            order = Order(
                id=self._order_counter,
                location=location,
                weight=weight,
                arrival_time=arrival_time,
                deadline=deadline,
                service_time=self.service_time,
                assigned_depot_id=assigned_depot_id
            )
            orders.append(order)
        
        return orders
    
    def generate_clustered_orders(
        self,
        num_orders: int,
        cluster_std: float = 500.0
    ) -> List[Order]:
        """
        生成聚类订单，使用仓库位置作为聚类中心
        
        Args:
            num_orders: 订单数量
            cluster_std: 聚类标准差
            
        Returns:
            订单列表
        """
        # 使用仓库位置作为聚类中心
        centers = [depot.location for depot in self.depot_infos]
        num_clusters = len(centers)
        
        orders = []
        for _ in range(num_orders):
            self._order_counter += 1
            
            # 随机选择聚类中心
            cluster_idx = self.rng.integers(num_clusters)
            center = centers[cluster_idx]
            
            # 在聚类中心附近生成位置
            x = self.rng.normal(center[0], cluster_std)
            y = self.rng.normal(center[1], cluster_std)
            
            # 限制在服务区域内
            min_x, max_x, min_y, max_y = self.area_bounds
            x = np.clip(x, min_x, max_x)
            y = np.clip(y, min_y, max_y)
            location = np.array([x, y])
            
            # 随机重量和截止时间
            weight = self.rng.uniform(0.5, self.max_weight)
            deadline = self.rng.uniform(
                self.min_deadline_offset,
                self.max_deadline_offset * 2
            )
            
            # 分配到最近的仓库
            nearest_depot = find_nearest_depot(location, self.depot_infos)
            assigned_depot_id = nearest_depot.id
            
            order = Order(
                id=self._order_counter,
                location=location,
                weight=weight,
                arrival_time=0.0,
                deadline=deadline,
                service_time=self.service_time,
                assigned_depot_id=assigned_depot_id
            )
            orders.append(order)
        
        return orders
    
    def get_orders_by_depot(self, orders: List[Order]) -> dict:
        """
        按仓库分组订单（多仓库场景）
        
        Args:
            orders: 订单列表
            
        Returns:
            字典，键为仓库ID，值为该仓库的订单列表
        """
        orders_by_depot = {depot.id: [] for depot in self.depot_infos}
        
        for order in orders:
            depot_id = order.assigned_depot_id
            if depot_id is not None and depot_id in orders_by_depot:
                orders_by_depot[depot_id].append(order)
            else:
                # 未分配的订单，分配到最近仓库
                nearest_depot = find_nearest_depot(order.location, self.depot_infos)
                order.assigned_depot_id = nearest_depot.id
                orders_by_depot[nearest_depot.id].append(order)
        
        return orders_by_depot
