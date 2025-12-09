"""
Main Simulator for DRPUDEC
Implements the MDP main loop (Algorithm 1) and CFA policy (Algorithm 2)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from copy import deepcopy

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.data_structures import (
    Order, Drone, Battery, Depot, Trip, State,
    DroneStatus, BatteryStatus
)
from models.energy_model import EnergyModel, euclidean_distance
from algorithms.trip_builder import TripBuilder
from algorithms.trip_selector import TripSelector
from algorithms.trip_assigner import TripAssigner
from algorithms.battery_manager import BatteryManager
from simulation.order_generator import OrderGenerator
from config import (
    T_HORIZON, PSI, UPSILON,
    DRONE_WEIGHT, MAX_PAYLOAD, E_MAX, E_MIN, CHARGE_RATE,
    DRONE_SPEED_MEAN, M_MAX_TRIPS,
    SERVICE_TIME, LATE_PENALTY_RATE, DISTANCE_COST_RATE,
    DEFAULT_DEPOT_INFOS, NUM_DEPOTS, DRONES_PER_DEPOT, BATTERIES_PER_DEPOT
)
from depot import DepotInfo, create_depot_infos


@dataclass
class SimulationResult:
    """
    仿真运行结果
    
    包含仿真统计数据和详细日志，用于分析和可视化
    """
    total_orders: int = 0           # 总订单数
    served_orders: int = 0          # 已服务订单数
    total_distance: float = 0.0     # 总飞行距离（米）
    total_lateness: float = 0.0     # 总延迟时间（分钟）
    total_cost: float = 0.0         # 总成本（算法1，第5行输出）
    total_trips: int = 0            # 总行程数
    failed_trips: int = 0           # 因能耗不足失败的行程数
    average_lateness: float = 0.0   # 平均延迟（分钟）
    service_rate: float = 0.0       # 服务率
    max_lateness: float = 0.0       # 最大延迟（分钟）
    total_energy: float = 0.0       # 总能耗（Wh）
    
    # 详细日志
    order_log: List[Dict] = field(default_factory=list)    # 订单日志
    trip_log: List[Dict] = field(default_factory=list)     # 行程日志
    decision_log: List[Dict] = field(default_factory=list) # 决策日志
    
    # 可视化数据
    trips: List[Trip] = field(default_factory=list)        # 所有行程对象
    orders: List[Order] = field(default_factory=list)      # 所有订单对象
    lateness_distribution: List[float] = field(default_factory=list)  # 延迟分布


class Simulator:
    """
    DRPUDEC多仓库仿真引擎
    
    每个仓库独立运作，有自己的无人机、电池和订单
    
    实现算法:
    - Algorithm 1: MDP主循环
    - Algorithm 2: CFA策略
    
    仿真以PSI分钟为间隔进行离散时间步。
    每个决策周期:
    1. 执行CFA策略生成决策
    2. 推进时间并观察新订单
    3. 更新系统状态
    """
    
    def __init__(
        self,
        time_horizon: float = T_HORIZON,
        decision_interval: float = PSI,
        buffer_threshold: float = UPSILON,
        max_trips_per_drone: int = M_MAX_TRIPS,
        seed: Optional[int] = None,
        depot_infos: List[DepotInfo] = None,
        drones_per_depot: int = DRONES_PER_DEPOT,
        batteries_per_depot: int = BATTERIES_PER_DEPOT
    ):
        """
        初始化多仓库仿真器
        
        Args:
            time_horizon: 仿真时长（分钟）
            decision_interval: 决策周期间隔（分钟）
            buffer_threshold: 行程取消缓冲阈值（分钟）
            max_trips_per_drone: 每架无人机每周期最大行程数(M参数)
            seed: 随机种子
            depot_infos: 仓库配置列表
            drones_per_depot: 每个仓库的无人机数量
            batteries_per_depot: 每个仓库的电池数量
        """
        self.time_horizon = time_horizon
        self.decision_interval = decision_interval
        self.buffer_threshold = buffer_threshold
        self.max_trips_per_drone = max_trips_per_drone
        self.seed = seed
        
        # 多仓库配置
        self.depot_infos = depot_infos if depot_infos is not None else DEFAULT_DEPOT_INFOS
        self.drones_per_depot = drones_per_depot
        self.batteries_per_depot = batteries_per_depot
        
        # 初始化系统组件
        self._initialize_system()
    
    def _initialize_system(self):
        """初始化所有系统组件，为每个仓库创建独立的无人机、电池和算法组件"""
        # 多仓库列表
        self.depots: List[Depot] = []
        self.drones: List[Drone] = []
        self.trip_builders: Dict[int, TripBuilder] = {}
        self.battery_managers: Dict[int, BatteryManager] = {}
        
        # 全局组件
        self.energy_model = EnergyModel()
        self.trip_selector = TripSelector(
            len(self.depot_infos) * self.drones_per_depot, 
            self.max_trips_per_drone
        )
        self.trip_assigner = TripAssigner()
        self.order_generator = OrderGenerator(seed=self.seed, depot_infos=self.depot_infos)
        
        drone_id_counter = 0
        battery_id_counter = 0
        
        # 为每个仓库创建资源
        for depot_info in self.depot_infos:
            # 创建仓库
            depot = Depot(
                location=depot_info.location.copy(),
                available_batteries=[],
                charging_batteries=[],
                num_charging_stations=depot_info.num_charging_stations,
                id=depot_info.id,
                name=depot_info.name
            )
            
            # 创建该仓库的电池
            for _ in range(self.batteries_per_depot):
                battery = Battery(
                    id=battery_id_counter,
                    current_charge=E_MAX,
                    max_charge=E_MAX,
                    cycle_count=0,
                    status=BatteryStatus.AVAILABLE
                )
                depot.available_batteries.append(battery)
                battery_id_counter += 1
            
            # 创建该仓库的无人机
            for _ in range(self.drones_per_depot):
                # 为无人机分配电池
                battery = depot.available_batteries.pop(0) if depot.available_batteries else None
                if battery:
                    battery.status = BatteryStatus.IN_USE
                
                drone = Drone(
                    id=drone_id_counter,
                    location=depot_info.location.copy(),
                    status=DroneStatus.IDLE,
                    battery=battery,
                    weight=DRONE_WEIGHT,
                    max_payload=MAX_PAYLOAD,
                    available_time=0.0,
                    home_depot_id=depot_info.id,
                    home_depot_location=depot_info.location.copy()
                )
                self.drones.append(drone)
                depot.drones.append(drone)
                drone_id_counter += 1
            
            self.depots.append(depot)
            
            # 为该仓库创建TripBuilder
            self.trip_builders[depot_info.id] = TripBuilder(
                depot, self.energy_model,
                depot_id=depot_info.id,
                depot_location=depot_info.location
            )
            
            # 为该仓库创建BatteryManager
            self.battery_managers[depot_info.id] = BatteryManager(depot)
        
        # 状态跟踪
        self.current_time = 0.0
        self.pending_orders: List[Order] = []
        self.served_orders: List[Order] = []
        self.active_trips: List[Trip] = []
        self.executing_trips: List[Trip] = []
        self.completed_trips: List[Trip] = []
    
    def reset(self):
        """重置仿真到初始状态"""
        self._initialize_system()
    
    def run(
        self,
        initial_orders: Optional[List[Order]] = None,
        dynamic_orders: bool = True,
        verbose: bool = False
    ) -> SimulationResult:
        """
        运行完整仿真 (Algorithm 1: MDP主循环)
        
        Args:
            initial_orders: 初始订单列表
            dynamic_orders: 是否动态生成订单
            verbose: 是否输出详细信息
            
        Returns:
            SimulationResult 仿真结果
        """
        result = SimulationResult()
        
        # 初始化订单
        if initial_orders:
            self.pending_orders.extend(initial_orders)
            result.total_orders += len(initial_orders)
        
        total_drones = len(self.depot_infos) * self.drones_per_depot
        total_batteries = len(self.depot_infos) * self.batteries_per_depot
        
        if verbose:
            print(f"开始仿真，初始订单: {len(self.pending_orders)}")
            print(f"仿真时长: {self.time_horizon} 分钟")
            print(f"决策周期: {self.decision_interval} 分钟")
            print(f"仓库数: {len(self.depot_infos)}, 无人机: {total_drones}, 电池: {total_batteries}")
            print("-" * 50)
        
        # Main simulation loop (Algorithm 1)
        # Line 2: k ← 0, c ← 0
        decision_epoch = 0
        total_cost = 0.0
        
        # Line 3: while t_k < T do
        while self.current_time < self.time_horizon:
            decision_epoch += 1
            
            if verbose:
                print(f"\n=== Decision Epoch {decision_epoch} (t = {self.current_time:.1f}) ===")
                print(f"Pending orders: {len(self.pending_orders)}")
                print(f"Active trips: {len(self.active_trips)}")
                print(f"Executing trips: {len(self.executing_trips)}")
            
            # Line 4: Deterministic transition - call Algorithm 2 (CFA policy)
            # s_k^x ← (t_k, D_k^x, ∅, Δ_k^x, B_k^x), c(s_k, x) ← process_decision_state(s_k, v)
            decision_info, epoch_cost = self._execute_cfa_policy(verbose)
            result.decision_log.append(decision_info)
            
            # Line 5: c ← c + c(s_k, x) - update total cost
            total_cost += epoch_cost
            
            # Lines 6-10: Stochastic transition to pre-decision state s_{k+1}
            # Line 6: U_{k+1} = {r_new = (i_new, q_new, l_new)} - newly revealed requests
            # Line 7: D_{k+1} ← D_k^x - update outstanding customers set
            # Line 8: Δ_{k+1} ← Δ_k^x - update drones' states  
            # Line 9: B_{k+1} ← B_k^x - update batteries' states
            # Line 10: t_{k+1} ← t_k + Ψ - update next time period
            new_orders = self._advance_time(dynamic_orders)
            result.total_orders += len(new_orders)
            
            if verbose and new_orders:
                print(f"New orders arrived: {len(new_orders)}")
            
            # Line 11: k ← k + 1 - advance to next epoch
        
        # Line 13: return c
        result.total_cost = total_cost
        
        # Finalize results
        result.served_orders = len(self.served_orders)
        result.total_trips = len(self.completed_trips)
        result.total_distance = sum(t.total_distance for t in self.completed_trips)
        result.total_energy = sum(t.expected_energy for t in self.completed_trips)
        
        # 保存行程和订单数据用于可视化
        result.trips = list(self.completed_trips)
        result.orders = list(self.served_orders) + list(self.pending_orders)
        
        # Calculate lateness
        total_lateness = 0.0
        max_lateness = 0.0
        lateness_list = []
        
        for order in self.served_orders:
            if hasattr(order, 'actual_delivery_time') and order.actual_delivery_time:
                lateness = max(0, order.actual_delivery_time - order.deadline)
                total_lateness += lateness
                max_lateness = max(max_lateness, lateness)
                lateness_list.append(lateness)
        
        result.total_lateness = total_lateness
        result.average_lateness = total_lateness / max(1, result.served_orders)
        result.max_lateness = max_lateness
        result.lateness_distribution = lateness_list
        result.service_rate = result.served_orders / max(1, result.total_orders)
        
        if verbose:
            print("\n" + "=" * 50)
            print("SIMULATION COMPLETE")
            print(f"Total orders: {result.total_orders}")
            print(f"Served orders: {result.served_orders} ({result.service_rate:.1%})")
            print(f"Total trips: {result.total_trips}")
            print(f"Total distance: {result.total_distance:.1f} m")
            print(f"Total energy: {result.total_energy:.1f} Wh")
            print(f"Average lateness: {result.average_lateness:.1f} min")
            print(f"Max lateness: {result.max_lateness:.1f} min")
        
        return result
    
    def _execute_cfa_policy(self, verbose: bool = False) -> Tuple[Dict, float]:
        """
        执行CFA策略 (Algorithm 2: process_decision_state)
        每个仓库独立执行CFA策略
        
        输入: 决策前状态 s_k = (t_k, D_k, U_k, Δ_k, B_k); 阈值 υ
        输出: 决策后状态 s_k^x; 成本 c(s_k, x)
        
        Returns:
            (decision_info字典, epoch_cost)元组
        """
        decision_info = {
            "time": self.current_time,
            "pending_orders": len(self.pending_orders),
            "trips_built": 0,
            "trips_selected": 0,
            "trips_assigned": 0,
            "epoch_cost": 0.0,
            "depot_stats": {}
        }
        
        # Line 1: τ(s_k, x) ← 0, ψ(s_k, x) ← 0
        tau_cost = 0.0  # 总距离
        psi_cost = 0.0  # 总延迟
        
        # 按仓库分组订单
        orders_by_depot = self.order_generator.get_orders_by_depot(
            [o for o in self.pending_orders if not o.is_served and o.assigned_trip_id is None]
        )
        
        all_new_trips = []
        all_selected_trips = []
        
        # 为每个仓库执行CFA策略
        for depot in self.depots:
            depot_id = depot.id
            depot_orders = orders_by_depot.get(depot_id, [])
            
            if not depot_orders:
                continue
            
            trip_builder = self.trip_builders.get(depot_id)
            if trip_builder is None:
                continue
            
            # 为该仓库构建trips
            depot_trips = trip_builder.build_trips(
                depot_orders,
                self.current_time,
                insertion_strategy="urgency"
            )
            all_new_trips.extend(depot_trips)
            
            decision_info["depot_stats"][depot_id] = {
                "orders": len(depot_orders),
                "trips_built": len(depot_trips)
            }
            
            if verbose:
                print(f"  仓库 {depot_id} ({depot.name}): {len(depot_orders)} 订单, {len(depot_trips)} 行程")
        
        decision_info["trips_built"] = len(all_new_trips)
        
        # 选择和分配trips（考虑仓库约束）
        all_candidate_trips = all_new_trips + [t for t in self.active_trips if not t.is_executed]
        
        if all_candidate_trips:
            # 按仓库分组可用无人机
            for depot in self.depots:
                depot_id = depot.id
                # 获取该仓库的trips和无人机
                depot_trips = [t for t in all_candidate_trips if t.depot_id == depot_id]
                depot_drones = [d for d in self.drones 
                               if d.home_depot_id == depot_id and d.is_available(self.current_time)]
                depot_orders = orders_by_depot.get(depot_id, [])
                
                if not depot_trips or not depot_drones:
                    continue
                
                # 为该仓库选择trips
                selected, assignments = self.trip_selector.select_trips(
                    depot_trips,
                    depot_orders,
                    depot_drones,
                    self.current_time
                )
                all_selected_trips.extend(selected)
                
                # 分配trips到无人机
                if selected:
                    trip_assignments = self.trip_assigner.assign_trips(
                        selected,
                        depot_drones,
                        self.current_time,
                        assignments
                    )
                    
                    for drone_id, trip_schedule in trip_assignments.items():
                        for trip, start_time in trip_schedule:
                            self._dispatch_trip(drone_id, trip, start_time)
                            decision_info["trips_assigned"] += 1
            
            decision_info["trips_selected"] = len(all_selected_trips)
            
            if verbose:
                print(f"选择 {len(all_selected_trips)} 行程, 分配 {decision_info['trips_assigned']} 行程")
        
        # Execute drone state machine for Ψ time units
        for drone in self.drones:
            tau_drone, psi_drone = self._execute_drone_state_machine(drone)
            tau_cost += tau_drone
            psi_cost += psi_drone
        
        # Cancel stale trips
        self._cleanup_stale_trips()
        
        # Update battery charging for all depots
        for depot_id, battery_manager in self.battery_managers.items():
            battery_manager.update_charging(self.current_time)
        
        # Calculate epoch cost
        epoch_cost = DISTANCE_COST_RATE * tau_cost + LATE_PENALTY_RATE * psi_cost
        decision_info["epoch_cost"] = epoch_cost
        decision_info["tau"] = tau_cost
        decision_info["psi"] = psi_cost
        
        return decision_info, epoch_cost
    
    def _execute_drone_state_machine(self, drone: Drone) -> Tuple[float, float]:
        """
        Execute drone state machine for Ψ time units (Algorithm 2, Lines 6-7)
        
        Updates drone state, battery state, and calculates costs.
        
        Args:
            drone: Drone to execute state machine for
            
        Returns:
            Tuple of (distance_cost, lateness_cost) for this drone
        """
        tau = 0.0  # Distance traveled
        psi = 0.0  # Lateness incurred
        
        # Check if drone has a current trip being executed
        if drone.current_trip and drone.current_trip.planned_start_time:
            trip = drone.current_trip
            
            # Calculate distance contribution
            tau += trip.total_distance
            
            # Calculate lateness for each order in trip
            psi += trip.max_lateness(trip.planned_start_time, DRONE_SPEED_MEAN)
            
            # Update battery usage (ω^k_β tracking)
            if drone.battery:
                # Energy consumed during this epoch
                pass  # Already handled in _dispatch_trip
        
        return tau, psi
    
    def _cleanup_stale_trips(self):
        """
        Cancel trips that are planned too far in the future
        
        If a trip's start time is > current_time + PSI + UPSILON,
        cancel it and return orders to pending pool.
        """
        threshold = self.current_time + self.decision_interval + self.buffer_threshold
        
        still_active = []
        for trip in self.active_trips:
            if trip.planned_start_time and trip.planned_start_time > threshold:
                # Cancel this trip
                trip.is_cancelled = True
                for order in trip.orders:
                    order.assigned_trip_id = None
            else:
                still_active.append(trip)
        
        self.active_trips = still_active
    
    def _dispatch_trip(self, drone_id: int, trip: Trip, start_time: float):
        """
        Dispatch a trip to a drone
        多仓库场景：使用无人机所属仓库的BatteryManager
        
        Updates drone state and marks orders as assigned.
        """
        drone = next((d for d in self.drones if d.id == drone_id), None)
        if drone is None:
            return
        
        # 获取无人机所属仓库的BatteryManager
        depot_id = drone.home_depot_id
        battery_manager = self.battery_managers.get(depot_id)
        
        # Check battery
        if drone.battery is None or not battery_manager.can_complete_trip(
            drone.battery, trip.expected_energy, trip.energy_std
        ):
            # Need battery swap
            urgent_ratio = len([o for o in self.pending_orders if o.is_urgent(self.current_time)]) / max(1, len(self.pending_orders))
            success, ready_time = battery_manager.swap_battery(drone, self.current_time, urgent_ratio)
            
            if not success:
                # Cannot dispatch, return orders to pool
                for order in trip.orders:
                    order.assigned_trip_id = None
                return
            
            start_time = max(start_time, ready_time)
        
        # Update drone state
        drone.status = DroneStatus.EN_ROUTE
        drone.current_trip = trip
        
        # Calculate completion time
        duration = trip.expected_duration(DRONE_SPEED_MEAN)
        completion_time = start_time + duration
        drone.available_time = completion_time
        
        # Update trip
        trip.drone_id = drone_id
        trip.planned_start_time = start_time
        trip.is_executed = True
        
        # Mark orders as assigned
        for order in trip.orders:
            order.assigned_trip_id = trip.id
        
        # Move to executing trips
        self.executing_trips.append(trip)
        
        # Simulate energy consumption (simplified)
        if drone.battery:
            # Use expected energy (actual would be stochastic)
            drone.battery.current_charge -= trip.expected_energy
            drone.battery.current_charge = max(E_MIN, drone.battery.current_charge)
    
    def _advance_time(self, dynamic_orders: bool) -> List[Order]:
        """
        Advance simulation time and observe new state
        
        Args:
            dynamic_orders: Whether to generate new orders
            
        Returns:
            List of newly arrived orders
        """
        old_time = self.current_time
        new_time = min(self.current_time + self.decision_interval, self.time_horizon)
        
        # Generate new orders
        new_orders = []
        if dynamic_orders:
            new_orders = self.order_generator.generate_orders_in_interval(old_time, new_time)
            self.pending_orders.extend(new_orders)
        
        # Update executing trips
        completed = []
        still_executing = []
        
        for trip in self.executing_trips:
            if trip.planned_start_time is None:
                still_executing.append(trip)
                continue
            
            completion_time = trip.planned_start_time + trip.expected_duration(DRONE_SPEED_MEAN)
            
            if completion_time <= new_time:
                # Trip completed
                completed.append(trip)
                
                # Mark orders as served
                for order in trip.orders:
                    order.is_served = True
                    order.actual_delivery_time = completion_time
                    self.served_orders.append(order)
                    
                    # Remove from pending
                    if order in self.pending_orders:
                        self.pending_orders.remove(order)
                
                # Update drone
                drone = next((d for d in self.drones if d.id == trip.drone_id), None)
                if drone:
                    drone.status = DroneStatus.IDLE
                    # 无人机返回其所属仓库
                    drone.location = drone.home_depot_location.copy()
                    drone.current_trip = None
                    drone.total_distance += trip.total_distance
                    
                    # 获取无人机所属仓库的BatteryManager
                    depot_id = drone.home_depot_id
                    battery_manager = self.battery_managers.get(depot_id)
                    
                    # Return battery for charging if low
                    if drone.battery and drone.battery.current_charge < E_MAX * 0.3:
                        battery_manager.return_battery(
                            drone.battery,
                            new_time,
                            drone.battery.current_charge
                        )
                        # Get new battery
                        urgent_ratio = len([o for o in self.pending_orders if o.is_urgent(new_time)]) / max(1, len(self.pending_orders))
                        new_battery = battery_manager.get_available_battery(urgent_ratio)
                        drone.battery = new_battery
            else:
                still_executing.append(trip)
        
        self.executing_trips = still_executing
        self.completed_trips.extend(completed)
        
        # 更新时间
        self.current_time = new_time
        
        # 更新所有仓库的电池充电状态
        for battery_manager in self.battery_managers.values():
            battery_manager.update_charging(new_time)
        
        return new_orders
    
    def get_state(self) -> State:
        """获取当前仿真状态"""
        return State(
            time=self.current_time,
            pending_orders=self.pending_orders.copy(),
            active_trips=self.active_trips.copy(),
            executing_trips=self.executing_trips.copy(),
            drones=self.drones.copy(),
            depot=self.depots[0] if self.depots else None
        )
