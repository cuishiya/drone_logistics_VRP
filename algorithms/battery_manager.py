"""
电池管理模块
实现电池均衡老化策略和电池预留策略

主要特性：
- 优先使用循环次数较少的电池（均衡老化）
- 预留10%的电池用于紧急情况
- 跟踪电池充电状态和可用性
"""

from typing import List, Optional, Tuple
from heapq import heappush, heappop
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.data_structures import Battery, BatteryStatus, Drone, Depot
from config import (
    E_MAX, E_MIN, CHARGE_RATE, BATTERY_SWAP_TIME,
    BATTERY_RESERVE_RATIO, URGENT_CUSTOMER_THRESHOLD
)


@dataclass
class BatteryEvent:
    """电池状态变化事件"""
    time: float
    battery_id: int
    event_type: str  # "available" 表示可用, "depleted" 表示耗尽
    
    def __lt__(self, other):
        return self.time < other.time


class BatteryManager:
    """
    管理电池分配、充电和更换
    
    实现内容：
    - 均衡老化：优先使用循环次数最少的电池
    - 预留策略：预留10%电池用于紧急情况
    - 充电队列管理
    """
    
    def __init__(
        self,
        depot: Depot,
        e_max: float = E_MAX,
        e_min: float = E_MIN,
        charge_rate: float = CHARGE_RATE,
        swap_time: float = BATTERY_SWAP_TIME,
        reserve_ratio: float = BATTERY_RESERVE_RATIO,
        urgent_threshold: float = URGENT_CUSTOMER_THRESHOLD
    ):
        self.depot = depot
        self.e_max = e_max
        self.e_min = e_min
        self.charge_rate = charge_rate
        self.swap_time = swap_time
        self.reserve_ratio = reserve_ratio
        self.urgent_threshold = urgent_threshold
        
        # 可用电池的优先队列（按循环次数排序）
        # 格式: (cycle_count, battery_id, battery)
        self._available_heap: List[Tuple[int, int, Battery]] = []
        
        # 使用当前可用电池初始化堆
        for battery in depot.available_batteries:
            heappush(self._available_heap, (battery.cycle_count, battery.id, battery))
        
        # 记录预留电池
        self._reserved_batteries: List[Battery] = []
        self._update_reserves()
    
    def _update_reserves(self):
        """根据预留比例更新预留电池池"""
        total_batteries = len(self.depot.available_batteries) + len(self.depot.charging_batteries)
        reserve_count = max(1, int(total_batteries * self.reserve_ratio))
        
        # 根据需要在预留池和可用池之间移动电池
        current_reserve = len(self._reserved_batteries)
        
        if current_reserve < reserve_count and self._available_heap:
            # 需要更多预留电池——从可用池中取出（循环次数最高的）
            # 重新构建堆以获得循环次数最高的电池
            all_available = []
            while self._available_heap:
                item = heappop(self._available_heap)
                all_available.append(item)
            
            # 按循环次数降序排序
            all_available.sort(key=lambda x: x[0], reverse=True)
            
            # 将一部分移动到预留池
            while len(self._reserved_batteries) < reserve_count and all_available:
                _, _, battery = all_available.pop(0)
                self._reserved_batteries.append(battery)
            
            # 剩余的重新放回堆中
            for item in all_available:
                heappush(self._available_heap, item)
    
    def get_available_battery(
        self,
        urgent_ratio: float = 0.0
    ) -> Optional[Battery]:
        """
        获取一块适合无人机使用的最佳电池
        
        使用均衡老化策略：返回循环次数最少的电池。
        当 urgent_ratio > urgent_threshold 时，可以动用预留电池。
        
        Args:
            urgent_ratio: 紧急客户比例（0 到 1）
            
        Returns:
            如果有电池则返回 Battery，否则返回 None
        """
        # 优先尝试普通可用电池
        if self._available_heap:
            _, _, battery = heappop(self._available_heap)
            battery.status = BatteryStatus.IN_USE
            return battery
        
        # 如果足够紧急，使用预留电池
        if urgent_ratio >= self.urgent_threshold and self._reserved_batteries:
            battery = self._reserved_batteries.pop(0)
            battery.status = BatteryStatus.IN_USE
            return battery
        
        return None
    
    def return_battery(
        self,
        battery: Battery,
        current_time: float,
        remaining_charge: float
    ):
        """
        将使用后的电池返回仓库进行充电
        
        Args:
            battery: 归还的电池
            current_time: 当前仿真时间
            remaining_charge: 剩余电量
        """
        battery.current_charge = remaining_charge
        battery.status = BatteryStatus.CHARGING
        battery.charge_start_time = current_time
        battery.cycle_count += 1
        
        self.depot.charging_batteries.append(battery)
    
    def update_charging(self, current_time: float):
        """更新所有充电电池，并将充满的电池移动到可用池"""
        still_charging = []
        
        for battery in self.depot.charging_batteries:
            if battery.charge_start_time is None:
                battery.charge_start_time = current_time
            
            # Calculate new charge level
            elapsed = current_time - battery.charge_start_time
            new_charge = battery.current_charge + elapsed * self.charge_rate
            
            if new_charge >= self.e_max:
                # Fully charged
                battery.current_charge = self.e_max
                battery.status = BatteryStatus.AVAILABLE
                battery.charge_start_time = None
                
                # Add to available heap
                heappush(
                    self._available_heap,
                    (battery.cycle_count, battery.id, battery)
                )
                
                # Also update depot's available list
                self.depot.available_batteries.append(battery)
            else:
                battery.current_charge = new_charge
                still_charging.append(battery)
        
        self.depot.charging_batteries = still_charging
        
        # 更新预留池
        self._update_reserves()
    
    def get_time_to_next_available(self, current_time: float) -> float:
        """
        获取下一块电池变为可用状态所需的时间
        
        Returns:
            时间（分钟）；如果没有电池在充电，则返回 float('inf')
        """
        if self._available_heap:
            return 0.0
        
        if not self.depot.charging_batteries:
            return float('inf')
        
        min_time = float('inf')
        for battery in self.depot.charging_batteries:
            if battery.charge_start_time is None:
                continue
            
            remaining = self.e_max - battery.current_charge
            time_to_full = remaining / self.charge_rate
            available_at = battery.charge_start_time + time_to_full
            wait_time = max(0, available_at - current_time)
            min_time = min(min_time, wait_time)
        
        return min_time
    
    def get_battery_status(self) -> dict:
        """
        获取电池状态摘要
        
        Returns:
            字典，包含 available/charging/reserved 等数量信息
        """
        return {
            "available": len(self._available_heap),
            "charging": len(self.depot.charging_batteries),
            "reserved": len(self._reserved_batteries),
            "total": (
                len(self._available_heap) +
                len(self.depot.charging_batteries) +
                len(self._reserved_batteries)
            )
        }
    
    def swap_battery(
        self,
        drone: Drone,
        current_time: float,
        urgent_ratio: float = 0.0
    ) -> Tuple[bool, float]:
        """
        为无人机更换电池
        
        Args:
            drone: 需要更换电池的无人机
            current_time: 当前仿真时间
            urgent_ratio: 紧急客户比例
            
        Returns:
            (success, time_when_ready) 二元组
        """
        # 先归还旧电池
        if drone.battery is not None:
            remaining = drone.battery.current_charge
            self.return_battery(drone.battery, current_time, remaining)
        
        # 获取新电池
        new_battery = self.get_available_battery(urgent_ratio)
        
        if new_battery is None:
            # 没有可用电池，需要等待
            wait_time = self.get_time_to_next_available(current_time)
            drone.battery = None
            return False, current_time + wait_time + self.swap_time
        
        # 更换成功
        drone.battery = new_battery
        ready_time = current_time + self.swap_time
        
        return True, ready_time
    
    def estimate_energy_for_trip(
        self,
        expected_energy: float,
        energy_std: float,
        alpha_quantile: float = 1.2816
    ) -> float:
        """
        估计执行一个行程所需的电量
        
        使用机会约束：E[energy] + Φ^{-1}(α) * std
        
        Args:
            expected_energy: 预期能耗
            energy_std: 能耗标准差
            alpha_quantile: 安全概率对应的分位数
            
        Returns:
            所需的电量水平
        """
        required = expected_energy + alpha_quantile * energy_std + self.e_min
        return min(required, self.e_max)
    
    def can_complete_trip(
        self,
        battery: Battery,
        expected_energy: float,
        energy_std: float
    ) -> bool:
        """
        检查电池电量是否足以完成一个行程
        
        Args:
            battery: 待检查的电池
            expected_energy: 该行程的预期能耗
            energy_std: 能耗标准差
            
        Returns:
            如果在安全裕度下可以完成行程则返回 True
        """
        required = self.estimate_energy_for_trip(expected_energy, energy_std)
        return battery.current_charge >= required
