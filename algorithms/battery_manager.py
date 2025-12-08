"""
Battery Management Module
Implements balanced aging policy and battery reservation strategy

Key features:
- Prioritize batteries with fewer cycles (balanced aging)
- Reserve 10% of batteries for emergencies
- Track charging status and availability
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
    """Event for battery state change"""
    time: float
    battery_id: int
    event_type: str  # "available", "depleted"
    
    def __lt__(self, other):
        return self.time < other.time


class BatteryManager:
    """
    Manages battery allocation, charging, and swapping
    
    Implements:
    - Balanced aging: prioritize least-used batteries
    - Reserve policy: keep 10% for emergencies
    - Charging queue management
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
        
        # Priority queue for available batteries (by cycle count)
        # Format: (cycle_count, battery_id, battery)
        self._available_heap: List[Tuple[int, int, Battery]] = []
        
        # Initialize heap with available batteries
        for battery in depot.available_batteries:
            heappush(self._available_heap, (battery.cycle_count, battery.id, battery))
        
        # Track reserved batteries
        self._reserved_batteries: List[Battery] = []
        self._update_reserves()
    
    def _update_reserves(self):
        """Update reserved battery pool based on reserve ratio"""
        total_batteries = len(self.depot.available_batteries) + len(self.depot.charging_batteries)
        reserve_count = max(1, int(total_batteries * self.reserve_ratio))
        
        # Move batteries to/from reserve as needed
        current_reserve = len(self._reserved_batteries)
        
        if current_reserve < reserve_count and self._available_heap:
            # Need more reserves - take from available (highest cycle count)
            # Rebuild heap to get highest cycle count
            all_available = []
            while self._available_heap:
                item = heappop(self._available_heap)
                all_available.append(item)
            
            # Sort by cycle count descending
            all_available.sort(key=lambda x: x[0], reverse=True)
            
            # Move some to reserve
            while len(self._reserved_batteries) < reserve_count and all_available:
                _, _, battery = all_available.pop(0)
                self._reserved_batteries.append(battery)
            
            # Put rest back in heap
            for item in all_available:
                heappush(self._available_heap, item)
    
    def get_available_battery(
        self,
        urgent_ratio: float = 0.0
    ) -> Optional[Battery]:
        """
        Get the best available battery for a drone
        
        Uses balanced aging: returns battery with fewest cycles.
        If urgent_ratio > urgent_threshold, may use reserve batteries.
        
        Args:
            urgent_ratio: Ratio of urgent customers (0 to 1)
            
        Returns:
            Battery if available, None otherwise
        """
        # First try regular available batteries
        if self._available_heap:
            _, _, battery = heappop(self._available_heap)
            battery.status = BatteryStatus.IN_USE
            return battery
        
        # If urgent enough, use reserve batteries
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
        Return a used battery to depot for charging
        
        Args:
            battery: Battery being returned
            current_time: Current simulation time
            remaining_charge: Remaining charge level
        """
        battery.current_charge = remaining_charge
        battery.status = BatteryStatus.CHARGING
        battery.charge_start_time = current_time
        battery.cycle_count += 1
        
        self.depot.charging_batteries.append(battery)
    
    def update_charging(self, current_time: float):
        """
        Update all charging batteries and move completed ones to available
        """
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
        
        # Update reserves
        self._update_reserves()
    
    def get_time_to_next_available(self, current_time: float) -> float:
        """
        Get time until next battery becomes available
        
        Returns:
            Time in minutes, or float('inf') if no batteries charging
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
        Get summary of battery status
        
        Returns:
            Dict with counts of available, charging, in_use, reserved
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
        Swap drone's battery with a fresh one
        
        Args:
            drone: Drone needing battery swap
            current_time: Current simulation time
            urgent_ratio: Ratio of urgent customers
            
        Returns:
            Tuple of (success, time_when_ready)
        """
        # Return old battery
        if drone.battery is not None:
            remaining = drone.battery.current_charge
            self.return_battery(drone.battery, current_time, remaining)
        
        # Get new battery
        new_battery = self.get_available_battery(urgent_ratio)
        
        if new_battery is None:
            # No battery available, need to wait
            wait_time = self.get_time_to_next_available(current_time)
            drone.battery = None
            return False, current_time + wait_time + self.swap_time
        
        # Swap successful
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
        Estimate required battery charge for a trip
        
        Uses chance constraint: E[energy] + Φ^{-1}(α) * std
        
        Args:
            expected_energy: Expected energy consumption
            energy_std: Standard deviation of energy
            alpha_quantile: Quantile for safety probability
            
        Returns:
            Required battery charge level
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
        Check if battery has enough charge to complete a trip
        
        Args:
            battery: Battery to check
            expected_energy: Expected trip energy consumption
            energy_std: Standard deviation of energy
            
        Returns:
            True if battery can complete trip with safety margin
        """
        required = self.estimate_energy_for_trip(expected_energy, energy_std)
        return battery.current_charge >= required
