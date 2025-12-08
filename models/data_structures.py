"""
Core data structures for DRPUDEC
Defines: Order, Drone, Battery, Depot, Trip, State
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple
from enum import Enum
import numpy as np


class DroneStatus(Enum):
    """Drone operational status"""
    IDLE = "idle"  # At depot, available
    EN_ROUTE = "en_route"  # Flying to customer or returning
    SERVING = "serving"  # Serving a customer
    CHARGING = "charging"  # Waiting for battery swap/charge


class BatteryStatus(Enum):
    """Battery status"""
    IN_USE = "in_use"  # Mounted on a drone
    CHARGING = "charging"  # Being charged
    AVAILABLE = "available"  # Fully charged and available


@dataclass
class Order:
    """
    Represents a customer order/request
    
    Attributes:
        id: Unique order identifier
        location: (x, y) coordinates of customer
        weight: Package weight (kg), denoted as q_r
        arrival_time: Time when order arrived (minutes)
        deadline: Soft deadline for delivery (minutes), denoted as l_r
        service_time: Time required to serve customer (minutes), denoted as η
        is_served: Whether order has been served
        assigned_trip_id: ID of trip this order is assigned to (if any)
    """
    id: int
    location: np.ndarray
    weight: float
    arrival_time: float
    deadline: float
    service_time: float = 2.0
    is_served: bool = False
    assigned_trip_id: Optional[int] = None
    
    def urgency(self, current_time: float) -> float:
        """
        Calculate urgency score (lower = more urgent)
        Urgency = deadline - current_time
        """
        return self.deadline - current_time
    
    def is_urgent(self, current_time: float, threshold: float = 30.0) -> bool:
        """Check if order is urgent (less than threshold minutes to deadline)"""
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
    Represents a drone battery
    
    Attributes:
        id: Unique battery identifier
        current_charge: Current charge level (0 to E_max)
        max_charge: Maximum charge capacity
        cycle_count: Number of charge/discharge cycles (for aging)
        status: Current battery status
        charge_start_time: Time when charging started (if charging)
    """
    id: int
    current_charge: float
    max_charge: float
    cycle_count: int = 0
    status: BatteryStatus = BatteryStatus.AVAILABLE
    charge_start_time: Optional[float] = None
    
    def time_to_full_charge(self, charge_rate: float) -> float:
        """Calculate time needed to fully charge"""
        if self.status == BatteryStatus.AVAILABLE:
            return 0.0
        remaining = self.max_charge - self.current_charge
        return remaining / charge_rate
    
    def update_charge(self, current_time: float, charge_rate: float):
        """Update charge level based on elapsed time"""
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
    Represents a drone in the fleet
    
    Attributes:
        id: Unique drone identifier
        location: Current (x, y) coordinates
        status: Current operational status
        battery: Currently mounted battery
        weight: Drone self-weight (kg), denoted as ν
        max_payload: Maximum payload capacity (kg), denoted as Q
        current_trip: Currently executing trip (if any)
        available_time: Time when drone becomes available (for assignment)
        total_distance: Total distance traveled (for statistics)
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
    
    def remaining_energy(self) -> float:
        """Get remaining battery energy"""
        if self.battery is None:
            return 0.0
        return self.battery.current_charge
    
    def is_available(self, current_time: float) -> bool:
        """Check if drone is available for new assignment"""
        return (self.status == DroneStatus.IDLE and 
                self.available_time <= current_time)


@dataclass
class Trip:
    """
    Represents a planned trip (route) for a drone
    
    Attributes:
        id: Unique trip identifier
        orders: List of orders to serve (in visit order)
        drone_id: Assigned drone ID (if assigned)
        planned_start_time: Planned departure time from depot
        expected_energy: Expected energy consumption
        energy_std: Standard deviation of energy consumption
        total_distance: Total trip distance
        total_weight: Total payload weight
        is_executed: Whether trip has been executed
        is_cancelled: Whether trip has been cancelled
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
    
    def get_customer_ids(self) -> Set[int]:
        """Get set of customer IDs in this trip"""
        return {order.id for order in self.orders}
    
    def get_route(self, depot_location: np.ndarray) -> List[np.ndarray]:
        """Get full route including depot at start and end"""
        route = [depot_location]
        for order in self.orders:
            route.append(order.location)
        route.append(depot_location)
        return route
    
    def expected_duration(self, speed: float) -> float:
        """Calculate expected trip duration"""
        travel_time = self.total_distance / speed / 60.0  # Convert to minutes
        service_time = sum(o.service_time for o in self.orders)
        return travel_time + service_time
    
    def max_lateness(self, start_time: float, speed: float) -> float:
        """Calculate maximum lateness across all orders"""
        if not self.orders:
            return 0.0
        
        current_time = start_time
        max_late = 0.0
        
        # Simplified: assume sequential visits
        for i, order in enumerate(self.orders):
            if i == 0:
                # Travel from depot
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
    Represents the central depot/warehouse
    
    Attributes:
        location: (x, y) coordinates
        available_batteries: List of available batteries
        charging_batteries: List of batteries currently charging
        num_charging_stations: Number of charging stations
    """
    location: np.ndarray
    available_batteries: List[Battery] = field(default_factory=list)
    charging_batteries: List[Battery] = field(default_factory=list)
    num_charging_stations: int = 5
    
    def get_best_battery(self) -> Optional[Battery]:
        """
        Get the best available battery (least used, for balanced aging)
        Returns None if no battery available
        """
        if not self.available_batteries:
            return None
        # Sort by cycle count (ascending) to balance aging
        self.available_batteries.sort(key=lambda b: b.cycle_count)
        return self.available_batteries[0]
    
    def return_battery(self, battery: Battery, current_time: float):
        """Return a used battery to depot for charging"""
        battery.status = BatteryStatus.CHARGING
        battery.charge_start_time = current_time
        battery.cycle_count += 1
        self.charging_batteries.append(battery)
    
    def update_charging(self, current_time: float, charge_rate: float):
        """Update all charging batteries"""
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
    MDP State at decision epoch k
    
    Attributes:
        time: Current time t_k
        pending_orders: Orders waiting to be assigned
        active_trips: Trips that are planned but not yet started
        executing_trips: Trips currently being executed
        drones: Current state of all drones
        depot: Current depot state
    """
    time: float
    pending_orders: List[Order] = field(default_factory=list)
    active_trips: List[Trip] = field(default_factory=list)
    executing_trips: List[Trip] = field(default_factory=list)
    drones: List[Drone] = field(default_factory=list)
    depot: Optional[Depot] = None
    
    def get_unassigned_orders(self) -> List[Order]:
        """Get orders not yet assigned to any trip"""
        return [o for o in self.pending_orders if o.assigned_trip_id is None]
    
    def get_available_drones(self) -> List[Drone]:
        """Get drones available for new assignments"""
        return [d for d in self.drones if d.is_available(self.time)]
    
    def get_urgent_orders(self, threshold: float = 30.0) -> List[Order]:
        """Get orders that are urgent"""
        return [o for o in self.pending_orders 
                if not o.is_served and o.is_urgent(self.time, threshold)]
