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
    T_HORIZON, PSI, UPSILON, NUM_DRONES, NUM_BATTERIES,
    DRONE_WEIGHT, MAX_PAYLOAD, E_MAX, E_MIN, CHARGE_RATE,
    DRONE_SPEED_MEAN, M_MAX_TRIPS, DEPOT_LOCATION,
    SERVICE_TIME, LATE_PENALTY_RATE, DISTANCE_COST_RATE
)


@dataclass
class SimulationResult:
    """Results from a simulation run"""
    total_orders: int = 0
    served_orders: int = 0
    total_distance: float = 0.0
    total_lateness: float = 0.0
    total_cost: float = 0.0  # Total cost c (Algorithm 1, line 5)
    total_trips: int = 0
    failed_trips: int = 0  # Trips that failed due to energy
    average_lateness: float = 0.0
    service_rate: float = 0.0
    
    # Detailed logs
    order_log: List[Dict] = field(default_factory=list)
    trip_log: List[Dict] = field(default_factory=list)
    decision_log: List[Dict] = field(default_factory=list)


class Simulator:
    """
    Main simulation engine for DRPUDEC
    
    Implements:
    - Algorithm 1: MDP main loop
    - Algorithm 2: CFA policy
    
    The simulation proceeds in discrete time steps of PSI minutes.
    At each decision epoch:
    1. Execute CFA policy to generate decisions
    2. Advance time and observe new orders
    3. Update system state
    """
    
    def __init__(
        self,
        num_drones: int = NUM_DRONES,
        num_batteries: int = NUM_BATTERIES,
        time_horizon: float = T_HORIZON,
        decision_interval: float = PSI,
        buffer_threshold: float = UPSILON,
        max_trips_per_drone: int = M_MAX_TRIPS,
        seed: Optional[int] = None
    ):
        self.num_drones = num_drones
        self.num_batteries = num_batteries
        self.time_horizon = time_horizon
        self.decision_interval = decision_interval
        self.buffer_threshold = buffer_threshold
        self.max_trips_per_drone = max_trips_per_drone
        self.seed = seed
        
        # Initialize components
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize all system components"""
        # Create depot
        self.depot = Depot(
            location=DEPOT_LOCATION.copy(),
            available_batteries=[],
            charging_batteries=[]
        )
        
        # Create batteries
        for i in range(self.num_batteries):
            battery = Battery(
                id=i,
                current_charge=E_MAX,
                max_charge=E_MAX,
                cycle_count=0,
                status=BatteryStatus.AVAILABLE
            )
            self.depot.available_batteries.append(battery)
        
        # Create drones
        self.drones: List[Drone] = []
        for i in range(self.num_drones):
            # Assign a battery to each drone
            battery = self.depot.available_batteries.pop(0) if self.depot.available_batteries else None
            if battery:
                battery.status = BatteryStatus.IN_USE
            
            drone = Drone(
                id=i,
                location=DEPOT_LOCATION.copy(),
                status=DroneStatus.IDLE,
                battery=battery,
                weight=DRONE_WEIGHT,
                max_payload=MAX_PAYLOAD,
                available_time=0.0
            )
            self.drones.append(drone)
        
        # Initialize algorithm components
        self.energy_model = EnergyModel()
        self.trip_builder = TripBuilder(self.depot, self.energy_model)
        self.trip_selector = TripSelector(self.num_drones, self.max_trips_per_drone)
        self.trip_assigner = TripAssigner()
        self.battery_manager = BatteryManager(self.depot)
        self.order_generator = OrderGenerator(seed=self.seed)
        
        # State tracking
        self.current_time = 0.0
        self.pending_orders: List[Order] = []
        self.served_orders: List[Order] = []
        self.active_trips: List[Trip] = []
        self.executing_trips: List[Trip] = []
        self.completed_trips: List[Trip] = []
    
    def reset(self):
        """Reset simulation to initial state"""
        self._initialize_system()
    
    def run(
        self,
        initial_orders: Optional[List[Order]] = None,
        dynamic_orders: bool = True,
        verbose: bool = False
    ) -> SimulationResult:
        """
        Run the full simulation (Algorithm 1: MDP Main Loop)
        
        Args:
            initial_orders: Optional list of orders at time 0
            dynamic_orders: Whether to generate dynamic orders
            verbose: Print progress information
            
        Returns:
            SimulationResult with statistics
        """
        result = SimulationResult()
        
        # Initialize with any initial orders
        if initial_orders:
            self.pending_orders.extend(initial_orders)
            result.total_orders += len(initial_orders)
        
        if verbose:
            print(f"Starting simulation with {len(self.pending_orders)} initial orders")
            print(f"Time horizon: {self.time_horizon} minutes")
            print(f"Decision interval: {self.decision_interval} minutes")
            print(f"Drones: {self.num_drones}, Batteries: {self.num_batteries}")
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
        
        # Calculate lateness
        total_lateness = 0.0
        for order in self.served_orders:
            if hasattr(order, 'actual_delivery_time') and order.actual_delivery_time:
                lateness = max(0, order.actual_delivery_time - order.deadline)
                total_lateness += lateness
        
        result.total_lateness = total_lateness
        result.average_lateness = total_lateness / max(1, result.served_orders)
        result.service_rate = result.served_orders / max(1, result.total_orders)
        
        if verbose:
            print("\n" + "=" * 50)
            print("SIMULATION COMPLETE")
            print(f"Total orders: {result.total_orders}")
            print(f"Served orders: {result.served_orders} ({result.service_rate:.1%})")
            print(f"Total trips: {result.total_trips}")
            print(f"Total distance: {result.total_distance:.1f} m")
            print(f"Average lateness: {result.average_lateness:.1f} min")
        
        return result
    
    def _execute_cfa_policy(self, verbose: bool = False) -> Tuple[Dict, float]:
        """
        Execute CFA policy (Algorithm 2: process_decision_state)
        
        Input: Pre-decision state s_k = (t_k, D_k, U_k, Δ_k, B_k); threshold υ
        Output: Post-decision state s_k^x; cost c(s_k, x)
        
        Returns:
            Tuple of (decision_info dict, epoch_cost)
        """
        decision_info = {
            "time": self.current_time,
            "pending_orders": len(self.pending_orders),
            "trips_built": 0,
            "trips_selected": 0,
            "trips_assigned": 0,
            "epoch_cost": 0.0
        }
        
        # Line 1: τ(s_k, x) ← 0, ψ(s_k, x) ← 0
        tau_cost = 0.0  # Total distance
        psi_cost = 0.0  # Total lateness
        
        # Get all pending orders: D_k ∪ U_k
        all_pending = [o for o in self.pending_orders if not o.is_served]
        new_orders = [o for o in all_pending if o.assigned_trip_id is None]
        
        # Line 2: (D'_k, U'_k) ← update_trips(D_k ∪ U_k, υ) - Algorithm 5
        # Update existing scheduled trips with new orders
        if self.active_trips and new_orders:
            self.active_trips, remaining_orders = self.trip_builder.update_trips(
                self.active_trips,
                new_orders,
                self.current_time
            )
            new_orders = remaining_orders  # Orders not inserted into existing trips
        
        if verbose:
            print(f"After update_trips: {len(new_orders)} orders still unassigned")
        
        # Line 3: Γ_k ← build_trips(D'_k ∪ U'_k) - Algorithm 3
        # Build new candidate trips from remaining unassigned orders
        unassigned_orders = [o for o in self.pending_orders 
                           if not o.is_served and o.assigned_trip_id is None]
        
        new_trips = []
        if unassigned_orders:
            new_trips = self.trip_builder.build_trips(
                unassigned_orders,
                self.current_time,
                insertion_strategy="urgency"
            )
            decision_info["trips_built"] = len(new_trips)
            
            if verbose:
                print(f"Built {len(new_trips)} candidate trips")
        
        # Line 4: Γ*_k ← select_trips(D'_k ∪ U'_k, Γ_k, |Δ|, M) - Algorithm 6
        # Select best trips using SPBP + DTSSP
        all_candidate_trips = new_trips + [t for t in self.active_trips if not t.is_executed]
        
        selected_trips = []
        drone_assignments = {}
        
        if all_candidate_trips:
            available_drones = [d for d in self.drones if d.is_available(self.current_time)]
            
            selected_trips, drone_assignments = self.trip_selector.select_trips(
                all_candidate_trips,
                unassigned_orders,
                available_drones,
                self.current_time
            )
            
            decision_info["trips_selected"] = len(selected_trips)
            
            if verbose:
                print(f"Selected {len(selected_trips)} trips")
        
        # Line 5: {θ^k_δ : δ ∈ Δ_k} ← assign_trips(Γ*_k, Δ_k) - Algorithm 7
        # Assign selected trips to drones
        if selected_trips:
            available_drones = [d for d in self.drones if d.is_available(self.current_time)]
            assignments = self.trip_assigner.assign_trips(
                selected_trips,
                available_drones,
                self.current_time,
                drone_assignments
            )
            
            # Execute assignments and dispatch trips
            for drone_id, trip_schedule in assignments.items():
                for trip, start_time in trip_schedule:
                    self._dispatch_trip(drone_id, trip, start_time)
                    decision_info["trips_assigned"] += 1
            
            if verbose:
                print(f"Assigned {decision_info['trips_assigned']} trips to drones")
        
        # Lines 6-8: Execute drone state machine for Ψ time units
        # For each δ ∈ Δ_k: execute_drone_state_machine(δ, θ^k_δ, B^x_k, τ, ψ)
        for drone in self.drones:
            tau_drone, psi_drone = self._execute_drone_state_machine(drone)
            tau_cost += tau_drone
            psi_cost += psi_drone
        
        # Line 8: D^x_k ← D^x_k ∪ cancel_scheduled_trips(δ)
        # Cancel trips starting after [t_k, t_k + υ] and return customers to D^x_k
        self._cleanup_stale_trips()
        
        # Update battery charging
        self.battery_manager.update_charging(self.current_time)
        
        # Line 10: return s^x_k, μ^c · τ(s_k, x) + μ^l · ψ(s_k, x)
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
        
        Updates drone state and marks orders as assigned.
        """
        drone = next((d for d in self.drones if d.id == drone_id), None)
        if drone is None:
            return
        
        # Check battery
        if drone.battery is None or not self.battery_manager.can_complete_trip(
            drone.battery, trip.expected_energy, trip.energy_std
        ):
            # Need battery swap
            urgent_ratio = len([o for o in self.pending_orders if o.is_urgent(self.current_time)]) / max(1, len(self.pending_orders))
            success, ready_time = self.battery_manager.swap_battery(drone, self.current_time, urgent_ratio)
            
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
                    drone.location = DEPOT_LOCATION.copy()
                    drone.current_trip = None
                    drone.total_distance += trip.total_distance
                    
                    # Return battery for charging if low
                    if drone.battery and drone.battery.current_charge < E_MAX * 0.3:
                        self.battery_manager.return_battery(
                            drone.battery,
                            new_time,
                            drone.battery.current_charge
                        )
                        # Get new battery
                        urgent_ratio = len([o for o in self.pending_orders if o.is_urgent(new_time)]) / max(1, len(self.pending_orders))
                        new_battery = self.battery_manager.get_available_battery(urgent_ratio)
                        drone.battery = new_battery
            else:
                still_executing.append(trip)
        
        self.executing_trips = still_executing
        self.completed_trips.extend(completed)
        
        # Update time
        self.current_time = new_time
        
        # Update battery charging
        self.battery_manager.update_charging(new_time)
        
        return new_orders
    
    def get_state(self) -> State:
        """Get current simulation state"""
        return State(
            time=self.current_time,
            pending_orders=self.pending_orders.copy(),
            active_trips=self.active_trips.copy(),
            executing_trips=self.executing_trips.copy(),
            drones=self.drones.copy(),
            depot=self.depot
        )
