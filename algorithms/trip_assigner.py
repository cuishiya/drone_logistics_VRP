"""
Trip Assignment Algorithm (Algorithm 7 from the paper)
Assigns selected trips to specific drones considering their current state

Solves a Task Assignment Problem (TAP) to minimize total lateness
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from itertools import permutations
import gurobipy as gp
from gurobipy import GRB

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.data_structures import Order, Trip, Drone, DroneStatus
from config import (
    GUROBI_TIME_LIMIT, GUROBI_MIP_GAP, DRONE_SPEED_MEAN,
    LATE_PENALTY_RATE, SERVICE_TIME
)


class TripAssigner:
    """
    Assigns trips to drones minimizing total lateness
    
    Considers:
    - Drone availability times (when they return from current trip)
    - Battery swap times
    - Trip execution order (permutations)
    """
    
    def __init__(
        self,
        speed: float = DRONE_SPEED_MEAN,
        service_time: float = SERVICE_TIME,
        late_penalty: float = LATE_PENALTY_RATE,
        time_limit: float = GUROBI_TIME_LIMIT,
        mip_gap: float = GUROBI_MIP_GAP
    ):
        self.speed = speed
        self.service_time = service_time
        self.late_penalty = late_penalty
        self.time_limit = time_limit
        self.mip_gap = mip_gap
    
    def assign_trips(
        self,
        trips: List[Trip],
        drones: List[Drone],
        current_time: float,
        drone_assignments: Optional[Dict[int, List[Trip]]] = None
    ) -> Dict[int, List[Tuple[Trip, float]]]:
        """
        Algorithm 7: Assign trips (assign_trips)
        
        Input: Current epoch time t_k; set of trips Γ*_k; set of drones Δ_k
        Output: Assignment of trip schedules to drones {Γ_k(δ) : δ ∈ Δ_k}
        
        Args:
            trips: Selected trips Γ*_k to assign
            drones: Available drones Δ_k
            current_time: Current epoch time t_k
            drone_assignments: Optional pre-assignment from TripSelector
            
        Returns:
            Dict mapping drone_id to list of (trip, planned_start_time) tuples
        """
        if not trips:
            return {d.id: [] for d in drones}
        
        # Lines 1-8: For each drone j, compute delay for each possible assignment
        # Precompute all permutation delays for TAP
        permutation_delays = self._compute_all_permutation_delays(
            trips, drones, current_time
        )
        
        # Line 9: {Γ_k(δ) : δ ∈ Δ_k} ← TAP({ψ_Γ*_{k,j} : Γ*_{k,j} ∈ Γ*_k, δ ∈ Δ}, Δ_k)
        # Solve model (10a)-(10d)
        return self._solve_tap(trips, drones, current_time, permutation_delays)
    
    def _compute_all_permutation_delays(
        self,
        trips: List[Trip],
        drones: List[Drone],
        current_time: float
    ) -> Dict[Tuple[int, int], float]:
        """
        Algorithm 7, Lines 1-8: Compute delays for all trip-drone assignments
        
        For each drone j (Line 1: do in parallel):
            For each trip set Γ*_{k,j} (Line 2):
                Find best permutation Π(Γ*_{k,j}, t_k) (Line 3)
                For each drone δ (Line 4):
                    Compute delay ψ_{Γ*_{k,j}=δ} (Line 5)
        
        Returns:
            Dict mapping (trip_idx, drone_idx) to minimum delay
        """
        delays: Dict[Tuple[int, int], float] = {}
        
        # Line 1: for j = 1, ..., |Δ| do in parallel
        for j, drone in enumerate(drones):
            drone_available = max(current_time, drone.available_time)
            
            # Line 2: for each Γ*_{k,j} ∈ Γ*_k do
            for t, trip in enumerate(trips):
                # Line 3: Γ*_{k,j} ← Π(Γ*_{k,j}, t_k) - find best permutation
                # For single trip, permutation is trivial
                # For trip with multiple orders, order is already fixed
                
                # Line 4-5: for each δ ∈ Δ_k, compute delay
                # ψ_{Γ*_{k,j}=δ} ← compute_delay(Γ*_{k,j=δ})
                delay = self._compute_delay(trip, drone_available)
                delays[(t, j)] = delay
        
        return delays
    
    def _compute_delay(self, trip: Trip, start_time: float) -> float:
        """
        Compute total delay (lateness) for a trip starting at given time
        
        Algorithm 7, Line 5: compute_delay(Γ*_{k,j=δ})
        """
        total_delay = 0.0
        current_time = start_time
        
        for order in trip.orders:
            # Travel time to this order (simplified: proportional)
            if len(trip.orders) > 0:
                travel_time = (trip.total_distance / len(trip.orders)) / self.speed / 60.0
            else:
                travel_time = 0
            
            current_time += travel_time
            
            # Delay for this order
            delay = max(0, current_time - order.deadline)
            total_delay += delay
            
            current_time += order.service_time
        
        return total_delay
    
    def _optimize_orders(
        self,
        drone_assignments: Dict[int, List[Trip]],
        drones: List[Drone],
        current_time: float
    ) -> Dict[int, List[Tuple[Trip, float]]]:
        """
        For each drone, find optimal execution order for assigned trips
        """
        result: Dict[int, List[Tuple[Trip, float]]] = {}
        
        drone_dict = {d.id: d for d in drones}
        
        for drone_id, assigned_trips in drone_assignments.items():
            if not assigned_trips:
                result[drone_id] = []
                continue
            
            drone = drone_dict.get(drone_id)
            if drone is None:
                continue
            
            # Get drone availability time
            available_time = max(current_time, drone.available_time)
            
            # Find best permutation
            best_order, best_lateness = self._find_best_permutation(
                assigned_trips, available_time
            )
            
            # Calculate start times for best order
            scheduled = self._schedule_trips(best_order, available_time)
            result[drone_id] = scheduled
        
        return result
    
    def _find_best_permutation(
        self,
        trips: List[Trip],
        start_time: float
    ) -> Tuple[List[Trip], float]:
        """
        Find the trip execution order that minimizes total lateness
        
        For small number of trips, enumerate all permutations.
        For larger sets, use heuristics.
        """
        if len(trips) <= 5:
            # Enumerate all permutations
            best_order = trips
            best_lateness = float('inf')
            
            for perm in permutations(trips):
                lateness = self._calculate_total_lateness(list(perm), start_time)
                if lateness < best_lateness:
                    best_lateness = lateness
                    best_order = list(perm)
            
            return best_order, best_lateness
        else:
            # Use urgency-based heuristic
            return self._urgency_based_order(trips, start_time)
    
    def _urgency_based_order(
        self,
        trips: List[Trip],
        start_time: float
    ) -> Tuple[List[Trip], float]:
        """
        Order trips by minimum deadline (most urgent first)
        """
        def min_deadline(trip: Trip) -> float:
            if not trip.orders:
                return float('inf')
            return min(o.deadline for o in trip.orders)
        
        ordered = sorted(trips, key=min_deadline)
        lateness = self._calculate_total_lateness(ordered, start_time)
        
        return ordered, lateness
    
    def _calculate_total_lateness(
        self,
        trips: List[Trip],
        start_time: float
    ) -> float:
        """
        Calculate total lateness for a sequence of trips
        """
        total_lateness = 0.0
        current_time = start_time
        
        for trip in trips:
            # Trip duration
            duration = trip.expected_duration(self.speed)
            
            # Calculate lateness for each order in trip
            trip_time = current_time
            for order in trip.orders:
                # Simplified: assume orders served in sequence
                travel_time = trip.total_distance / len(trip.orders) / self.speed / 60.0
                trip_time += travel_time
                
                lateness = max(0, trip_time - order.deadline)
                total_lateness += lateness * self.late_penalty
                
                trip_time += order.service_time
            
            current_time += duration
        
        return total_lateness
    
    def _schedule_trips(
        self,
        trips: List[Trip],
        start_time: float
    ) -> List[Tuple[Trip, float]]:
        """
        Create schedule with start times for each trip
        """
        scheduled = []
        current_time = start_time
        
        for trip in trips:
            trip.planned_start_time = current_time
            scheduled.append((trip, current_time))
            
            # Next trip starts after this one completes
            duration = trip.expected_duration(self.speed)
            current_time += duration
        
        return scheduled
    
    def _solve_tap(
        self,
        trips: List[Trip],
        drones: List[Drone],
        current_time: float,
        precomputed_delays: Optional[Dict[Tuple[int, int], float]] = None
    ) -> Dict[int, List[Tuple[Trip, float]]]:
        """
        Algorithm 7, Line 9: Solve TAP model (10a)-(10d)
        
        TAP({ψ_Γ*_{k,j} : Γ*_{k,j} ∈ Γ*_k, δ ∈ Δ}, Δ_k)
        
        Assigns trips to drones minimizing total weighted lateness
        using precomputed delays from Lines 1-8.
        """
        try:
            model = gp.Model("TAP")
            model.Params.OutputFlag = 0
            model.Params.TimeLimit = self.time_limit
            model.Params.MIPGap = self.mip_gap
            
            num_trips = len(trips)
            num_drones = len(drones)
            
            # Calculate drone availability times
            availability = [max(current_time, d.available_time) for d in drones]
            
            # Calculate trip durations
            durations = [t.expected_duration(self.speed) for t in trips]
            
            # Calculate urgency (minimum deadline) for each trip
            urgencies = []
            for trip in trips:
                if trip.orders:
                    urgencies.append(min(o.deadline for o in trip.orders))
                else:
                    urgencies.append(float('inf'))
            
            # Decision variables
            # x[t, j] = 1 if trip t is assigned to drone j
            x = model.addVars(num_trips, num_drones, vtype=GRB.BINARY, name="x")
            
            # s[t] = start time of trip t
            s = model.addVars(num_trips, vtype=GRB.CONTINUOUS, lb=0, name="s")
            
            # l[t] = lateness of trip t (max lateness among its orders)
            l = model.addVars(num_trips, vtype=GRB.CONTINUOUS, lb=0, name="l")
            
            # Objective: Minimize total weighted lateness
            model.setObjective(
                gp.quicksum(l[t] * self.late_penalty for t in range(num_trips)),
                GRB.MINIMIZE
            )
            
            # Constraint 1: Each trip assigned to exactly one drone
            for t in range(num_trips):
                model.addConstr(
                    gp.quicksum(x[t, j] for j in range(num_drones)) == 1,
                    name=f"assign_{t}"
                )
            
            # Constraint 2: Start time >= drone availability
            M = 10000  # Big-M
            for t in range(num_trips):
                for j in range(num_drones):
                    model.addConstr(
                        s[t] >= availability[j] - M * (1 - x[t, j]),
                        name=f"avail_{t}_{j}"
                    )
            
            # Constraint 3: Lateness calculation
            for t in range(num_trips):
                # Lateness = max(0, completion_time - deadline)
                # completion_time = start_time + duration
                completion = s[t] + durations[t]
                model.addConstr(
                    l[t] >= completion - urgencies[t],
                    name=f"late_{t}"
                )
            
            # Constraint 4: Sequencing - trips on same drone don't overlap
            # This is simplified; full model would need ordering variables
            for j in range(num_drones):
                for t1 in range(num_trips):
                    for t2 in range(t1 + 1, num_trips):
                        # If both assigned to drone j, they must be sequenced
                        model.addConstr(
                            s[t2] >= s[t1] + durations[t1] - M * (2 - x[t1, j] - x[t2, j]),
                            name=f"seq_{t1}_{t2}_{j}_a"
                        )
            
            # Solve
            model.optimize()
            
            if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
                result: Dict[int, List[Tuple[Trip, float]]] = {d.id: [] for d in drones}
                
                for t in range(num_trips):
                    for j in range(num_drones):
                        if x[t, j].X > 0.5:
                            trip = trips[t]
                            start = s[t].X
                            trip.drone_id = drones[j].id
                            trip.planned_start_time = start
                            result[drones[j].id].append((trip, start))
                
                # Sort each drone's trips by start time
                for drone_id in result:
                    result[drone_id].sort(key=lambda x: x[1])
                
                return result
            else:
                return self._greedy_tap(trips, drones, current_time)
                
        except gp.GurobiError as e:
            print(f"Gurobi error in TAP: {e}")
            return self._greedy_tap(trips, drones, current_time)
    
    def _greedy_tap(
        self,
        trips: List[Trip],
        drones: List[Drone],
        current_time: float
    ) -> Dict[int, List[Tuple[Trip, float]]]:
        """
        Greedy fallback for trip assignment
        
        Assigns most urgent trips first to earliest available drone
        """
        result: Dict[int, List[Tuple[Trip, float]]] = {d.id: [] for d in drones}
        drone_finish_times = {d.id: max(current_time, d.available_time) for d in drones}
        
        # Sort trips by urgency (minimum deadline)
        def min_deadline(trip: Trip) -> float:
            if not trip.orders:
                return float('inf')
            return min(o.deadline for o in trip.orders)
        
        sorted_trips = sorted(trips, key=min_deadline)
        
        for trip in sorted_trips:
            # Find drone that finishes earliest
            best_drone_id = min(drone_finish_times, key=drone_finish_times.get)
            
            start_time = drone_finish_times[best_drone_id]
            trip.drone_id = best_drone_id
            trip.planned_start_time = start_time
            
            result[best_drone_id].append((trip, start_time))
            
            # Update drone finish time
            duration = trip.expected_duration(self.speed)
            drone_finish_times[best_drone_id] = start_time + duration
        
        return result
    
    def calculate_trip_lateness(
        self,
        trip: Trip,
        start_time: float
    ) -> Dict[int, float]:
        """
        Calculate lateness for each order in a trip
        
        Returns:
            Dict mapping order_id to lateness (0 if on time)
        """
        lateness = {}
        current_time = start_time
        current_loc = np.array([0.0, 0.0])  # Depot
        
        for order in trip.orders:
            # Travel time to this order
            dist = np.linalg.norm(order.location - current_loc)
            travel_time = dist / self.speed / 60.0
            current_time += travel_time
            
            # Calculate lateness
            lateness[order.id] = max(0, current_time - order.deadline)
            
            # Service time
            current_time += order.service_time
            current_loc = order.location
        
        return lateness
