"""
Trip Selection Algorithm (Algorithm 6 from the paper)
Implements SPBP (Set Packing Based Problem) and DTSSP (Disjoint Trip Set Selection Problem)

Uses Gurobi to solve:
1. SPBP: Maximize weighted customer coverage
2. DTSSP: Minimize routing cost for selected customers
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
import gurobipy as gp
from gurobipy import GRB

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.data_structures import Order, Trip, Drone
from config import (
    M_MAX_TRIPS, T_HORIZON, GUROBI_TIME_LIMIT, GUROBI_MIP_GAP,
    DRONE_SPEED_MEAN, LATE_PENALTY_RATE
)


class TripSelector:
    """
    Selects optimal trips from candidate pool using MILP models
    
    Two-phase approach:
    1. SPBP: Select trips to maximize customer coverage (weighted by urgency)
    2. DTSSP: Among selected customers, minimize routing cost
    """
    
    def __init__(
        self,
        num_drones: int,
        max_trips_per_drone: int = M_MAX_TRIPS,
        time_horizon: float = T_HORIZON,
        time_limit: float = GUROBI_TIME_LIMIT,
        mip_gap: float = GUROBI_MIP_GAP,
        speed: float = DRONE_SPEED_MEAN
    ):
        self.num_drones = num_drones
        self.max_trips_per_drone = max_trips_per_drone
        self.time_horizon = time_horizon
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.speed = speed
    
    def select_trips(
        self,
        candidate_trips: List[Trip],
        pending_orders: List[Order],
        drones: List[Drone],
        current_time: float,
        urgency_threshold: float = None
    ) -> Tuple[List[Trip], Dict[int, List[Trip]]]:
        """
        Algorithm 6: Select trips (select_trips)
        
        Input: Set of outstanding and newly arrived requests D_k ∪ U_k;
               Set of trips Γ_k; number of drones n; maximum number of trips per drone M
        Output: Set of trips Γ*_k without repeated requests
        
        Args:
            candidate_trips: Pool of feasible trips Γ_k from TripBuilder
            pending_orders: All pending orders (D_k ∪ U_k)
            drones: Available drones Δ
            current_time: Current simulation time
            urgency_threshold: Threshold l for priority classification
            
        Returns:
            Tuple of:
            - List of selected trips Γ*_k
            - Dict mapping drone_id to list of assigned trips
        """
        if not candidate_trips:
            return [], {}
        
        # Line 1: D^pr_k ∪ U^pr_k, D^npr_k ∪ U^npr_k ← split(D_k ∪ U_k)
        # Split requests into priority and non-priority
        priority_orders, non_priority_orders = self._split_orders(
            pending_orders, current_time, urgency_threshold
        )
        
        # Line 2: D'_k ∪ U'_k ← SPBP(D^pr_k ∪ U^pr_k, D^npr_k ∪ U^npr_k, Γ_k, n, M)
        # Solve model (8a)-(8h) and find set D'_k ∪ U'_k
        selected_trips_phase1, selected_customers = self._solve_spbp(
            candidate_trips, pending_orders, priority_orders, non_priority_orders,
            drones, current_time
        )
        
        if not selected_trips_phase1:
            return [], {}
        
        # Line 3: Γ*_k ← DTSSP(D_k ∪ U_k, D'_k ∪ U'_k, Γ_k, n, M)
        # Solve model (9a)-(9b) and find set of trips Γ*_k
        final_trips, drone_assignments = self._solve_dtssp(
            candidate_trips, selected_customers, drones, current_time
        )
        
        # Line 4: return Γ*_k
        return final_trips, drone_assignments
    
    def _split_orders(
        self,
        orders: List[Order],
        current_time: float,
        urgency_threshold: float = None
    ) -> Tuple[List[Order], List[Order]]:
        """
        Split orders into priority and non-priority sets (Algorithm 6, Line 1)
        
        Priority orders: deadline within [t_k, t_k + l]
        Non-priority orders: deadline after t_k + l
        
        Args:
            orders: All pending orders
            current_time: Current time t_k
            urgency_threshold: Threshold l (default from config)
            
        Returns:
            Tuple of (priority_orders, non_priority_orders)
        """
        from config import PSI
        if urgency_threshold is None:
            # Default: orders due within 2 decision intervals are priority
            urgency_threshold = 2 * PSI
        
        priority_orders = []
        non_priority_orders = []
        
        for order in orders:
            # Check if deadline is within [t_k, t_k + l]
            if order.deadline <= current_time + urgency_threshold:
                priority_orders.append(order)
            else:
                non_priority_orders.append(order)
        
        return priority_orders, non_priority_orders
    
    def _solve_spbp(
        self,
        trips: List[Trip],
        all_orders: List[Order],
        priority_orders: List[Order],
        non_priority_orders: List[Order],
        drones: List[Drone],
        current_time: float
    ) -> Tuple[List[Trip], Set[int]]:
        """
        Solve Set Packing Based Problem (SPBP) - Model (8a)-(8h)
        
        Maximize: Σ w_r * z_r  (weighted customer coverage)
        
        Priority orders (D^pr ∪ U^pr) get higher weights than
        non-priority orders (D^npr ∪ U^npr)
        
        Subject to:
        - Each customer served at most once
        - Each drone gets at most M trips
        - Time constraints
        - Symmetry breaking
        
        Returns:
            Tuple of (selected_trips, selected_customer_ids as D'_k ∪ U'_k)
        """
        try:
            model = gp.Model("SPBP")
            model.Params.OutputFlag = 0  # Suppress output
            model.Params.TimeLimit = self.time_limit
            model.Params.MIPGap = self.mip_gap
            
            num_trips = len(trips)
            num_drones = len(drones)
            num_orders = len(all_orders)
            
            # Create order ID to index mapping
            order_id_to_idx = {o.id: i for i, o in enumerate(all_orders)}
            priority_ids = {o.id for o in priority_orders}
            
            # Decision variables
            # y[γ, j] = 1 if trip γ is assigned to drone j
            y = model.addVars(num_trips, num_drones, vtype=GRB.BINARY, name="y")
            
            # z[r] = 1 if customer r is served
            z = model.addVars(num_orders, vtype=GRB.BINARY, name="z")
            
            # Calculate weights for customers
            # Priority orders get much higher weights than non-priority
            weights = []
            for order in all_orders:
                if order.id in priority_ids:
                    # Priority: high base weight + urgency bonus
                    urgency = order.urgency(current_time)
                    weight = 1000.0 + max(0, 100.0 - urgency)
                else:
                    # Non-priority: lower weight based on urgency
                    urgency = order.urgency(current_time)
                    weight = max(1.0, 100.0 - urgency)
                weights.append(weight)
            
            # Objective: Maximize weighted customer coverage
            model.setObjective(
                gp.quicksum(weights[i] * z[i] for i in range(num_orders)),
                GRB.MAXIMIZE
            )
            
            # Constraint 1: Link z and y
            # If customer r is in trip γ and trip γ is assigned, then z[r] = 1
            for i, order in enumerate(all_orders):
                covering_trips = []
                for t, trip in enumerate(trips):
                    if order.id in trip.get_customer_ids():
                        covering_trips.append(t)
                
                if covering_trips:
                    model.addConstr(
                        z[i] <= gp.quicksum(
                            y[t, j] for t in covering_trips for j in range(num_drones)
                        ),
                        name=f"cover_{i}"
                    )
            
            # Constraint 2: Each trip assigned to at most one drone
            for t in range(num_trips):
                model.addConstr(
                    gp.quicksum(y[t, j] for j in range(num_drones)) <= 1,
                    name=f"trip_once_{t}"
                )
            
            # Constraint 3: Each drone gets at most M trips
            for j in range(num_drones):
                model.addConstr(
                    gp.quicksum(y[t, j] for t in range(num_trips)) <= self.max_trips_per_drone,
                    name=f"max_trips_{j}"
                )
            
            # Constraint 4: Each customer served at most once
            for i, order in enumerate(all_orders):
                covering_trips = []
                for t, trip in enumerate(trips):
                    if order.id in trip.get_customer_ids():
                        covering_trips.append(t)
                
                if len(covering_trips) > 1:
                    model.addConstr(
                        gp.quicksum(
                            y[t, j] for t in covering_trips for j in range(num_drones)
                        ) <= 1,
                        name=f"once_{i}"
                    )
            
            # Constraint 5: Time constraint - trips must complete before horizon
            for j, drone in enumerate(drones):
                # Sum of trip durations for this drone
                model.addConstr(
                    gp.quicksum(
                        y[t, j] * trips[t].expected_duration(self.speed)
                        for t in range(num_trips)
                    ) <= self.time_horizon - current_time,
                    name=f"time_{j}"
                )
            
            # Constraint 6: Symmetry breaking - order drones by total trip duration
            for j in range(num_drones - 1):
                model.addConstr(
                    gp.quicksum(
                        y[t, j] * trips[t].expected_duration(self.speed)
                        for t in range(num_trips)
                    ) >= gp.quicksum(
                        y[t, j+1] * trips[t].expected_duration(self.speed)
                        for t in range(num_trips)
                    ),
                    name=f"symmetry_{j}"
                )
            
            # Solve
            model.optimize()
            
            if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
                selected_trips = []
                selected_customers = set()
                
                for t in range(num_trips):
                    for j in range(num_drones):
                        if y[t, j].X > 0.5:
                            selected_trips.append(trips[t])
                            selected_customers.update(trips[t].get_customer_ids())
                
                return selected_trips, selected_customers
            else:
                return [], set()
                
        except gp.GurobiError as e:
            print(f"Gurobi error in SPBP: {e}")
            return [], set()
    
    def _solve_dtssp(
        self,
        trips: List[Trip],
        must_serve_customers: Set[int],
        drones: List[Drone],
        current_time: float
    ) -> Tuple[List[Trip], Dict[int, List[Trip]]]:
        """
        Solve Disjoint Trip Set Selection Problem (DTSSP)
        
        Given the customers that must be served (from SPBP),
        select trips to minimize total routing cost.
        
        Minimize: Σ c_γ * x_γ  (total routing cost)
        Subject to:
        - All must-serve customers are covered
        - No customer overlap between selected trips
        - Drone capacity constraints
        
        Returns:
            Tuple of (selected_trips, drone_to_trips_mapping)
        """
        if not must_serve_customers:
            return [], {}
        
        # Filter trips to those covering at least one must-serve customer
        relevant_trips = [
            t for t in trips
            if t.get_customer_ids() & must_serve_customers
        ]
        
        if not relevant_trips:
            return [], {}
        
        try:
            model = gp.Model("DTSSP")
            model.Params.OutputFlag = 0
            model.Params.TimeLimit = self.time_limit
            model.Params.MIPGap = self.mip_gap
            
            num_trips = len(relevant_trips)
            num_drones = len(drones)
            
            # Decision variables
            # x[γ, j] = 1 if trip γ is assigned to drone j
            x = model.addVars(num_trips, num_drones, vtype=GRB.BINARY, name="x")
            
            # Objective: Minimize total routing cost (distance)
            model.setObjective(
                gp.quicksum(
                    x[t, j] * relevant_trips[t].total_distance
                    for t in range(num_trips)
                    for j in range(num_drones)
                ),
                GRB.MINIMIZE
            )
            
            # Constraint 1: Each must-serve customer is covered exactly once
            for cust_id in must_serve_customers:
                covering_trips = [
                    t for t, trip in enumerate(relevant_trips)
                    if cust_id in trip.get_customer_ids()
                ]
                
                if covering_trips:
                    model.addConstr(
                        gp.quicksum(
                            x[t, j] for t in covering_trips for j in range(num_drones)
                        ) == 1,
                        name=f"must_cover_{cust_id}"
                    )
            
            # Constraint 2: Each trip assigned to at most one drone
            for t in range(num_trips):
                model.addConstr(
                    gp.quicksum(x[t, j] for j in range(num_drones)) <= 1,
                    name=f"trip_once_{t}"
                )
            
            # Constraint 3: Each drone gets at most M trips
            for j in range(num_drones):
                model.addConstr(
                    gp.quicksum(x[t, j] for t in range(num_trips)) <= self.max_trips_per_drone,
                    name=f"max_trips_{j}"
                )
            
            # Constraint 4: No customer overlap (customers not in must_serve should not be duplicated)
            all_customers = set()
            for trip in relevant_trips:
                all_customers.update(trip.get_customer_ids())
            
            for cust_id in all_customers:
                if cust_id not in must_serve_customers:
                    covering_trips = [
                        t for t, trip in enumerate(relevant_trips)
                        if cust_id in trip.get_customer_ids()
                    ]
                    
                    if len(covering_trips) > 1:
                        model.addConstr(
                            gp.quicksum(
                                x[t, j] for t in covering_trips for j in range(num_drones)
                            ) <= 1,
                            name=f"no_overlap_{cust_id}"
                        )
            
            # Constraint 5: Balance drone usage
            avg_trips = num_trips / num_drones if num_drones > 0 else 0
            for j in range(num_drones):
                model.addConstr(
                    gp.quicksum(x[t, j] for t in range(num_trips)) <= avg_trips + 2,
                    name=f"balance_{j}"
                )
            
            # Solve
            model.optimize()
            
            if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
                selected_trips = []
                drone_assignments: Dict[int, List[Trip]] = {d.id: [] for d in drones}
                
                for t in range(num_trips):
                    for j in range(num_drones):
                        if x[t, j].X > 0.5:
                            trip = relevant_trips[t]
                            trip.drone_id = drones[j].id
                            selected_trips.append(trip)
                            drone_assignments[drones[j].id].append(trip)
                
                return selected_trips, drone_assignments
            else:
                # Fallback: greedy assignment
                return self._greedy_assignment(relevant_trips, must_serve_customers, drones)
                
        except gp.GurobiError as e:
            print(f"Gurobi error in DTSSP: {e}")
            return self._greedy_assignment(relevant_trips, must_serve_customers, drones)
    
    def _greedy_assignment(
        self,
        trips: List[Trip],
        must_serve: Set[int],
        drones: List[Drone]
    ) -> Tuple[List[Trip], Dict[int, List[Trip]]]:
        """
        Greedy fallback when MILP fails
        
        Assigns trips greedily to cover all must-serve customers
        """
        selected_trips = []
        drone_assignments: Dict[int, List[Trip]] = {d.id: [] for d in drones}
        covered = set()
        drone_trip_counts = {d.id: 0 for d in drones}
        
        # Sort trips by efficiency (customers per distance)
        sorted_trips = sorted(
            trips,
            key=lambda t: len(t.orders) / max(t.total_distance, 1),
            reverse=True
        )
        
        for trip in sorted_trips:
            trip_customers = trip.get_customer_ids()
            new_coverage = trip_customers & must_serve - covered
            
            if not new_coverage:
                continue
            
            # Find drone with fewest trips
            available_drones = [
                d for d in drones
                if drone_trip_counts[d.id] < self.max_trips_per_drone
            ]
            
            if not available_drones:
                break
            
            best_drone = min(available_drones, key=lambda d: drone_trip_counts[d.id])
            
            trip.drone_id = best_drone.id
            selected_trips.append(trip)
            drone_assignments[best_drone.id].append(trip)
            drone_trip_counts[best_drone.id] += 1
            covered.update(trip_customers)
            
            if covered >= must_serve:
                break
        
        return selected_trips, drone_assignments
