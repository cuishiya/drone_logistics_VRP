"""
Trip Construction Algorithm (Algorithm 3 & 4 from the paper)
Backward Label-setting Heuristic for generating feasible trips

Key concepts:
- Build trips backward from depot to handle payload-dependent energy correctly
- Use urgency-based insertion priority
- Apply Pareto domination rules to prune inferior trips
- Enforce chance constraints for energy feasibility
"""

import numpy as np
from typing import List, Set, Tuple, Optional, Dict
from dataclasses import dataclass, field
from copy import deepcopy

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.data_structures import Order, Trip, Depot
from models.energy_model import EnergyModel, euclidean_distance, compute_route_distance
from config import MAX_PAYLOAD, SERVICE_TIME, DRONE_SPEED_MEAN


@dataclass
class Label:
    """
    Label for backward label-setting algorithm
    
    Represents a partial trip built backward from depot
    
    Attributes:
        orders: Orders in this partial trip (in reverse visit order)
        total_weight: Total payload weight
        expected_energy: Expected energy consumption
        energy_std: Standard deviation of energy consumption
        total_distance: Total route distance
        last_location: Location of the last added customer (first to visit)
        visited_ids: Set of visited customer IDs
        earliest_start: Earliest feasible start time
        is_dominated: Whether this label is dominated by another
    """
    orders: List[Order] = field(default_factory=list)
    total_weight: float = 0.0
    expected_energy: float = 0.0
    energy_std: float = 0.0
    total_distance: float = 0.0
    last_location: Optional[np.ndarray] = None
    visited_ids: Set[int] = field(default_factory=set)
    earliest_start: float = 0.0
    is_dominated: bool = False
    
    def copy(self) -> 'Label':
        """Create a deep copy of this label"""
        new_label = Label(
            orders=self.orders.copy(),
            total_weight=self.total_weight,
            expected_energy=self.expected_energy,
            energy_std=self.energy_std,
            total_distance=self.total_distance,
            last_location=self.last_location.copy() if self.last_location is not None else None,
            visited_ids=self.visited_ids.copy(),
            earliest_start=self.earliest_start,
            is_dominated=False
        )
        return new_label


class TripBuilder:
    """
    Builds feasible trips using backward label-setting heuristic
    
    Algorithm 3: BuildTrips
    - Generates a pool of feasible trips from pending orders
    - Uses urgency-based priority for insertion
    - Applies energy chance constraints
    - Prunes dominated labels
    """
    
    def __init__(
        self,
        depot: Depot,
        energy_model: EnergyModel,
        max_payload: float = MAX_PAYLOAD,
        service_time: float = SERVICE_TIME,
        speed: float = DRONE_SPEED_MEAN,
        max_customers_per_trip: int = 5,
        max_labels: int = 100
    ):
        self.depot = depot
        self.energy_model = energy_model
        self.max_payload = max_payload
        self.service_time = service_time
        self.speed = speed
        self.max_customers_per_trip = max_customers_per_trip
        self.max_labels = max_labels
        self._trip_counter = 0
    
    def build_trips(
        self,
        pending_orders: List[Order],
        current_time: float,
        insertion_strategy: str = "urgency",
        max_vertices_per_iteration: int = 10  # Σ parameter from paper
    ) -> List[Trip]:
        """
        Build a pool of feasible trips (Algorithm 3: Backward heuristic)
        
        Input: Set of outstanding and newly arrived requests D_k ∪ U_k at epoch k
        Output: Set of trips Γ_k
        
        Args:
            pending_orders: List of orders waiting to be served (D_k ∪ U_k)
            current_time: Current simulation time
            insertion_strategy: "urgency" or "distance"
            max_vertices_per_iteration: Σ - max size of vertex set V
            
        Returns:
            List of feasible Trip objects (Γ_k)
        """
        if not pending_orders:
            return []
        
        # Line 1: Define maximum size of set V, represented by parameter Σ
        sigma = max_vertices_per_iteration
        
        # Line 2: Γ_k ← ∅
        completed_trips: List[Trip] = []
        
        # Line 3: Γ'_k ← {i_0, i_{p+1}} - set of trips being built
        # Initially, set Γ'_k has only the empty trip (depot to depot)
        initial_trip = Trip(
            id=self._get_next_trip_id(),
            orders=[],
            total_distance=0.0,
            expected_energy=0.0
        )
        trips_being_built: List[Trip] = [initial_trip]
        
        # Line 4: while Γ'_k ≠ ∅ do
        while trips_being_built:
            # Line 5: γ ← select_trip(Γ'_k) - select a trip from set Γ'_k
            gamma = trips_being_built.pop(0)
            
            # Line 6: Γ'_k ← Γ'_k \ γ - remove γ from Γ'_k (already done by pop)
            
            # Line 7: V ← select_vertices_to_insert(γ, Σ) - Algorithm 4
            # Select vertices that can be inserted into γ
            vertices_to_insert = self._select_vertices_to_insert(
                gamma, pending_orders, current_time, sigma, insertion_strategy
            )
            
            # Line 8: if V = ∅ and γ is not dominated by any trip in Γ_k then
            if not vertices_to_insert:
                if gamma.orders and not self._is_dominated_by_any(gamma, completed_trips):
                    # Line 9: Γ_k ← Γ_k ∪ γ - add γ to the set of trips to be returned
                    completed_trips.append(gamma)
            # Line 10: else if V ≠ ∅ then
            elif vertices_to_insert:
                # Line 11: for each i ∈ V do
                for order in vertices_to_insert:
                    # Line 12: γ' ← γ ∪ {i} - i is inserted after the last backward-inserted vertex
                    new_trip = self._insert_order_into_trip(gamma, order, current_time)
                    
                    if new_trip is not None:
                        # Line 13: Γ'_k ← Γ'_k ∪ γ' - add γ' to the set of trips being built
                        trips_being_built.append(new_trip)
                # Line 14: end
            # Line 15: end
            
            # Limit to prevent explosion
            if len(trips_being_built) > self.max_labels:
                trips_being_built = self._select_best_trips(trips_being_built)
        # Line 16: end
        
        # Line 17: return Γ_k
        return completed_trips
    
    def _get_next_trip_id(self) -> int:
        """Get next unique trip ID"""
        self._trip_counter += 1
        return self._trip_counter
    
    def _select_vertices_to_insert(
        self,
        trip: Trip,
        all_orders: List[Order],
        current_time: float,
        sigma: int,
        strategy: str
    ) -> List[Order]:
        """
        Algorithm 4: select_vertices_to_insert
        
        Input: Trip γ; Σ, the maximum size of set V
        Output: Set of vertices V
        
        Selects up to Σ vertices that can be inserted into trip γ,
        exploring neighbors of the first vertex i_r in the trip.
        """
        # Line 1: Let i_r ∈ γ be the first location to be visited by trip γ, after depot i_0
        # In backward construction, trip.orders[0] is the first customer after depot
        if trip.orders:
            i_r = trip.orders[0]
            i_r_location = i_r.location
        else:
            # Empty trip, use depot as reference
            i_r = None
            i_r_location = self.depot.location
        
        # Get IDs already in trip
        trip_order_ids = {o.id for o in trip.orders}
        current_weight = sum(o.weight for o in trip.orders)
        
        # Build neighbor sets I^d(i_r) and I^l(i_r)
        # I^d: sorted by distance from i_r
        # I^l: sorted by urgency (deadline)
        available_orders = [o for o in all_orders if o.id not in trip_order_ids]
        
        if strategy == "distance":
            # I^d(i_r): neighbors sorted by distance
            neighbor_set = sorted(
                available_orders,
                key=lambda o: euclidean_distance(o.location, i_r_location)
            )
        else:  # "urgency" or default
            # I^l(i_r): neighbors sorted by urgency (deadline proximity)
            neighbor_set = sorted(
                available_orders,
                key=lambda o: o.urgency(current_time)
            )
        
        # Line 2: V ← ∅
        V: List[Order] = []
        
        # Line 3: while |V| < Σ and there is a neighbor vertex of i_r to be explored do
        while len(V) < sigma and neighbor_set:
            # Line 4: i_p ← select_first_neighbor(I^d(i_r)) or select_first_neighbor(I^l(i_r))
            # Select from the correct neighbor set, following the sorting policy
            i_p = neighbor_set[0]
            
            # Line 5: if inequalities (6) and (7) hold when inserting i_p into γ then
            # Check constraint (6): Weight constraint Q - sum(q) >= q_{i_p}
            weight_feasible = (current_weight + i_p.weight <= self.max_payload)
            
            # Check constraint (7): Energy chance constraint
            # Also check max customers constraint
            energy_feasible = False
            if weight_feasible and len(trip.orders) < self.max_customers_per_trip:
                test_orders = [i_p] + trip.orders  # Backward insertion
                energy_result = self.energy_model.compute_trip_energy_from_orders(
                    self.depot.location,
                    test_orders
                )
                energy_feasible = energy_result.is_feasible
            
            if weight_feasible and energy_feasible:
                # Line 6: V ← V ∪ {i_p}
                V.append(i_p)
            # Line 7: end
            
            # Line 8: I^d(i_r) ← I^d(i_r) \ i_p or I^l(i_r) ← I^l(i_r) \ i_p
            # Remove i_p from the correct neighbor set
            neighbor_set.pop(0)
        # Line 9: end
        
        # Line 10: return V
        return V
    
    def _insert_order_into_trip(
        self,
        trip: Trip,
        order: Order,
        current_time: float
    ) -> Optional[Trip]:
        """
        Create a new trip by inserting order at the front (backward insertion)
        
        Returns None if insertion is infeasible
        """
        # Create new order list with backward insertion
        new_orders = [order] + trip.orders
        
        # Check energy feasibility
        energy_result = self.energy_model.compute_trip_energy_from_orders(
            self.depot.location,
            new_orders
        )
        
        if not energy_result.is_feasible:
            return None
        
        # Calculate new distance
        route = [self.depot.location] + [o.location for o in new_orders] + [self.depot.location]
        new_distance = compute_route_distance(route)
        
        # Create new trip
        new_trip = Trip(
            id=self._get_next_trip_id(),
            orders=new_orders,
            expected_energy=energy_result.expected_energy,
            energy_std=energy_result.energy_std,
            total_distance=new_distance,
            total_weight=sum(o.weight for o in new_orders)
        )
        
        return new_trip
    
    def _is_dominated_by_any(self, trip: Trip, trip_set: List[Trip]) -> bool:
        """Check if trip is dominated by any trip in the set"""
        trip_customers = trip.get_customer_ids()
        
        for other in trip_set:
            other_customers = other.get_customer_ids()
            
            # Check if other dominates trip
            # Domination: same or more customers, less or equal cost
            if other_customers >= trip_customers:  # Superset or equal
                if other.total_distance <= trip.total_distance:
                    if other.expected_energy <= trip.expected_energy:
                        # Strictly better in at least one criterion
                        if (other.total_distance < trip.total_distance or
                            other.expected_energy < trip.expected_energy or
                            len(other_customers) > len(trip_customers)):
                            return True
        
        return False
    
    def _select_best_trips(self, trips: List[Trip]) -> List[Trip]:
        """Select best trips when exceeding max_labels"""
        # Score by: more customers, less energy, less distance
        def score(trip: Trip) -> Tuple[int, float, float]:
            return (
                -len(trip.orders),
                trip.expected_energy,
                trip.total_distance
            )
        
        sorted_trips = sorted(trips, key=score)
        return sorted_trips[:self.max_labels]
    
    def _sort_orders(
        self,
        orders: List[Order],
        current_time: float,
        strategy: str
    ) -> List[Order]:
        """Sort orders by insertion priority"""
        if strategy == "urgency":
            # Sort by urgency (ascending = most urgent first)
            return sorted(orders, key=lambda o: o.urgency(current_time))
        elif strategy == "distance":
            # Sort by distance from depot (ascending)
            return sorted(
                orders,
                key=lambda o: euclidean_distance(o.location, self.depot.location)
            )
        else:
            return orders
    
    def _extend_label(
        self,
        label: Label,
        order: Order,
        current_time: float
    ) -> Optional[Label]:
        """
        Try to extend a label by adding an order at the front
        
        This is the backward extension: we're adding a customer that will be
        visited BEFORE the customers already in the label.
        
        Returns None if extension is infeasible.
        """
        # Check if already visited
        if order.id in label.visited_ids:
            return None
        
        # Check payload constraint
        new_weight = label.total_weight + order.weight
        if new_weight > self.max_payload:
            return None
        
        # Check max customers
        if len(label.orders) >= self.max_customers_per_trip:
            return None
        
        # Calculate new distance
        if label.last_location is not None:
            # Distance from new order to previous first customer
            if len(label.orders) > 0:
                dist_to_next = euclidean_distance(order.location, label.orders[0].location)
            else:
                # First customer, distance to depot (return leg)
                dist_to_next = euclidean_distance(order.location, self.depot.location)
            
            # Distance from depot to new order
            dist_from_depot = euclidean_distance(self.depot.location, order.location)
        else:
            dist_to_next = 0
            dist_from_depot = euclidean_distance(self.depot.location, order.location)
        
        # Build the new route for energy calculation
        new_orders = [order] + label.orders
        
        # Check energy feasibility with chance constraint
        energy_result = self.energy_model.compute_trip_energy_from_orders(
            self.depot.location,
            new_orders
        )
        
        if not energy_result.is_feasible:
            return None
        
        # Calculate new total distance
        new_distance = compute_route_distance(
            [self.depot.location] + [o.location for o in new_orders] + [self.depot.location]
        )
        
        # Calculate earliest start time (must allow enough time to reach deadline)
        travel_time = new_distance / self.speed / 60.0  # minutes
        service_time = len(new_orders) * self.service_time
        total_trip_time = travel_time + service_time
        
        # Earliest start based on most urgent order's deadline
        min_deadline = min(o.deadline for o in new_orders)
        earliest_start = max(current_time, min_deadline - total_trip_time)
        
        # Create extended label
        extended = Label(
            orders=new_orders,
            total_weight=new_weight,
            expected_energy=energy_result.expected_energy,
            energy_std=energy_result.energy_std,
            total_distance=new_distance,
            last_location=order.location,
            visited_ids=label.visited_ids | {order.id},
            earliest_start=earliest_start,
            is_dominated=False
        )
        
        return extended
    
    def _prune_dominated(self, labels: List[Label]) -> List[Label]:
        """
        Apply Pareto domination rules (Algorithm 4)
        
        A label L1 dominates L2 if:
        - L1 serves the same or more customers
        - L1 has lower or equal energy consumption
        - L1 has lower or equal distance
        """
        if len(labels) <= 1:
            return labels
        
        # Mark dominated labels
        for i, l1 in enumerate(labels):
            if l1.is_dominated:
                continue
            for j, l2 in enumerate(labels):
                if i == j or l2.is_dominated:
                    continue
                
                # Check if l1 dominates l2
                if self._dominates(l1, l2):
                    l2.is_dominated = True
        
        return [l for l in labels if not l.is_dominated]
    
    def _dominates(self, l1: Label, l2: Label) -> bool:
        """Check if l1 Pareto-dominates l2"""
        # l1 must be at least as good in all criteria
        same_customers = l1.visited_ids == l2.visited_ids
        more_customers = l1.visited_ids.issuperset(l2.visited_ids) and len(l1.visited_ids) > len(l2.visited_ids)
        
        if not (same_customers or more_customers):
            return False
        
        # l1 must have lower or equal energy and distance
        better_energy = l1.expected_energy <= l2.expected_energy
        better_distance = l1.total_distance <= l2.total_distance
        
        # Must be strictly better in at least one criterion
        strictly_better = (
            l1.expected_energy < l2.expected_energy or
            l1.total_distance < l2.total_distance or
            more_customers
        )
        
        return better_energy and better_distance and strictly_better
    
    def _select_best_labels(self, labels: List[Label]) -> List[Label]:
        """Select top labels when exceeding max_labels"""
        # Score by: more customers, less energy, less distance
        def score(label: Label) -> Tuple[int, float, float]:
            return (
                -len(label.orders),  # More customers = better (negative for ascending sort)
                label.expected_energy,
                label.total_distance
            )
        
        sorted_labels = sorted(labels, key=score)
        return sorted_labels[:self.max_labels]
    
    def _label_to_trip(self, label: Label) -> Optional[Trip]:
        """Convert a label to a Trip object"""
        if not label.orders:
            return None
        
        self._trip_counter += 1
        
        trip = Trip(
            id=self._trip_counter,
            orders=label.orders.copy(),
            expected_energy=label.expected_energy,
            energy_std=label.energy_std,
            total_distance=label.total_distance,
            total_weight=label.total_weight
        )
        
        return trip
    
    def _create_single_customer_trip(
        self,
        order: Order,
        current_time: float
    ) -> Optional[Trip]:
        """Create a trip serving only one customer"""
        # Check energy feasibility
        energy_result = self.energy_model.compute_trip_energy_from_orders(
            self.depot.location,
            [order]
        )
        
        if not energy_result.is_feasible:
            return None
        
        distance = 2 * euclidean_distance(self.depot.location, order.location)
        
        self._trip_counter += 1
        
        return Trip(
            id=self._trip_counter,
            orders=[order],
            expected_energy=energy_result.expected_energy,
            energy_std=energy_result.energy_std,
            total_distance=distance,
            total_weight=order.weight
        )
    
    def _remove_duplicates(self, trips: List[Trip]) -> List[Trip]:
        """Remove trips with identical customer sets, keeping the best"""
        seen: Dict[frozenset, Trip] = {}
        
        for trip in trips:
            key = frozenset(trip.get_customer_ids())
            if key not in seen:
                seen[key] = trip
            else:
                # Keep the one with lower expected energy
                if trip.expected_energy < seen[key].expected_energy:
                    seen[key] = trip
        
        return list(seen.values())
    
    def update_trips(
        self,
        existing_trips: List[Trip],
        new_orders: List[Order],
        current_time: float,
        threshold_v: float = None
    ) -> Tuple[List[Trip], List[Order]]:
        """
        Algorithm 5: Trip update Heuristic (update_trips)
        
        Input: Set of outstanding and newly arrived requests D_k ∪ U_k at epoch k; Threshold υ
        Output: Updated set of outstanding and newly arrived requests D'_k ∪ U'_k at epoch k
        
        Args:
            existing_trips: Trips from previous epoch Γ(s_{k-1}, s_k)
            new_orders: Newly arrived orders (D_k ∪ U_k)
            current_time: Current time t_k
            threshold_v: Threshold υ for dispatching time window
            
        Returns:
            Tuple of (updated_trips, remaining_orders as D'_k ∪ U'_k)
        """
        from config import UPSILON
        if threshold_v is None:
            threshold_v = UPSILON
        
        # Line 1: D'_k ← D_k, U'_k ← U_k
        remaining_orders = new_orders.copy()  # D'_k ∪ U'_k
        
        # Line 2: Γ'(s_{k-1}, s_k) ← {γ ∈ Γ(s_{k-1}, s_k) : no lateness and dispatching time ∈ [t_k, t_k + υ]}
        # Filter trips: no lateness AND dispatching time in [t_k, t_k + υ]
        eligible_trips = []
        for trip in existing_trips:
            if trip.is_executed or trip.is_cancelled:
                continue
            
            # Check dispatching time is in [t_k, t_k + υ]
            if trip.planned_start_time is not None:
                if not (current_time <= trip.planned_start_time <= current_time + threshold_v):
                    continue
            
            # Check no lateness: max_lateness should be 0
            if trip.planned_start_time is not None:
                lateness = trip.max_lateness(trip.planned_start_time, self.speed)
                if lateness > 0:
                    continue
            
            eligible_trips.append(trip)
        
        # Line 3: for each r ∈ D_k ∪ U_k do
        for order in new_orders:
            if order in remaining_orders:  # Still not assigned
                # Line 4: γ ← find_best_trip_for_request(Γ'(s_{k-1}, s_k), i_r)
                best_trip, best_position, delta_psi = self._find_best_trip_for_request(
                    eligible_trips, order, current_time
                )
                
                # Line 5: if γ ≠ ∅ and Δψ(γ, i_r) = 0 then
                if best_trip is not None and delta_psi == 0:
                    # Line 6: γ ← γ ∪ {i_r} - i_r is inserted on γ in the correct position
                    best_trip.orders.insert(best_position, order)
                    best_trip.total_weight += order.weight
                    
                    # Recalculate trip properties
                    energy_result = self.energy_model.compute_trip_energy_from_orders(
                        self.depot.location, best_trip.orders
                    )
                    best_trip.expected_energy = energy_result.expected_energy
                    best_trip.energy_std = energy_result.energy_std
                    best_trip.total_distance = compute_route_distance(
                        best_trip.get_route(self.depot.location)
                    )
                    
                    order.assigned_trip_id = best_trip.id
                    
                    # Line 7: D'_k ∪ U'_k ← (D'_k ∪ U'_k) \ {r}
                    remaining_orders.remove(order)
                # Line 8: end
        # Line 9: end
        
        # Line 10: return D'_k ∪ U'_k
        return existing_trips, remaining_orders
    
    def _find_best_trip_for_request(
        self,
        eligible_trips: List[Trip],
        order: Order,
        current_time: float
    ) -> Tuple[Optional[Trip], Optional[int], float]:
        """
        Find the best trip to insert a request (Algorithm 5, Line 4)
        
        Returns:
            Tuple of (best_trip, best_position, delta_psi)
            where delta_psi is the additional lateness from insertion
        """
        best_trip = None
        best_position = None
        best_delta_psi = float('inf')
        best_insertion_cost = float('inf')
        
        for trip in eligible_trips:
            # Check weight constraint
            if trip.total_weight + order.weight > self.max_payload:
                continue
            
            # Try each insertion position
            for pos in range(len(trip.orders) + 1):
                # Check energy feasibility
                test_orders = trip.orders[:pos] + [order] + trip.orders[pos:]
                energy_result = self.energy_model.compute_trip_energy_from_orders(
                    self.depot.location, test_orders
                )
                
                if not energy_result.is_feasible:
                    continue
                
                # Calculate Δψ(γ, i_r) - additional lateness from insertion
                original_lateness = trip.max_lateness(
                    trip.planned_start_time or current_time, self.speed
                )
                
                # Create temporary trip to calculate new lateness
                new_lateness = self._calculate_trip_lateness(
                    test_orders, trip.planned_start_time or current_time
                )
                
                delta_psi = new_lateness - original_lateness
                
                # Calculate insertion cost (distance)
                insertion_cost = self._calculate_insertion_cost(
                    trip.orders, order, pos
                )
                
                # Prefer: delta_psi = 0, then minimum insertion cost
                if delta_psi < best_delta_psi or (delta_psi == best_delta_psi and insertion_cost < best_insertion_cost):
                    best_trip = trip
                    best_position = pos
                    best_delta_psi = delta_psi
                    best_insertion_cost = insertion_cost
        
        return best_trip, best_position, best_delta_psi
    
    def _calculate_trip_lateness(
        self,
        orders: List[Order],
        start_time: float
    ) -> float:
        """Calculate total lateness for a trip"""
        current_time = start_time
        current_loc = self.depot.location
        total_lateness = 0.0
        
        for order in orders:
            # Travel time
            dist = euclidean_distance(current_loc, order.location)
            travel_time = dist / self.speed / 60.0  # Convert to minutes
            current_time += travel_time
            
            # Calculate lateness for this order
            lateness = max(0, current_time - order.deadline)
            total_lateness = max(total_lateness, lateness)
            
            current_time += order.service_time
            current_loc = order.location
        
        return total_lateness
    
    def _calculate_insertion_cost(
        self,
        current_orders: List[Order],
        new_order: Order,
        position: int
    ) -> float:
        """Calculate additional distance from inserting order at position"""
        if not current_orders:
            # First customer
            return 2 * euclidean_distance(self.depot.location, new_order.location)
        
        if position == 0:
            # Insert at beginning
            prev_loc = self.depot.location
            next_loc = current_orders[0].location
        elif position == len(current_orders):
            # Insert at end
            prev_loc = current_orders[-1].location
            next_loc = self.depot.location
        else:
            prev_loc = current_orders[position - 1].location
            next_loc = current_orders[position].location
        
        # Original distance
        original_dist = euclidean_distance(prev_loc, next_loc)
        
        # New distance through new_order
        new_dist = (
            euclidean_distance(prev_loc, new_order.location) +
            euclidean_distance(new_order.location, next_loc)
        )
        
        return new_dist - original_dist
    
    def _check_deadline_feasibility(
        self,
        orders: List[Order],
        start_time: float
    ) -> bool:
        """Check if all orders can be served before their deadlines"""
        current_time = start_time
        current_loc = self.depot.location
        
        for order in orders:
            # Travel time
            dist = euclidean_distance(current_loc, order.location)
            travel_time = dist / self.speed / 60.0
            current_time += travel_time
            
            # Allow some lateness (soft deadline)
            # Here we just check if it's not too late
            if current_time > order.deadline + 60:  # Allow up to 60 min late
                return False
            
            current_time += order.service_time
            current_loc = order.location
        
        return True
