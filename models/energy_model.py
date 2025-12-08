"""
Energy Consumption Model with Chance Constraints
Based on: Chagas et al. (2025) - Section on Random Energy Model

Energy consumption formula:
e_{i_r, i_p}(q) = (d_{i_r, i_p} / ξ) * ((ν + q)^{3/2} * sqrt(g^3 / (2ρζn)))

Where:
- d: distance between nodes
- ξ: random speed (affected by wind)
- ν: drone self-weight
- q: payload weight
- g: gravity
- ρ: air density
- ζ: rotor disc area parameter
- n: number of rotors
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
    """Result of energy consumption calculation"""
    expected_energy: float  # E[CE]
    energy_std: float  # SDV(CE)
    is_feasible: bool  # Whether chance constraint is satisfied
    safety_margin: float  # E_max - E_min - (E[CE] + Φ^{-1}(α) * SDV)


class EnergyModel:
    """
    Stochastic energy consumption model for drone routing
    
    Implements the chance-constrained energy model from the paper.
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
        
        # Precompute constant factor: sqrt(g^3 / (2ρζn))
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
        Compute energy consumption for a single edge
        
        Args:
            distance: Distance between nodes (meters)
            payload: Current payload weight (kg)
            speed: Flight speed (m/s), if None uses mean speed
            
        Returns:
            Tuple of (expected_energy, energy_variance)
        """
        if speed is None:
            speed = self.speed_mean
        
        # Total weight = drone weight + payload
        total_weight = self.drone_weight + payload
        
        # Power consumption: (ν + q)^{3/2} * power_factor
        power = (total_weight ** 1.5) * self.power_factor
        
        # Energy = Power * Time = Power * (Distance / Speed)
        # E[e] = d * power * E[1/ξ]
        # For random speed ξ ~ N(μ, σ²), we approximate E[1/ξ] ≈ 1/μ
        # and Var(1/ξ) ≈ σ²/μ⁴
        
        expected_energy = distance * power / self.speed_mean
        
        # Variance approximation using delta method
        # Var(e) ≈ (d * power)² * Var(1/ξ) ≈ (d * power)² * σ²/μ⁴
        energy_variance = ((distance * power) ** 2) * (self.speed_std ** 2) / (self.speed_mean ** 4)
        
        return expected_energy, energy_variance
    
    def compute_trip_energy(
        self,
        route: List[np.ndarray],
        payloads: List[float]
    ) -> EnergyResult:
        """
        Compute total energy consumption for a trip
        
        Args:
            route: List of locations [depot, customer1, customer2, ..., depot]
            payloads: Payload weight at each leg of the journey
                     payloads[i] is the weight when traveling from route[i] to route[i+1]
                     
        Returns:
            EnergyResult with expected energy, std, and feasibility
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
            # Assuming independence, variances add
            total_variance += var_e
        
        total_std = np.sqrt(total_variance)
        
        # Check chance constraint:
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
        Compute trip energy from a list of orders
        
        Args:
            depot_location: Location of depot
            orders: List of orders in visit sequence
            order_weights: Optional list of order weights (uses order.weight if None)
            
        Returns:
            EnergyResult
        """
        if not orders:
            return EnergyResult(0.0, 0.0, True, self.e_max - self.e_min)
        
        # Build route
        route = [depot_location]
        for order in orders:
            route.append(order.location)
        route.append(depot_location)
        
        # Build payloads (decreasing as we deliver)
        if order_weights is None:
            order_weights = [o.weight for o in orders]
        
        # Payload at each leg
        # Leg 0: depot -> customer 1, carrying all packages
        # Leg 1: customer 1 -> customer 2, carrying all except first
        # ...
        # Last leg: last customer -> depot, carrying nothing
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
        Check if inserting a new order at given position is energy-feasible
        
        Args:
            depot_location: Depot location
            current_orders: Current orders in trip
            new_order: Order to insert
            insert_position: Position to insert (0 = first customer)
            
        Returns:
            Tuple of (is_feasible, energy_result)
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
        Calculate maximum additional distance that can be traveled
        while satisfying chance constraint
        
        Args:
            current_energy_used: Expected energy already consumed
            current_energy_std: Current energy standard deviation
            payload: Current payload weight
            
        Returns:
            Maximum additional distance (meters)
        """
        available = self.e_max - self.e_min
        
        # We need: (current + new_exp) + α_q * sqrt(current_var + new_var) ≤ available
        # This is complex to solve exactly, so we use a conservative estimate
        
        # Conservative: assume new_std adds linearly (overestimate)
        remaining_budget = available - current_energy_used - self.alpha_quantile * current_energy_std
        
        if remaining_budget <= 0:
            return 0.0
        
        # Energy per meter at current payload
        total_weight = self.drone_weight + payload
        power = (total_weight ** 1.5) * self.power_factor
        energy_per_meter = power / self.speed_mean
        
        # Add safety factor for variance
        safety_factor = 0.8
        
        return remaining_budget * safety_factor / energy_per_meter


# Utility functions for distance calculations
def euclidean_distance(loc1: np.ndarray, loc2: np.ndarray) -> float:
    """Calculate Euclidean distance between two locations"""
    return np.linalg.norm(loc1 - loc2)


def compute_route_distance(route: List[np.ndarray]) -> float:
    """Compute total distance of a route"""
    if len(route) < 2:
        return 0.0
    total = 0.0
    for i in range(len(route) - 1):
        total += euclidean_distance(route[i], route[i+1])
    return total
