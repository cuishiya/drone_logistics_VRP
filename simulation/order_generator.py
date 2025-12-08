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
    LAMBDA_ORDERS, AREA_SIZE, DEPOT_LOCATION,
    SERVICE_TIME, MAX_PAYLOAD, T_HORIZON
)


class OrderGenerator:
    """
    Generates customer orders dynamically using Poisson process
    
    Orders arrive according to Poisson process with rate λ.
    Locations are uniformly distributed in the service area.
    Deadlines are set based on arrival time + random service window.
    """
    
    def __init__(
        self,
        arrival_rate: float = LAMBDA_ORDERS,
        area_size: float = AREA_SIZE,
        depot_location: np.ndarray = DEPOT_LOCATION,
        service_time: float = SERVICE_TIME,
        max_weight: float = MAX_PAYLOAD,
        min_deadline_offset: float = 30.0,  # Minimum 30 min to deadline
        max_deadline_offset: float = 120.0,  # Maximum 2 hours to deadline
        seed: Optional[int] = None
    ):
        self.arrival_rate = arrival_rate
        self.area_size = area_size
        self.depot_location = depot_location
        self.service_time = service_time
        self.max_weight = max_weight
        self.min_deadline_offset = min_deadline_offset
        self.max_deadline_offset = max_deadline_offset
        
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
        """Generate a single random order"""
        self._order_counter += 1
        
        # Random arrival time within interval
        arrival_time = self.rng.uniform(start_time, end_time)
        
        # Random location in service area
        x = self.rng.uniform(-self.area_size / 2, self.area_size / 2)
        y = self.rng.uniform(-self.area_size / 2, self.area_size / 2)
        location = np.array([x, y])
        
        # Random weight (uniform between 0.5 and max_weight)
        weight = self.rng.uniform(0.5, self.max_weight)
        
        # Random deadline
        deadline_offset = self.rng.uniform(
            self.min_deadline_offset,
            self.max_deadline_offset
        )
        deadline = arrival_time + deadline_offset
        
        return Order(
            id=self._order_counter,
            location=location,
            weight=weight,
            arrival_time=arrival_time,
            deadline=deadline,
            service_time=self.service_time
        )
    
    def generate_initial_orders(
        self,
        num_orders: int,
        time_horizon: float = T_HORIZON
    ) -> List[Order]:
        """
        Generate a set of initial orders for testing
        
        Args:
            num_orders: Number of orders to generate
            time_horizon: Time horizon for deadlines
            
        Returns:
            List of Order objects
        """
        orders = []
        for _ in range(num_orders):
            self._order_counter += 1
            
            # All arrive at time 0
            arrival_time = 0.0
            
            # Random location
            x = self.rng.uniform(-self.area_size / 2, self.area_size / 2)
            y = self.rng.uniform(-self.area_size / 2, self.area_size / 2)
            location = np.array([x, y])
            
            # Random weight
            weight = self.rng.uniform(0.5, self.max_weight)
            
            # Random deadline within horizon
            deadline = self.rng.uniform(
                self.min_deadline_offset,
                min(self.max_deadline_offset * 2, time_horizon)
            )
            
            order = Order(
                id=self._order_counter,
                location=location,
                weight=weight,
                arrival_time=arrival_time,
                deadline=deadline,
                service_time=self.service_time
            )
            orders.append(order)
        
        return orders
    
    def generate_clustered_orders(
        self,
        num_orders: int,
        num_clusters: int = 3,
        cluster_std: float = 500.0
    ) -> List[Order]:
        """
        Generate orders clustered around random centers
        
        Useful for testing routing efficiency
        
        Args:
            num_orders: Total number of orders
            num_clusters: Number of cluster centers
            cluster_std: Standard deviation of cluster spread
            
        Returns:
            List of Order objects
        """
        # Generate cluster centers
        centers = []
        for _ in range(num_clusters):
            cx = self.rng.uniform(-self.area_size / 3, self.area_size / 3)
            cy = self.rng.uniform(-self.area_size / 3, self.area_size / 3)
            centers.append(np.array([cx, cy]))
        
        orders = []
        for _ in range(num_orders):
            self._order_counter += 1
            
            # Pick random cluster
            center = centers[self.rng.integers(num_clusters)]
            
            # Generate location around cluster center
            x = self.rng.normal(center[0], cluster_std)
            y = self.rng.normal(center[1], cluster_std)
            
            # Clip to service area
            x = np.clip(x, -self.area_size / 2, self.area_size / 2)
            y = np.clip(y, -self.area_size / 2, self.area_size / 2)
            location = np.array([x, y])
            
            # Random weight and deadline
            weight = self.rng.uniform(0.5, self.max_weight)
            deadline = self.rng.uniform(
                self.min_deadline_offset,
                self.max_deadline_offset * 2
            )
            
            order = Order(
                id=self._order_counter,
                location=location,
                weight=weight,
                arrival_time=0.0,
                deadline=deadline,
                service_time=self.service_time
            )
            orders.append(order)
        
        return orders
