"""
Configuration parameters for DRPUDEC (Dynamic Drone Routing Problem with Uncertain Demand and Energy Consumption)
Based on: Chagas et al. (2025)
"""

import numpy as np

# =============================================================================
# Time Parameters
# =============================================================================
T_HORIZON = 480  # Total time horizon in minutes (8 hours)
PSI = 30  # Decision epoch interval in minutes
UPSILON = 5  # Buffer threshold for trip cancellation (minutes)

# =============================================================================
# Drone Physical Parameters
# =============================================================================
NUM_DRONES = 3  # Number of drones in the fleet
DRONE_WEIGHT = 2.0  # Drone self-weight including battery (kg), denoted as ν
MAX_PAYLOAD = 5.0  # Maximum payload capacity Q (kg)
DRONE_SPEED_MEAN = 15.0  # Mean speed (m/s)
DRONE_SPEED_STD = 2.0  # Speed standard deviation due to wind

# =============================================================================
# Battery Parameters
# =============================================================================
NUM_BATTERIES = 5  # Total number of batteries (b)
E_MAX = 500000.0  # Maximum battery capacity (Joules, ~140 Wh for typical drone battery)
E_MIN = 50000.0  # Minimum safe battery level (10% reserve)
CHARGE_RATE = 1000.0  # Charging rate (Joules per minute)
BATTERY_SWAP_TIME = 2.0  # Time to swap battery (minutes)

# =============================================================================
# Energy Consumption Model Parameters
# =============================================================================
GRAVITY = 9.81  # Gravitational acceleration (m/s^2)
AIR_DENSITY = 1.225  # Air density ρ (kg/m^3)
NUM_ROTORS = 4  # Number of rotors (n)
ROTOR_DISC_AREA = 0.1  # Blade disc area parameter ζ (m^2)

# =============================================================================
# Service Parameters
# =============================================================================
SERVICE_TIME = 2.0  # Fixed service time η at each customer (minutes)

# =============================================================================
# Cost Parameters (μ^c, μ^l from paper)
# =============================================================================
DISTANCE_COST_RATE = 0.001  # μ^c: monetary cost per unit of distance (per meter)
LATE_PENALTY_RATE = 1.0  # μ^l: penalty cost per minute of lateness

# =============================================================================
# Chance Constraint Parameters
# =============================================================================
ALPHA = 0.90  # Safety probability for chance constraints
# Φ^(-1)(0.90) ≈ 1.2816
ALPHA_QUANTILE = 1.2816

# =============================================================================
# CFA Policy Parameters
# =============================================================================
M_MAX_TRIPS = 2  # Maximum trips per drone per decision epoch

# =============================================================================
# Order Generation (Poisson Process)
# =============================================================================
LAMBDA_ORDERS = 0.5  # Average order arrival rate per minute

# =============================================================================
# Problem Instance Parameters
# =============================================================================
DEPOT_LOCATION = np.array([0.0, 0.0])  # Depot coordinates
AREA_SIZE = 10000.0  # Service area size (meters), customers in [-AREA_SIZE/2, AREA_SIZE/2]

# =============================================================================
# Solver Parameters
# =============================================================================
GUROBI_TIME_LIMIT = 60  # Time limit for Gurobi solver (seconds)
GUROBI_MIP_GAP = 0.01  # MIP gap tolerance

# =============================================================================
# Battery Reservation Policy
# =============================================================================
BATTERY_RESERVE_RATIO = 0.10  # Reserve 10% of batteries for emergencies
URGENT_CUSTOMER_THRESHOLD = 0.10  # Threshold for using reserve batteries
