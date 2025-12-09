"""
DRPUDEC: Dynamic Drone Routing Problem with Uncertain Demand and Energy Consumption
Main entry point for running simulations

Based on: Chagas et al. (2025) - "A dynamic drone routing problem with uncertain 
demand and energy consumption"

This implementation includes:
- MDP-based dynamic decision making
- Cost Function Approximation (CFA) policy
- Chance-constrained energy model
- Backward label-setting trip construction
- MILP-based trip selection (SPBP + DTSSP)
- Task assignment problem (TAP)
- Battery management with balanced aging
"""

import numpy as np
import argparse
from typing import Optional

from simulation.simulator import Simulator
from simulation.order_generator import OrderGenerator
from config import (
    NUM_DRONES, NUM_BATTERIES, T_HORIZON, PSI, M_MAX_TRIPS,
    NUM_DEPOTS, DRONES_PER_DEPOT, BATTERIES_PER_DEPOT, DEFAULT_DEPOT_INFOS
)


def run_single_simulation(
    num_drones: int = NUM_DRONES,
    num_batteries: int = NUM_BATTERIES,
    num_initial_orders: int = 10,
    time_horizon: float = T_HORIZON,
    decision_interval: float = PSI,
    max_trips_per_drone: int = M_MAX_TRIPS,
    dynamic_orders: bool = True,
    seed: Optional[int] = None,
    verbose: bool = True,
    multi_depot_mode: bool = True,  # 多仓库模式
    drones_per_depot: int = DRONES_PER_DEPOT,  # 每个仓库的无人机数
    batteries_per_depot: int = BATTERIES_PER_DEPOT  # 每个仓库的电池数
):
    """
    Run a single simulation instance
    支持多仓库场景
    
    Args:
        num_drones: Number of drones in fleet (单仓库模式)
        num_batteries: Total number of batteries (单仓库模式)
        num_initial_orders: Number of orders at time 0
        time_horizon: Total simulation time (minutes)
        decision_interval: Time between decision epochs (minutes)
        max_trips_per_drone: Maximum trips per drone per epoch (M parameter)
        dynamic_orders: Whether to generate dynamic orders
        seed: Random seed for reproducibility
        verbose: Print progress information
        multi_depot_mode: 是否启用多仓库模式
        drones_per_depot: 每个仓库的无人机数量
        batteries_per_depot: 每个仓库的电池数量
    """
    print("=" * 60)
    if multi_depot_mode:
        print("DRPUDEC Multi-Depot Simulation")
        print(f"多仓库无人机路径规划 - {NUM_DEPOTS}个仓库")
    else:
        print("DRPUDEC Simulation")
        print("Dynamic Drone Routing with Uncertain Demand and Energy")
    print("=" * 60)
    
    # Create simulator
    simulator = Simulator(
        num_drones=num_drones,
        num_batteries=num_batteries,
        time_horizon=time_horizon,
        decision_interval=decision_interval,
        max_trips_per_drone=max_trips_per_drone,
        seed=seed,
        multi_depot_mode=multi_depot_mode,
        drones_per_depot=drones_per_depot,
        batteries_per_depot=batteries_per_depot
    )
    
    # Generate initial orders
    order_gen = OrderGenerator(seed=seed)
    initial_orders = order_gen.generate_initial_orders(num_initial_orders)
    
    print(f"\nConfiguration:")
    if multi_depot_mode:
        print(f"  Mode: Multi-Depot (多仓库模式)")
        print(f"  Depots: {NUM_DEPOTS}")
        print(f"  Drones per depot: {drones_per_depot}")
        print(f"  Batteries per depot: {batteries_per_depot}")
        print(f"  Total drones: {NUM_DEPOTS * drones_per_depot}")
        print(f"  Total batteries: {NUM_DEPOTS * batteries_per_depot}")
    else:
        print(f"  Mode: Single-Depot (单仓库模式)")
        print(f"  Drones: {num_drones}")
        print(f"  Batteries: {num_batteries}")
    print(f"  Initial orders: {num_initial_orders}")
    print(f"  Time horizon: {time_horizon} minutes")
    print(f"  Decision interval: {decision_interval} minutes")
    print(f"  Max trips per drone (M): {max_trips_per_drone}")
    print(f"  Dynamic orders: {dynamic_orders}")
    print(f"  Seed: {seed}")
    
    # Run simulation
    result = simulator.run(
        initial_orders=initial_orders,
        dynamic_orders=dynamic_orders,
        verbose=verbose
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total orders:     {result.total_orders}")
    print(f"Served orders:    {result.served_orders}")
    print(f"Service rate:     {result.service_rate:.1%}")
    print(f"Total trips:      {result.total_trips}")
    print(f"Total distance:   {result.total_distance:.1f} meters")
    print(f"Total cost:       {result.total_cost:.1f}")  # Algorithm 1 output
    print(f"Total lateness:   {result.total_lateness:.1f} minutes")
    print(f"Avg lateness:     {result.average_lateness:.1f} minutes")
    
    return result


def run_comparison(
    num_runs: int = 5,
    seed_base: int = 42,
    multi_depot_mode: bool = True  # 多仓库模式
):
    """
    Run comparison between CFA and Myopic policies
    支持多仓库场景
    
    CFA: Uses M parameter to limit trips per drone
    Myopic: M = infinity (no limit)
    """
    print("=" * 60)
    if multi_depot_mode:
        print(f"POLICY COMPARISON: CFA vs Myopic (Multi-Depot, {NUM_DEPOTS} depots)")
    else:
        print("POLICY COMPARISON: CFA vs Myopic")
    print("=" * 60)
    
    cfa_results = []
    myopic_results = []
    
    for i in range(num_runs):
        seed = seed_base + i
        print(f"\n--- Run {i+1}/{num_runs} (seed={seed}) ---")
        
        # CFA policy (M = 2)
        print("\nRunning CFA policy (M=2)...")
        simulator_cfa = Simulator(
            max_trips_per_drone=2,
            seed=seed,
            multi_depot_mode=multi_depot_mode
        )
        order_gen = OrderGenerator(seed=seed)
        initial_orders = order_gen.generate_initial_orders(10)
        result_cfa = simulator_cfa.run(
            initial_orders=initial_orders,
            dynamic_orders=True,
            verbose=False
        )
        cfa_results.append(result_cfa)
        
        # Myopic policy (M = large number)
        print("Running Myopic policy (M=100)...")
        simulator_myopic = Simulator(
            max_trips_per_drone=100,  # Effectively unlimited
            seed=seed,
            multi_depot_mode=multi_depot_mode
        )
        order_gen = OrderGenerator(seed=seed)
        initial_orders = order_gen.generate_initial_orders(10)
        result_myopic = simulator_myopic.run(
            initial_orders=initial_orders,
            dynamic_orders=True,
            verbose=False
        )
        myopic_results.append(result_myopic)
    
    # Aggregate results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    def avg(results, attr):
        return np.mean([getattr(r, attr) for r in results])
    
    print(f"\n{'Metric':<20} {'CFA (M=2)':<15} {'Myopic':<15}")
    print("-" * 50)
    print(f"{'Service Rate':<20} {avg(cfa_results, 'service_rate'):.1%}{'':<10} {avg(myopic_results, 'service_rate'):.1%}")
    print(f"{'Avg Lateness':<20} {avg(cfa_results, 'average_lateness'):.1f} min{'':<5} {avg(myopic_results, 'average_lateness'):.1f} min")
    print(f"{'Total Distance':<20} {avg(cfa_results, 'total_distance'):.0f} m{'':<6} {avg(myopic_results, 'total_distance'):.0f} m")
    print(f"{'Total Trips':<20} {avg(cfa_results, 'total_trips'):.1f}{'':<10} {avg(myopic_results, 'total_trips'):.1f}")


def main():
    """Main entry point with command line arguments"""
    parser = argparse.ArgumentParser(
        description="DRPUDEC: Dynamic Drone Routing Simulation"
    )
    
    parser.add_argument(
        "--mode", type=str, default="single",
        choices=["single", "compare"],
        help="Simulation mode: single run or policy comparison"
    )
    parser.add_argument(
        "--drones", type=int, default=NUM_DRONES,
        help="Number of drones"
    )
    parser.add_argument(
        "--batteries", type=int, default=NUM_BATTERIES,
        help="Number of batteries"
    )
    parser.add_argument(
        "--orders", type=int, default=10,
        help="Number of initial orders"
    )
    parser.add_argument(
        "--horizon", type=float, default=T_HORIZON,
        help="Time horizon in minutes"
    )
    parser.add_argument(
        "--interval", type=float, default=PSI,
        help="Decision interval in minutes"
    )
    parser.add_argument(
        "--max-trips", type=int, default=M_MAX_TRIPS,
        help="Max trips per drone per epoch (M parameter)"
    )
    parser.add_argument(
        "--no-dynamic", action="store_true",
        help="Disable dynamic order generation"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output"
    )
    parser.add_argument(
        "--runs", type=int, default=5,
        help="Number of runs for comparison mode"
    )
    parser.add_argument(
        "--multi-depot", action="store_true", default=True,
        help="Enable multi-depot mode (多仓库模式)"
    )
    parser.add_argument(
        "--single-depot", action="store_true",
        help="Use single-depot mode (单仓库模式)"
    )
    parser.add_argument(
        "--drones-per-depot", type=int, default=DRONES_PER_DEPOT,
        help="Number of drones per depot (每个仓库的无人机数)"
    )
    parser.add_argument(
        "--batteries-per-depot", type=int, default=BATTERIES_PER_DEPOT,
        help="Number of batteries per depot (每个仓库的电池数)"
    )
    
    args = parser.parse_args()
    
    # 确定是否使用多仓库模式
    multi_depot = not args.single_depot
    
    if args.mode == "single":
        run_single_simulation(
            num_drones=args.drones,
            num_batteries=args.batteries,
            num_initial_orders=args.orders,
            time_horizon=args.horizon,
            decision_interval=args.interval,
            max_trips_per_drone=args.max_trips,
            dynamic_orders=not args.no_dynamic,
            seed=args.seed,
            verbose=not args.quiet,
            multi_depot_mode=multi_depot,
            drones_per_depot=args.drones_per_depot,
            batteries_per_depot=args.batteries_per_depot
        )
    elif args.mode == "compare":
        run_comparison(
            num_runs=args.runs,
            seed_base=args.seed,
            multi_depot_mode=multi_depot
        )


if __name__ == "__main__":
    main()
