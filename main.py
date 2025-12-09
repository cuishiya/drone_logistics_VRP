"""
DRPUDEC: 动态无人机路径规划问题（考虑不确定需求和能耗）
主程序入口

基于论文: Chagas et al. (2025) - "A dynamic drone routing problem with uncertain 
demand and energy consumption"

实现功能:
- 基于MDP的动态决策
- 成本函数近似(CFA)策略
- 机会约束能耗模型
- 反向标号设置行程构建
- 基于MILP的行程选择(SPBP + DTSSP)
- 任务分配问题(TAP)
- 电池管理（平衡老化）
- 多仓库调度
"""

import numpy as np
from typing import Optional

from simulation.simulator import Simulator
from simulation.order_generator import OrderGenerator
from config import (
    T_HORIZON, PSI, M_MAX_TRIPS,
    NUM_DEPOTS, DRONES_PER_DEPOT, BATTERIES_PER_DEPOT, DEFAULT_DEPOT_INFOS
)
from visualization import Visualizer, plot_all


def run_simulation(
    num_initial_orders: int = 10,
    time_horizon: float = T_HORIZON,
    decision_interval: float = PSI,
    max_trips_per_drone: int = M_MAX_TRIPS,
    dynamic_orders: bool = True,
    seed: Optional[int] = None,
    verbose: bool = True,
    drones_per_depot: int = DRONES_PER_DEPOT,
    batteries_per_depot: int = BATTERIES_PER_DEPOT,
    visualize: bool = True,
    save_plots: str = None
):
    """
    运行多仓库仿真
    
    Args:
        num_initial_orders: 初始订单数量
        time_horizon: 仿真时长（分钟）
        decision_interval: 决策周期间隔（分钟）
        max_trips_per_drone: 每架无人机每周期最大行程数(M参数)
        dynamic_orders: 是否动态生成订单
        seed: 随机种子
        verbose: 是否输出详细信息
        drones_per_depot: 每个仓库的无人机数量
        batteries_per_depot: 每个仓库的电池数量
        visualize: 是否生成可视化图表
        save_plots: 图表保存目录
    """
    print("=" * 60)
    print("DRPUDEC 多仓库无人机路径规划仿真")
    print(f"仓库数量: {NUM_DEPOTS}")
    print("=" * 60)
    
    # 创建仿真器
    simulator = Simulator(
        time_horizon=time_horizon,
        decision_interval=decision_interval,
        max_trips_per_drone=max_trips_per_drone,
        seed=seed,
        drones_per_depot=drones_per_depot,
        batteries_per_depot=batteries_per_depot
    )
    
    # 生成初始订单
    order_gen = OrderGenerator(seed=seed)
    initial_orders = order_gen.generate_initial_orders(num_initial_orders)
    
    print(f"\n配置信息:")
    print(f"  仓库数量: {NUM_DEPOTS}")
    print(f"  每仓库无人机: {drones_per_depot}")
    print(f"  每仓库电池: {batteries_per_depot}")
    print(f"  无人机总数: {NUM_DEPOTS * drones_per_depot}")
    print(f"  电池总数: {NUM_DEPOTS * batteries_per_depot}")
    print(f"  初始订单: {num_initial_orders}")
    print(f"  仿真时长: {time_horizon} 分钟")
    print(f"  决策周期: {decision_interval} 分钟")
    print(f"  每无人机最大行程数(M): {max_trips_per_drone}")
    print(f"  动态订单: {dynamic_orders}")
    print(f"  随机种子: {seed}")
    
    # 运行仿真
    result = simulator.run(
        initial_orders=initial_orders,
        dynamic_orders=dynamic_orders,
        verbose=verbose
    )
    
    # 输出结果摘要
    print("\n" + "=" * 60)
    print("仿真结果")
    print("=" * 60)
    print(f"总订单数:     {result.total_orders}")
    print(f"已服务:       {result.served_orders}")
    print(f"服务率:       {result.service_rate:.1%}")
    print(f"总行程数:     {result.total_trips}")
    print(f"总飞行距离:   {result.total_distance:.1f} 米")
    print(f"总成本:       {result.total_cost:.1f}")
    print(f"总延迟:       {result.total_lateness:.1f} 分钟")
    print(f"平均延迟:     {result.average_lateness:.1f} 分钟")
    
    # 可视化
    if visualize:
        print("\n生成可视化图表...")
        
        from models.data_structures import Depot
        
        # 准备统计数据
        stats = {
            'total_orders': result.total_orders,
            'served_orders': result.served_orders,
            'service_rate': result.service_rate,
            'total_lateness': result.total_lateness,
            'avg_lateness': result.average_lateness,
            'max_lateness': getattr(result, 'max_lateness', result.average_lateness * 2),
            'total_distance': result.total_distance,
            'total_energy': getattr(result, 'total_energy', result.total_distance * 0.01),
        }
        
        # 获取仓库位置（DEFAULT_DEPOT_INFOS 是 DepotInfo 对象列表）
        depots = []
        for info in DEFAULT_DEPOT_INFOS:
            depot = Depot(
                id=info.id,
                location=info.location.copy(),
                name=info.name
            )
            depots.append(depot)
        
        # 获取行程和订单数据
        trips = getattr(result, 'trips', [])
        orders = getattr(result, 'orders', initial_orders)
        
        # 绘制图表
        plot_all(
            trips=trips,
            orders=orders,
            depots=depots,
            stats=stats,
            horizon=time_horizon,
            save_dir=save_plots,
            show=True
        )
    
    return result


if __name__ == "__main__":
    # 直接运行仿真，所有参数在 config.py 中配置
    run_simulation()
