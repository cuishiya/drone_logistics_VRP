"""
可视化绘图模块
提供行程路线、订单分布、无人机调度甘特图和仿真统计图的绘制功能

主要功能：
- plot_trips: 绘制行程路线图（仓库→客户→仓库）
- plot_orders: 绘制订单分布图（按紧急度着色）
- plot_drone_schedule: 绘制无人机调度甘特图
- plot_simulation_stats: 绘制仿真结果统计图
- plot_all: 一次性绘制所有图表
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.data_structures import Order, Trip, Drone, Depot, DroneStatus


# ============================================================================
# 颜色配置
# ============================================================================

# 行程路线颜色（最多支持20条不同颜色的路线）
TRIP_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
]

# 紧急度颜色映射（从绿色到红色）
URGENCY_CMAP = plt.cm.RdYlGn_r  # 反转：绿色=不紧急，红色=紧急

# 无人机状态颜色
DRONE_STATUS_COLORS = {
    'flying': '#2ca02c',      # 绿色 - 飞行中
    'delivering': '#1f77b4',  # 蓝色 - 配送中
    'returning': '#ff7f0e',   # 橙色 - 返回中
    'charging': '#d62728',    # 红色 - 充电中
    'idle': '#7f7f7f',        # 灰色 - 空闲
    'waiting': '#9467bd',     # 紫色 - 等待中
}


# ============================================================================
# 行程路线图
# ============================================================================

def plot_trips(
    trips: List[Trip],
    depot_location: np.ndarray = None,
    depots: List[Depot] = None,
    orders: List[Order] = None,
    title: str = "行程路线图",
    figsize: Tuple[int, int] = (12, 10),
    show_legend: bool = True,
    show_order_ids: bool = True,
    save_path: str = None,
    ax: plt.Axes = None
) -> plt.Figure:
    """
    绘制行程路线图
    
    Args:
        trips: 行程列表
        depot_location: 单仓库位置（单仓库模式）
        depots: 仓库列表（多仓库模式）
        orders: 额外的订单列表（可选，用于显示未分配订单）
        title: 图表标题
        figsize: 图表大小
        show_legend: 是否显示图例
        show_order_ids: 是否显示订单ID
        save_path: 保存路径（可选）
        ax: 现有的Axes对象（可选）
        
    Returns:
        matplotlib Figure对象
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure
    
    # 收集所有仓库位置
    depot_locations = []
    if depots:
        for depot in depots:
            depot_locations.append((depot.location, depot.id, depot.name))
    elif depot_location is not None:
        depot_locations.append((depot_location, 0, "仓库"))
    else:
        # 默认原点
        depot_locations.append((np.array([0.0, 0.0]), 0, "仓库"))
    
    # 绘制仓库
    for loc, depot_id, name in depot_locations:
        ax.scatter(loc[0], loc[1], c='red', s=200, marker='s', 
                   zorder=10, edgecolors='black', linewidths=2)
        ax.annotate(f'{name}', (loc[0], loc[1]), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold', color='red')
    
    # 绘制每条行程
    for i, trip in enumerate(trips):
        color = TRIP_COLORS[i % len(TRIP_COLORS)]
        
        # 获取行程的仓库位置
        if trip.depot_location is not None:
            trip_depot = trip.depot_location
        elif depot_location is not None:
            trip_depot = depot_location
        else:
            trip_depot = np.array([0.0, 0.0])
        
        # 构建路线点
        route_points = [trip_depot]
        for order in trip.orders:
            route_points.append(order.location)
        route_points.append(trip_depot)
        
        # 绘制路线
        for j in range(len(route_points) - 1):
            start = route_points[j]
            end = route_points[j + 1]
            
            # 绘制箭头
            ax.annotate('', xy=(end[0], end[1]), xytext=(start[0], start[1]),
                       arrowprops=dict(arrowstyle='->', color=color, lw=1.5, 
                                      connectionstyle='arc3,rad=0.1'))
        
        # 绘制订单点
        for order in trip.orders:
            ax.scatter(order.location[0], order.location[1], 
                      c=color, s=100, marker='o', zorder=5,
                      edgecolors='black', linewidths=1)
            if show_order_ids:
                ax.annotate(f'{order.id}', (order.location[0], order.location[1]),
                           xytext=(3, 3), textcoords='offset points',
                           fontsize=8, color=color)
    
    # 绘制未分配的订单（如果有）
    if orders:
        assigned_ids = set()
        for trip in trips:
            for order in trip.orders:
                assigned_ids.add(order.id)
        
        unassigned = [o for o in orders if o.id not in assigned_ids]
        if unassigned:
            for order in unassigned:
                ax.scatter(order.location[0], order.location[1],
                          c='gray', s=80, marker='x', zorder=4, alpha=0.6)
                if show_order_ids:
                    ax.annotate(f'{order.id}', (order.location[0], order.location[1]),
                               xytext=(3, 3), textcoords='offset points',
                               fontsize=7, color='gray')
    
    # 图例
    if show_legend and trips:
        legend_elements = []
        for i, trip in enumerate(trips[:10]):  # 最多显示10条
            color = TRIP_COLORS[i % len(TRIP_COLORS)]
            label = f'行程 {trip.id} ({len(trip.orders)}单)'
            legend_elements.append(mpatches.Patch(color=color, label=label))
        
        # 添加仓库和未分配订单图例
        legend_elements.append(plt.scatter([], [], c='red', s=100, marker='s', 
                                           label='仓库'))
        if orders:
            legend_elements.append(plt.scatter([], [], c='gray', s=60, marker='x',
                                              label='未分配订单'))
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    ax.set_xlabel('X 坐标 (米)', fontsize=10)
    ax.set_ylabel('Y 坐标 (米)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    return fig


# ============================================================================
# 订单分布图
# ============================================================================

def plot_orders(
    orders: List[Order],
    depot_location: np.ndarray = None,
    depots: List[Depot] = None,
    current_time: float = 0.0,
    title: str = "订单分布图（按紧急度着色）",
    figsize: Tuple[int, int] = (12, 10),
    show_deadlines: bool = False,
    save_path: str = None,
    ax: plt.Axes = None
) -> plt.Figure:
    """
    绘制订单分布图，按紧急度着色
    
    Args:
        orders: 订单列表
        depot_location: 单仓库位置
        depots: 仓库列表（多仓库模式）
        current_time: 当前时间（用于计算紧急度）
        title: 图表标题
        figsize: 图表大小
        show_deadlines: 是否显示截止时间
        save_path: 保存路径
        ax: 现有的Axes对象
        
    Returns:
        matplotlib Figure对象
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure
    
    if not orders:
        ax.text(0.5, 0.5, '无订单数据', ha='center', va='center', 
                transform=ax.transAxes, fontsize=14)
        return fig
    
    # 计算紧急度
    urgencies = [order.urgency(current_time) for order in orders]
    min_urg, max_urg = min(urgencies), max(urgencies)
    
    # 归一化紧急度用于着色
    if max_urg > min_urg:
        norm_urgencies = [(u - min_urg) / (max_urg - min_urg) for u in urgencies]
    else:
        norm_urgencies = [0.5] * len(urgencies)
    
    # 绘制仓库
    if depots:
        for depot in depots:
            ax.scatter(depot.location[0], depot.location[1], c='red', s=200, 
                      marker='s', zorder=10, edgecolors='black', linewidths=2)
            ax.annotate(depot.name, (depot.location[0], depot.location[1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, fontweight='bold', color='red')
    elif depot_location is not None:
        ax.scatter(depot_location[0], depot_location[1], c='red', s=200, 
                  marker='s', zorder=10, edgecolors='black', linewidths=2)
        ax.annotate('仓库', (depot_location[0], depot_location[1]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold', color='red')
    
    # 绘制订单点
    for order, norm_urg in zip(orders, norm_urgencies):
        color = URGENCY_CMAP(norm_urg)
        size = 50 + norm_urg * 80  # 更紧急的订单更大
        
        ax.scatter(order.location[0], order.location[1], c=[color], s=size,
                  marker='o', zorder=5, edgecolors='black', linewidths=0.5,
                  alpha=0.8)
        
        # 显示订单ID
        ax.annotate(f'{order.id}', (order.location[0], order.location[1]),
                   xytext=(3, 3), textcoords='offset points', fontsize=7)
        
        # 显示截止时间
        if show_deadlines:
            deadline_str = f'{order.deadline:.0f}min'
            ax.annotate(deadline_str, (order.location[0], order.location[1]),
                       xytext=(3, -10), textcoords='offset points', 
                       fontsize=6, color='gray')
    
    # 颜色条
    sm = plt.cm.ScalarMappable(cmap=URGENCY_CMAP, 
                                norm=plt.Normalize(vmin=min_urg, vmax=max_urg))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('紧急度（越高越紧急）', fontsize=10)
    
    ax.set_xlabel('X 坐标 (米)', fontsize=10)
    ax.set_ylabel('Y 坐标 (米)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # 添加统计信息
    stats_text = f'订单总数: {len(orders)}\n'
    stats_text += f'平均紧急度: {np.mean(urgencies):.2f}\n'
    stats_text += f'最大紧急度: {max_urg:.2f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    return fig


# ============================================================================
# 无人机调度甘特图
# ============================================================================

@dataclass
class DroneEvent:
    """无人机事件记录"""
    drone_id: int
    start_time: float
    end_time: float
    event_type: str  # 'flying', 'delivering', 'charging', 'idle', 'waiting'
    trip_id: Optional[int] = None
    description: str = ""


def plot_drone_schedule(
    events: List[DroneEvent] = None,
    drones: List[Drone] = None,
    trips: List[Trip] = None,
    horizon: float = 480.0,
    title: str = "无人机调度甘特图",
    figsize: Tuple[int, int] = (14, 8),
    save_path: str = None,
    ax: plt.Axes = None
) -> plt.Figure:
    """
    绘制无人机调度甘特图
    
    Args:
        events: 无人机事件列表（如果提供，直接使用）
        drones: 无人机列表（用于从无人机状态推断事件）
        trips: 行程列表（用于从行程推断事件）
        horizon: 仿真时长（分钟）
        title: 图表标题
        figsize: 图表大小
        save_path: 保存路径
        ax: 现有的Axes对象
        
    Returns:
        matplotlib Figure对象
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure
    
    # 如果没有提供事件，从trips推断
    if events is None and trips:
        events = _infer_events_from_trips(trips, horizon)
    
    if not events:
        ax.text(0.5, 0.5, '无调度数据', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        return fig
    
    # 获取所有无人机ID
    drone_ids = sorted(set(e.drone_id for e in events))
    drone_y_map = {did: i for i, did in enumerate(drone_ids)}
    
    # 绘制每个事件
    for event in events:
        y = drone_y_map[event.drone_id]
        width = event.end_time - event.start_time
        color = DRONE_STATUS_COLORS.get(event.event_type, '#7f7f7f')
        
        # 绘制条形
        ax.barh(y, width, left=event.start_time, height=0.6, 
                color=color, edgecolor='black', linewidth=0.5, alpha=0.8)
        
        # 如果条形足够宽，显示行程ID
        if width > horizon * 0.05 and event.trip_id is not None:
            ax.text(event.start_time + width/2, y, f'T{event.trip_id}',
                   ha='center', va='center', fontsize=8, color='white',
                   fontweight='bold')
    
    # 设置Y轴
    ax.set_yticks(range(len(drone_ids)))
    ax.set_yticklabels([f'无人机 {did}' for did in drone_ids])
    
    # 设置X轴
    ax.set_xlim(0, horizon)
    ax.set_xlabel('时间 (分钟)', fontsize=10)
    ax.set_ylabel('无人机', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # 图例
    legend_elements = []
    for status, color in DRONE_STATUS_COLORS.items():
        status_cn = {
            'flying': '飞行中',
            'delivering': '配送中', 
            'returning': '返回中',
            'charging': '充电中',
            'idle': '空闲',
            'waiting': '等待中'
        }.get(status, status)
        legend_elements.append(mpatches.Patch(color=color, label=status_cn))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    return fig


def _infer_events_from_trips(trips: List[Trip], horizon: float) -> List[DroneEvent]:
    """
    从行程列表推断无人机事件
    
    包含更详细的状态：出发飞行、配送服务、返回飞行、空闲等待
    """
    from config import DRONE_SPEED_MEAN, SERVICE_TIME
    
    events = []
    speed = DRONE_SPEED_MEAN * 60  # 转换为 m/min
    
    # 按无人机分组
    drone_trips: Dict[int, List[Trip]] = {}
    for trip in trips:
        if trip.drone_id is not None:
            if trip.drone_id not in drone_trips:
                drone_trips[trip.drone_id] = []
            drone_trips[trip.drone_id].append(trip)
    
    # 为每架无人机生成详细事件
    for drone_id, drone_trip_list in drone_trips.items():
        # 按开始时间排序
        sorted_trips = sorted(drone_trip_list, 
                             key=lambda t: t.planned_start_time or 0)
        
        current_time = 0.0
        for trip in sorted_trips:
            start = trip.planned_start_time or current_time
            
            # 空闲等待时间
            if start > current_time + 0.1:
                events.append(DroneEvent(
                    drone_id=drone_id,
                    start_time=current_time,
                    end_time=start,
                    event_type='idle'
                ))
            
            # 获取仓库位置
            depot_loc = trip.depot_location if trip.depot_location is not None else np.array([0.0, 0.0])
            
            # 计算各阶段时间
            current_pos = depot_loc
            trip_time = start
            
            for i, order in enumerate(trip.orders):
                # 飞行到客户位置
                fly_dist = np.linalg.norm(order.location - current_pos)
                fly_time = fly_dist / speed if speed > 0 else 0
                
                if fly_time > 0.1:
                    events.append(DroneEvent(
                        drone_id=drone_id,
                        start_time=trip_time,
                        end_time=trip_time + fly_time,
                        event_type='flying',
                        trip_id=trip.id
                    ))
                    trip_time += fly_time
                
                # 服务时间
                service_duration = order.service_time if hasattr(order, 'service_time') else SERVICE_TIME
                events.append(DroneEvent(
                    drone_id=drone_id,
                    start_time=trip_time,
                    end_time=trip_time + service_duration,
                    event_type='delivering',
                    trip_id=trip.id
                ))
                trip_time += service_duration
                current_pos = order.location
            
            # 返回仓库
            return_dist = np.linalg.norm(depot_loc - current_pos)
            return_time = return_dist / speed if speed > 0 else 0
            
            if return_time > 0.1:
                events.append(DroneEvent(
                    drone_id=drone_id,
                    start_time=trip_time,
                    end_time=trip_time + return_time,
                    event_type='returning',
                    trip_id=trip.id
                ))
                trip_time += return_time
            
            current_time = trip_time
        
        # 剩余时间为空闲
        if current_time < horizon:
            events.append(DroneEvent(
                drone_id=drone_id,
                start_time=current_time,
                end_time=horizon,
                event_type='idle'
            ))
    
    return events


# ============================================================================
# 综合绘图函数
# ============================================================================

def plot_all(
    trips: List[Trip] = None,
    orders: List[Order] = None,
    depot_location: np.ndarray = None,
    depots: List[Depot] = None,
    stats: Dict[str, Any] = None,
    events: List[DroneEvent] = None,
    current_time: float = 0.0,
    horizon: float = 480.0,
    save_dir: str = None,
    show: bool = True
) -> Dict[str, plt.Figure]:
    """
    一次性绘制所有可视化图表
    
    Args:
        trips: 行程列表
        orders: 订单列表
        depot_location: 单仓库位置
        depots: 仓库列表（多仓库模式）
        stats: 仿真统计数据
        events: 无人机事件列表
        current_time: 当前时间
        horizon: 仿真时长
        save_dir: 保存目录（可选）
        show: 是否显示图表
        
    Returns:
        包含所有Figure对象的字典
    """
    figures = {}
    
    # 创建保存目录
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # 1. 行程路线图
    if trips:
        save_path = os.path.join(save_dir, 'trip_routes.png') if save_dir else None
        figures['trips'] = plot_trips(
            trips=trips,
            depot_location=depot_location,
            depots=depots,
            orders=orders,
            save_path=save_path
        )
    
    # 2. 订单分布图
    if orders:
        save_path = os.path.join(save_dir, 'order_distribution.png') if save_dir else None
        figures['orders'] = plot_orders(
            orders=orders,
            depot_location=depot_location,
            depots=depots,
            current_time=current_time,
            save_path=save_path
        )
    
    # 3. 无人机调度甘特图
    if events or trips:
        save_path = os.path.join(save_dir, 'drone_schedule.png') if save_dir else None
        figures['schedule'] = plot_drone_schedule(
            events=events,
            trips=trips,
            horizon=horizon,
            save_path=save_path
        )
    
    if show:
        plt.show()
    
    return figures


# ============================================================================
# Visualizer 类（便于集成到仿真器中）
# ============================================================================

class Visualizer:
    """
    可视化器类，用于在仿真过程中收集数据并生成图表
    
    使用方法：
        viz = Visualizer()
        viz.record_trip(trip)
        viz.record_order(order)
        viz.record_drone_event(event)
        viz.plot_all()
    """
    
    def __init__(
        self,
        depot_location: np.ndarray = None,
        depots: List[Depot] = None,
        horizon: float = 480.0
    ):
        """
        初始化可视化器
        
        Args:
            depot_location: 单仓库位置
            depots: 仓库列表（多仓库模式）
            horizon: 仿真时长
        """
        self.depot_location = depot_location
        self.depots = depots
        self.horizon = horizon
        
        self.trips: List[Trip] = []
        self.orders: List[Order] = []
        self.events: List[DroneEvent] = []
        self.stats: Dict[str, Any] = {}
        
        self._lateness_list: List[float] = []
    
    def record_trip(self, trip: Trip):
        """记录一个行程"""
        self.trips.append(trip)
    
    def record_trips(self, trips: List[Trip]):
        """记录多个行程"""
        self.trips.extend(trips)
    
    def record_order(self, order: Order):
        """记录一个订单"""
        self.orders.append(order)
    
    def record_orders(self, orders: List[Order]):
        """记录多个订单"""
        self.orders.extend(orders)
    
    def record_drone_event(self, event: DroneEvent):
        """记录一个无人机事件"""
        self.events.append(event)
    
    def record_lateness(self, lateness: float):
        """记录一个延迟值"""
        self._lateness_list.append(lateness)
    
    def set_stats(self, stats: Dict[str, Any]):
        """设置仿真统计数据"""
        self.stats = stats
        if self._lateness_list:
            self.stats['lateness_distribution'] = self._lateness_list
    
    def plot_trips(self, **kwargs) -> plt.Figure:
        """绘制行程路线图"""
        return plot_trips(
            trips=self.trips,
            depot_location=self.depot_location,
            depots=self.depots,
            orders=self.orders,
            **kwargs
        )
    
    def plot_orders(self, current_time: float = 0.0, **kwargs) -> plt.Figure:
        """绘制订单分布图"""
        return plot_orders(
            orders=self.orders,
            depot_location=self.depot_location,
            depots=self.depots,
            current_time=current_time,
            **kwargs
        )
    
    def plot_schedule(self, **kwargs) -> plt.Figure:
        """绘制无人机调度甘特图"""
        return plot_drone_schedule(
            events=self.events if self.events else None,
            trips=self.trips,
            horizon=self.horizon,
            **kwargs
        )
    
    def plot_stats(self, **kwargs) -> plt.Figure:
        """绘制仿真统计图"""
        if self._lateness_list and 'lateness_distribution' not in self.stats:
            self.stats['lateness_distribution'] = self._lateness_list
        return plot_simulation_stats(stats=self.stats, **kwargs)
    
    def plot_all(self, save_dir: str = None, show: bool = True) -> Dict[str, plt.Figure]:
        """绘制所有图表"""
        if self._lateness_list and 'lateness_distribution' not in self.stats:
            self.stats['lateness_distribution'] = self._lateness_list
        
        return plot_all(
            trips=self.trips,
            orders=self.orders,
            depot_location=self.depot_location,
            depots=self.depots,
            stats=self.stats if self.stats else None,
            events=self.events if self.events else None,
            horizon=self.horizon,
            save_dir=save_dir,
            show=show
        )
    
    def clear(self):
        """清空所有记录"""
        self.trips.clear()
        self.orders.clear()
        self.events.clear()
        self.stats.clear()
        self._lateness_list.clear()


