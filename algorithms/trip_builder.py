"""
行程构造算法（论文中的算法3和算法4）
使用反向标号（Backward Label-setting）的启发式方法生成可行行程

核心思想：
- 从仓库反向构建行程，以正确处理与载重相关的能耗
- 使用基于紧急度的插入优先级
- 应用帕累托支配规则剪枝劣质行程
- 通过机会约束保证能耗可行性
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
    反向标号算法中的标签结构
    
    表示从仓库反向构建的一个部分行程
    
    属性:
        orders: 此部分行程中的订单（按访问顺序的反向存储）
        total_weight: 当前总载重
        expected_energy: 预期能耗
        energy_std: 能耗标准差
        total_distance: 总路程距离
        last_location: 最后添加的客户位置（将被最先访问）
        visited_ids: 已访问客户ID集合
        earliest_start: 最早可行的出发时间
        is_dominated: 是否被其他标签支配
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
        """创建当前标签的深拷贝"""
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
    使用反向标号启发式构建可行行程
    支持多仓库场景：每个仓库独立构建路径
    
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
        max_labels: int = 100,
        depot_id: int = None,  # 多仓库：仓库ID
        depot_location: np.ndarray = None  # 多仓库：仓库位置
    ):
        self.depot = depot
        self.energy_model = energy_model
        self.max_payload = max_payload
        self.service_time = service_time
        self.speed = speed
        self.max_customers_per_trip = max_customers_per_trip
        self.max_labels = max_labels
        self._trip_counter = 0
        
        # 多仓库支持
        self.depot_id = depot_id if depot_id is not None else getattr(depot, 'id', 0)
        self.depot_location = depot_location if depot_location is not None else depot.location
    
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
            expected_energy=0.0,
            depot_id=self.depot_id,  # 多仓库：设置出发仓库
            depot_location=self.depot_location  # 多仓库：设置仓库位置
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
        """获取下一个唯一的行程ID"""
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
        算法4: select_vertices_to_insert
        
        输入: 行程 γ；Σ 为集合 V 的最大大小
        输出: 顶点集合 V
        
        从行程 γ 的第一个顶点 i_r 出发，在其邻域中选出最多 Σ 个可插入的订单。
        """
        # Line 1: 令 i_r ∈ γ 为行程 γ 中在仓库 i_0 之后访问的第一个位置
        # 在反向构造中，trip.orders[0] 就是仓库之后访问的第一个客户
        if trip.orders:
            i_r = trip.orders[0]
            i_r_location = i_r.location
        else:
            # 空行程，使用仓库位置作为参考点
            i_r = None
            i_r_location = self.depot.location
        
        # 获取当前行程中已有的订单ID
        trip_order_ids = {o.id for o in trip.orders}
        current_weight = sum(o.weight for o in trip.orders)
        
        # 构造邻居集合 I^d(i_r) 和 I^l(i_r)
        # I^d: 按与 i_r 的距离排序
        # I^l: 按紧急度（截止时间）排序
        available_orders = [o for o in all_orders if o.id not in trip_order_ids]
        
        if strategy == "distance":
            # I^d(i_r): 按距离排序的邻居集合
            neighbor_set = sorted(
                available_orders,
                key=lambda o: euclidean_distance(o.location, i_r_location)
            )
        else:  # "urgency" or default
            # I^l(i_r): 按紧急度排序的邻居集合（截止时间更早更优先）
            neighbor_set = sorted(
                available_orders,
                key=lambda o: o.urgency(current_time)
            )
        
        # Line 2: V ← ∅
        V: List[Order] = []
        
        # Line 3: while |V| < Σ 且 i_r 还有邻居待探索时执行
        while len(V) < sigma and neighbor_set:
            # Line 4: i_p ← select_first_neighbor(I^d(i_r)) 或 select_first_neighbor(I^l(i_r))
            # 根据排序策略，从对应邻居集合中选择第一个元素
            i_p = neighbor_set[0]
            
            # Line 5: 如果 i_p 插入 γ 后仍满足不等式 (6) 和 (7)
            # 检查约束 (6): 载重约束 Q - sum(q) >= q_{i_p}
            weight_feasible = (current_weight + i_p.weight <= self.max_payload)
            
            # 检查约束 (7): 能耗机会约束
            # 同时检查最大客户数约束
            energy_feasible = False
            if weight_feasible and len(trip.orders) < self.max_customers_per_trip:
                test_orders = [i_p] + trip.orders  # 反向插入
                energy_result = self.energy_model.compute_trip_energy_from_orders(
                    self.depot.location,
                    test_orders
                )
                energy_feasible = energy_result.is_feasible
            
            if weight_feasible and energy_feasible:
                # Line 6: V ← V ∪ {i_p}
                V.append(i_p)
            # Line 7: end
            
            # Line 8: I^d(i_r) ← I^d(i_r) \ i_p 或 I^l(i_r) ← I^l(i_r) \ i_p
            # 从相应的邻居集合中移除 i_p
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
        通过在行程前端插入订单（反向插入）创建新行程
        
        如果插入后不可行，则返回 None
        """
        # 使用反向插入创建新的订单列表
        new_orders = [order] + trip.orders
        
        # 检查能耗可行性
        energy_result = self.energy_model.compute_trip_energy_from_orders(
            self.depot.location,
            new_orders
        )
        
        if not energy_result.is_feasible:
            return None
        
        # 计算新行程的总距离
        route = [self.depot_location] + [o.location for o in new_orders] + [self.depot_location]
        new_distance = compute_route_distance(route)
        
        # 创建新的行程对象
        new_trip = Trip(
            id=self._get_next_trip_id(),
            orders=new_orders,
            expected_energy=energy_result.expected_energy,
            energy_std=energy_result.energy_std,
            total_distance=new_distance,
            total_weight=sum(o.weight for o in new_orders),
            depot_id=self.depot_id,  # 多仓库：设置出发仓库
            depot_location=self.depot_location  # 多仓库：设置仓库位置
        )
        
        return new_trip
    
    def _is_dominated_by_any(self, trip: Trip, trip_set: List[Trip]) -> bool:
        """检查该行程是否被集合中的其他行程支配"""
        trip_customers = trip.get_customer_ids()
        
        for other in trip_set:
            other_customers = other.get_customer_ids()
            
            # 检查 other 是否支配 trip
            # 支配条件：客户数不小于且成本（距离、能耗）不大于
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
        """当标签数量超过上限时选取质量最好的若干行程"""
        # 评分规则：客户更多、能耗更低、距离更短
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
        """根据插入优先级对订单排序"""
        if strategy == "urgency":
            # 按紧急度排序（升序 = 最紧急优先）
            return sorted(orders, key=lambda o: o.urgency(current_time))
        elif strategy == "distance":
            # 按与仓库的距离排序（升序）
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
        通过在前端添加订单尝试扩展一个标签
        
        这是反向扩展：新增的客户会在标签中已有客户之前被访问。
        
        如果扩展不可行，则返回 None。
        """
        # 检查是否已访问过
        if order.id in label.visited_ids:
            return None
        
        # 检查载重约束
        new_weight = label.total_weight + order.weight
        if new_weight > self.max_payload:
            return None
        
        # 检查最大客户数
        if len(label.orders) >= self.max_customers_per_trip:
            return None
        
        # Calculate new distance
        if label.last_location is not None:
            # 新订单到先前第一个客户的距离
            if len(label.orders) > 0:
                dist_to_next = euclidean_distance(order.location, label.orders[0].location)
            else:
                # 第一个客户，与仓库之间的距离（返程段）
                dist_to_next = euclidean_distance(order.location, self.depot.location)
            
            # 仓库到新订单的距离
            dist_from_depot = euclidean_distance(self.depot.location, order.location)
        else:
            dist_to_next = 0
            dist_from_depot = euclidean_distance(self.depot.location, order.location)
        
        # 构建新路线用于能耗计算
        new_orders = [order] + label.orders
        
        # 在机会约束下检查能耗可行性
        energy_result = self.energy_model.compute_trip_energy_from_orders(
            self.depot.location,
            new_orders
        )
        
        if not energy_result.is_feasible:
            return None
        
        # 计算新的总距离
        new_distance = compute_route_distance(
            [self.depot.location] + [o.location for o in new_orders] + [self.depot.location]
        )
        
        # 计算最早出发时间（必须留出足够时间满足截止时间）
        travel_time = new_distance / self.speed / 60.0  # 单位：分钟
        service_time = len(new_orders) * self.service_time
        total_trip_time = travel_time + service_time
        
        # 根据最紧急订单的截止时间计算最早出发时间
        min_deadline = min(o.deadline for o in new_orders)
        earliest_start = max(current_time, min_deadline - total_trip_time)
        
        # 构造扩展后的标签
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
        应用帕累托支配规则（算法4）
        
        标签 L1 支配 L2 的条件：
        - L1 服务的客户数不少于 L2
        - L1 的能耗不高于 L2
        - L1 的距离不长于 L2
        """
        if len(labels) <= 1:
            return labels
        
        # 标记被支配的标签
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
        """检查 l1 是否在帕累托意义上支配 l2"""
        # l1 在所有指标上都不能比 l2 更差
        same_customers = l1.visited_ids == l2.visited_ids
        more_customers = l1.visited_ids.issuperset(l2.visited_ids) and len(l1.visited_ids) > len(l2.visited_ids)
        
        if not (same_customers or more_customers):
            return False
        
        # l1 的能耗和距离必须不高于 l2
        better_energy = l1.expected_energy <= l2.expected_energy
        better_distance = l1.total_distance <= l2.total_distance
        
        # 至少在一个指标上要严格更好
        strictly_better = (
            l1.expected_energy < l2.expected_energy or
            l1.total_distance < l2.total_distance or
            more_customers
        )
        
        return better_energy and better_distance and strictly_better
    
    def _select_best_labels(self, labels: List[Label]) -> List[Label]:
        """当标签数量超过上限时选取评分最优的若干标签"""
        # 评分规则：客户更多、能耗更低、距离更短
        def score(label: Label) -> Tuple[int, float, float]:
            return (
                -len(label.orders),  # More customers = better (negative for ascending sort)
                label.expected_energy,
                label.total_distance
            )
        
        sorted_labels = sorted(labels, key=score)
        return sorted_labels[:self.max_labels]
    
    def _label_to_trip(self, label: Label) -> Optional[Trip]:
        """将标签转换为 Trip 对象"""
        if not label.orders:
            return None
        
        self._trip_counter += 1
        
        trip = Trip(
            id=self._trip_counter,
            orders=label.orders.copy(),
            expected_energy=label.expected_energy,
            energy_std=label.energy_std,
            total_distance=label.total_distance,
            total_weight=label.total_weight,
            depot_id=self.depot_id,  # 多仓库：设置出发仓库
            depot_location=self.depot_location  # 多仓库：设置仓库位置
        )
        
        return trip
    
    def _create_single_customer_trip(
        self,
        order: Order,
        current_time: float
    ) -> Optional[Trip]:
        """创建仅服务单个客户的行程"""
        # 检查能耗可行性
        energy_result = self.energy_model.compute_trip_energy_from_orders(
            self.depot.location,
            [order]
        )
        
        if not energy_result.is_feasible:
            return None
        
        distance = 2 * euclidean_distance(self.depot_location, order.location)
        
        self._trip_counter += 1
        
        return Trip(
            id=self._trip_counter,
            orders=[order],
            expected_energy=energy_result.expected_energy,
            energy_std=energy_result.energy_std,
            total_distance=distance,
            total_weight=order.weight,
            depot_id=self.depot_id,  # 多仓库：设置出发仓库
            depot_location=self.depot_location  # 多仓库：设置仓库位置
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
        # 过滤行程：无延迟且发车时间在 [t_k, t_k + υ] 区间内
        eligible_trips = []
        for trip in existing_trips:
            if trip.is_executed or trip.is_cancelled:
                continue
            
            # 检查发车时间是否在 [t_k, t_k + υ] 范围内
            if trip.planned_start_time is not None:
                if not (current_time <= trip.planned_start_time <= current_time + threshold_v):
                    continue
            
            # 检查是否无延迟：max_lateness 应为 0
            if trip.planned_start_time is not None:
                lateness = trip.max_lateness(trip.planned_start_time, self.speed)
                if lateness > 0:
                    continue
            
            eligible_trips.append(trip)
        
        # Line 3: for each r ∈ D_k ∪ U_k do
        for order in new_orders:
            if order in remaining_orders:  # 仍未被分配
                # Line 4: γ ← find_best_trip_for_request(Γ'(s_{k-1}, s_k), i_r)
                best_trip, best_position, delta_psi = self._find_best_trip_for_request(
                    eligible_trips, order, current_time
                )
                
                # Line 5: if γ ≠ ∅ and Δψ(γ, i_r) = 0 then
                if best_trip is not None and delta_psi == 0:
                    # Line 6: γ ← γ ∪ {i_r} - 将 i_r 插入到 γ 的合适位置
                    best_trip.orders.insert(best_position, order)
                    best_trip.total_weight += order.weight
                    
                    # 重新计算行程属性
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
        为给定请求查找最合适的插入行程（算法5，第4行）
        
        Returns:
            (best_trip, best_position, delta_psi) 三元组
            其中 delta_psi 为插入导致的新增延迟
        """
        best_trip = None
        best_position = None
        best_delta_psi = float('inf')
        best_insertion_cost = float('inf')
        
        # 遍历所有可行的行程
        for trip in eligible_trips:
            # 检查载重约束
            if trip.total_weight + order.weight > self.max_payload:
                continue
            
            # 尝试每一个插入位置
            for pos in range(len(trip.orders) + 1):
                # 检查能耗可行性
                test_orders = trip.orders[:pos] + [order] + trip.orders[pos:]
                energy_result = self.energy_model.compute_trip_energy_from_orders(
                    self.depot.location, test_orders
                )
                
                if not energy_result.is_feasible:
                    continue
                
                # 计算 Δψ(γ, i_r) - 插入带来的新增延迟
                original_lateness = trip.max_lateness(
                    trip.planned_start_time or current_time, self.speed
                )
                
                # 创建临时行程以计算新的延迟
                new_lateness = self._calculate_trip_lateness(
                    test_orders, trip.planned_start_time or current_time
                )
                
                delta_psi = new_lateness - original_lateness
                
                # 计算插入代价（距离增加）
                insertion_cost = self._calculate_insertion_cost(
                    trip.orders, order, pos
                )
                
                # 优先选择 delta_psi = 0，其次选择插入代价最小
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
        """计算一个行程的总最大延迟"""
        current_time = start_time
        current_loc = self.depot.location
        total_lateness = 0.0
        
        for order in orders:
            # 旅行时间
            dist = euclidean_distance(current_loc, order.location)
            travel_time = dist / self.speed / 60.0  # 转换为分钟
            current_time += travel_time
            
            # 计算该订单的延迟
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
        """计算在给定位置插入订单所增加的额外距离"""
        if not current_orders:
            # 第一个客户
            return 2 * euclidean_distance(self.depot_location, new_order.location)
        
        if position == 0:
            # 插入到开头
            prev_loc = self.depot_location
            next_loc = current_orders[0].location
        elif position == len(current_orders):
            # 插入到末尾
            prev_loc = current_orders[-1].location
            next_loc = self.depot_location
        else:
            prev_loc = current_orders[position - 1].location
            next_loc = current_orders[position].location
        
        # 原始距离
        original_dist = euclidean_distance(prev_loc, next_loc)
        
        # 通过新订单后的新距离
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
        """检查在给定出发时间下是否能在截止时间前（允许一定宽限）完成所有订单"""
        current_time = start_time
        current_loc = self.depot_location
        
        for order in orders:
            # 旅行时间
            dist = euclidean_distance(current_loc, order.location)
            travel_time = dist / self.speed / 60.0
            current_time += travel_time
            
            # 允许一定的延迟（软截止时间）
            # 这里只检查是否“过于”延迟
            if current_time > order.deadline + 60:  # Allow up to 60 min late
                return False
            
            current_time += order.service_time
            current_loc = order.location
        
        return True
