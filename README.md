# DRPUDEC: Dynamic Drone Routing Problem with Uncertain Demand and Energy Consumption

基于论文 **"A dynamic drone routing problem with uncertain demand and energy consumption"** (Chagas et al., 2025) 的复现实现。

## 项目概述

本项目实现了一个动态无人机路径规划系统，用于同日送达（Same-day Delivery）场景。系统特点：

- **多仓库支持**：支持多个仓库独立运作，订单自动分配到最近仓库
- **动态需求**：客户订单按泊松过程随机到达
- **软时间窗**：允许延迟送达，但有惩罚成本
- **能源不确定性**：考虑风速等因素对能耗的随机影响
- **电池管理**：支持电池更换和均衡老化策略

## 核心算法

### 1. MDP框架 (Algorithm 1)
- 离散时间决策
- 状态包含：订单、无人机、电池状态
- 动作包含：路径规划、任务分配、电池管理

### 2. CFA策略 (Algorithm 2)
- **成本函数近似**：通过参数M限制每架无人机的最大行程数
- 保留运力应对未来紧急订单

### 3. 路径构建 (Algorithm 3-4)
- **反向标签设定启发式**：从仓库反向构建路径
- 按紧急程度优先插入
- 帕累托支配规则剪枝

### 4. 路径选择 (Algorithm 5-6)
- **SPBP模型**：最大化加权客户覆盖
- **DTSSP模型**：最小化路由成本

### 5. 任务分配 (Algorithm 7)
- 求解指派问题(TAP)
- 最小化总延迟

### 6. 能源模型
- 考虑载重和风速的能耗公式
- **机会约束**：保证α概率下安全返航

## 项目结构

```
splitwise/
├── config.py                 # 配置参数（包含多仓库配置）
├── depot.py                  # 多仓库定义和工具函数
├── main.py                   # 主入口
├── requirements.txt          # 依赖
├── README.md
├── models/
│   ├── __init__.py
│   ├── data_structures.py    # 数据结构定义（支持多仓库）
│   └── energy_model.py       # 能源消耗模型
├── algorithms/
│   ├── __init__.py
│   ├── trip_builder.py       # 路径构建算法（多仓库支持）
│   ├── trip_selector.py      # 路径选择(SPBP+DTSSP)
│   ├── trip_assigner.py      # 任务分配(TAP)（多仓库支持）
│   └── battery_manager.py    # 电池管理
└── simulation/
    ├── __init__.py
    ├── order_generator.py    # 订单生成器（多仓库支持）
    └── simulator.py          # 仿真引擎（多仓库支持）
```

## 安装

```bash
# 安装依赖
pip install -r requirements.txt

# 注意：需要安装Gurobi求解器
```

## 使用方法

### 多仓库模式（默认）
```bash
# 使用默认的9个仓库，每个仓库1架无人机
python main.py --mode single --orders 20 --seed 42

# 自定义每个仓库的无人机和电池数量
python main.py --mode single --drones-per-depot 2 --batteries-per-depot 3 --orders 30
```

### 单仓库模式
```bash
python main.py --mode single --single-depot --drones 3 --batteries 5 --orders 10 --seed 42
```

### 策略对比 (CFA vs Myopic)
```bash
# 多仓库模式对比
python main.py --mode compare --runs 5

# 单仓库模式对比
python main.py --mode compare --single-depot --runs 5
```

### 参数说明
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--multi-depot` | 启用多仓库模式 | True |
| `--single-depot` | 使用单仓库模式 | False |
| `--drones-per-depot` | 每个仓库的无人机数 | 1 |
| `--batteries-per-depot` | 每个仓库的电池数 | 2 |
| `--drones` | 无人机数量(单仓库模式) | 3 |
| `--batteries` | 电池数量(单仓库模式) | 5 |
| `--orders` | 初始订单数 | 10 |
| `--horizon` | 时间范围(分钟) | 480 |
| `--interval` | 决策间隔(分钟) | 30 |
| `--max-trips` | 每架无人机最大行程数(M) | 2 |
| `--seed` | 随机种子 | 42 |

## 关键公式

### 能耗模型
$$\tilde{e}_{ij}(q) = \frac{d_{ij}}{\tilde{\xi}} \left( (\nu + q)^{3/2} \sqrt{\frac{g^3}{2\rho \zeta n}} \right)$$

### 机会约束
$$\mathbb{E}[\tilde{CE}_{\gamma}] + \Phi^{-1}(\alpha) \cdot SDV(\tilde{CE}_{\gamma}) \le E_{max} - E_{min}$$

## 输出示例

### 多仓库模式
```
============================================================
DRPUDEC Multi-Depot Simulation
多仓库无人机路径规划 - 9个仓库
============================================================

Configuration:
  Mode: Multi-Depot (多仓库模式)
  Depots: 9
  Drones per depot: 1
  Batteries per depot: 2
  Total drones: 9
  Total batteries: 18
  Initial orders: 20
  ...

============================================================
RESULTS SUMMARY
============================================================
Total orders:     35
Served orders:    32
Service rate:     91.4%
Total trips:      18
Total distance:   52340.5 meters
Total lateness:   12.3 minutes
Avg lateness:     0.4 minutes
```

### 单仓库模式
```
============================================================
RESULTS SUMMARY
============================================================
Total orders:     25
Served orders:    23
Service rate:     92.0%
Total trips:      12
Total distance:   45230.5 meters
Total lateness:   15.3 minutes
Avg lateness:     0.7 minutes
```

## 多仓库特性

### 仓库配置
项目默认配置了9个顺丰门店作为仓库站点（位于深圳地区）：

| ID | 仓库名称 | 经度 | 纬度 |
|----|----------|------|------|
| 0 | 冠城高新科技店 | 113.939 | 22.798 |
| 1 | 新健兴科技店 | 113.950 | 22.773 |
| 2 | 碧眼新村店 | 113.952 | 22.757 |
| 3 | 南庄店 | 113.910 | 22.774 |
| 4 | 西田店 | 113.901 | 22.801 |
| 5 | 东发集配店 | 113.922 | 22.751 |
| 6 | 中央山店 | 113.899 | 22.777 |
| 7 | 大鸿科技店 | 113.990 | 22.740 |
| 8 | 汇业科技园店 | 113.942 | 22.736 |

### 多仓库运作逻辑
1. **订单分配**：新订单自动分配到最近的仓库
2. **独立运作**：每个仓库独立管理其无人机、电池和订单
3. **路径规划**：每个仓库独立构建和执行配送路径
4. **无人机返航**：无人机完成任务后返回其所属仓库

## 参考文献

Chagas, G. O., et al. (2025). A dynamic drone routing problem with uncertain demand and energy consumption. *Computers & Operations Research*.
