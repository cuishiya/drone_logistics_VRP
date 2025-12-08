# DRPUDEC: Dynamic Drone Routing Problem with Uncertain Demand and Energy Consumption

基于论文 **"A dynamic drone routing problem with uncertain demand and energy consumption"** (Chagas et al., 2025) 的复现实现。

## 项目概述

本项目实现了一个动态无人机路径规划系统，用于同日送达（Same-day Delivery）场景。系统特点：

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
├── config.py                 # 配置参数
├── main.py                   # 主入口
├── requirements.txt          # 依赖
├── README.md
├── models/
│   ├── __init__.py
│   ├── data_structures.py    # 数据结构定义
│   └── energy_model.py       # 能源消耗模型
├── algorithms/
│   ├── __init__.py
│   ├── trip_builder.py       # 路径构建算法
│   ├── trip_selector.py      # 路径选择(SPBP+DTSSP)
│   ├── trip_assigner.py      # 任务分配(TAP)
│   └── battery_manager.py    # 电池管理
└── simulation/
    ├── __init__.py
    ├── order_generator.py    # 订单生成器
    └── simulator.py          # 仿真引擎
```

## 安装

```bash
# 安装依赖
pip install -r requirements.txt

# 注意：需要安装Gurobi求解器
# 学术用户可申请免费许可证：https://www.gurobi.com/academia/
```

## 使用方法

### 单次仿真
```bash
python main.py --mode single --drones 3 --batteries 5 --orders 10 --seed 42
```

### 策略对比 (CFA vs Myopic)
```bash
python main.py --mode compare --runs 5
```

### 参数说明
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--drones` | 无人机数量 | 3 |
| `--batteries` | 电池数量 | 5 |
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

## 参考文献

Chagas, G. O., et al. (2025). A dynamic drone routing problem with uncertain demand and energy consumption. *Computers & Operations Research*.
