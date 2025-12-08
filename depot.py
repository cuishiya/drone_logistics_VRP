import matplotlib.pyplot as plt

# 数据准备
stations = [
    {"name": "冠城高新科技店", "lon": 113.939042, "lat": 22.798145},
    {"name": "新健兴科技店", "lon": 113.949883, "lat": 22.772652},
    {"name": "碧眼新村店", "lon": 113.951536, "lat": 22.7567584},
    {"name": "南庄店", "lon": 113.910375, "lat": 22.7740880},
    {"name": "西田店", "lon": 113.900603, "lat": 22.8009850},
    {"name": "东发集配店", "lon": 113.922249, "lat": 22.7509074},
    {"name": "中央山店", "lon": 113.898593, "lat": 22.7765857},
    {"name": "大鸿科技店", "lon": 113.989764, "lat": 22.7398432},
    {"name": "汇业科技园店", "lon": 113.942272, "lat": 22.736281}
]

lons = [s["lon"] for s in stations]
lats = [s["lat"] for s in stations]
names = [s["name"] for s in stations]

# 绘图
plt.figure(figsize=(10, 8))
plt.scatter(lons, lats, c='blue', s=100, alpha=0.7, label='SF Stations (Nodes)')

# 添加标签 (使用索引 0, 1, 2... 代表站点，避免中文字体在某些环境下乱码)
for i, txt in enumerate(names):
    # xytext=(5, 5) 表示标签相对于点的偏移量
    plt.annotate(i, (lons[i], lats[i]), xytext=(5, 5), textcoords='offset points', fontsize=12, weight='bold')

plt.title('Spatial Distribution of SF Express Stations', fontsize=14)
plt.xlabel('Longitude (E)', fontsize=12)
plt.ylabel('Latitude (N)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# 保存图片
plt.savefig('sf_stations_map.png')
plt.show()