import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16  # 同时设置字体大小

# 定义用户数量
users = np.arange(0, 70)

# 定义不同策略下的BS效用
utility_qdrdfl = [0.5911909786339378, 0.6506519049167567, 0.6642735637230694, 0.6764606605321748, 0.6811083317985978, 0.6932654635780342]
utility_random = [0.489446867831107, 0.516213638127866, 0.6397813029750996, 0.5931272842685523, 0.6554285116835066, 0.6451347379252996]
utility_fix = [0.5459358033561617, 0.5900354723340313, 0.6289863290741216, 0.6374246078666332, 0.6353505149063667, 0.6493243468111762]


# 绘制图表
plt.plot(users[10:70:10], utility_qdrdfl, 'b--s', label='QD-RDFL')  # 蓝色虚线，方形标记
plt.plot(users[10:70:10], utility_random, 'g--^', label='Random')  # 绿色虚线，三角形标记
plt.plot(users[10:70:10], utility_fix, 'r--o', label='Fixed')  # 红色虚线，圆形标记

# 添加图例
plt.legend()

plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)

# 设置标题和坐标轴标签
plt.xlabel(r'$N$')
plt.ylabel(r'$U_s$')

# 强制显示完全
plt.tight_layout()

# 显示网格
plt.grid(True)

# 显示图表
plt.show()
