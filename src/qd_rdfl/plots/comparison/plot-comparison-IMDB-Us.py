import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16  # 同时设置字体大小

# 定义用户数量
users = [10,20,30,40,50,60]

# 定义不同策略下的BS效用
utility_qdrdfl = [0.5592983421682765, 0.6557128293215773, 0.6645593591495016, 0.7008315408643211, 0.6873540615551421, 0.7055771405777336]

utility_random = [0.46724682301673504, 0.6094059091599233, 0.6642652286412261, 0.6959617871350949, 0.670571019127804, 0.6978501105102085]

utility_fix = [0.5365736949557522, 0.6193404109385809, 0.6268523332529554, 0.6575324807658285, 0.6461544098689904, 0.6615330860067301]

# 绘制图表
plt.plot(users, utility_qdrdfl, 'b--s', label='QD-RDFL')  # 蓝色虚线，方形标记
plt.plot(users, utility_random, 'g--^', label='Random')  # 绿色虚线，三角形标记
plt.plot(users, utility_fix, 'r--o', label='Fixed')  # 红色虚线，圆形标记

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
