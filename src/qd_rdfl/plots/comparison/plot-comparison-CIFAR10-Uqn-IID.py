import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16  # 同时设置字体大小

# 定义用户数量
users = np.arange(0, 70)

# 定义不同策略下的BS效用
utility_qdrdfl = [0.014667493426695022, 0.004857711156804179, 0.002553201276535444, 0.0019337789459559496, 0.001166044269566302, 0.000752784006678675]
utility_random = [0.008359205988637352, 0.002198257525473446, 0.001874337476451493, 0.0010937838025212921, 0.001156313467747716, 0.0008672695729946111]
utility_fix = [0.01092027727350362, 0.0032429515450467126, 0.0019226388763371202, 0.001145813568274584, 0.0008635518135624042, 0.0007886934156823381]

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
plt.ylabel(r'Average of  $U_n$')

# 强制显示完全
plt.tight_layout()

# 显示网格
plt.grid(True)

# 显示图表
plt.show()
