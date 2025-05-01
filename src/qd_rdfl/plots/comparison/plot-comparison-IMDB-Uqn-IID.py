import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16  # 同时设置字体大小

# 定义用户数量
users = [10,20,30,40,50,60]

# 定义不同策略下的BS效用
utility_qdrdfl = [0.014380408670253609, 0.0045020379157513, 0.0023171975642439853, 0.0017810851213730995, 0.0012198803838360161, 0.0011972558290411957]

utility_random = [0.01204527158767429, 0.003630792270178074, 0.0019395336439433845, 0.0012098184047831168, 0.0010324367870356843, 0.000634864377557004]

utility_fix = [0.010909213855280975, 0.0032511095954177007, 0.0016362401432434245, 0.0012599707868298448, 0.0008564151413257726, 0.0008342843713122457]

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
plt.ylabel(r'Average of  $U_n$')

# 强制显示完全
plt.tight_layout()

# 显示网格
plt.grid(True)

# 显示图表
plt.show()
