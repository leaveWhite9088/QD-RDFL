import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16  # 同时设置字体大小

# 定义用户数量
users = np.arange(0, 70)

# 定义不同策略下的BS效用
utility_qdrdfl = [0.014399130091867062, 0.004520437647834774, 0.002566049391130565, 0.0016994562633446063, 0.001239036457614322, 0.0013485415465878078]
utility_random = [0.01441354475682306, 0.003015592662699895, 0.0022826347583449888, 0.0010845358613851524, 0.0009873284504611402, 0.0009924497073003468]
utility_fix = [0.010690639545586711, 0.0030961051889048796, 0.0017668155929634814, 0.0014066570181528932, 0.0008409500284330807,0.0006945177894479126]

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
