import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16  # 同时设置字体大小

# 定义用户数量
users = np.arange(0, 70)

# 定义不同策略下的BS效用
utility_qdrdfl = [0.013888904975080973, 0.003618421224931645, 0.0016283533929213187, 0.0009214748398204242, 0.0005918370325082023, 0.0004119587710614936]
utility_random = [0.008023977842834696, 0.0035489522510946726, 0.0012593924607740966, 0.0008284765587383836, 0.0005460392541126513, 0.0005460392541126513]
utility_fix = [0.010000000000007104, 0.0025000000000000135, 0.001111111111111111, 0.0006249999999999922, 0.00039999999999989154, 0.0002777777777777768]

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
