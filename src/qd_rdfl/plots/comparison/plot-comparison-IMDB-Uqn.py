import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16  # 同时设置字体大小

# 定义用户数量
users = [10,20,30,40,50,60]

# 定义不同策略下的BS效用
utility_qdrdfl = [0.014361430502911077, 0.004351432410918487, 0.0024564277656011326, 0.0013051890256982114, 0.0012856080511178059, 0.0010078013346734786]

utility_random = [0.00764301188849863, 0.0029527698616976794, 0.0023899307791023375, 0.001166382800309329, 0.000838993952082483, 0.0005432680610288657]

utility_fix = [0.010862568059949496, 0.0031024960103432143, 0.001742837261888397, 0.0009083468798841615, 0.0009010237682940581, 0.0006996747304051828]

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
