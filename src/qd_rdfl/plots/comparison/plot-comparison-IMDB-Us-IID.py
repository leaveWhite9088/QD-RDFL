import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16  # 同时设置字体大小

# 定义用户数量
users = [10,20,30,40,50,60]

# 定义不同策略下的BS效用
utility_qdrdfl = [0.5715333370628699, 0.6333026895285576, 0.6733055049059622, 0.6699418125410352, 0.6841332323137947, 0.6983824109506411]

utility_random = [0.5615852643155914, 0.6178230805149456, 0.6619459220296005, 0.6131687052240256, 0.6739538156084988, 0.5869381571350399]

utility_fix = [0.5519508556387043, 0.6002551081113294, 0.6342673377468961, 0.631416939732588, 0.643431608207929, 0.655466673006798]

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
