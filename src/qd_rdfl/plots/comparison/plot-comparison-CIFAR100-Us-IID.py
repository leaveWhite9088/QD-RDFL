import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16  # 同时设置字体大小

# 定义用户数量
users = np.arange(0, 70)

# 定义不同策略下的BS效用
utility_qdrdfl = [0.555158958139429, 0.6252185935075907, 0.6630926627850176, 0.6843373444139751, 0.6891740440493792, 0.6903342363100233]
utility_random = [0.5664825152480899, 0.5677388286217808, 0.6559866143351585, 0.6245848728968075, 0.6829380650748049, 0.6733704210641462]
utility_fix = [0.5263256560630223, 0.6192924649082934, 0.6272222867141861, 0.6274865990549929, 0.6362425051615259,0.6364616167946497]


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
