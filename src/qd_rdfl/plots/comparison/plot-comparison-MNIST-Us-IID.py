import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16  # 同时设置字体大小

# 定义用户数量
users = np.arange(0, 70)

# 定义不同策略下的BS效用
utility_qdrdfl = [0.6384366516514275, 0.7151251726638783, 0.7404557091167423, 0.7530733353655703, 0.7606282246567844, 0.7656581864367016]
utility_random = [0.5565476131439678, 0.7149695894568477, 0.7162317824606541, 0.748461318729682, 0.7579235622175451, 0.7579235622175451]
utility_fix = [0.6046347154308931, 0.6695734314391386, 0.6908501556089104, 0.7014209958827129, 0.7077421117661165, 0.7119473272235717]


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
