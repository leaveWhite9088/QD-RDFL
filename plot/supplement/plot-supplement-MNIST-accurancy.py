import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 14  # 同时设置字体大小

# 准备数据

accuracy_dict = {}

accurancy_list = accuracy_dict[-0.5]

accurancy_list2 = accuracy_dict[0.0]

accurancy_list3 = accuracy_dict[0.5]

x = np.arange(1, len(accurancy_list) + 1)  # 轮数

plt.plot(x, accurancy_list, label='fn(Reported 50% lower)')
plt.plot(x, accurancy_list2, label='fn(Normal)')
plt.plot(x, accurancy_list3, label='fn(Reported 50% higher)')

plt.legend()
plt.xlabel('round')
plt.ylabel('Accurancy')

plt.show()
