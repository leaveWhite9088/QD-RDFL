import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 20  # 同时设置字体大小

# 准备数据
accuracy_dict = {
    -0.5: [38.55, 42.45, 43.65, 44.15, 44.67, 45.09, 45.44, 45.84, 46.27, 46.91, 47.0, 47.19, 47.2, 47.5, 47.67, 47.71,
           48.03, 48.03, 48.18, 48.21, 48.46, 48.46, 48.59, 48.87, 49.49, 49.49, 49.49, 49.49, 49.49, 49.49, 49.49,
           49.49, 49.49, 49.49, 49.49, 49.49, 49.49, 49.49, 49.49, 49.49, 49.49, 49.49, 49.49, 49.49, 49.49, 49.49,
           49.49, 49.49, 49.49, 49.49, 49.49, 49.49, 49.49, 49.49, 49.49, 49.49, 49.49, 49.49, 49.49, 49.49, 49.49,
           49.49, 49.49, 49.49, 49.49, 49.49, 49.49, 49.49, 49.49, 49.49, 49.49, 49.49, 49.49, 49.49, 49.49, 49.49,
           49.49, 49.49, 49.49, 49.49, 49.49, 49.49, 49.49, 49.49, 49.49, 49.49, 49.49, 49.49, 49.49, 49.49, 49.49,
           49.54, 49.54, 49.54, 49.54, 49.54, 49.54, 49.54, 49.54, 49.54, 49.54],
    0.0: [39.71, 41.6, 42.52, 43.57, 44.47, 44.91, 45.6, 45.6, 45.68, 45.91, 45.91, 46.05, 46.05, 46.49, 46.49, 46.49,
          46.49, 46.5, 46.56, 46.64, 46.82, 46.85, 47.24, 47.67, 47.75, 48.13, 48.13, 48.13, 48.13, 48.26, 48.26, 48.26,
          48.26, 48.26, 48.26, 48.26, 48.26, 48.47, 48.47, 48.47, 48.47, 48.47, 48.47, 48.47, 48.47, 48.47, 48.47,
          48.47, 48.47, 48.47, 48.47, 48.47, 48.47, 48.47, 48.47, 48.47, 48.47, 48.47, 48.47, 48.47, 48.47, 48.47,
          48.47, 48.47, 48.47, 48.47, 48.47, 48.47, 48.47, 48.47, 48.47, 48.47, 48.47, 48.47, 48.47, 48.47, 48.47,
          48.47, 48.47, 48.47, 48.54, 48.54, 48.54, 48.54, 48.54, 48.54, 48.54, 48.54, 48.54, 48.55, 48.55, 48.55,
          48.55, 48.62, 48.62, 48.76, 48.76, 48.88, 48.88, 48.95, 48.95],
    0.5: [39.56, 42.1, 43.53, 44.63, 44.74, 45.05, 45.05, 45.38, 45.38, 45.4, 45.54, 45.54, 45.54, 45.54, 45.54, 45.67,
          46.01, 46.39, 46.39, 46.39, 46.39, 46.39, 46.39, 46.39, 46.39, 46.39, 46.39, 46.39, 46.39, 46.39, 46.39,
          46.39, 46.39, 46.39, 46.39, 46.39, 46.39, 46.39, 46.39, 46.39, 46.39, 46.39, 46.39, 46.39, 46.39, 46.39,
          46.39, 46.39, 46.39, 46.39, 46.39, 46.39, 46.39, 46.39, 46.39, 46.45, 46.57, 46.57, 46.59, 46.68, 46.81,
          46.81, 46.81, 46.81, 46.81, 46.81, 46.81, 46.83, 46.83, 46.83, 46.83, 46.9, 46.9, 47.22, 47.22, 47.25, 47.25,
          47.25, 47.25, 47.25, 47.25, 47.25, 47.25, 47.25, 47.25, 47.25, 47.25, 47.25, 47.37, 47.41, 47.41, 47.41,
          47.41, 47.41, 47.41, 47.41, 47.41, 47.41, 47.41, 47.49, 47.49]}

accurancy_list = accuracy_dict[-0.5]

accurancy_list2 = accuracy_dict[0.0]

accurancy_list3 = accuracy_dict[0.5]

x = np.arange(1, len(accurancy_list) + 1)  # 轮数

# plt.figure(figsize=(8, 6))

plt.plot(x[0:80], accurancy_list[0:80], label=r'20% Data owners with -60% $f_n$', marker='o', markersize=2)
plt.plot(x[0:80], accurancy_list2[0:80], label=r'100% Data owners with $f_n$', marker='o', markersize=2)
plt.plot(x[0:80], accurancy_list3[0:80], label=r'20% Data owners with +60% $f_n$', marker='o', markersize=2)

plt.legend()
plt.xlabel(r'$Round$')
plt.ylabel(r'$Accurancy$')

# 显示网格
plt.grid(True)

plt.tight_layout()
plt.show()
