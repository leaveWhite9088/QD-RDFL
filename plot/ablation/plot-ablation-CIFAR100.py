import matplotlib.pyplot as plt
import matplotlib

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16  # 同时设置字体大小

# 假设这是您的算法和对比算法的精度数据
# 精度值
accuracy_your_algorithm = [19.07, 19.11, 19.11, 19.11, 19.11, 19.11, 19.23, 19.23, 19.23, 19.23, 19.23, 19.23, 19.28,
                           19.29, 19.29, 19.29, 19.29, 19.29, 19.29, 19.29, 19.29, 19.29, 19.29, 19.29, 19.29, 19.29,
                           19.29, 19.29, 19.29, 19.29, 19.29, 19.29, 19.29, 19.29, 19.29, 19.29, 19.29, 19.29, 19.29,
                           19.29, 19.29, 19.29, 19.29, 19.29, 19.29, 19.29, 19.29, 19.29, 19.29, 19.29, 19.29, 19.29,
                           19.29, 19.29, 19.29, 19.29, 19.29, 19.29, 19.29, 19.29, 19.29, 19.29, 19.29, 19.29, 19.3,
                           19.3, 19.3, 19.3, 19.3, 19.3, 19.3, 19.3, 19.3, 19.3, 19.3, 19.3, 19.3, 19.39, 19.39, 19.39,
                           19.39, 19.39, 19.39, 19.39, 19.39, 19.39, 19.39, 19.39, 19.39, 19.39, 19.39, 19.39, 19.39,
                           19.39, 19.39, 19.39, 19.39, 19.39, 19.39, 19.39, 19.39]

accuracy_your_algorithm = [x / 100 for x in accuracy_your_algorithm]

accuracy_comparison_algorithm = [19.15, 19.16, 19.16, 19.23, 19.23, 19.23, 19.23, 19.23, 19.25, 19.25, 19.25, 19.25,
                                 19.25, 19.25, 19.25, 19.25, 19.25, 19.25, 19.25, 19.25, 19.25, 19.25, 19.25, 19.25,
                                 19.25, 19.25, 19.25, 19.25, 19.25, 19.25, 19.25, 19.25, 19.25, 19.25, 19.25, 19.25,
                                 19.25, 19.25, 19.25, 19.25, 19.25, 19.25, 19.25, 19.25, 19.25, 19.25, 19.25, 19.25,
                                 19.26, 19.26, 19.26, 19.26, 19.26, 19.26, 19.26, 19.26, 19.26, 19.26, 19.26, 19.26,
                                 19.26, 19.26, 19.26, 19.26, 19.26, 19.26, 19.26, 19.26, 19.26, 19.26, 19.26, 19.26,
                                 19.26, 19.26, 19.26, 19.26, 19.26, 19.26, 19.26, 19.26, 19.26, 19.26, 19.26, 19.26,
                                 19.26, 19.26, 19.26, 19.26, 19.26, 19.26, 19.26, 19.26, 19.26, 19.26, 19.26, 19.26,
                                 19.26, 19.26, 19.26, 19.26, 19.26]

accuracy_comparison_algorithm = [x / 100 for x in accuracy_comparison_algorithm]

# 测试次数或版本号
tests = range(1, len(accuracy_your_algorithm) + 1)

# 创建折线图
plt.plot(tests[:70:5], accuracy_your_algorithm[:70:5], marker='o', label='With dynamic adjustment')  # 绘制您的算法精度折线图
plt.plot(tests[:70:5], accuracy_comparison_algorithm[:70:5], marker='s',
         label='Without dynamic adjustment')  # 绘制对比算法精度折线图

# 添加标题和标签
plt.xlabel('Training round')  # x轴标签
plt.ylabel('Accuracy')  # y轴标签

# 显示图例
plt.legend()

plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)

# 显示网格
plt.grid(True)

# 显示图表
plt.show()
