import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 26  # 同时设置字体大小

# 数据
client_dict = {0.0: [0.5870544592103093, 0.5870544592103093, 0.5870544592103093, 0.5870544592103093, 0.5870544592103093,
                     0.5870544592103093],
               0.2: [0.6098651659872869, 0.6099133332687039, 0.6099794389309185, 0.6186135296229187, 0.6270211825140655,
                     0.6352893938816722],
               0.4: [0.6660470003772354, 0.6748325262856303, 0.6745793671707181, 0.685885627327566, 0.699835254831304,
                     0.7134732800272416],
               0.6: [0.7131791425257026, 0.7187629627370964, 0.7192043303603768, 0.7203163450544141, 0.7219016570928758,
                     0.7257738083003535],
               0.8: [0.7240850203689877, 0.724001289851218, 0.723890024604503, 0.7235039963107335, 0.7231233051901738,
                     0.7223782656079591],
               1.0: [0.7237349750777962, 0.7237349750738018, 0.723734975074948, 0.7237349750739384, 0.7237349750723197,
                     0.7237349750725066]}

years = ["-60%", "-40%", "-20%", "+20%", "+40%", "+60%"]
clients0 = client_dict[0.0]
clients20 = client_dict[0.2]
clients40 = client_dict[0.4]
clients60 = client_dict[0.6]
clients80 = client_dict[0.8]
clients100 = client_dict[1.0]

# 设置柱状图的宽度
bar_width = 0.1

# 设置每组柱状图的x轴位置
r1 = np.arange(len(years))
r2 = [x - bar_width for x in r1]
r3 = [x for x in r1]
r4 = [x + bar_width for x in r1]
r5 = [x + 2 * bar_width for x in r1]
r6 = [x + 3 * bar_width for x in r1]

# 创建图形和轴
fig, ax = plt.subplots(figsize=(12, 8))

# 绘制柱状图
plt.bar(r2, clients20, color='#FFCCCC', width=bar_width, label='20% Data owners', edgecolor='white', linewidth=0.5)
plt.bar(r3, clients40, color='#FF8888', width=bar_width, label='40% Data owners', edgecolor='white', linewidth=0.5)
plt.bar(r4, clients60, color='#FF3333', width=bar_width, label='60% Data owners', edgecolor='white', linewidth=0.5)
plt.bar(r5, clients80, color='#FF0000', width=bar_width, label='80% Data owners', alpha=0.8, edgecolor='white',
        linewidth=0.5)
plt.bar(r6, clients100, color='#CC0000', width=bar_width, label='100% Data owners', alpha=0.8, edgecolor='white',
        linewidth=0.5)

# 添加格子背景
ax.grid(axis='y', linestyle='--', alpha=0.7)

# 添加图例
plt.legend(loc='upper right')

# 添加标签和标题
plt.xlabel(r'$Report$ $deviation$ $ratio$')
plt.ylabel(r'$U_s$')

# 设置x轴刻度
plt.xticks([r + bar_width for r in range(len(years))], years)

# 设置y轴的显示范围，不从0开始
plt.ylim(ymin=0.5)  # 只设置y轴的最小值

plt.tight_layout()
# 显示图表
plt.show()
