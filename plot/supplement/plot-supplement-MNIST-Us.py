import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 14  # 同时设置字体大小

# 数据
client_dict = {0.2: [0.5607447142080892, 0.5750701075500249, 0.5891204078006735, 0.6164291149349614, 0.6297037231952121,
                     0.6427352453230908],
               0.4: [0.6427314714038199, 0.6433939208614474, 0.6440956525864605, 0.6456103647521945, 0.6464203161880786,
                     0.6472634421160808],
               0.6: [0.6669099419039954, 0.6678035384816927, 0.6687728344825796, 0.6751619843392633, 0.6855465808162298,
                     0.6980189731347919]}

years = ["-60%", "-40%", "-20%", "+20%", "+40%", "+60%"]
clients20 = client_dict[0.2]
clients40 = client_dict[0.4]
clients60 = client_dict[0.6]

# 设置柱状图的宽度
bar_width = 0.2

# 设置每组柱状图的x轴位置
r1 = np.arange(len(years))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# 绘制柱状图
plt.bar(r1, clients20, color='#5470C6', width=bar_width, label='20% Clients')
plt.bar(r2, clients40, color='#91CC75', width=bar_width, label='40% Clients')
plt.bar(r3, clients60, color='#FAC858', width=bar_width, label='60% Clients')

# 添加图例
plt.legend()

# 添加标签和标题
plt.xlabel(r'$Report$ $deviation$ $ratio$')
plt.ylabel(r'$U_s$')

# 设置x轴刻度
plt.xticks([r + bar_width for r in range(len(years))], years)

# 显示图表
plt.show()
