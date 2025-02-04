import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 14  # 同时设置字体大小

# 数据
years = ["-60%", "-40%", "-20%", "Normal", "+20%", "+40%", "+60%"]
user0 = [0.017187037902624047, 0.017460842296343898, 0.0177354069851397, 0.018010734462383382, 0.018286827027694907, 0.01856368678044204, 0.018841315626311778]
user1 = [0.007853735437941545, 0.009244377537614909, 0.010688579987770785, 0.012186466987738065, 0.013737748353553902, 0.01534179398387972, 0.01699769433167153]
user2 = [0.004945055121943917, 0.006028451394949036, 0.0071668564133289675, 0.008360463263313078, 0.00960908412941619, 0.010912213023864609, 0.012269079715933912]

# 设置柱状图的宽度
bar_width = 0.2

# 设置每组柱状图的x轴位置
r1 = np.arange(len(years))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# 绘制柱状图
plt.bar(r1, user0, color='#5470C6', width=bar_width, label='User 0')
plt.bar(r2, user1, color='#91CC75', width=bar_width, label='User 1')
plt.bar(r3, user2, color='#FAC858', width=bar_width, label='User 2')

# 添加图例
plt.legend()

# 添加标签和标题
plt.xlabel(r'$Report$ $deviation$ $ratio$')
plt.ylabel(r'$U_n$')

# 设置x轴刻度
plt.xticks([r + bar_width for r in range(len(years))], years)

# 显示图表
plt.show()