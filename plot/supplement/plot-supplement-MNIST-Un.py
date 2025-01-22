import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 14  # 同时设置字体大小

# 数据
years = ["-60%", "-40%", "-20%", "Normal", "+20%", "+40%", "+60%"]
user0 = [0.012590268364477686, 0.013306403916628781, 0.014030835616521325, 0.014763289017004494, 0.015503795978128121,
         0.016252376939088542, 0.017009037763627555]
user1 = [0.0072790873819382484, 0.008192379932023153, 0.009130998302226995, 0.0100954056977164, 0.011084720725804076,
          0.012099805018759291, 0.01314024507918407]
user2 = [0.006130925350980365, 0.007211710069612867, 0.008335696474114501, 0.009503050539907978, 0.010713688763342027,
          0.011967294065170775, 0.013263371933078777]

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
