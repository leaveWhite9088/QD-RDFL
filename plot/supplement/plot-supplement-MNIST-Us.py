import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 14  # 同时设置字体大小

# 数据
client_dict = {0.0: [0.5881472263157428, 0.5881472263157428, 0.5881472263157428, 0.5881472263157428, 0.5881472263157428,
                     0.5881472263157428],
               0.2: [0.6293755720152212, 0.6346301411876545, 0.6342287130896733, 0.6489500467815987, 0.663563971996842,
                     0.6771684566117215],
               0.4: [0.6969160488211423, 0.7006142245818123, 0.7010499854940009, 0.7039539742916616, 0.7158643527078534,
                     0.7267371911349145],
               0.6: [0.7369990874920145, 0.7382353045066115, 0.7387903690569504, 0.7402519553621991, 0.7469671185827522,
                     0.7559229743868845],
               0.8: [0.7604199820310875, 0.7612318922125552, 0.7620657793103309, 0.7641396497087845, 0.7651544505316246,
                     0.7683491712501902],
               1.0: [0.7652625150471464, 0.7652658686021923, 0.7652625162687618, 0.7652658694522301, 0.7652658686180867,
                     0.7652658697700463]}

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
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]
r5 = [x + bar_width for x in r4]
r6 = [x + bar_width for x in r5]

plt.figure(figsize=(12, 8))

# 绘制柱状图
plt.bar(r1, clients0, color='grey', width=bar_width, label='0% Clients')
plt.bar(r2, clients20, color='#5470C6', width=bar_width, label='20% Clients')
plt.bar(r3, clients40, color='#91CC75', width=bar_width, label='40% Clients')
plt.bar(r4, clients60, color='#FAC858', width=bar_width, label='60% Clients')
plt.bar(r5, clients80, color='#EF6765', width=bar_width, label='80% Clients')
plt.bar(r6, clients100, color='#73C0DF', width=bar_width, label='100% Clients')

# 添加图例
plt.legend()

# 添加标签和标题
plt.xlabel(r'$Report$ $deviation$ $ratio$')
plt.ylabel(r'$U_s$')

# 设置x轴刻度
plt.xticks([r + bar_width for r in range(len(years))], years)

# 显示图表
plt.show()
