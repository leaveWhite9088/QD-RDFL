import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16  # 同时设置字体大小

# 定义用户数量
users = np.arange(0, 28)

# 定义不同策略下的BS效用
utility_qdrdfl = [0.057858878285647286, 0.27706405941477463, 0.40485498188886093, 0.4828679481396754, 0.5349229377000839, 0.5720167967835781, 0.5997554909808431, 0.6212692405438973, 0.6384366516514275, 0.6524513801741887, 0.664107478120116, 0.6739533938346711, 0.6823799762240124, 0.689673222396475, 0.6960471935080605, 0.70166527514426, 0.7066543244972798, 0.7111143320652527, 0.7151251726638783, 0.7187514192618063, 0.7220458382172021, 0.7250519687128474, 0.7278060545079619, 0.7303385100514157, 0.7326750468021841, 0.7348375481896703, 0.736844756295101, 0.7387128158713641]

utility_random = [0.057069476480451464, 0.2552820970178957, 0.3607340439575695, 0.4048837603989329, 0.5308367012548867, 0.5328324143204496, 0.5977900863767338, 0.5987719186026093, 0.6254023354682343, 0.6391430823607136, 0.663759593734305, 0.6678753886777471, 0.5979346762474714, 0.6206058737331539, 0.5987636666023055, 0.6996888371554335, 0.6313058604194567, 0.7092063077105133, 0.6635978770786908, 0.624408573719953, 0.678085034926621, 0.7199606138455208, 0.7043840869911395, 0.6304208687900404, 0.7326744790644271, 0.6973513493187697, 0.6910307517966836, 0.7050001416495373]

utility_fix = [0.013662770270411073, 0.2770640594149767, 0.39903946983855554, 0.46946666225531763, 0.5153395089258672, 0.5475980210155589, 0.5715216485559353, 0.5899719167996649, 0.6046347154308931, 0.6165679123126311, 0.6264689153528735, 0.6348161685166598, 0.6419488409726766, 0.6481140722106598, 0.6534962056134064, 0.6582355435256662, 0.6624407589831223, 0.66619733369446, 0.6695734314391386, 0.6726240724522123, 0.6753941558381165, 0.6779206846029253, 0.6802344284052824, 0.6823611831060645, 0.6843227367571094, 0.6861376188694821, 0.6878216876431673, 0.6893885946201903]


# 绘制图表
plt.plot(users, utility_qdrdfl, 'b--s', label='QD-RDFL')  # 蓝色虚线，方形标记
plt.plot(users, utility_random, 'g--^', label='Random')  # 绿色虚线，三角形标记
plt.plot(users, utility_fix, 'r--o', label='Fixed')  # 红色虚线，圆形标记

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
