import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16  # 同时设置字体大小

# 定义用户数量
users = np.arange(0, 28)

# 定义不同策略下的BS效用
utility_qdrdfl = [0.12500014232854578, 0.11111122274995593, 0.0729167558742089, 0.05000510707422636, 0.036111071025532425, 0.027210858109573736, 0.021205335585483764, 0.01697528939988656, 0.013888904975080973, 0.01157025942885555, 0.009785361866063805, 0.008382649046969904, 0.00726060093138492, 0.006349209507790461, 0.005598960578756931, 0.004974050002467469, 0.004448075758269578, 0.004001231346860829, 0.003618421224931645, 0.003287982010077168, 0.003000787224487368, 0.0027496134534936035, 0.0025286836504975576, 0.0023333334281220265, 0.002159763399046828, 0.0020048549979291737, 0.0018660252395822283, 0.0017411254846685796]

utility_random = [0.1094249840119361, 0.07603820737960243, 0.04537016972121677, 0.02705737007458293, 0.0322159275644621, 0.018702672672425553, 0.019676834874219062, 0.013017744209634185, 0.011422162238821466, 0.009511061850689425, 0.009497334245135417, 0.007374955929251559, 0.004226132368889397, 0.003937920585694406, 0.0031218273591195195, 0.004634616025809972, 0.0027054121233921666, 0.003734179364806549, 0.002433859473250455, 0.0018698122189651685, 0.002091706527087047, 0.002454385052941662, 0.001961357364834132, 0.0013064000402868415, 0.002157271505313043, 0.0014447560895615534, 0.0012938688542031642, 0.001279288642698164]

utility_fix = [0.25, 0.11111111111111109, 0.06250000000000017, 0.03999999999999713, 0.027777777777768204, 0.020408163265306114, 0.015625000000000028, 0.012345679012373132, 0.010000000000007104, 0.008264462809917349, 0.0069444444444444475, 0.005917159763313631, 0.0051020408163265285, 0.004444444444444444, 0.003906250000000291, 0.0034602076124564804, 0.0030864197530864395, 0.0027700831024930718, 0.0025000000000000135, 0.0022675736961451113, 0.002066115702479337, 0.00189035916824196, 0.0017361111111111143, 0.001600000000000006, 0.0014792899408284056, 0.0013717421124828512, 0.00127551020408164, 0.0011890606420927605]

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
plt.ylabel(r'Average of  $U_n$')

# 强制显示完全
plt.tight_layout()

# 显示网格
plt.grid(True)

# 显示图表
plt.show()
