import matplotlib.pyplot as plt


# 绘制折线图的函数。
def plot_line_chart(x_data, y_data, title='Line Chart', xlabel='X Axis', ylabel='Y Axis', marker='o', linestyle='-',
                    color='b', grid=False):
    """
    绘制折线图的函数。

    参数:
    x_data (list): x轴的数据点列表。
    y_data (list): y轴的数据点列表。
    title (str): 图表的标题，默认为'Line Chart'。
    xlabel (str): x轴的标签，默认为'X Axis'。
    ylabel (str): y轴的标签，默认为'Y Axis'。
    marker (str): 数据点的标记类型，默认为'o'。
    linestyle (str): 线条的样式，默认为'-'。
    color (str): 线条的颜色，默认为'b'（蓝色）。
    grid (bool): 是否显示网格，默认为False。
    """

    # 绘制折线图
    plt.plot(x_data, y_data, marker=marker, linestyle=linestyle, color=color)

    # 添加标题和轴标签
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # 根据参数决定是否显示网格
    if grid:
        plt.grid(True)

    # 显示图表
    plt.show()


if __name__ == "__main__":
    # 准备数据
    U_Eta_list = [0.031244260958471104, 0.19849051995974254, 0.30744187366111686, 0.40928886172434464,
                  0.47435968832776587, 0.5443773323988008, 0.5181017503417205, 0.5103458071746161, 0.580597701913917,
                  0.5644380461571937, 0.5503825835202109]

    U_qn_list = [0.06956763478257169, 0.061916876865433644, 0.043644741632349715, 0.04664061982456258,
                 0.03122658803664724, 0.025424181674234818, 0.020532088248475883, 0.01129113726744688,
                 0.0150539149580327, 0.009104632916548928, 0.007009816993491541]

    x = range(2, len(U_Eta_list) + 2)  # 客户端数量

    # 绘制折线图
    plot_line_chart(x, U_Eta_list, title='U_Eta', xlabel='N', ylabel='U(Eta)', marker='o', linestyle='--',
                    color='g', grid=True)

    # 绘制折线图
    plot_line_chart(x, U_qn_list, title='U_qn', xlabel='N', ylabel='U(qn)/N', marker='o', linestyle='--',
                    color='g', grid=True)
