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
    x = [2, 3, 4, 5]  # 客户端数量
    U_Eta_list = [0.03380193622101457, 0.22688791031943378, 0.3238870128044695, 0.4198541294490401]
    U_qn_list = [0.09749147610231523, 0.10292951601937202, 0.06765377798332572, 0.04841829611527748]

    # 绘制折线图
    plot_line_chart(x, U_Eta_list, title='U_Eta', xlabel='N', ylabel='U(Eta)', marker='o', linestyle='--',
                    color='g', grid=True)

    # 绘制折线图
    plot_line_chart(x, U_qn_list, title='U_qn', xlabel='N', ylabel='U(qn)/N', marker='o', linestyle='--',
                    color='g', grid=True)
