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
    U_Eta_list = [0.0315632007924026, 0.18136166435634088, 0.3323691960880061, 0.4232070060612041, 0.4323111371288503,
                  0.4666774099182086, 0.5293532815729336, 0.5502397045721201, 0.5901206543436763]

    U_qn_list = [0.07134105231502777, 0.053113632601736314, 0.05283460594614542, 0.03961961150630344,
                 0.0235831572513448, 0.01816828776515914, 0.017210745507677372, 0.01564275644802689,
                 0.012895256283734623]

    x = range(2, len(U_Eta_list) + 2)  # 客户端数量

    # 绘制折线图
    plot_line_chart(x, U_Eta_list, title='U_Eta', xlabel='N', ylabel='U(Eta)', marker='o', linestyle='--',
                    color='g', grid=True)

    # 绘制折线图
    plot_line_chart(x, U_qn_list, title='U_qn', xlabel='N', ylabel='U(qn)/N', marker='o', linestyle='--',
                    color='g', grid=True)
