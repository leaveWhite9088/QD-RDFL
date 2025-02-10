import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # 准备数据
    U_qn_list = {
        'CPC1': ['DataOwner3', 'DataOwner8', 'DataOwner5', 'DataOwner2', 'DataOwner1', 'DataOwner7', 'DataOwner10',
                 'DataOwner9', 'DataOwner4', 'DataOwner6'],
        'CPC2': ['DataOwner8', 'DataOwner3', 'DataOwner5', 'DataOwner2', 'DataOwner1', 'DataOwner7', 'DataOwner10',
                 'DataOwner9', 'DataOwner4', 'DataOwner6'],
        'CPC3': ['DataOwner8', 'DataOwner5', 'DataOwner3', 'DataOwner2', 'DataOwner1', 'DataOwner7', 'DataOwner10',
                 'DataOwner9', 'DataOwner4', 'DataOwner6'],
        'CPC4': ['DataOwner5', 'DataOwner8', 'DataOwner2', 'DataOwner3', 'DataOwner1', 'DataOwner7', 'DataOwner10',
                 'DataOwner9', 'DataOwner4', 'DataOwner6'],
        'CPC5': ['DataOwner5', 'DataOwner2', 'DataOwner1', 'DataOwner8', 'DataOwner7', 'DataOwner3', 'DataOwner10',
                 'DataOwner9', 'DataOwner4', 'DataOwner6'],
        'CPC6': ['DataOwner2', 'DataOwner1', 'DataOwner5', 'DataOwner7', 'DataOwner10', 'DataOwner8', 'DataOwner9',
                 'DataOwner4', 'DataOwner6', 'DataOwner3'],
        'CPC7': ['DataOwner1', 'DataOwner2', 'DataOwner7', 'DataOwner10', 'DataOwner5', 'DataOwner9', 'DataOwner4',
                 'DataOwner6', 'DataOwner8', 'DataOwner3'],
        'CPC8': ['DataOwner7', 'DataOwner1', 'DataOwner10', 'DataOwner9', 'DataOwner2', 'DataOwner4', 'DataOwner6',
                 'DataOwner5', 'DataOwner8', 'DataOwner3'],
        'CPC9': ['DataOwner10', 'DataOwner7', 'DataOwner9', 'DataOwner4', 'DataOwner6', 'DataOwner1', 'DataOwner2',
                 'DataOwner5', 'DataOwner8', 'DataOwner3'],
        'CPC10': ['DataOwner6', 'DataOwner4', 'DataOwner9', 'DataOwner10', 'DataOwner7', 'DataOwner1', 'DataOwner2',
                  'DataOwner5', 'DataOwner8', 'DataOwner3']}

    x = np.arange(len(U_qn_list))  # 客户端数量

    plt.plot(x, U_qn_list, label='UEta')

    plt.legend()
    plt.title('UEta')
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')

    plt.show()
