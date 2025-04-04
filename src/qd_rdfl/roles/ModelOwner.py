import math


class ModelOwner:
    def __init__(self, Alpha, model, g=None):
        """
        :param Alpha: 预设参数
        :param g: 一个凹函数，用于评价模型质量
        """
        if g is None:
            self.g = self.defaultg
        else:
            self.g = g
        self.Alpha = Alpha
        self.model = model

    # 效用公式
    def cal_mo_utility(self, Eta, sumqn):
        """
        计算ModelOwner的效用
        :param Eta: ModelOwner的总支付
        :param sumqn: 所有DataOwner的总的模型贡献
        :return: ModelOwner的效用
        """
        # 默认模型评价函数

        return self.Alpha * self.g(sumqn) - Eta

    # 默认模型评价函数
    def defaultg(self, x):
        return 0.5 * math.log(1 + x)
