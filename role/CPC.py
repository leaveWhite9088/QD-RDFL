class CPC:
    def __init__(self, Lambda, Epsilon, SigmaM):
        """
        :param Lambda: 市场调节因子
        :param Epsilon: CPC的单位计算费用
        :param Sigma: 当前CPC的算力因子
        """
        self.Lambda = Lambda
        self.Epsilon = Epsilon
        self.SigmaM = SigmaM
        self.imgData = None
        self.labelData = None

    # 效用函数
    def cal_cpc_utility(self, Rho, sumdm, sumxn, dm):
        """
        计算CPC的效用
        :param Rho: DataOwner支付的单位数据的训练价格
        :param sumdm: 所有CPC承接到的数据总量
        :param sumxn: 所有DataOwner提供的数据总量
        :param dm: 当前CPC承接的数据量
        :return: 当前CPC的效用
        """
        return self.Lambda * (dm / sumdm) * Rho * sumxn - self.Epsilon * self.SigmaM * dm