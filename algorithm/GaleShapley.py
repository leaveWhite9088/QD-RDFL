import numpy as np
from scipy.optimize import minimize


class GaleShapley:

    # Gale - Shapley 婚姻匹配算法。
    @staticmethod
    def gale_shapley(proposals, preferences):
        """
        Gale-Shapley 婚姻匹配算法。
        这个场景下， DataOwner 充当提议者， CPC 充当被提议者。
        每个 DataOwner 主动向其首选的 CPC 提出请求，CPC 根据其优先顺序选择是否接受或者更喜欢其他 DataOwner 的请求。
        :param proposals: 提议者的初始顺序表，每个提议者是一个列表，表示他们的优先选择。
        :param preferences: 被提议者的优先顺序表，每个被提议者是一个列表，表示她们的优先选择。
        :return: dict, 匹配结果，key 是提议者，value 是匹配的被提议者。
        """
        # 初始化
        free_proposers = list(proposals.keys())  # 所有提议者都开始时是空闲的
        engaged = {}  # 存储匹配结果，key 是提议者，value 是匹配的被提议者
        proposal_count = {proposer: 0 for proposer in proposals}  # 跟踪每个提议者已经提议过的人数

        # 被提议者的当前订婚者
        current_engaged = {receiver: None for receiver in preferences}  # key 是被提议者，value 是当前订婚者

        # 开始匹配
        while free_proposers:
            proposer = free_proposers.pop(0)  # 从空闲提议者中取出一个
            # 提议者将向下一个优先的被提议者提出请求
            receiver = proposals[proposer][proposal_count[proposer]]
            proposal_count[proposer] += 1

            # 如果被提议者还没有订婚
            if current_engaged[receiver] is None:
                current_engaged[receiver] = proposer
                engaged[proposer] = receiver
            else:
                # 如果被提议者已经订婚了，检查她是否更喜欢现有的订婚者
                current_partner = current_engaged[receiver]
                if preferences[receiver].index(proposer) < preferences[receiver].index(current_partner):
                    # 如果被提议者更喜欢新提议者，则替换原有订婚者
                    free_proposers.append(current_partner)  # 原有订婚者变为空闲
                    current_engaged[receiver] = proposer
                    engaged[proposer] = receiver
                else:
                    # 否则，保持现有订婚
                    free_proposers.append(proposer)

        return engaged

    # 根据CPC的算力因子和DataOwner的数量生成每个DataOwner的提议列表。
    @staticmethod
    def make_proposals(computational_powers, num_dataowners):
        """
        根据CPC的算力因子和DataOwner的数量生成每个DataOwner的提议列表。
        :param computational_powers:
        :param num_dataowners:
        :return:
        """
        # 生成CPC名称列表（假设CPC名称为CPC1, CPC2, ..., CPCM）
        cpcs = [f"CPC{i + 1}" for i in range(len(computational_powers))]

        # 将CPC与算力因子配对，并按算力因子排序
        cpcs_sorted = sorted(zip(cpcs, computational_powers), key=lambda x: x[1], reverse=True)

        # 获取排序后的CPC名称列表
        sorted_cpcs = [x[0] for x in cpcs_sorted]

        # 创建一个包含num_dataowners个DataOwner的字典，值为排序后的CPC列表
        proposals = {}

        for i in range(1, num_dataowners + 1):
            proposals[f"DataOwner{i}"] = sorted_cpcs

        return proposals

    # 根据 DataOwner 对每个 CPC 的效用，生成 CPC 对每个 DataOwner 的优先顺序。
    @staticmethod
    def make_preferences(xn_array, CPCs, Rho, dataowners):
        """
        根据 DataOwner 对每个 CPC 的效用，生成 CPC 对每个 DataOwner 的优先顺序。
        :param xn_array: 一个长度为 N 的数组，每个元素代表一个 DataOwner 对所有 CPC 的数据量。
        :param CPCs:
        :param Rho:
        :return:
        """
        preferences = {}

        N = len(xn_array)  # DataOwner 数量
        M = len(CPCs)  # CPC 数量
        sumdm = sum(xn_array)
        sumxn = sum(xn_array)

        # TODO 这里需要修改获取 CPC 偏好列表的方式：先算出 dm*，再取得最近的 do
        bestDm_list = GaleShapley.nash_equilibrium(CPCs, Rho, sumxn, xn_array)

        # data_volume_list 是数据量列表
        data_volume_list = []
        for do in dataowners:
            data_volume_list.append(len(do.imgData))
        data_volume_list = [x * y for x, y in zip(data_volume_list, xn_array)]

        # 对 bestDm_list 归一化，然后根据这个值确定偏好
        min_value = min(bestDm_list)
        max_value = max(bestDm_list)
        minVal = min(data_volume_list)
        maxVal = max(data_volume_list)

        # normalized_arr 是最优数据量列表
        normalized_lst = [minVal + (x - min_value) * (maxVal - minVal) / (max_value - min_value) for x in bestDm_list]

        print(f"normalized_lst:{normalized_lst}")

        # 对每个 CPC，计算 diff_list
        for index in range(len(CPCs)):

            diff_list = [x - normalized_lst[index] for x in data_volume_list]

            diff_list = [abs(x) for x in diff_list]

            # 使用enumerate获取索引和值，然后根据值排序
            sorted_indices = [index for value, index in sorted((value, index) for index, value in enumerate(diff_list))]

            sorted_dataowners = []
            for index2 in sorted_indices:
                sorted_dataowners.append((f"DataOwner{index2 + 1}"))

            # 形成 preferences
            preferences[f"CPC{index + 1}"] = sorted_dataowners

        return preferences

    @staticmethod
    def update_bestDm(cpc, Rho, sumdm, sumxn, xn_array, learning_rate=0.01):
        # 使用梯度下降法来优化dm
        dm = cpc.bestDm  # 初始的dm值
        for _ in range(100):  # 最大迭代次数
            utility_current = cpc.cal_cpc_utility(Rho, sumdm, sumxn, dm)
            # 计算效用对dm的梯度
            gradient = (cpc.cal_cpc_utility(Rho, sumdm, sumxn, dm + 0.001) - utility_current) / 0.001
            # 更新dm值
            dm = dm + learning_rate * gradient
            # # 确保dm值在[0,1]之间
            # dm = np.clip(dm, min(xn_array), max(xn_array))
        cpc.bestDm = dm

    @staticmethod
    def nash_equilibrium(cpcs, Rho, sumxn, xn_array):
        iteration = 0
        converged = False
        while not converged:
            iteration += 1
            sumdm = sum(cpc.bestDm for cpc in cpcs)  # 计算所有CPC的bestDm之和
            prev_bestDm_values = [cpc.bestDm for cpc in cpcs]

            # 每个CPC计算自己通过调整bestDm得到的最优效用
            for cpc in cpcs:
                GaleShapley.update_bestDm(cpc, Rho, sumdm, sumxn, xn_array)

            # 检查是否收敛：如果所有CPC的bestDm值都没有变化，则认为收敛
            converged = all(prev == cpc.bestDm for prev, cpc in zip(prev_bestDm_values, cpcs))

            # 防止无限循环，可以设置一个最大迭代次数
            if iteration > 100:
                converged = True
                print("最大迭代次数已达到，退出")

        return_list = [cpc.bestDm for cpc in cpcs]

        return return_list  # 返回每个CPC的最优bestDm值