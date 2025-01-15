from role.CPC import CPC


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
    def make_preferences(xn_array, CPCs, Rho):
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

        # 对每个 CPC，计算其对所有 DataOwner 的效用
        for cpc in CPCs:
            utility_list = []
            for i in range(N):
                # 每个 DataOwner 提供的具体数据量为 xn_array[i]
                dm = xn_array[i]
                # 计算该 DataOwner 对当前 CPC 的效用
                utility = cpc.cal_cpc_utility(Rho, sumdm, sumxn, dm)
                utility_list.append((f"DataOwner{i + 1}", utility))

            # 按照效用值降序排列 DataOwner
            sorted_dataowners = [item[0] for item in sorted(utility_list, key=lambda x: x[1], reverse=True)]
            preferences[f"CPC{CPCs.index(cpc) + 1}"] = sorted_dataowners

        return preferences
