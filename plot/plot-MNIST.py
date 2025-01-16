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
    U_Eta_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.15285673453012114, 0.1611690360336873,
                  0.1611690360336873, 0.1611690360336873, 0.1611690360336873, 0.1611690360336873, 0.1611690360336873,
                  0.1611690360336873, 0.21573146763191528, 0.21573146763191528, 0.2302098123104489, 0.2372722952806745,
                  0.24421911586688416, 0.2510518316129473, 0.25777196880940006, 0.26438102330883356,
                  0.27088046141856353, 0.27727172049219656, 0.28355620987446256, 0.2897353115269479, 0.295810380736725,
                  0.295810380736725, 0.3076537138499581, 0.3134245610842403, 0.3134245610842403, 0.3134245610842403,
                  0.3301488201150091, 0.3301488201150091, 0.3408201260223009, 0.3460158141262499, 0.3511196963862255,
                  0.35613287545654276, 0.3610564342345392, 0.3610564342345392, 0.3610564342345392, 0.3610564342345392,
                  0.3610564342345392, 0.3843665008270294, 0.38877403025715107, 0.3930990044634607, 0.3973423641456786,
                  0.3973423641456786, 0.4055879231287358, 0.40959192533381683, 0.41351791946630423, 0.4173667698614162,
                  0.4211393263042197, 0.4248364249955151, 0.42845888820827593, 0.43200752492426864, 0.4354831310221291,
                  0.4354831310221291, 0.442218371200111, 0.44547953413748587, 0.44867072469973246, 0.45179267743247176,
                  0.4548461153665311, 0.4578317502816396, 0.46075028290452824, 0.4636024031654795, 0.46638879040945125,
                  0.4691101136050986, 0.4717670315655649, 0.4743601931450787, 0.47689023744286785, 0.4793577940068625,
                  0.48176348297026816, 0.4841079153438992, 0.486391693091085, 0.48861540934906167, 0.49077964860734535,
                  0.4928849868574189, 0.49493199176667, 0.4969212228341201, 0.49885323155173233, 0.5007285615399467,
                  0.502547748726821, 0.5043113214633171, 0.506019800681, 0.5076737000408355, 0.5092735259846805,
                  0.5108197780460677, 0.5123129487952933, 0.5137535240576601, 0.5151419830226942, 0.516478798351405,
                  0.5177644363214511, 0.5189993569034468, 0.520184013908995, 0.5213188550800976, 0.5224043222036907,
                  0.5234408512155813, 0.5244288723046828, 0.5253688100081948, 0.5262610833205366, 0.5271061057787574,
                  0.5279042855633724, 0.528656025586596, 0.5293617235869295, 0.5300217722131502, 0.5306365591135662,
                  0.5312064670186711, 0.5317318738255825, 0.5322131526780516, 0.5326506720463844, 0.5330447958045457,
                  0.533395883307209, 0.5337042894635102, 0.533970364810362, 0.5341944555839624, 0.5343769037898578,
                  0.5345180472726745, 0.5346182197779281, 0.53467775103165, 0.5346969667802199, 0.5346761888860612,
                  0.5346157353629202, 0.5345159204482197, 0.5343770546614148, 0.5341994448608098, 0.5339833943035714,
                  0.5337292026991927, 0.5334371662659396, 0.5331075777836256, 0.5327407268661579, 0.5323368989300961,
                  0.5318963774019971, 0.5314194416211506, 0.5309063680157937, 0.5303574296303541, 0.5297728968008206,
                  0.5291530365343804, 0.5284981130507957, 0.5278083873998571, 0.527084117871282, 0.5263255598396348,
                  0.5255329659130079, 0.5247065858371143, 0.5238466666694293, 0.5229534527671136, 0.5220271858058365,
                  0.521068104856343, 0.520076446395755, 0.5190524443575446, 0.5179963301651496, 0.5169083327654505,
                  0.5157886786659502, 0.5146375919576969, 0.5134552944063095, 0.5122420053702561, 0.5109979419490549,
                  0.5097233189576322, 0.5084183489697016, 0.5070832423479312, 0.5057182072769069, 0.5043234497909963,
                  0.5028991737981212, 0.5014455811331806, 0.4999628715315112, 0.4984512427421799, 0.4969108904723216,
                  0.49534200846010545, 0.49374478847522085, 0.49211942041665147, 0.49046609217560655,
                  0.48878498990506136, 0.48707629777236616, 0.48534019819790863, 0.48357687177885667,
                  0.48178649733261647, 0.4799692519220462, 0.47812531087241084, 0.4762548478135902, 0.4743580346657099,
                  0.4724350416959784, 0.4704860375248412, 0.46851118914356094, 0.4665106619504231, 0.4644846197498307,
                  0.46243322479245097, 0.4603566377816817, 0.45825501789899525, 0.4561285228214409, 0.4539773087428378,
                  0.45180153038825344, 0.4496013410361739, 0.4473768925260746, 0.4451283353050499, 0.44285581840510524,
                  0.44055948948042456, 0.4382394948538595, 0.43589597948335745, 0.4335290869771318, 0.4311389596614008,
                  0.42872573855919915, 0.4262895634127264, 0.42383057270369706, 0.4213489036473015, 0.41884469225091436,
                  0.4163180732837799, 0.4137691803350023, 0.41119814572431324, 0.4086051007119136, 0.40599017531683623,
                  0.4033534984310134, 0.4006951978068227, 0.3980154000775822, 0.3953142307663784, 0.39259181430013346,
                  0.389848274022337, 0.387083732205487, 0.3842983100635493, 0.38149212776394803, 0.3786653044434227,
                  0.37581795820223407, 0.3729502061513217, 0.37006216438637063, 0.36715394802023305, 0.3642256711884224,
                  0.36127744706062925, 0.358309387852104, 0.35532160483199116, 0.35231420834151006, 0.34928730778438677,
                  0.34624101166913634, 0.34317542759218655, 0.34009066225748086, 0.3369868214871272, 0.3338640102293384,
                  0.3307223325688087, 0.3275618917359724, 0.32438279011757976, 0.3211851292654191, 0.31796900988694476,
                  0.31473453190909817, 0.3114817944174706, 0.3082108957133416, 0.304921933289525, 0.30161500387293927,
                  0.29829020342519863, 0.29494762711205924, 0.2915873693077713, 0.28820952381896525,
                  0.28481418345795717, 0.2814014404437062, 0.27797138625089746, 0.27452411162793355,
                  0.27105970661141265, 0.2675782605342074, 0.2640798620312692, 0.2605645990483567, 0.2570325588486164,
                  0.2534838280193332, 0.24991849248126385, 0.2463366374898177, 0.242738347647395, 0.23912370690841156,
                  0.23549279858516714, 0.2318457053548899, 0.22818250926629124, 0.2245032917454628, 0.2208081336025609,
                  0.21709711503791507, 0.2133703156480964, 0.2096278144328938, 0.20586968979819797, 0.2020960195642294,
                  0.19830688097471194, 0.194502350695394, 0.19068250482551807, 0.18684741890015788, 0.18299716789729636,
                  0.1791318262434567, 0.17525146781607326, 0.17135616595929548, 0.16744599346972633,
                  0.16352102262107948, 0.15958132515952705, 0.155626972309999, 0.15165803478318374, 0.1476745827789938,
                  0.14367668599085004, 0.13966441361215365, 0.13563783434013432, 0.13159701638070453]

    U_qn_list = [0.04671615854245892, 0.0007010638630982058, 0.040981938363289505, 0.04492919217725455,
                 -0.05432581676287327, -0.008712123480250236, -0.04519022485665579, -0.011923888214612162,
                 -0.024683745474462927, -0.014112332996909142, -0.010608875445599908, -0.016277388240144967,
                 0.00039305102677768467, -0.0026861195948625276, -0.02336667614465443, -0.027201399737610193,
                 -0.024052334743571522, -0.012385238073735764, -0.020360510367609226, -0.026986512879144338,
                 -0.016191497077641347, -0.021731675909437764, -0.014725197527862305, -0.02445393086396847,
                 -0.028317388378682456, -0.016535222317823308, -0.00785317072115215, -0.016336915867181306,
                 -0.0042334901069458475, -0.012198555121058177, -0.006472029100955394, -0.012786747524753703,
                 -0.009926526963524735, -0.0169129219239011, -0.00955761090694196, -0.01989948624085096,
                 -0.01354098735447, -0.007786545978629147, -0.022244246992646072, -0.017388660770533613,
                 -0.02405815649470414, -0.009701553353042876, -0.01931885877390557, -0.021434864361626296,
                 -0.021881292861396062, -0.017115478051479564, -0.015532406091026138, -0.016430311456823634,
                 -0.02070847843205578, -0.012148246024238625, -0.014051222769410213, -0.01556940684822575,
                 -0.013218054213452178, -0.017810844491018926, -0.014618674749876045, -0.013454021496915282,
                 -0.015407898574733132, -0.01549733759786819, -0.011151985507525454, -0.01744805206219055,
                 -0.015361556391394431, -0.01569684778671993, -0.017720962001047162, -0.01807303642067104,
                 -0.01582007796942693, -0.018617674465429997, -0.03151368565253098, -0.019484646668602515,
                 -0.028358495642899557, -0.01760583157389113, -0.015848470606103223, -0.02978309857770902,
                 -0.033463987120205464, -0.048427663260002314, -0.044017696380845295, -0.04142702611546847,
                 -0.04293631573760224, -0.049188932983019, -0.039881706759293235, -0.0394571301149887,
                 -0.043675786823758285, -0.04555949236171567, -0.057092888019079566, -0.01706806420575025,
                 -0.020598564533088944, -0.05265561516032314, -0.06855796911051593, -0.06794237347323831,
                 -0.08014233102997416, -0.05937655278497722, -0.06536578231200425, -0.06968391422331296,
                 -0.07900741926503027, -0.05676338654401114, -0.07409963549036543, -0.07276318123701789,
                 -0.07614636029130106, -0.06358985594025504, -0.07865088987595904, -0.08893165942718886]

    x = range(2, len(U_Eta_list) + 2)  # 客户端数量

    # 绘制折线图
    plot_line_chart(x, U_Eta_list, title='U_Eta', xlabel='N', ylabel='U(Eta)', marker='o', linestyle='--',
                    color='g', grid=True)

    # 绘制折线图
    plot_line_chart(x, U_qn_list, title='U_qn', xlabel='N', ylabel='U(qn)/N', marker='o', linestyle='--',
                    color='g', grid=True)
