import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16  # 同时设置字体大小

# 定义用户数量
users = np.arange(0, 100)

# 定义不同策略下的BS效用
utility_qdrdfl = [0.03380193622101457, 0.21096639694708064, 0.36682164285517005, 0.42908201328951967,
                  0.4646270567586128,
                  0.514592709171201, 0.5572623557501122, 0.5785974045612368, 0.5810544508335453, 0.6082460720788319,
                  0.6124597283955593, 0.6375345117541453, 0.6165856945244734, 0.5645971835385035, 0.6150555892156151,
                  0.657150322289537, 0.6358907422261297, 0.6615825715827717, 0.6641486527094433, 0.6759926873556956,
                  0.666793064524539, 0.6631633672588644, 0.6740991042240938, 0.6789843965859073, 0.6777646282929568,
                  0.6655932048478923, 0.6839144843563079, 0.6807220153480706, 0.6799432285945572, 0.7055007854547448,
                  0.6966111964564883, 0.6771110680139096, 0.699719666910533, 0.6949497022272071, 0.6892036061740503,
                  0.7054940686735482, 0.7012479502150584, 0.6953260012053017, 0.7106268766880264, 0.6943264777861531,
                  0.6979458450280895, 0.7070943386732751, 0.71154639356021, 0.6938039369535536, 0.7107745627332682,
                  0.7112818801870919, 0.7052413571838563, 0.693975297284521, 0.7157929260822342, 0.7125520110484216,
                  0.7028185291097697, 0.70416789572212, 0.7147621764235699, 0.7003901956029748, 0.710322325182414,
                  0.7174279265988279, 0.7165147467388977, 0.3238064550608395, 0.716259397618157, 0.700649376954043,
                  0.7219499927190562, 0.715601683326438, 0.7181917093107786, 0.7176531124847818, 0.7210469507771669,
                  0.7198379341631631, 0.7243721422699345, 0.7202478907136509, 0.7116850488271063, 0.5814270428884964,
                  0.7239880772752494, 0.5734456391999649, 0.47676714037890777, 0.4889784334168823, 0.5060161129040739,
                  0.31048592511628215, 0.4104957163363303, 0.473304972004986, 0.40480760095478585, 0.2140184607642328,
                  0.35664793839603304, 0.3926920086722636, 0.20706441129104025, 0.24428643969539765, -6.772649166341273,
                  0.7285524017508305, 0.10260689803029654, 0.26318182498937226, 0.0094384044700484, 0.02120341789358271,
                  -0.054828677277433346, -0.24882941024814453, -0.11919672897370104, -0.010302973338702781,
                  -27.731692135343046, -0.2700917397531759, -0.5436498932237299, -0.31369616153723534,
                  -0.2422783437350562, -0.44509436660700086]

utility_random = [0.03303926550176223, 0.20993498076642014, 0.36604934728129535, 0.3808481815397483,
                  0.46303803779049346, 0.5040631914785416, 0.48369739654740307, 0.49364024997106504, 0.5361774697519724,
                  0.5464890752167533, 0.6062119776414285, 0.6371178939992632, 0.5973196470917745, 0.5869565992096408,
                  0.5743563862549972, 0.6539513064173696, 0.6345727375593022, 0.6605273568707999, 0.6438641835259758,
                  0.6724987577476853, 0.660945723804147, 0.6612538623919222, 0.6606882748051686, 0.6549571913934973,
                  0.5902713043465778, 0.6208817174672019, 0.6839127429007865, 0.6806597106673056, 0.6549080253715367,
                  0.644228362699978, 0.6703841753506303, 0.5821214345288325, 0.6986912235742184, 0.6856961503950376,
                  0.6438786914213893, 0.5833162732245373, 0.7010410600158645, 0.6907859406714472, 0.6916441756141478,
                  0.6910216740680513, 0.6968927275454262, 0.6726930655799837, 0.6895475933062654, 0.5922140702356258,
                  0.7086849896890297, 0.5836730555162598, 0.6799266856590727, 0.6806742370430889, 0.7047569091246111,
                  0.5920791232580943, 0.6887087089662386, 0.6999968420581113, 0.6452049884318617, 0.680121273163069,
                  0.6546156922337911, 0.6961223376025516, 0.6997753483151647, 0.5653101398098204, 0.694266595449452,
                  0.6985834189936218, 0.72165674983968, 0.6827752818443702, 0.7179964390856435, 0.7040059382294004,
                  0.6669877574486466, 0.6747202983668006, 0.6437230639739594, 0.713358020687801, 0.7113944448015841,
                  0.7113944448015841, 0.717616627091352, 0.717616627091352, 0.7109059456707221, 0.7131173769886938,
                  0.7131173769886938, 0.7131173769886938, 0.7080440236278804, 0.7083997966033913, 0.7053863216575129,
                  0.7053863216575129, 0.7053863216575129, 0.7178726075389918, 0.6950684197884105, 0.6950684197884105,
                  -2.4954123574936764, -2.4954123574936764, 0.6661542524996995, 0.6661542524996995, 0.6661542524996995,
                  0.6661542524996995, 0.6661542524996995, 0.6661542524996995, 0.6661542524996995, 0.6661542524996995,
                  -12.102364013381223, -12.102364013381223, -12.102364013381223, 0.611553494529169, 0.611553494529169,
                  0.611553494529169]

utility_fix = [-0.030586172447248705, 0.2086776258120231, 0.36376458576678816, 0.42116167872651866, 0.45319854113960245,
               0.49751435698461943, 0.534805502747435, 0.5532880896268004, 0.5554102540301371, 0.5788135975276387,
               0.5824273124816048, 0.6038654540039763, 0.5859626217368255, 0.6045476510630097, 0.5846519264048196,
               0.6205618658186394, 0.6024634463388818, 0.6243259769260807, 0.6265038529890921, 0.6365432666468773,
               0.6287471625449748, 0.6256677436308427, 0.634939592234752, 0.6390758840987898, 0.6380434508651629,
               0.6277294276488403, 0.6432466319144214, 0.640546266612336, 0.639887306380853, 0.6614687408089759,
               0.653972180754292, 0.6374901778167847, 0.6565947286582652, 0.652569884799612, 0.6477173122826925,
               0.6614630804767576, 0.6578836375404284, 0.6528875124908702, 0.6657868832387499, 0.6520437894313142,
               0.6550983519891522, 0.6628114873232609, 0.6665611158628102, 0.6516026457426216, 0.6659112420165885,
               0.6663384067393201, 0.6612501124664114, 0.6517473170295518, 0.6701353127196454, 0.6674077227848323,
               0.6592078970477449, 0.6603453800273307, 0.669267964848784, 0.6571602706617228, 0.6655304274316773,
               0.6715108505018617, 0.6707426265271594, 0.6732554249577423, 0.6705277924034798, 0.6573788552949067,
               0.6753135836745114, 0.6699743971275949, 0.6721533130871282, 0.6717002750914591, 0.6745543899869149,
               0.6735378056286645, 0.6773494138058778, 0.6738825317490127, 0.6666778542254048, 0.6609749899596737,
               0.6770266522918742, 0.6697326740168248, 0.6686656034917737, 0.6689537622438047, 0.6740136617340773,
               0.6722670740393779, 0.6696895556737166, 0.6794999648381235, 0.6729051688322538, 0.6710583605843767,
               0.6736796088092187, 0.6747003631379491, 0.6747003631379491, 0.6747003631379491, 0.6747003631379491,
               0.6808612891483121, 0.6713449721887972, 0.6819471287245014, 0.6749011153668159, 0.6731384779474099,
               0.6780445940700957, 0.6754376112001736, 0.6759760005926594, 0.6725068331837405, 0.67774001415284,
               0.674270226473209, 0.6737348524093254, 0.6737348524093254, 0.6777548987244286, 0.6777548987244286]

# 绘制图表
plt.plot(users[10:70:10], utility_qdrdfl[8:68:10], 'b--s', label='QD-RDFL')  # 蓝色虚线，方形标记
plt.plot(users[10:70:10], utility_random[8:68:10], 'g--^', label='Random')  # 绿色虚线，三角形标记
plt.plot(users[10:70:10], utility_fix[8:68:10], 'r--o', label='Fix')  # 红色虚线，圆形标记

# 添加图例
plt.legend()

plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)

# 设置标题和坐标轴标签
plt.xlabel(r'$N$')
plt.ylabel(r'$U_s$')

# 显示网格
plt.grid(True)

# 显示图表
plt.show()
