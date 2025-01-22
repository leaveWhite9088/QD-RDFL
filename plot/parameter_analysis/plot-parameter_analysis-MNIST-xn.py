import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16  # 同时设置字体大小

# 定义支付值范围
xn_list = np.arange(0, 1.001, 0.001)

# 定义BS效用函数，这里假设一个二次函数来模拟效用变化
un_list = [0.0, 0.00015649706182115216, 0.0003109898520949465, 0.00046348357657587927, 0.0006139834230059612,
           0.0007624945611925515, 0.0009090221430857991, 0.00105357130285567, 0.0011961471569686011,
           0.001336754804263712, 0.001475399326028682, 0.0016120857860751816, 0.00174681923081396,
           0.0018796046893295067, 0.0020104471734543553, 0.0021393516778430086, 0.0022663231800454445,
           0.0023913666405802995, 0.002514487003007642, 0.0026356891940013775, 0.0027549781234212783,
           0.002872358684384683, 0.002987835753337758, 0.0031014141901264773, 0.0032130988380671653,
           0.003322894524016743, 0.003430806058442565, 0.003536838235491939, 0.0036409958330612495,
           0.003743283612864825, 0.0038437063205033067, 0.003942268685531783, 0.00403897542152757, 0.004133831226157597,
           0.004226840781245508, 0.004318008752838366, 0.004407339791273057, 0.0044948385312423705,
           0.0045805095918607375, 0.004664357576729582, 0.004746387074002427, 0.004826602656449637,
           0.004905008881522843, 0.00498161029141897, 0.005056411413144127, 0.005129416758576921, 0.005200630824531718,
           0.0052700580928213905, 0.005337703030319815, 0.00540357008902409, 0.0054676637061164685,
           0.005529988304025804, 0.005590548290488918, 0.005649348058611545, 0.00570639198692896, 0.005761684439466355,
           0.005815229765798929, 0.005867032301111567, 0.005917096366258333, 0.005965426267821726, 0.006012026298171422,
           0.00605690073552291, 0.006100053843995826, 0.006141489873671913, 0.006181213060652779, 0.006219227627117213,
           0.006255537781378537, 0.006290147717941355, 0.006323061617558093, 0.0063542836472854775,
           0.0063838179605403556, 0.0064116686971557035, 0.006437839983435917, 0.006462335932212113,
           0.006485160642897059, 0.006506318201539865, 0.006525812680880372, 0.006543648140403341, 0.006559828626392275,
           0.006574358171983027, 0.006587240797217328, 0.006598480509095617, 0.006608081301630056, 0.006616047155897101,
           0.006622382040089803, 0.006627089909569825, 0.006630174706919401, 0.0066316403619927355,
           0.006631490791967509, 0.006629729901395792, 0.00662636158225495, 0.006621389713998185, 0.006614818163604944,
           0.006606650785630955, 0.006596891422258083, 0.006585543903343941, 0.006572612046471249,
           0.0065580996569970795, 0.006542010528101633, 0.006524348440836922, 0.0065051171641752475,
           0.006484320455057424, 0.006461962058440671, 0.006438045707346407, 0.006412575122907743, 0.006385554014416817,
           0.006356986079371782, 0.006326875003523691, 0.006295224460923121, 0.006262038113966473, 0.006227319613442278,
           0.006191072598577124, 0.006153300697081268, 0.006114007525194282, 0.00607319668773032, 0.006030871778123156,
           0.005987036378471131, 0.005941694059581781, 0.0058948483810163516, 0.005846502891133837,
           0.005796661127135255, 0.005745326615107321, 0.005692502870066202, 0.005638193396000746, 0.005582401685915811,
           0.005525131221875257, 0.005466385475044772, 0.005406167905734421, 0.005344481963441061, 0.005281331086890523,
           0.00521671870407972, 0.005150648232318394, 0.005083123078270718, 0.00501414663799668, 0.004943722296993386,
           0.004871853430236078, 0.004798543402219052, 0.004723795566996092, 0.004647613268221135, 0.004569999839188543,
           0.004490958602873046, 0.00441049287196979, 0.004328605948933972, 0.004245301126020534, 0.004160581685323184,
           0.004074450898813947, 0.003986912028381917, 0.003897968325872081, 0.0038076230331240968,
           0.003715879382010595, 0.003622740594475482, 0.0035282098825720754, 0.0034322904485010464,
           0.003334985484648084, 0.00323629817362156, 0.003136231688289859, 0.0030347891918187375,
           0.0029319738377081017, 0.002827788769829309, 0.002722237122461474, 0.0026153220203284366,
           0.002507046578634653, 0.0023974139031019415, 0.0022864270900050943, 0.0021740892262081535,
           0.0020604033891997997, 0.0019453726471291566, 0.0018290000588410404, 0.0017112886739113764,
           0.0015922415326819483, 0.0014718616662958428, 0.0013501520967316993, 0.0012271158368385993,
           0.0011027558903704826, 0.0009770752520203707, 0.0008500769074545611, 0.0007217638333466003,
           0.000592138997411118, 0.00046120535843752197, 0.00032896586632344404, 0.00019542346210824069,
           6.058107800616108e-05, -7.555836256065152e-05, -0.00021299194492904738, -0.00035171676316231304,
           -0.0004917299200174197, -0.0006330285269123825, -0.0007756097038942589, -0.0009194705796067293,
           -0.0010646082912581234, -0.0012110199845892788, -0.0013587028138422608, -0.0015076539417282486,
           -0.0016578705393963666, -0.0018093497864021257, -0.001962088870676365, -0.002116084988494138,
           -0.0022713353444438755, -0.002427837151396328, -0.0025855876304742564, -0.002744584011021678,
           -0.002904823530573669, -0.0030663034348258345, -0.0032290209776044687, -0.0033929734208365536,
           -0.003558158034519726, -0.003724572096692691, -0.0038922128934056888, -0.004061077718690881,
           -0.004231163874533206, -0.004402468670841125, -0.004574989425417342, -0.004748723463930293,
           -0.0049236681198849, -0.005099820734594335, -0.005277178657150855, -0.005455739244398045,
           -0.00563549986090206, -0.00581645787892357, -0.005998610678389721, -0.006181955646866161,
           -0.006366490179529172, -0.006552211679138192, -0.006739117556008006, -0.006927205227981376,
           -0.007116472120401618, -0.007306915666085684, -0.0074985333052967895, -0.0076913224857173,
           -0.007885280662422361, -0.008080405297852644, -0.00827669386178817, -0.008474143831321279,
           -0.008672752690830676, -0.008872517931955037, -0.009073437053566724, -0.009275507561745722,
           -0.009478726969753909, -0.009683092798008941, -0.009888602574058825, -0.01009525383255594,
           -0.01030304411523189, -0.010511970970872003, -0.010722031955289868, -0.010933224631302557,
           -0.011145546568704978, -0.011358995344245476, -0.01157356854160077, -0.011789263751351142,
           -0.012006078570955786, -0.012224010604728552, -0.012443057463813051, -0.012663216766158891,
           -0.012884486136496842, -0.01310686320631535, -0.01333034561383592, -0.0135549310039898,
           -0.013780617028393671, -0.014007401345326104, -0.014235281619704138, -0.014464255523059633,
           -0.014694320733515759, -0.014925474935763794, -0.015157715821039891, -0.015391041087102153,
           -0.015625448438207262, -0.015860935585087776, -0.016097500244929452, -0.016335140141348126,
           -0.016573853004367645, -0.016813636570396584, -0.01705448858220665, -0.017296406788909702,
           -0.01753938894593582, -0.017783432815011047, -0.018028536164135356, -0.018274696767560716,
           -0.018521912405769225, -0.01877018086545096, -0.019019499939483098, -0.01926986742690745,
           -0.019521281132909574, -0.019773738868796853, -0.02002723845197757, -0.02028177770593964,
           -0.020537354460229085, -0.02079396655042931, -0.02105161181814008, -0.021310288110956477,
           -0.021569993282448352, -0.021830725192139244, -0.022092481705486333, -0.02235526069385907,
           -0.022619060034519467, -0.0228838776106014, -0.023149711311090282, -0.023416559030802975,
           -0.02368441867036758, -0.02395328813620351, -0.02422316534050145, -0.024494048201203322,
           -0.024765934641983012, -0.02503882259222595, -0.02531270998701024, -0.025587594767086275,
           -0.02586347487885804, -0.026140348274363345, -0.026418212911254513, -0.026697066752779275,
           -0.026976907767761515, -0.027257733930582284, -0.02753954322116081, -0.027822333624935358,
           -0.02810610313284484, -0.028390849741309565, -0.028676571452212973, -0.028963266272882815,
           -0.029250932216072445, -0.029539567299942893, -0.02982916954804382, -0.03011973698929593,
           -0.03041126765797214, -0.030703759593679547, -0.030997210841341716, -0.03129161945118025,
           -0.03158698347869704, -0.03188330098465636, -0.032180570035067135, -0.03247878870116522,
           -0.03277795505939546, -0.03307806719139472, -0.03337912318397379, -0.03368112112910032, -0.03398405912388114,
           -0.034287935270545467, -0.034592747676427005, -0.03489849445394755, -0.035205173720599225,
           -0.03551278359892779, -0.035821322216515794, -0.036130787705965395, -0.03644117820488202,
           -0.03675249185585683, -0.03706472680645101, -0.0373778812091784, -0.03769195322148944, -0.03800694100575419,
           -0.038322842729246465, -0.03863965656412727, -0.03895738068742838, -0.03927601328103619,
           -0.03959555253167574, -0.03991599663089446, -0.040237343775046175, -0.04055959216527533,
           -0.04088274000750053, -0.04120678551239959, -0.04153172689539297, -0.04185756237662863, -0.04218429018096587,
           -0.04251190853796022, -0.04284041568184771, -0.04316980985152946, -0.043500089290556154,
           -0.04383125224711282, -0.044163296974003674, -0.0444962217286366, -0.04483002477300835, -0.04516470437368908,
           -0.04550025880180769, -0.0458366863330365, -0.046173985247576765, -0.04651215383014318, -0.04685119036994967,
           -0.047191093160694475, -0.047531860500545364, -0.04787349069212504, -0.04821598204249655,
           -0.04855933286314901, -0.04890354146998277, -0.04924860618329535, -0.049594525327766925,
           -0.04994129723244606, -0.050288920230735557, -0.05063739266037798, -0.050986712863442196,
           -0.05133687918630858, -0.0516878899796554, -0.05203974359844493, -0.0523924384019091, -0.05274597275353626,
           -0.05310034502105687, -0.053455553576430226, -0.05381159679583025, -0.054168473059632294,
           -0.0545261807523994, -0.05488471826286878, -0.055244083983938264, -0.05560427631265302, -0.05596529365019204,
           -0.05632713440185516, -0.05668979697704918, -0.05705327978927516, -0.057417581256115235,
           -0.05778269979921924, -0.0581486338442917, -0.05851538182107896, -0.05888294216335621, -0.05925131330891459,
           -0.05962049369954797, -0.05999048178104055, -0.060361276003154274, -0.06073287481961537,
           -0.061105276688102395, -0.0614784800702331, -0.06185248343155236, -0.06222728524151916, -0.06260288397349473,
           -0.06297927810472925, -0.06335646611635037, -0.06373444649335025, -0.06411321772457357, -0.06449277830270528,
           -0.06487312672425827, -0.06525426148956137, -0.06563618110274705, -0.06601888407173984, -0.06640236890824358,
           -0.06678663412773012, -0.06717167824942699, -0.0675574997963061, -0.06794409729507095, -0.06833146927614564,
           -0.06871961427366319, -0.06910853082545315, -0.06949821747303048, -0.06988867276158389, -0.07027989523996425,
           -0.07067188346067305, -0.0710646359798508, -0.07145815135726574, -0.0718524281563026, -0.07224746494395085,
           -0.07264326029079399, -0.07303981277099758, -0.07343712096229854, -0.07383518344599393, -0.0742339988069296,
           -0.07463356563348916, -0.07503388251758303, -0.07543494805463752, -0.07583676084358365, -0.07623931948684609,
           -0.07664262259033289, -0.07704666876342398, -0.07745145661896069, -0.07785698477323477, -0.07826325184597815,
           -0.0786702564603517, -0.07907799724293485, -0.07948647282371496, -0.07989568183607676, -0.08030562291679183,
           -0.08071629470600805, -0.08112769584723933, -0.08153982498735474, -0.08195268077656898, -0.08236626186843121,
           -0.08278056691981517, -0.08319559459090864, -0.08361134354520394, -0.08402781244948682, -0.08444499997382682,
           -0.08486290479156716, -0.08528152557931462, -0.08570086101692947, -0.08612090978751535, -0.0865416705774098,
           -0.08696314207617367, -0.08738532297658164, -0.08780821197461264, -0.08823180776943934, -0.08865610906341875,
           -0.08908111456208267, -0.08950682297412782, -0.08993323301140593, -0.09036034338891447, -0.09078815282478686,
           -0.09121666004028312, -0.0916458637597799, -0.09207576271076162, -0.09250635562381021, -0.09293764123259662,
           -0.09336961827387075, -0.0938022854874524, -0.09423564161622144, -0.09466968540610965, -0.09510441560609018,
           -0.09553983096816915, -0.09597593024737627, -0.09641271220175546, -0.09685017559235615, -0.09728831918322378,
           -0.09772714174139097, -0.09816664203686859, -0.09860681884263639, -0.09904767093463474, -0.09948919709175474,
           -0.09993139609583018, -0.1003742667316283, -0.10081780778684124, -0.10126201805207657, -0.10170689632084934,
           -0.10215244138957302, -0.10259865205755064, -0.10304552712696624, -0.1034930654028765, -0.10394126569320161,
           -0.10439012680871718, -0.10483964756304553, -0.10528982677264698, -0.10574066325681175, -0.1061921558376509,
           -0.10664430334008884, -0.10709710459185401, -0.10755055842347105, -0.1080046636682524, -0.1084594191622898,
           -0.10891482374444611, -0.10937087625634734, -0.10982757554237393, -0.11028492044965299, -0.11074290982804996,
           -0.11120154253016062, -0.11166081741130268, -0.11212073332950806, -0.11258128914551457, -0.11304248372275844,
           -0.11350431592736543, -0.11396678462814364, -0.11442988869657539, -0.11489362700680916, -0.11535799843565175,
           -0.11582300186256073, -0.1162886361696363, -0.1167549002416135, -0.11722179296585472, -0.11768931323234172,
           -0.11815745993366822, -0.11862623196503164, -0.11909562822422609, -0.11956564761163452, -0.1200362890302209,
           -0.12050755138552316, -0.12097943358564495, -0.12145193454124875, -0.12192505316554797, -0.12239878837430002,
           -0.12287313908579794, -0.1233481042208639, -0.12382368270284144, -0.1242998734575882, -0.12477667541346832,
           -0.12525408750134542, -0.12573210865457518, -0.12621073780899844, -0.1266899739029333, -0.12716981587716847,
           -0.12765026267495566, -0.12813131324200305, -0.12861296652646742, -0.1290952214789477, -0.1295780770524771,
           -0.13006153220251704, -0.13054558588694948, -0.1310302370660698, -0.1315154847025803, -0.1320013277615828,
           -0.13248776521057237, -0.13297479601942924, -0.13346241916041324, -0.13395063360815596, -0.1344394383396545,
           -0.13492883233426406, -0.1354188145736921, -0.13590938404199038, -0.13640053972554922, -0.13689228061309017,
           -0.13738460569565975, -0.13787751396662212, -0.13837100442165323, -0.1388650760587336, -0.13935972787814221,
           -0.1398549588824491, -0.1403507680765098, -0.140847154467458, -0.1413441170646998, -0.1418416548799062,
           -0.1423397669270075, -0.14283845222218655, -0.14333770978387234, -0.14383753863273352, -0.14433793779167192,
           -0.14483890628581653, -0.14534044314251687, -0.1458425473913369, -0.14634521806404838, -0.14684845419462494,
           -0.14735225481923553, -0.14785661897623864, -0.1483615457061755, -0.1488670340517642, -0.14937308305789365,
           -0.14987969177161725, -0.15038685924214668, -0.1508945845208461, -0.15140286666122543, -0.15191170471893528,
           -0.15242109775176, -0.15293104481961217, -0.15344154498452617, -0.15395259731065275, -0.1544642008642526,
           -0.15497635471369076, -0.15548905792943002, -0.1560023095840261, -0.1565161087521209, -0.15703045451043685,
           -0.1575453459377713, -0.15806078211499025, -0.15857676212502314, -0.15909328505285647, -0.15961034998552853,
           -0.16012795601212315, -0.16064610222376452, -0.16116478771361104, -0.16168401157684997, -0.16220377291069127,
           -0.1627240708143627, -0.16324490438910338, -0.16376627273815875, -0.16428817496677478, -0.16481061018219245,
           -0.16533357749364186, -0.1658570760123373, -0.16638110485147117, -0.16690566312620897, -0.16743074995368323,
           -0.16795636445298856, -0.16848250574517598, -0.16900917295324774, -0.1695363652021512, -0.1700640816187744,
           -0.17059232133194002, -0.17112108347240018, -0.1716503671728314, -0.1721801715678286, -0.17271049579390063,
           -0.17324133898946426, -0.1737727002948396, -0.17430457885224404, -0.17483697380578772, -0.17536988430146788,
           -0.1759033094871642, -0.1764372485126327, -0.1769717005295015, -0.17750666469126514, -0.17804214015327968,
           -0.17857812607275753, -0.17911462160876213, -0.17965162592220324, -0.18018913817583176, -0.18072715753423452,
           -0.18126568316382946, -0.18180471423286026, -0.18234424991139186, -0.18288428937130524, -0.18342483178629226,
           -0.18396587633185096, -0.1845074221852805, -0.1850494685256765, -0.1855920145339257, -0.18613505939270147,
           -0.18667860228645855, -0.18722264240142883, -0.18776717892561584, -0.18831221104879026, -0.18885773796248495,
           -0.1894037588599906, -0.1899502729363503, -0.19049727938835537, -0.19104477741454007, -0.1915927662151774,
           -0.19214124499227425, -0.19269021294956623, -0.1932396692925138, -0.1937896132282968, -0.1943400439658104,
           -0.19489096071566014, -0.19544236269015758, -0.1959942491033152, -0.19654661917084237, -0.19709947211014056,
           -0.19765280714029865, -0.19820662348208828, -0.1987609203579601, -0.1993156969920381, -0.19987095261011623,
           -0.20042668643965278, -0.20098289770976685, -0.20153958565123353, -0.2020967494964795, -0.20265438847957828,
           -0.2032125018362464, -0.20377108880383843, -0.20433014862134347, -0.20488968052937928, -0.20544968377018935,
           -0.20601015758763797, -0.2065711012272058, -0.20713251393598597, -0.20769439496267916, -0.20825674355758972,
           -0.2088195589726214, -0.2093828404612732, -0.20994658727863458, -0.21051079868138178, -0.21107547392777326,
           -0.21164061227764575, -0.21220621299241005, -0.2127722753350464, -0.21333879857010074, -0.2139057819636807,
           -0.2144732247834511, -0.21504112629862981, -0.21560948577998362, -0.21617830249982473, -0.21674757573200598,
           -0.2173173047519169, -0.2178874888364799, -0.21845812726414593, -0.21902921931489083, -0.2196007642702108,
           -0.2201727614131186, -0.22074521002813996, -0.2213181094013088, -0.22189145882016414, -0.22246525757374513,
           -0.22303950495258829, -0.22361420024872225, -0.22418934275566504, -0.2247649317684196, -0.22534096658346947,
           -0.22591744649877588, -0.22649437081377322, -0.22707173882936527, -0.2276495498479214, -0.22822780317327263,
           -0.22880649811070797, -0.22938563396697076, -0.2299652100502544, -0.23054522567019875, -0.23112568013788648,
           -0.23170657276583928, -0.23228790286801415, -0.2328696697597994, -0.23345187275801083, -0.23403451118088892,
           -0.23461758434809388, -0.2352010915807028, -0.2357850322012055, -0.23636940553350128, -0.23695421090289503,
           -0.23753944763609336, -0.23812511506120115, -0.23871121250771854, -0.23929773930653586, -0.23988469478993157,
           -0.2404720782915677, -0.24105988914648646, -0.24164812669110702, -0.24223679026322154, -0.24282587920199183,
           -0.24341539284794567, -0.24400533054297335, -0.24459569163032424, -0.24518647545460337, -0.24577768136176736,
           -0.24636930869912177, -0.24696135681531695, -0.24755382506034485, -0.24814671278553568, -0.24874001934355428,
           -0.2493337440883967, -0.24992788637538688, -0.2505224455611731, -0.2511174210037249, -0.2517128120623292,
           -0.2523086180975873, -0.2529048384714113, -0.25350147254702105, -0.25409851968894037, -0.25469597926299403,
           -0.2552938506363043, -0.25589213317728754, -0.25649082625565145, -0.2570899292423907, -0.2576894415097847,
           -0.2582893624313938, -0.2588896913820562, -0.25949042773788433, -0.26009157087626233, -0.260693120175842,
           -0.26129507501654015, -0.2618974347795352, -0.2625001988472637, -0.2631033666034178, -0.26370693743294127,
           -0.26431091072202706, -0.2649152858581133, -0.2655200622298809, -0.2661252392272505, -0.2667308162413782,
           -0.2673367926646537, -0.2679431678906964, -0.26854994131435284, -0.2691571123316929, -0.26976468034000767,
           -0.2703726447378052, -0.27098100492480814, -0.2715897603019515, -0.2721989102713771, -0.2728084542364335,
           -0.2734183916016708, -0.27402872177283866, -0.2746394441568827, -0.2752505581619422, -0.2758620631973465,
           -0.27647395867361213, -0.27708624400244, -0.27769891859671214, -0.2783119818704891, -0.2789254332390071,
           -0.2795392721186739, -0.2801534979270679, -0.28076811008293323, -0.28138310800617794, -0.2819984911178709,
           -0.28261425884023883, -0.2832304105966632, -0.2838469458116779, -0.2844638639109658, -0.28508116432135566,
           -0.28569884647082056, -0.2863169097884738, -0.28693535370456624, -0.28755417765048386, -0.2881733810587448,
           -0.2887929633629964, -0.28941292399801266, -0.2900332623996912, -0.29065397800505033, -0.29127507025222676,
           -0.2918965385804724, -0.2925183824301517, -0.2931406012427392, -0.2937631944608161, -0.2943861615280683,
           -0.295009501889283, -0.29563321499034656, -0.29625730027824115, -0.2968817572010427, -0.2975065852079176,
           -0.29813178374912075, -0.29875735227599176, -0.29938329024095334, -0.3000095970975083, -0.3006362723002366,
           -0.3012633153047928, -0.301890725567904, -0.302518502547366, -0.30314664570204186, -0.303775154491859,
           -0.3044040283778058, -0.3050332668219301, -0.30566286928733577, -0.3062928352381806, -0.3069231641396736,
           -0.307553855458072, -0.3081849086606794, -0.30881632321584296, -0.3094480985929505, -0.31008023426242826,
           -0.3107127296957384, -0.3113455843653762, -0.3119787977448677, -0.3126123693087677, -0.31324629853265584,
           -0.31388058489313586, -0.3145152278678318, -0.3151502269353861, -0.3157855815754569, -0.31642129126871554,
           -0.31705735549684455, -0.3176937737425345, -0.31833054548948236, -0.3189676702223877, -0.3196051474269521,
           -0.32024297658987544, -0.32088115719885346, -0.32151968874257597, -0.3221585707107244, -0.3227978025939686,
           -0.3234373838839656, -0.32407731407335605, -0.3247175926557627, -0.325358219125788, -0.3259991929790108,
           -0.3266405137119852, -0.3272821808222378, -0.32792419380826443, -0.32856655216952935, -0.32920925540646184,
           -0.3298523030204542, -0.33049569451385963, -0.3311394293899893, -0.3317835071531109, -0.3324279273084456,
           -0.33307268936216616, -0.33371779282139447, -0.33436323719419947, -0.3350090219895947, -0.33565514671753593,
           -0.3363016108889193, -0.3369484140155785, -0.3375955556102831, -0.33824303518673615, -0.3388908522595713,
           -0.33953900634435175, -0.3401874969575671, -0.3408363236166313, -0.3414854858398807, -0.34213498314657187,
           -0.3427848150568791, -0.3434349810918923, -0.344085480773615, -0.34473631362496204, -0.34538747916975754,
           -0.3460389769327321, -0.3466908064395219, -0.347342967216665, -0.34799545879160076, -0.3486482806926663,
           -0.3493014324490954, -0.3499549135910158, -0.3506087236494474, -0.35126286215629976, -0.3519173286443702,
           -0.35257212264734195, -0.353227243699782, -0.3538826913371381, -0.35453846509573816, -0.3551945645127872,
           -0.35585098912636504, -0.35650773847542516, -0.35716481209979223, -0.35782220954015964, -0.358479930338088,
           -0.3591379740360028, -0.3597963401771924, -0.3604550283058062, -0.3611140379668527, -0.36177336870619636,
           -0.36243302007055755, -0.36309299160750885, -0.3637532828654737, -0.36441389339372454, -0.36507482274238034,
           -0.3657360704624053, -0.36639763610560616, -0.36705951922463065, -0.36772171937296527, -0.36838423610493376,
           -0.3690470689756946, -0.36971021754123934, -0.37037368135839055, -0.3710374599848001, -0.3717015529789468,
           -0.3723659599001351, -0.3730306803084922, -0.37369571376496746, -0.37436105983132917, -0.3750267180701633,
           -0.37569268804487144, -0.3763589693196696, -0.37702556145958455, -0.3776924640304541, -0.37835967659892356,
           -0.3790271987324446, -0.3796950299992735, -0.38036316996846853]

# 定义最佳响应线，这里假设它是一个常数
best_response = np.full_like(xn_list, max(un_list))

# 绘制BS效用曲线
plt.plot(xn_list[0:190:10], un_list[0:190:10], 'g--o', label='Data owner utility')

# 绘制最佳响应线
plt.plot(xn_list[0:190:10], best_response[0:190:10], 'r--', label='Optimal strategy')

# 添加图例
plt.legend()

plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)

# 设置标题和坐标轴标签
plt.xlabel(r'$x_n$')
plt.ylabel(r'$U_n$')

# 显示网格
plt.grid(True)

# 显示图表
plt.show()
