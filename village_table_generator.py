import re
import unicodedata

# Paste raw source text from the bilingual PDF here
RAW_TEXT = """
Islands 離島
North 北區
Sai Kung 西貢
Sha Tin 沙田
Tuen Mun 屯門
Tai Po 大埔
Tsuen Wan 荃灣
Kwai Tsing 葵青
Yuen Long 元朗
Village Improvement Section 地政總署
Lands Department 鄉村改善組
September 2009 Edition 二ＯＯ九年九月版
UNDER THE NEW TERRITORIES SMALL HOUSE POLICY
LIST OF RECOGNIZED VILLAGES
在新界小型屋宇政策下之認可鄉村名冊
位於離島區的認可鄉村
RECOGNIZED VILLAGES IN ISLANDS DISTRICT
Village Name 村名 District 地區
1 KO LONG 高塱 LAMMA NORTH 南丫島北
2 LO TIK WAN 蘆荻灣 LAMMA NORTH 南丫島北
3 PAK KOK KAU TSUEN 北角舊村 LAMMA NORTH 南丫島北
4 PAK KOK SAN TSUEN 北角新村 LAMMA NORTH 南丫島北
5 SHA PO 沙埔 LAMMA NORTH 南丫島北
6 TAI PENG 大坪 LAMMA NORTH 南丫島北
7 TAI WAN KAU TSUEN 大灣舊村 LAMMA NORTH 南丫島北
8 TAI WAN SAN TSUEN 大灣新村 LAMMA NORTH 南丫島北
9 TAI YUEN 大園 LAMMA NORTH 南丫島北
10 WANG LONG 橫塱 LAMMA NORTH 南丫島北
11 YUNG SHUE LONG 榕樹塱 LAMMA NORTH 南丫島北
12 YUNG SHUE WAN 榕樹灣 LAMMA NORTH 南丫島北
13 LO SO SHING 蘆鬚城 LAMMA SOUTH 南丫島南
14 LUK CHAU 鹿洲 LAMMA SOUTH 南丫島南
15 MO TAT 模達 LAMMA SOUTH 南丫島南
16 MO TAT WAN 模達灣 LAMMA SOUTH 南丫島南
17 PO TOI 蒲台 LAMMA SOUTH 南丫島南
18 SOK KWU WAN 索罟灣 LAMMA SOUTH 南丫島南
19 TUNG O 東澳 LAMMA SOUTH 南丫島南
20 YUNG SHUE HA 榕樹下 LAMMA SOUTH 南丫島南
21 CHUNG HAU 涌口 MUI WO 梅窩
22 LUK TEI TONG 鹿地塘 MUI WO 梅窩
23 MAN KOK TSUI 萬角咀 MUI WO 梅窩
24 MANG TONG 盲塘 MUI WO 梅窩
25 MUI WO KAU TSUEN 梅窩舊村 MUI WO 梅窩
26 NGAU KWU LONG 牛牯塱 MUI WO 梅窩
位於離島區的認可鄉村
RECOGNIZED VILLAGES IN ISLANDS DISTRICT
Village Name 村名 District 地區
27 PAK MONG 白芒 MUI WO 梅窩
28 PAK NGAN HEUNG 白銀鄉 MUI WO 梅窩
29 TAI HO 大蠔 MUI WO 梅窩
30 TAI TEI TONG 大地塘 MUI WO 梅窩
31 TUNG WAN TAU 東灣頭 MUI WO 梅窩
32 WONG FUNG TIN 黃蜂田 MUI WO 梅窩
33 CHEUNG SHA LOWER VILLAGE 長沙下村 SOUTH LANTAU 大嶼山南
34 CHEUNG SHA UPPER VILLAGE 長沙上村 SOUTH LANTAU 大嶼山南
35 HAM TIN 鹹田 SOUTH LANTAU 大嶼山南
36 LO UK 羅屋 SOUTH LANTAU 大嶼山南
37 MONG TUNG WAN 望東灣 SOUTH LANTAU 大嶼山南
38 PUI O KAU TSUEN (LO WAI) 杯澳舊村 SOUTH LANTAU 大嶼山南
39 PUI O SAN TSUEN (SAN WAI) 杯澳新村 SOUTH LANTAU 大嶼山南
40 SHAN SHEK WAN 䃟石灣 SOUTH LANTAU 大嶼山南
41 SHAP LONG 十塱 SOUTH LANTAU 大嶼山南
42 SHUI HAU 水口 SOUTH LANTAU 大嶼山南
43 SIU A CHAU 小亞洲 SOUTH LANTAU 大嶼山南
44 TAI A CHAU 大亞洲 SOUTH LANTAU 大嶼山南
45 TAI LONG 大浪 SOUTH LANTAU 大嶼山南
46 TONG FUK 塘福 SOUTH LANTAU 大嶼山南
47 FAN LAU 分流 TAI O 大澳
48 KEUNG SHAN, LOWER 下羗山 TAI O 大澳
49 KEUNG SHAN, UPPER 上羗山 TAI O 大澳
50 LEUNG UK 梁屋 TAI O 大澳
51 LUK WU 鹿湖 TAI O 大澳
52 NGONG PING 昂平 TAI O 大澳
位於離島區的認可鄉村
RECOGNIZED VILLAGES IN ISLANDS DISTRICT
Village Name 村名 District 地區
53 SAN TAU 䃟頭 TAI O 大澳
54 SHA LO WAN 沙螺灣 TAI O 大澳
55 SHAM WAT 深屈 TAI O 大澳
56 SHAN SHEK WAN 䃟石灣 TAI O 大澳
57 YI O 二澳 TAI O 大澳
58 TAI LONG WAN 大浪灣 TAI O 大澳
59 CHEK LAP KOK SAN TSUEN 赤鱲角新村 TUNG CHUNG 東涌
60 HA LING PEI 下嶺皮 TUNG CHUNG 東涌
61 LAM CHE 藍輋 TUNG CHUNG 東涌
62 LUNG TSENG TAU 龍井頭 TUNG CHUNG 東涌
63 MA WAN 馬灣 TUNG CHUNG 東涌
64 MA WAN CHUNG 馬灣涌 TUNG CHUNG 東涌
65 MOK KA 莫家 TUNG CHUNG 東涌
66 NGAU AU 牛凹 TUNG CHUNG 東涌
67 NIM YUEN 稔園 TUNG CHUNG 東涌
68 SHAN HA (PA MEI) 山下(壩尾) TUNG CHUNG 東涌
69 SHEK LAU PO 石榴埔 TUNG CHUNG 東涌
70 SHEK MUN KAP 石門甲 TUNG CHUNG 東涌
71 SHEUNG LING PEI 上嶺皮 TUNG CHUNG 東涌
72 TAI PO 低埔 TUNG CHUNG 東涌
73 TEI TONG TSAI 地塘仔 TUNG CHUNG 東涌
74 WONG KA WAI 黃家圍 TUNG CHUNG 東涌
75 WONG NEI UK 黃泥屋 TUNG CHUNG 東涌
位於北區的認可鄉村
RECOGNIZED VILLAGES IN NORTH DISTRICT
Village Name 村名 District 地區
1 SHUNG HIM TONG 崇謙堂 FANLING 粉嶺
2 TONG HANG 塘坑 FANLING 粉嶺
3 FAN LENG LAU 粉嶺樓 FANLING 粉嶺
4 FANLING 粉嶺 FANLING 粉嶺
5 FU TEI PAI 虎地排 FANLING 粉嶺
6 HOK TAU WAI 鶴藪圍 FANLING 粉嶺
7 HUNG LENG 孔嶺 FANLING 粉嶺
8 KAN TAU TSUEN 簡頭村 FANLING 粉嶺
9 KO PO 高莆 FANLING 粉嶺
10 KWAI TAU LENG 龜頭嶺 FANLING 粉嶺
11 KWAN TEI 軍地 FANLING 粉嶺
12 LAU SHUI HEUNG 流水响 FANLING 粉嶺
13 LENG PEI TSUEN 嶺皮村 FANLING 粉嶺
14 LENG TSAI 嶺仔 FANLING 粉嶺
15 LUNG YEUK TAU (Including SAN
UK TSUEN, SAN WAI, WING NING
TSUEN, WING NING WAI, MA WAT
TSUEN, TUNG KOK WAI & LO WAI)
龍躍頭(包括︰新屋村，
新圍，永寧村，永寧圍
，麻笏村，東閣圍及老
圍)
FANLING 粉嶺
16 MA LIU SHUI SAN TSUEN 馬料水新村 FANLING 粉嶺
17 MA MEI HA 馬尾下 FANLING 粉嶺
18 MA MEI HA, LENG TSUI 馬尾下嶺咀 FANLING 粉嶺
19 MA WAT WAI 麻笏圍 FANLING 粉嶺
20 SAN TONG PO 新塘莆 FANLING 粉嶺
21 SAN UK TSAI 新屋仔 FANLING 粉嶺
22 SIU HANG SAN TSUEN 小坑新村 FANLING 粉嶺
位於北區的認可鄉村
RECOGNIZED VILLAGES IN NORTH DISTRICT
Village Name 村名 District 地區
23 TAN CHUK HANG 丹竹坑 FANLING 粉嶺
24 TSZ TONG TSUEN 祠堂村 FANLING 粉嶺
25 WA MEI SHAN 畫眉山 FANLING 粉嶺
26 WO HOP SHEK 和合石 FANLING 粉嶺
27 AP CHAU 鴨洲 SHA TAU KOK 沙頭角
28 SHEK CHUNG AU 石涌凹 SHA TAU KOK 沙頭角
29 A MA WAT 亞媽笏 SHA TAU KOK 沙頭角
30 AU HA 凹下 SHA TAU KOK 沙頭角
31 FUNG HANG 鳳坑 SHA TAU KOK 沙頭角
32 HA WO HANG 下禾坑 SHA TAU KOK 沙頭角
33 KAI KUK SHUE HA 雞谷樹下 SHA TAU KOK 沙頭角
34 KAT O 吉澳 SHA TAU KOK 沙頭角
35 KAU TAM TSO 九担租 SHA TAU KOK 沙頭角
36 KONG HA 崗下 SHA TAU KOK 沙頭角
37 KOP TONG 蛤塘 SHA TAU KOK 沙頭角
38 KUK PO 谷埔 SHA TAU KOK 沙頭角
39 LAI CHI WO 荔枝窩 SHA TAU KOK 沙頭角
40 LAI TAU SHEK 犂頭石 SHA TAU KOK 沙頭角
41 LIN MA HANG 蓮麻坑 SHA TAU KOK 沙頭角
42 LOI TUNG 萊洞 SHA TAU KOK 沙頭角
43 LUK KENG CHAN UK 鹿頸陳屋 SHA TAU KOK 沙頭角
44 LUK KENG WONG UK 鹿頸黃屋 SHA TAU KOK 沙頭角
45 MA TSEUK LENG 麻雀嶺 SHA TAU KOK 沙頭角
46 MAN UK PIN 萬屋邊 SHA TAU KOK 沙頭角
47 MIU TIN, HA 下苗田 SHA TAU KOK 沙頭角
48 MIU TIN, SHEUNG 上苗田 SHA TAU KOK 沙頭角
49 MUI TSZ LAM 梅子林 SHA TAU KOK 沙頭角
50 MUK MIN TAU 木棉頭 SHA TAU KOK 沙頭角
51 NAM CHUNG 南涌 SHA TAU KOK 沙頭角
位於北區的認可鄉村
RECOGNIZED VILLAGES IN NORTH DISTRICT
Village Name 村名 District 地區
52 NGAU SHI WU 牛屎湖 SHA TAU KOK 沙頭角
53 SAM A 三椏 SHA TAU KOK 沙頭角
54 SAN KWAI TIN 新桂田 SHA TAU KOK 沙頭角
55 SAN TSUEN 新村 SHA TAU KOK 沙頭角
56 SHAN TSUI 山嘴 SHA TAU KOK 沙頭角
57 SHEK KIU TAU 石橋頭 SHA TAU KOK 沙頭角
58 SHEUNG WO HANG 上禾坑 SHA TAU KOK 沙頭角
59 SO LO PUN 鎖羅盤 SHA TAU KOK 沙頭角
60 TAI TONG WU 大塘湖 SHA TAU KOK 沙頭角
61 TAM SHUI HANG 担水坑 SHA TAU KOK 沙頭角
62 TONG TO 塘肚 SHA TAU KOK 沙頭角
63 TSAT MUK KIU 七木橋 SHA TAU KOK 沙頭角
64 WANG SHAN KEUK 橫山脚 SHA TAU KOK 沙頭角
65 WO HANG TAI LONG 禾坑大朗 SHA TAU KOK 沙頭角
66 WU KAU TANG 烏蛟騰 SHA TAU KOK 沙頭角
67 WU SHEK KOK 烏石角 SHA TAU KOK 沙頭角
68 YIM TSO HA 鹽灶下 SHA TAU KOK 沙頭角
69 YUN SHUE AU 榕樹凹 SHA TAU KOK 沙頭角
70 WA SHAN 華山 SHEUNG SHUI 上水
71 YING PUN 營盤 SHEUNG SHUI 上水
72 CHEUNG LEK 長瀝 SHEUNG SHUI 上水
73 HANG TAU 坑頭 SHEUNG SHUI 上水
74 HO SHEUNG HEUNG 河上鄉 SHEUNG SHUI 上水
75 KAM TSIN 金錢 SHEUNG SHUI 上水
76 LIN TONG MEI 蓮塘尾 SHEUNG SHUI 上水
77 LIU POK 料壆 SHEUNG SHUI 上水
78 NG UK TSUEN 吳屋村 SHEUNG SHUI 上水
79 PING KONG 丙崗 SHEUNG SHUI 上水
位於北區的認可鄉村
RECOGNIZED VILLAGES IN NORTH DISTRICT
Village Name 村名 District 地區
80 SHEUNG SHUI 上水 SHEUNG SHUI 上水
81 TAI TAU LENG 大頭嶺 SHEUNG SHUI 上水
82 TONG KUNG LENG 唐公嶺 SHEUNG SHUI 上水
83 TSIU KENG 蕉徑 SHEUNG SHUI 上水
84 TSUNG PAK LONG 松栢塱 SHEUNG SHUI 上水
85 YIN KONG 燕崗 SHEUNG SHUI 上水
86 KAI LENG 雞嶺 SHEUNG SHUI 上水
87 NGA YIU 瓦窰 TA KWU LING 打鼓嶺
88 CHUK YUEN 竹園 TA KWU LING 打鼓嶺
89 FUNG WONG WU 鳳凰湖 TA KWU LING 打鼓嶺
90 HEUNG YUEN WAI 香園圍 TA KWU LING 打鼓嶺
91 KAN TAU WAI 簡頭圍 TA KWU LING 打鼓嶺
92 LEI UK 李屋 TA KWU LING 打鼓嶺
93 LO SHUE LING 老鼠嶺 TA KWU LING 打鼓嶺
94 MUK WU 木湖 TA KWU LING 打鼓嶺
95 PING CHE 坪輋 TA KWU LING 打鼓嶺
96 PING YEUNG 坪洋 TA KWU LING 打鼓嶺
97 SAN UK LING 新屋嶺 TA KWU LING 打鼓嶺
98 SHAN KAI WAT 山雞笏 TA KWU LING 打鼓嶺
99 TAI PO TIN 大埔田 TA KWU LING 打鼓嶺
100 TONG FONG 塘坊 TA KWU LING 打鼓嶺
101 TSUNG YUEN HA 松園下 TA KWU LING 打鼓嶺
102 WO KENG SHAN 禾徑山 TA KWU LING 打鼓嶺
位於西貢區的認可鄉村
RECOGNIZED VILLAGES IN SAI KUNG DISTRICT
Village Name 村名 District 地區
1 FAT TAU CHAU 佛頭洲 HANG HAU 坑口
2 HA YEUNG (Including MAU PO,
SIU HANG HAU)
下洋(包括︰茅莆，小坑口) HANG HAU 坑口
3 MA YAU TONG 馬游塘 HANG HAU 坑口
4 MANG KUNG UK 孟公屋 HANG HAU 坑口
5 MAU WU TSAI 茅湖仔 HANG HAU 坑口
6 PAN LONG WAN 檳榔灣 HANG HAU 坑口
7 PO TOI O 布袋澳 HANG HAU 坑口
8 SEUNG SZ WAN 相思灣 HANG HAU 坑口
9 SHEUNG YEUNG 上洋 HANG HAU 坑口
10 TAI HANG HAU 大坑口 HANG HAU 坑口
11 TAI PO TSAI (KV175729) 大埔仔 HANG HAU 坑口
12 TAI WAN TAU (Including TAI AU
MUN)
大環頭(包括︰大㘭門) HANG HAU 坑口
13 TIN HA WAN 田下灣 HANG HAU 坑口
14 TSENG LAN SHUE 井欄樹 HANG HAU 坑口
15 TSEUNG KWAN O 將軍澳 HANG HAU 坑口
16 YAU YUE WAN 魷魚灣 HANG HAU 坑口
17 CHE KENG TUK 輋徑篤 SAI KUNG 西貢
18 CHUK YUEN 竹園 SAI KUNG 西貢
19 HEUNG CHUNG 响鐘 SAI KUNG 西貢
20 HING KENG SHEK (Including
SAM FAI TIN)
慶徑石(包括︰三塊田) SAI KUNG 西貢
21 HO CHUNG 蠔涌 SAI KUNG 西貢
位於西貢區的認可鄉村
RECOGNIZED VILLAGES IN SAI KUNG DISTRICT
Village Name 村名 District 地區
22 KAI HAM (Including WANG CHE) 界咸(包括︰橫輋) SAI KUNG 西貢
23 LONG KE 浪茄 SAI KUNG 西貢
24 LONG KENG 浪徑 SAI KUNG 西貢
25 LUNG MEI 龍尾 SAI KUNG 西貢
26 MA NAM WAT 麻南笏 SAI KUNG 西貢
27 MAN WO 蠻窩 SAI KUNG 西貢
28 MAU PING NEW VILLAGE 茅坪新村 SAI KUNG 西貢
29 MOK TSE CHE 莫遮輋 SAI KUNG 西貢
30 NAM A 南丫 SAI KUNG 西貢
31 NAM SHAN (Including KAK
HANG TUN, LONG MEI)
南山(包括︰隔坑墩，朗尾) SAI KUNG 西貢
32 NAM WAI 南圍 SAI KUNG 西貢
33 O TAU 澳頭 SAI KUNG 西貢
34 PAK A 北丫 SAI KUNG 西貢
35 PAK KONG 北港 SAI KUNG 西貢
36 PAK KONG AU 北港凹 SAI KUNG 西貢
37 PAK LAP 白腊 SAI KUNG 西貢
38 PAK TAM 北潭 SAI KUNG 西貢
39 PAK TAM CHUNG (SHEUNG
YIU)
北潭涌(上窰) SAI KUNG 西貢
40 PAK WAI 北圍 SAI KUNG 西貢
41 PIK UK 壁屋 SAI KUNG 西貢
42 PING TUN 坪墩 SAI KUNG 西貢
43 SAI WAN 西灣 SAI KUNG 西貢
位於西貢區的認可鄉村
RECOGNIZED VILLAGES IN SAI KUNG DISTRICT
Village Name 村名 District 地區
44 SHA HA 沙下 SAI KUNG 西貢
45 SHA KOK MEI (Including NGAU
LIU (KV190790))
沙角尾(包括︰牛寮) SAI KUNG 西貢
46 SHAN LIU 山寮 SAI KUNG 西貢
47 SHE TAU 蛇頭 SAI KUNG 西貢
48 SHEK HANG 石坑 SAI KUNG 西貢
49 SHEK LUNG TSAI NEW
VILLAGE
石壟仔新村 SAI KUNG 西貢
50 TA HO TUN 打蠔墩 SAI KUNG 西貢
51 TAI LAM WU (Including NGAU
LIU (KV149745))
大藍湖(包括︰牛寮) SAI KUNG 西貢
52 TAI LONG (Including LAM UK,
HAM TIN)
大浪(包括︰林屋，鹹田) SAI KUNG 西貢
53 TAI MONG TSAI 大網仔 SAI KUNG 西貢
54 TAI NO 大腦 SAI KUNG 西貢
55 TAI NO SHEUNG YEUNG
(Including TIN LIU)
大腦上陽(包括︰田寮) SAI KUNG 西貢
56 TAI PO TSAI (KV217795) 西貢大埔仔 SAI KUNG 西貢
57 TAI SHE WAN 大蛇灣 SAI KUNG 西貢
58 TAI WAN 大環 SAI KUNG 西貢
59 TAM WAT 氹笏 SAI KUNG 西貢
60 TIT KIM HANG 鐵鉗坑 SAI KUNG 西貢
61 TSAK YUE WU 鯽魚湖 SAI KUNG 西貢
62 TSAM CHUK WAN 斬竹灣 SAI KUNG 西貢
63 TSIU HANG 蕉坑 SAI KUNG 西貢
64 TSO WO HANG 早禾坑 SAI KUNG 西貢
位於西貢區的認可鄉村
RECOGNIZED VILLAGES IN SAI KUNG DISTRICT
Village Name 村名 District 地區
65 TUI MIN HOI (Including TSIU
LUNG, SHUI TSING TAU)
對面海(包括︰蕉壟，水井
頭)
SAI KUNG 西貢
66 TUNG A (or LEUNG SHUN WAN) 東丫(又名糧船灣) SAI KUNG 西貢
67 UK CHEUNG 屋場 SAI KUNG 西貢
68 WO LIU 禾寮 SAI KUNG 西貢
69 WO MEI 窩美(在集體官契中又稱窩
尾)
SAI KUNG 西貢
70 WONG CHUK SHAN NEW
VILLAGE
黃竹山新村 SAI KUNG 西貢
71 WONG CHUK WAN (Including
NGONG WO)
黃竹灣(包括︰昂窩) SAI KUNG 西貢
72 WONG KENG TEI 黃麖地 SAI KUNG 西貢
73 WONG KENG TSAI 黃麖仔 SAI KUNG 西貢
74 WONG MO YING 黃毛應 SAI KUNG 西貢
75 WONG YI (NAI) CHAU 黃宜 (泥) 洲 SAI KUNG 西貢
76 YIM TIN TSAI 鹽田仔 SAI KUNG 西貢
77 KAU SAI SAN TSUEN 滘西新村 SAI KUNG 西貢
位於沙田區的認可鄉村
RECOGNIZED VILLAGES IN SHA TIN DISTRICT
Village Name 村名 District 地區
1 AU PUI WAN 㘭背灣 SHA TIN 沙田
2 CHAP WAI KON 插桅杆 SHA TIN 沙田
3 CHEK NAI PING 赤坭坪 SHA TIN 沙田
4 CHEUNG LEK MEI 長瀝尾 SHA TIN 沙田
5 FA SAM HANG 花心坑 SHA TIN 沙田
6 FO TAN 火炭 SHA TIN 沙田
7 FU YUNG PIT 芙蓉 SHA TIN 沙田
8 HA KENG HAU 下徑口 SHA TIN 沙田
9 HO LEK PUI 河瀝背 SHA TIN 沙田
10 KAK TIN 隔田 SHA TIN 沙田
11 KAU TO 九肚 SHA TIN 沙田
12 KONG PUI TSUEN 崗背村 SHA TIN 沙田
13 KWUN YAM SHAN 觀音山 SHA TIN 沙田
14 LEI UK TSUEN 李屋村 SHA TIN 沙田
15 LO SHU TIN 老鼠田 SHA TIN 沙田
16 LOK LO HA 落路下 SHA TIN 沙田
17 MA NIU 馬尿 SHA TIN 沙田
18 MA ON SHAN TSUEN 馬鞍山村 SHA TIN 沙田
19 MAU PING 茅坪 SHA TIN 沙田
20 MAU TAT 茅撻 SHA TIN 沙田
21 MAU TSO NGAM 茂草岩 SHA TIN 沙田
22 MUI TSZ LAM 梅子林 SHA TIN 沙田
23 NAM SHAN 南山 SHA TIN 沙田
位於沙田區的認可鄉村
RECOGNIZED VILLAGES IN SHA TIN DISTRICT
Village Name 村名 District 地區
24 NGAU PEI SHA 牛皮沙 SHA TIN 沙田
25 NGAU WU TOK 牛湖托 SHA TIN 沙田
26 NGONG PING 昂平 SHA TIN 沙田
27 NIM AU 稔凹 SHA TIN 沙田
28 PAI TAU 排頭 SHA TIN 沙田
29 SAN TIN WAI 新田圍 SHA TIN 沙田
30 SHA TIN TAU 沙田頭 SHA TIN 沙田
31 SHA TIN WAI 沙田圍 SHA TIN 沙田
32 SHAN HA WAI (TSANG TAI UK) 山下圍(曾大屋) SHA TIN 沙田
33 SHAN MEI 山尾 SHA TIN 沙田
34 SHAP YI WAT 十二笏 SHA TIN 沙田
35 SHEK KWU LUNG 石古壟 SHA TIN 沙田
36 SHEK LAU TUNG 石榴洞 SHA TIN 沙田
37 SHEK LUNG TSAI 石壟仔 SHA TIN 沙田
38 SHEUNG KENG HAU 上徑口 SHA TIN 沙田
39 SHEUNG WO CHE 上禾輋 SHA TIN 沙田
40 SIU LEK YUEN 小瀝源 SHA TIN 沙田
41 TAI CHE 大輋 SHA TIN 沙田
42 TAI LAM LIU 大藍寮 SHA TIN 沙田
43 TAI SHUI HANG 大水坑 SHA TIN 沙田
44 TAI WAI 大圍 SHA TIN 沙田
45 TIN LIU 田寮 SHA TIN 沙田
位於沙田區的認可鄉村
RECOGNIZED VILLAGES IN SHA TIN DISTRICT
Village Name 村名 District 地區
46 TIN SAM 田心 SHA TIN 沙田
47 TO SHEK 多石 SHA TIN 沙田
48 WO LIU HANG 禾寮坑 SHA TIN 沙田
49 WO SHEUNG TUN 禾上墩 SHA TIN 沙田
50 WONG CHUK SHAN 黃竹山 SHA TIN 沙田
51 WONG CHUK YEUNG 黃竹洋 SHA TIN 沙田
52 WONG NAI TAU 黃泥頭 SHA TIN 沙田
53 WONG UK 王屋 SHA TIN 沙田
54 WU KAI SHA (Including CHEUNG
KANG)
烏溪沙(包括︰長徑) SHA TIN 沙田
55 YUEN CHAU KOK (TSE UK) 圓洲角(謝屋) SHA TIN 沙田
56 FUI YIU HA RESITE AREA 灰窰下新村 SHA TIN 沙田
57 HEUNG FAN LIU RESITE AREA 香粉寮新村 SHA TIN 沙田
58 KAK TIN KUNG MUI 隔田公廟 SHA TIN 沙田
59 KWAI TEI RESITE AREA 龜地新村 SHA TIN 沙田
60 PAT TSZ WO RESITE AREA 拔子窩新村 SHA TIN 沙田
61 SHA TIN WAI RESITE AREA 沙田圍新村 SHA TIN 沙田
62 TSOK POK HANG RESITE AREA 作壆坑新村 SHA TIN 沙田
63 HA WO CHE 下禾輋 SHA TIN 沙田
64 HIN TIN 顯田 SHA TIN 沙田
65 TUNG LO WAN 銅鑼灣 SHA TIN 沙田
位於屯門區的認可鄉村
RECOGNIZED VILLAGES IN TUEN MUN DISTRICT
Village Name 村名 District 地區
1 FU TEI 虎地 TUEN MUN 屯門
2 CHUNG UK TSUEN 鍾屋村 TUEN MUN 屯門
3 KEI LUN WAI 麒麟圍 TUEN MUN 屯門
4 LAM TEI 藍地 TUEN MUN 屯門
5 LAM TEI SAN TSUEN 藍地新村 TUEN MUN 屯門
6 LUNG KWU TAN 龍鼓灘 TUEN MUN 屯門
7 NAI WAI 泥圍 TUEN MUN 屯門
8 NIM WAN 稔灣 TUEN MUN 屯門
9 PO TONG HA 寶塘下 TUEN MUN 屯門
10 SAN HING TSUEN 新慶村 TUEN MUN 屯門
11 SHUN FUNG WAI 順風圍 TUEN MUN 屯門
12 SIU HANG TSUEN 小坑村 TUEN MUN 屯門
13 SO KWUN WAT 掃管笏 TUEN MUN 屯門
14 TAI LAM CHUNG 大欖涌 TUEN MUN 屯門
15 TIN FU CHAI 田夫仔 TUEN MUN 屯門
16 TO YUEN WAI 桃園圍 TUEN MUN 屯門
17 TSING CHUEN WAI 青磚圍 TUEN MUN 屯門
18 TSZ TIN TSUEN 紫田村 TUEN MUN 屯門
19 TUEN MUN KAU HUI 屯門舊墟 TUEN MUN 屯門
20 TUEN TSZ WAI 屯子圍 TUEN MUN 屯門
21 WONG KA WAI 黃家圍 TUEN MUN 屯門
22 YEUNG SIU HANG 楊小坑 TUEN MUN 屯門
23 TUEN MUN SAN HUI 屯門新墟 TUEN MUN 屯門
位於大埔區的認可鄉村
RECOGNIZED VILLAGES IN TAI PO DISTRICT
Village Name 村名 District 地區
1 CHE HA 輋下 SAI KUNG NORTH 西貢北
2 CHEK KENG 赤徑 SAI KUNG NORTH 西貢北
3 CHEUNG MUK TAU 樟木頭 SAI KUNG NORTH 西貢北
4 CHEUNG SHEUNG 嶂上 SAI KUNG NORTH 西貢北
5 HA YEUNG 下洋 SAI KUNG NORTH 西貢北
6 HOI HA 海下 SAI KUNG NORTH 西貢北
7 KAU LAU WAN 較流灣 SAI KUNG NORTH 西貢北
8 KEI LING HA LO WAI 企嶺下老圍 SAI KUNG NORTH 西貢北
9 KEI LING HA SAN WAI 企嶺下新圍 SAI KUNG NORTH 西貢北
10 KO TONG 高塘 SAI KUNG NORTH 西貢北
11 KWUN HANG 官坑 SAI KUNG NORTH 西貢北
12 LAI CHI CHONG 荔枝莊 SAI KUNG NORTH 西貢北
13 MA KWU LAM 馬牯纜 SAI KUNG NORTH 西貢北
14 NAI CHUNG 泥涌 SAI KUNG NORTH 西貢北
15 NAM SHAN TUNG 南山洞 SAI KUNG NORTH 西貢北
16 PAK SHA O 白沙澳 SAI KUNG NORTH 西貢北
17 PAK SHA O HA YEUNG 白沙澳下洋 SAI KUNG NORTH 西貢北
18 PAK TAM AU 北潭凹 SAI KUNG NORTH 西貢北
19 PING CHAU CHAU MEI 平洲洲尾 SAI KUNG NORTH 西貢北
20 PING CHAU CHAU TAU 平洲洲頭 SAI KUNG NORTH 西貢北
21 PING CHAU NAI TAU 平洲奶頭 SAI KUNG NORTH 西貢北
22 PING CHAU SHA TAU 平洲沙頭 SAI KUNG NORTH 西貢北
23 PING CHAU TAI TONG 平洲大塘 SAI KUNG NORTH 西貢北
24 SAI KENG 西徑 SAI KUNG NORTH 西貢北
位於大埔區的認可鄉村
RECOGNIZED VILLAGES IN TAI PO DISTRICT
Village Name 村名 District 地區
25 SAI O 西澳 SAI KUNG NORTH 西貢北
26 SHAM CHUNG 深涌 SAI KUNG NORTH 西貢北
27 TAI TAN 大灘 SAI KUNG NORTH 西貢北
28 TAI TUNG 大洞 SAI KUNG NORTH 西貢北
29 TAI TUNG WO LIU 大洞禾寮 SAI KUNG NORTH 西貢北
30 TAN KA WAN 蛋家灣 SAI KUNG NORTH 西貢北
31 TAP MUN 塔門 SAI KUNG NORTH 西貢北
32 TO KWA PENG 土瓜坪 SAI KUNG NORTH 西貢北
33 TSENG TAU (SAI KUNG NORTH) 井頭 SAI KUNG NORTH 西貢北
34 TUNG SAM KEI 東心其 SAI KUNG NORTH 西貢北
35 UK TAU 屋頭 SAI KUNG NORTH 西貢北
36 WONG CHUK YEUNG 黃竹洋 SAI KUNG NORTH 西貢北
37 YUNG SHUE AU 榕樹凹 SAI KUNG NORTH 西貢北
38 SAM MUN TSAI SAN TSUEN 三門仔新村 TAI PO 大埔
39 SAN WAI TSAI 新圍仔 TAI PO 大埔
40 CHAI KEK 寨乪 TAI PO 大埔
41 CHEUNG SHUE TAN 樟樹灘 TAI PO 大埔
42 CHEUNG UK TEI 張屋地 TAI PO 大埔
43 CHUEN SHUI TSENG 泉水井 TAI PO 大埔
44 CHUNG UK TSUEN 鍾屋村 TAI PO 大埔
45 FONG MA PO 放馬莆 TAI PO 大埔
46 FUNG YUEN 鳳園 TAI PO 大埔
47 HA HANG 下坑 TAI PO 大埔
48 HA TEI HA 蝦地下 TAI PO 大埔
位於大埔區的認可鄉村
RECOGNIZED VILLAGES IN TAI PO DISTRICT
Village Name 村名 District 地區
49 HANG HA PO 坑下莆 TAI PO 大埔
50 KAU LIU HA 較寮下 TAI PO 大埔
51 KAU LUNG HANG 九龍坑 TAI PO 大埔
52 KO TIN HOM 高田墈 TAI PO 大埔
53 LAI CHI SHAN 荔枝山 TAI PO 大埔
54 LIN AU, CHENG UK 蓮凹鄭屋 TAI PO 大埔
55 LIN AU, LEI UK 蓮凹李屋 TAI PO 大埔
56 LO TSZ TIN 蘆慈田 TAI PO 大埔
57 LUNG A PAI 龍丫排 TAI PO 大埔
58 LUNG MEI 龍尾 TAI PO 大埔
59 MA PO MEI 麻布尾 TAI PO 大埔
60 NAM HANG 南坑 TAI PO 大埔
61 NAM WA PO 南華莆 TAI PO 大埔
62 NG TUNG CHAI 梧桐寨 TAI PO 大埔
63 PAN CHUNG 泮涌 TAI PO 大埔
64 PAK NGAU SHEK HA TSUEN 白牛石下村 TAI PO 大埔
65 PAK NGAU SHEK SHEUNG
TSUEN
白牛石上村 TAI PO 大埔
66 PING LONG 坪朗 TAI PO 大埔
67 PING SHAN CHAI 平山寨 TAI PO 大埔
68 PO SAM PAI 布心排 TAI PO 大埔
69 PUN SHAN CHAU 半山洲 TAI PO 大埔
位於大埔區的認可鄉村
RECOGNIZED VILLAGES IN TAI PO DISTRICT
Village Name
70 SAN TAU KOK 徹頭角 TAI PO 大埔
71 SAN TONG 新塘 TAI PO 大埔
村名
District 地區
72 SAN TSUEN (LAM TSUEN) 新村(林村) TAI PO 大埔
73 SAN UK КА 新屋家 TAI PO 大埔
74 SAN UK PAI 新屋排 TAI PO 大埔
75 SAN UK TSAI 新屋仔 TAI PO 大埔
76 |SHA LO TUNG CHEUNG UK 沙螺洞張屋 TAI PO 大埔
77 SHA LO TUNG LEI UK 沙螺洞李屋 TAI PO 大埔
78 SHAN LIU (Including LAI PEK
SHAN & LAI PEK SHAN SAN
TSUEN)
山寮(包括: 壁山及壁山 TAI PO 大埔
新村)
79 SHE SHAN 社山 TAI PO 大埔
80 |SHEK KWU LUNG 石古 TAI PO 大埔
81 SHUEN WAN CHAN UK 船灣陳屋 TAI PO 大埔
82 |SHUEN WAN CHIM UK 船灣詹屋 TAI PO 大埔
83 SHUEN WAN LEI UK 船灣李屋 TAI PO 大埔
84 SHUEN WAN SHA LAN 船灣沙欄 TAI PO 大埔
85 SHUEN WAN WAI HA 船灣圍下 TAI PO 大埔
86 SHUI WO (Including SHA PA) 水窩(包括:沙壩) TAI PO 大埔
87 SIU OM SHAN 小菴山 TAI PO 大埔
88 TA TIT YAN 打鐵 TAI PO 大埔
位於大埔區的認可鄉村
RECOGNIZED VILLAGES IN TAI PO DISTRICT
Village Name 村名 District 地區
89 TAI HANG 太坑 TAI PO 大埔
90 TAI MEI TUK 大美篤 TAI PO 大埔
91 TAI MONG CHE 大芒輋 TAI PO 大埔
92 TAI OM 大菴 TAI PO 大埔
93 TAI OM SHAN 大菴山 TAI PO 大埔
94 TAI PO HUI 大埔墟 TAI PO 大埔
95 TAI PO KAU 大埔滘 TAI PO 大埔
96 TAI PO KAU HUI 大埔舊墟 TAI PO 大埔
97 TAI PO MEI 大埔尾 TAI PO 大埔
98 TAI PO TAU 大埔頭 TAI PO 大埔
99 TAI PO TAU SHUI WAI 大埔頭水圍 TAI PO 大埔
100 TAI WO 大窩 TAI PO 大埔
101 TIN LIU HA 田寮下 TAI PO 大埔
102 TING KOK 汀角 TAI PO 大埔
103 TO YUEN TUNG 桃源洞 TAI PO 大埔
104 TONG SHEUNG TSUEN 塘上村 TAI PO 大埔
位於大埔區的認可鄉村
RECOGNIZED VILLAGES IN TAI PO DISTRICT
Village Name 村名 District 地區
105 TSENG TAU (Including A SHAN &
TUNG TSZ)
井頭(包括︰鴉山及洞梓) TAI PO 大埔
106 WAI TAU TSUEN 圍頭村 TAI PO 大埔
107 WAN TAU KOK 運頭角 TAI PO 大埔
108 WO LIU 禾寮 TAI PO 大埔
109 WONG YI AU, HA 下黃宜凹 TAI PO 大埔
110 WONG YI AU, SHEUNG 上黃宜凹 TAI PO 大埔
111 WONG YUE TAN 黃魚灘 TAI PO 大埔
112 WUN YIU 碗窰 TAI PO 大埔
113 YIN NGAM 燕岩 TAI PO 大埔
114 YUEN LENG, LEI UK 元嶺李屋 TAI PO 大埔
115 YUEN LENG, YIP UK 元嶺葉屋 TAI PO 大埔
116 YUEN TUN HA 元墩下 TAI PO 大埔
117 KAM SHAN 錦山 TAI PO 大埔
118 MUI SHUE HANG 梅樹坑 TAI PO 大埔
119 PAN CHUNG SAN TSUEN 泮涌新村 TAI PO 大埔
120 YUE KOK 魚角 TAI PO 大埔
位於荃灣區的認可鄉村
RECOGNIZED VILLAGES IN TSUEN WAN DISTRICT
Village Name 村名 District 地區
1 FA PENG 花坪 MA WAN & NORTH EAST LANTAU 馬灣及大嶼山東北
2 LUK KENG 鹿頸 MA WAN & NORTH EAST LANTAU 馬灣及大嶼山東北
3 MA WAN MAIN STREET 馬灣正街 MA WAN & NORTH EAST LANTAU 馬灣及大嶼山東北
4 PA TAU KU 扒頭鼓 MA WAN & NORTH EAST LANTAU 馬灣及大嶼山東北
5 TA PANG PO 打棚埔 MA WAN & NORTH EAST LANTAU 馬灣及大嶼山東北
6 TAI CHEUN 大串 MA WAN & NORTH EAST LANTAU 馬灣及大嶼山東北
7 TAI TSING CHAU 大青洲 MA WAN & NORTH EAST LANTAU 馬灣及大嶼山東北
8 TIN LIU 田寮 MA WAN & NORTH EAST LANTAU 馬灣及大嶼山東北
9 CHUEN LUNG 川龍 TSUEN WAN 荃灣
10 HA TONG LEK 下塘瀝 TSUEN WAN 荃灣
11 HOI PA 海壩 TSUEN WAN 荃灣
12 LO WAI 老圍 TSUEN WAN 荃灣
13 SAN TSUEN 新村 TSUEN WAN 荃灣
14 SHEK WAI KOK 石圍角 TSUEN WAN 荃灣
15 SHEUNG FA SHAN 上花山 TSUEN WAN 荃灣
16 SHEUNG KWAI CHUNG 上葵涌 TSUEN WAN 荃灣
17 SHEUNG TONG 上塘 TSUEN WAN 荃灣
18 TA CHUEN PING 打磚坪 TSUEN WAN 荃灣
19 TAI PAK TIN 大白田 TSUEN WAN 荃灣
20 TING KAU 汀九 TSUEN WAN 荃灣
位於荃灣區的認可鄉村
RECOGNIZED VILLAGES IN TSUEN WAN DISTRICT
Village Name 村名 District 地區
21 TSING FAI TONG 清快塘 TSUEN WAN 荃灣
22 TSING LUNG TAU 青龍頭 TSUEN WAN 荃灣
23 WO YI HOP 和宜合 TSUEN WAN 荃灣
24 YAU KOM TAU 油柑頭 TSUEN WAN 荃灣
25 YAU MA HOM 油麻磡 TSUEN WAN 荃灣
26 YI PEI CHUN 二陂圳 TSUEN WAN 荃灣
27 YUEN TUEN NEW VILLAGE 圓墪新村 TSUEN WAN 荃灣
28 HAM TIN NEW VILLAGE 咸田村 TSUEN WAN 荃灣
29 HO PUI NEW VILLAGE 河背村 TSUEN WAN 荃灣
30 HOI PA NEW VILLAGE 海壩村 TSUEN WAN 荃灣
31 KWAN MUN HAU NEW
VILLAGE
關門口村 TSUEN WAN 荃灣
32 MUK MIN HA 木棉下 TSUEN WAN 荃灣
33 PAK TIN PA 白田壩 TSUEN WAN 荃灣
34 SAI LAU KOK 西樓角 TSUEN WAN 荃灣
35 SHAM TSENG RE-SITE
VILLAGE
深井村 TSUEN WAN 荃灣
36 SAM TUNG UK 三棟屋 TSUEN WAN 荃灣
37 SHEK PIK SAN TSUEN 石碧新村 TSUEN WAN 荃灣
38 TAI UK WAI 大屋圍 TSUEN WAN 荃灣
39 YEUNG UK NEW VILLAGE 楊屋村 TSUEN WAN 荃灣
位於葵青區的認可鄉村
RECOGNIZED VILLAGES IN KWAI TSING DISTRICT
Village Name 村名 District 地區
1 KAU WAH KENG 九華徑 KWAI CHUNG 葵涌
2 CHUNG KWAI CHUNG
VILLAGE
中葵涌村 KWAI CHUNG 葵涌
3 HA KWAI CHUNG VILLAGE 下葵涌村 KWAI CHUNG 葵涌
4 CHUNG MEI 涌尾 TSING YI 青衣
5 LAM TIN 藍田 TSING YI 青衣
6 LO UK 老屋 TSING YI 青衣
7 SAN UK 新屋 TSING YI 青衣
8 TAI WONG HA 大王下 TSING YI 青衣
9 YIM TIN KOK 鹽田角 TSING YI 青衣
位於元朗區的認可鄉村
RECOGNIZED VILLAGES IN YUEN LONG DISTRICT
Village Name 村名 District 地區
1 FUNG KONG TSUEN 鳳降村 HA TSUEN 廈村
2 HA TSUEN SAN WAI 廈村新圍 HA TSUEN 廈村
3 HA TSUEN SHI 廈村市 HA TSUEN 廈村
4 HONG MEI TSUEN 巷尾村 HA TSUEN 廈村
5 LEI UK TSUEN 李屋村 HA TSUEN 廈村
6 LO UK TSUEN 羅屋村 HA TSUEN 廈村
7 SAN SANG TSUEN 新生村 HA TSUEN 廈村
8 SAN UK TSUEN 新屋村 HA TSUEN 廈村
9 SIK KONG TSUEN 錫降村 HA TSUEN 廈村
10 SIK KONG WAI 錫降圍 HA TSUEN 廈村
11 TIN SUM TSUEN 田心村 HA TSUEN 廈村
12 TSEUNG KONG WAI 祥降圍 HA TSUEN 廈村
13 TUNG TAU TSUEN 東頭村 HA TSUEN 廈村
14 FUNG KAT HEUNG 逢吉鄉 KAM TIN 錦田
15 CHI TONG TSUEN 祠堂村 KAM TIN 錦田
16 KAM HING WAI 錦慶圍 KAM TIN 錦田
17 KAM TIN SHI 錦田市 KAM TIN 錦田
18 KAT HING WAI 吉慶圍 KAM TIN 錦田
19 KO PO 高埔 KAM TIN 錦田
20 SHA PO TSUEN 沙埔村 KAM TIN 錦田
21 SHUI MEI TSUEN 水尾村 KAM TIN 錦田
22 SHUI TAU TSUEN 水頭村 KAM TIN 錦田
23 TAI HONG WAI 泰康圍 KAM TIN 錦田
24 WING LUNG WAI 永隆圍 KAM TIN 錦田
25 KAM TIN SAN TSUEN 錦田新村 KAM TIN 錦田
26 TSAT SING KONG 七星崗 PAT HEUNG 八鄉
位於元朗區的認可鄉村
RECOGNIZED VILLAGES IN YUEN LONG DISTRICT
Village Name 村名 District 地區
27 CHEUNG KONG TSUEN 長江村 PAT HEUNG 八鄉
28 NG KA TSUEN 吳家村 PAT HEUNG 八鄉
29 TAI KONG PO 大江埔 PAT HEUNG 八鄉
30 CHEUNG PO 長莆 PAT HEUNG 八鄉
31 CHUK HANG 竹坑 PAT HEUNG 八鄉
32 HA CHE 下輋 PAT HEUNG 八鄉
33 HO PUI 河背 PAT HEUNG 八鄉
34 LEUNG UK TSUEN 梁屋村 PAT HEUNG 八鄉
35 LIN FA TEI 蓮花地 PAT HEUNG 八鄉
36 LO UK TSUEN 羅屋村 PAT HEUNG 八鄉
37 MA ON KONG 馬鞍崗 PAT HEUNG 八鄉
38 NGAU KENG 牛徑 PAT HEUNG 八鄉
39 SHEK WU TONG 石湖塘 PAT HEUNG 八鄉
40 SHEUNG CHE 上輋 PAT HEUNG 八鄉
41 SHEUNG TSUEN 上村 PAT HEUNG 八鄉
42 SHUI LAU TIN 水流田 PAT HEUNG 八鄉
43 SHUI TSAN TIN 水盞田 PAT HEUNG 八鄉
44 TA SHEK WU 打石湖 PAT HEUNG 八鄉
45 TAI KEK 大乪 PAT HEUNG 八鄉
46 TAI WOR 大窩 PAT HEUNG 八鄉
47 TIN SAM 田心 PAT HEUNG 八鄉
48 WANG TOI SHAN 橫台山 PAT HEUNG 八鄉
49 YUEN KONG 元崗 PAT HEUNG 八鄉
50 YUEN KONG SAN TSUEN 元崗新村 PAT HEUNG 八鄉
51 KAM TSIN WAI 金錢圍 PAT HEUNG 八鄉
52 KAP LUNG 甲龍 PAT HEUNG 八鄉
53 FUNG CHI TSUEN 鳳池村 PING SHAN 屏山
位於元朗區的認可鄉村
RECOGNIZED VILLAGES IN YUEN LONG DISTRICT
Village Name 村名 District 地區
54 FUNG KA WAI 馮家圍 PING SHAN 屏山
55 WING NING TSUEN 永寧村 PING SHAN 屏山
56 CHUNG SUM WAI 中心圍 PING SHAN 屏山
57 FUI SHA WAI 灰沙圍 PING SHAN 屏山
58 FUK HING TSUEN 福慶村 PING SHAN 屏山
59 HA MEI SAN TSUEN 虾尾新村 PING SHAN 屏山
60 HANG MEI TSUEN 坑尾村 PING SHAN 屏山
61 HANG TAU TSUEN 坑頭村 PING SHAN 屏山
62 HUNG UK TSUEN 洪屋村 PING SHAN 屏山
63 KIU TAU WAI 橋頭圍 PING SHAN 屏山
64 LAM HAU TSUEN 欖口村 PING SHAN 屏山
65 LAM UK TSUEN 林屋村 PING SHAN 屏山
66 MONG TSENG TSUEN 輞井村 PING SHAN 屏山
67 MONG TSENG WAI 輞井圍 PING SHAN 屏山
68 NG UK TSUEN 吳屋村 PING SHAN 屏山
69 NGAU HOM TSUEN 牛磡村 PING SHAN 屏山
70 SAI TAU WAI 西頭圍 PING SHAN 屏山
71 SAN HING TSUEN 新慶村 PING SHAN 屏山
72 SHA KONG WAI 沙江圍 PING SHAN 屏山
73 SHAN HA 山下 PING SHAN 屏山
74 SHEK PO TSUEN 石埔村 PING SHAN 屏山
75 SHEUNG CHEUNG WAI 上章圍 PING SHAN 屏山
76 SHING UK TSUEN 盛屋村 PING SHAN 屏山
77 SHUI PIN TSUEN 水邊村 PING SHAN 屏山
78 SHUI PIN WAI 水邊圍 PING SHAN 屏山
79 TAI TSENG WAI 大井圍 PING SHAN 屏山
位於元朗區的認可鄉村
RECOGNIZED VILLAGES IN YUEN LONG DISTRICT
Village Name 村名 District 地區
80 TONG FONG TSUEN 塘坊村 PING SHAN 屏山
81 TUNG TAU TSUEN 東頭村 PING SHAN 屏山
82 YEUNG UK TSUEN 楊屋村 PING SHAN 屏山
83 SHUI TIN TSUEN 水田村 PING SHAN 屏山
84 PING SHAN SAN TSUEN 屏山新村 PING SHAN 屏山
85 SAN TIN HA SAN WAI 新田下新圍 SAN TIN 新田
86 SAN TIN SHEUNG SAN WAI 新田上新圍 SAN TIN 新田
87 SHEUNG CHUK YUEN 上竹園 SAN TIN 新田
88 WAI TSAI 圍仔 SAN TIN 新田
89 CHAU TAU 洲頭 SAN TIN 新田
90 CHING LOONG TSUEN 青龍村 SAN TIN 新田
91 FAN TIN (SAN YI CHO AND MING
TAK TONG)
蕃田(野祖及明德堂) SAN TIN 新田
92 HA CHUK YUEN 下竹園 SAN TIN 新田
93 LOK MA CHAU 落馬洲 SAN TIN 新田
94 MAI PO TSUEN 米埔村 SAN TIN 新田
95 ON LOONG TSUEN 安龍村 SAN TIN 新田
96 POK WAI 壆圍 SAN TIN 新田
97 POON UK TSUEN 潘屋村 SAN TIN 新田
98 SAN LOONG TSUEN 新龍村 SAN TIN 新田
99 SHEK WU WAI 石湖圍 SAN TIN 新田
100 TUNG CHUN WAI 東鎮圍 SAN TIN 新田
101 WING PING TSUEN 永平村 SAN TIN 新田
102 YAN SAU WAI 仁壽圍 SAN TIN 新田
103 CHUK HANG (TAI WAI WO LIU) 竹坑(大圍禾寮) SHAP PAT HEUNG 十八鄉
位於元朗區的認可鄉村
RECOGNIZED VILLAGES IN YUEN LONG DISTRICT
Village Name 村名 District 地區
104 LUNG TIN TSUEN 龍田村 SHAP PAT HEUNG 十八鄉
105 NGA YIU TAU 瓦窰頭 SHAP PAT HEUNG 十八鄉
106 SHUNG CHING SAN TSUEN 崇正新村 SHAP PAT HEUNG 十八鄉
107 HA YAU TIN TSUEN 下攸田村 SHAP PAT HEUNG 十八鄉
108 HUNG TSO TIN TSUEN 紅棗田村 SHAP PAT HEUNG 十八鄉
109 KONG TAU SAN TSUEN 港頭新村 SHAP PAT HEUNG 十八鄉
110 KONG TAU TSUEN 港頭村 SHAP PAT HEUNG 十八鄉
111 MA TIN TSUEN 馬田村 SHAP PAT HEUNG 十八鄉
112 MUK KIU TAU TSUEN 木橋頭村 SHAP PAT HEUNG 十八鄉
113 NAM HANG TSUEN 南坑村 SHAP PAT HEUNG 十八鄉
114 NAM PIN WAI 南邊圍 SHAP PAT HEUNG 十八鄉
115 PAK SHA TSUEN 白沙村 SHAP PAT HEUNG 十八鄉
116 SAI PIN WAI 西邊圍 SHAP PAT HEUNG 十八鄉
117 SHAM CHUNG TSUEN 深涌村 SHAP PAT HEUNG 十八鄉
118 SHAN PUI TSUEN 山貝村 SHAP PAT HEUNG 十八鄉
119 SHEUNG YAU TIN TSUEN 上攸田村 SHAP PAT HEUNG 十八鄉
120 SHUI TSIU LO WAI 水蕉老圍 SHAP PAT HEUNG 十八鄉
121 SHUI TSIU SAN TSUEN 水蕉新村 SHAP PAT HEUNG 十八鄉
122 TAI KIU 大橋 SHAP PAT HEUNG 十八鄉
123 TAI TONG TSUEN 大棠村 SHAP PAT HEUNG 十八鄉
124 TAI WAI TSUEN 大圍村 SHAP PAT HEUNG 十八鄉
125 TIN LIU TSUEN 田寮村 SHAP PAT HEUNG 十八鄉
126 TONG TAU PO TSUEN 塘頭埔村 SHAP PAT HEUNG 十八鄉
127 TSOI UK TSUEN 蔡屋村 SHAP PAT HEUNG 十八鄉
128 TUNG TAU TSUEN 東頭村 SHAP PAT HEUNG 十八鄉
129 WONG NAI TUN TSUEN 黃泥墩村 SHAP PAT HEUNG 十八鄉
位於元朗區的認可鄉村
RECOGNIZED VILLAGES IN YUEN LONG DISTRICT
Village Name 村名 District 地區
130 WONG UK TSUEN 黃屋村 SHAP PAT HEUNG 十八鄉
131 YEUNG UK TSUEN 楊屋村 SHAP PAT HEUNG 十八鄉
132 YING LUNG WAI 英龍圍 SHAP PAT HEUNG 十八鄉
"""

# Use standard characters in the known districts
KNOWN_DISTRICTS_EN = [
    "MA WAN & NORTH EAST LANTAU", "SAI KUNG NORTH", "SHAP PAT HEUNG",
    "LAMMA NORTH", "LAMMA SOUTH", "SOUTH LANTAU", "SHA TAU KOK",
    "SHEUNG SHUI", "TA KWU LING", "TUNG CHUNG", "TUEN MUN",
    "HANG HAU", "SAI KUNG", "SHA TIN", "TAI PO", "TSUEN WAN",
    "KWAI CHUNG", "TSING YI", "HA TSUEN", "KAM TIN", "PAT HEUNG",
    "PING SHAN", "SAN TIN", "MUI WO", "TAI O", "FANLING", "ISLANDS", "NORTH", "YUEN LONG"
]

KNOWN_DISTRICTS_ZH = [
    "馬灣及大嶼山東北", "西貢北", "十八鄉", "南丫島北", "南丫島南",
    "大嶼山南", "沙頭角", "上水", "打鼓嶺", "東涌", "屯門",
    "坑口", "西貢", "沙田", "大埔", "荃灣", "葵涌", "青衣",
    "廈村", "錦田", "八鄉", "屏山", "新田", "梅窩", "大澳", "粉嶺", "離島", "北區", "元朗"
]


def normalize_text(text):
    """Fix OCR misread Greek homoglyphs AND CJK Compatibility Characters."""
    char_map = {
        '\u03a4': 'T', '\u0391': 'A', '\u0399': 'I', '\u03a1': 'P',
        '\u039f': 'O', '\u039d': 'N', '\u039c': 'M', '\u039a': 'K',
        '\u0395': 'E', '\u0397': 'H', '\u03a5': 'Y', '\u03a7': 'X',
        '\u0392': 'B', '\u03a6': 'F', '\u03a0': 'P',
    }
    for greek, ascii_char in char_map.items():
        text = text.replace(greek, ascii_char)

    text = unicodedata.normalize('NFKC', text)
    return text


def extract_district_from_end(text, district_list):
    """Removes the matching district strictly from the tail end of the string."""
    for d in sorted(district_list, key=len, reverse=True):
        # If the district has English letters, handle variable whitespace
        if re.search(r'[a-zA-Z]', d):
            d_pattern = r'\s+'.join(re.escape(word) for word in d.split())
        else:
            d_pattern = re.escape(d)

        # Removed the inline (?i) that crashes Python 3.11+
        pattern = r'^(.*?)\s*' + d_pattern + r'[\s\(\)\.,;&\-]*$'

        # Added flags=re.IGNORECASE here instead
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(), d

    return text, ""


def parse_blocks(raw_text):
    lines = raw_text.split('\n')
    blocks = []
    current_indices = []
    current_block_lines = []

    ignore_phrases = [
        "LIST OF RECOGNIZED VILLAGES", "UNDER THE NEW TERRITORIES",
        "Village Improvement Section", "Lands Department",
        "September 2009 Edition", "RECOGNIZED VILLAGES IN",
        "地政總署", "鄉村改善組", "二ＯＯ九年九月版", "位於",
        "Village Name", "村名", "District", "地區"
    ]

    for line in lines:
        line = line.replace('|', '').strip()
        if not line:
            continue

        if any(p in line for p in ignore_phrases):
            continue
        if re.match(r'^(Islands|North|Sai Kung|Sha Tin|Tuen Mun|Tai Po|Tsuen Wan|Kwai Tsing|Yuen Long).*$', line):
            continue

        match = re.match(r'^(\d+)\s+(.*)', line)
        if match:
            if current_indices:
                blocks.append((current_indices, current_block_lines))
            current_indices = [int(match.group(1))]
            current_block_lines = [match.group(2)]
        else:
            if current_indices:
                current_block_lines.append(line)

    if current_indices:
        blocks.append((current_indices, current_block_lines))

    return blocks


def extract_bilingual_pairs(raw_text):
    normalized = normalize_text(raw_text)
    blocks = parse_blocks(normalized)
    results = []

    for indices, lines in blocks:
        combined_text = " ".join(lines).strip()
        if not combined_text:
            continue

        combined_text, chi_district = extract_district_from_end(combined_text, KNOWN_DISTRICTS_ZH)

        combined_text, eng_district = extract_district_from_end(combined_text, KNOWN_DISTRICTS_EN)

        eng_village = ""
        chi_village = ""

        match = re.match(r'^([a-zA-Z0-9\s\(\)\.,;&\-\'\/"]+)\s*(.*)$', combined_text)

        if match:
            eng_village = match.group(1).strip()
            chi_village = match.group(2).strip()
        else:
            eng_village = combined_text

        results.append({
            "indices": indices,
            "eng_village": eng_village,
            "zh_village": chi_village,
            "eng_district": eng_district,
            "zh_district": chi_district
        })

    return results


if __name__ == "__main__":
    extracted_pairs = extract_bilingual_pairs(RAW_TEXT)

    print(f"{'Idx':<4} | {'Eng Village':<50} | {'Chi Village':<25} | {'Eng District':<15} | {'Chi District'}")
    print("-" * 125)

    for row in extracted_pairs:
        idx_str = ", ".join(map(str, row['indices']))
        eng_v = row['eng_village'][:47] + "..." if len(row['eng_village']) > 50 else row['eng_village']
        zh_v = row['zh_village'][:22] + "..." if len(row['zh_village']) > 25 else row['zh_village']
        print(f"{idx_str:<4} | {eng_v:<50} | {zh_v:<25} | {row['eng_district']:<15} | {row['zh_district']}")