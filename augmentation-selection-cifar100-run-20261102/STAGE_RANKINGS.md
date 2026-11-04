# 全部48设置的三阶段排序

S=标准预处理，W=弱增强，R=弱增强后RandAugment。CV总正确数分母100B；折模型查询总正确数分母4×9990；全标签模型分母9990。相等以`=`保留；不新增选择规则。
折查询和全标签查询均是封存选择后的事后解释；训练集规模和图像改变，不能纯因果拆分验证噪声与重训练效应。

|编码器|B|episode|CV排序 / 正确数S,W,R|折查询排序 / 正确数S,W,R|全标签排序 / 正确数S,W,R|原选择|全标签错选|同对折阶段已差|
|:---|---:|---:|:---|:---|:---|:---|:---|:---|
|dinov2_small|8|0|R > S > W / 559,547,560|W > S > R / 28409,28437,27980|S > W > R / 7316,7304,7210|ra|True|True|
|dinov2_small|16|0|S > W > R / 1199,1196,1177|S > W > R / 29976,29850,29484|S > W > R / 7601,7557,7495|clean|False|False|
|dinov2_small|8|1|R > S > W / 578,572,579|S > W > R / 28606,28496,28114|S > W > R / 7382,7322,7216|ra|True|True|
|dinov2_small|16|1|S > W > R / 1204,1201,1183|S > W > R / 30215,30062,29637|S > W > R / 7662,7625,7511|clean|False|False|
|dinov2_small|8|2|W > R > S / 564,580,571|S > W > R / 28354,28345,27991|S > W > R / 7334,7315,7228|weak|True|True|
|dinov2_small|16|2|S > W > R / 1221,1213,1203|S > W > R / 30243,30043,29730|S > W > R / 7638,7628,7564|clean|False|False|
|dinov2_small|8|3|S > W > R / 567,562,551|W > S > R / 28251,28367,27871|W > S > R / 7260,7304,7156|clean|True|True|
|dinov2_small|16|3|S > W > R / 1220,1209,1200|W > S > R / 30102,30126,29669|W > S > R / 7641,7666,7525|clean|True|True|
|dinov2_small|8|4|W > S > R / 572,585,570|W > S > R / 28627,28668,28140|S > W > R / 7364,7336,7223|weak|True|False|
|dinov2_small|16|4|W > R > S / 1189,1203,1195|S > W > R / 30314,30303,29918|W > S > R / 7675,7691,7598|weak|False|False|
|dinov2_small|8|5|S > R > W / 587,582,583|W > S > R / 28615,28742,28409|S > W > R / 7378,7354,7328|clean|False|False|
|dinov2_small|16|5|W > S > R / 1214,1224,1195|W > S > R / 30126,30183,29909|S > W > R / 7699,7674,7573|weak|True|False|
|dinov2_small|8|6|W > R > S / 576,583,577|W > S > R / 28646,28706,28413|S > W > R / 7376,7345,7288|weak|True|False|
|dinov2_small|16|6|W > S > R / 1240,1254,1225|S > W > R / 30463,30437,30113|W > S > R / 7729,7740,7649|weak|False|False|
|dinov2_small|8|7|W > S > R / 554,559,551|S > W > R / 28482,28325,27996|S > W > R / 7324,7287,7188|weak|True|True|
|dinov2_small|16|7|S > W > R / 1184,1171,1153|S > W > R / 30184,30061,29618|S > W > R / 7650,7613,7543|clean|False|False|
|dinov2_small|8|8|S > W > R / 578,571,565|S > W > R / 28790,28680,28173|S > W > R / 7419,7400,7291|clean|False|False|
|dinov2_small|16|8|S > W > R / 1245,1244,1207|S > W > R / 30317,30160,29831|S > W > R / 7688,7655,7567|clean|False|False|
|dinov2_small|8|9|W > S > R / 557,566,550|S > W > R / 28730,28542,28129|S > W > R / 7407,7367,7235|weak|True|True|
|dinov2_small|16|9|S > W > R / 1204,1202,1194|S > W > R / 30148,30087,29653|S > W > R / 7663,7641,7521|clean|False|False|
|dinov2_small|8|10|S > W > R / 562,557,556|S > W > R / 28641,28480,28052|S > W > R / 7395,7350,7238|clean|False|False|
|dinov2_small|16|10|W > S > R / 1207,1216,1183|S > W > R / 30482,30260,29841|S > W > R / 7741,7697,7563|weak|True|True|
|dinov2_small|8|11|S > R > W / 560,552,555|W > S > R / 28448,28460,28099|W > S > R / 7338,7352,7270|clean|True|True|
|dinov2_small|16|11|S > W > R / 1188,1176,1164|S > W > R / 30107,29982,29576|S > W > R / 7678,7601,7533|clean|False|False|
|deit_tiny|8|0|W > S > R / 417,421,406|S > W > R / 21043,20980,20787|S > W > R / 5496,5468,5416|weak|True|True|
|deit_tiny|16|0|S > W > R / 905,904,891|S > W > R / 22684,22468,22163|S > W > R / 5839,5759,5691|clean|False|False|
|deit_tiny|8|1|S > W > R / 399,388,382|S > W > R / 20592,20218,19921|S > W > R / 5365,5269,5215|clean|False|False|
|deit_tiny|16|1|W > S > R / 855,865,845|S > W > R / 22521,22354,22072|S > W > R / 5785,5730,5660|weak|True|True|
|deit_tiny|8|2|W > S > R / 404,405,395|S > W > R / 20517,20286,20049|S > W > R / 5352,5252,5195|weak|True|True|
|deit_tiny|16|2|W > S > R / 895,897,888|S > W > R / 22633,22379,22095|S > W > R / 5759,5691,5618|weak|True|True|
|deit_tiny|8|3|W > S > R / 420,422,416|W > S > R / 20496,20507,20432|S > W > R / 5379,5366,5352|weak|True|False|
|deit_tiny|16|3|S > W > R / 912,903,893|S > W > R / 22604,22381,22119|S > W > R / 5813,5778,5696|clean|False|False|
|deit_tiny|8|4|W > S > R / 428,434,427|W > S > R / 20758,20767,20300|S > W > R / 5412,5401,5279|weak|True|False|
|deit_tiny|16|4|S > W > R / 904,892,886|S > W > R / 22787,22559,22204|S > W > R / 5808,5737,5660|clean|False|False|
|deit_tiny|8|5|W > R > S / 409,421,416|S > W > R / 20767,20563,20200|W > S > R / 5371,5372,5302|weak|False|False|
|deit_tiny|16|5|W > R > S / 890,915,901|S > W > R / 22613,22405,22091|S > W > R / 5809,5759,5647|weak|True|True|
|deit_tiny|8|6|W > S > R / 415,419,403|W > S > R / 20974,20979,20835|S > W > R / 5495,5468,5456|weak|True|False|
|deit_tiny|16|6|W > S > R / 911,923,906|S > W > R / 22737,22644,22417|W > S > R / 5788,5804,5745|weak|False|False|
|deit_tiny|8|7|S > R > W / 423,406,413|S > W > R / 20575,20478,20149|S > W > R / 5341,5301,5208|clean|False|False|
|deit_tiny|16|7|S > W > R / 889,882,878|S > W > R / 22500,22360,21995|S > W > R / 5740,5674,5593|clean|False|False|
|deit_tiny|8|8|W > S > R / 429,437,428|S > W > R / 21096,20956,20667|S > W > R / 5505,5409,5348|weak|True|True|
|deit_tiny|16|8|S > W > R / 950,944,925|S > W > R / 22770,22569,22378|S > W > R / 5831,5781,5704|clean|False|False|
|deit_tiny|8|9|S > W > R / 433,427,421|S > W > R / 20940,20806,20400|S > W > R / 5445,5419,5295|clean|False|False|
|deit_tiny|16|9|S > W > R / 917,916,900|S > W > R / 22673,22615,22231|S > W > R / 5797,5790,5695|clean|False|False|
|deit_tiny|8|10|S > W > R / 413,410,405|S > W > R / 20635,20561,20300|W > S > R / 5360,5363,5307|clean|True|False|
|deit_tiny|16|10|S > W > R / 891,889,881|S > W > R / 22627,22580,22202|S > W > R / 5786,5779,5677|clean|False|False|
|deit_tiny|8|11|S > W > R / 400,394,391|S > W > R / 20963,20873,20678|S > W > R / 5463,5417,5381|clean|False|False|
|deit_tiny|16|11|S > W > R / 915,904,886|W > S > R / 22747,22761,22491|W > S > R / 5816,5817,5762|clean|True|True|
