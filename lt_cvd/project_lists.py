TRANS_KEYWORDS = ['orthotropic liver transplant', 'heterotropic liver transplant', 'liver transplant']
TRANS_CONCEPT_LIST = [467458,2100972, 4067459]

METAB_CODES = ['K75.8','K76.0']
METAB_CONCEPT_CODES = [45533616,35208359]

ALD_CODES = ['K70']
ALD_CONCEPT_CODES = [35208330,35208331,35208332,45562507,45552946,45605952,\
                    45538545,45586730,45552947]

CANCER_CODES = ['C22']
CANCER_CONCEPT_CODES = [35206146,35206147,35206148,35206149,35206150,35206151,\
                        45585987,35206152]

HEP_CODES = ["B15", "B16", "B17", "B18", "B19"]
HEP_CONCEPT_CODES = list(range(35205759,35205776))+\
                        [45576259,45605221,45552212,45566540,45581151]
                        
FULM_CODES = ["K72.0"]
FULM_CONCEPT_CODES = [45567324,45543338]

IMMUNE_CODES = ["K75.4", "K74.3", "K83.01"]
IMMUNE_CONCEPT_CODES = [35208356,35208349] # <-- TODO: add PSC

RE_TX_CODES = ["Z94.4", "T86.4"]
RE_TX_CONCEPT_CODES = [45561100,45551553,45594968,45609395,45590132,35225408]

NONSMOKER_CODE = 40770157

DM_CODES = ['E10', 'E11']
DM_CONCEPT_CODES = [1326492,1326493,35206878,35206879,35206881,35206882,37200148,37200166,37200167,37200170] + \
                    list(range(37200191,37200255))+list(range(45533017,45533024))+[45537962,45542736,45542737,45542738] + \
                    list(range(45547621,45547628)) + [45552379,45552382,45552385,45557113]+[45561949,45566731,45576439,45576443,\
                    45581350] + list(range(45581352,45581356)) + [45586139, 45586140, 45591027, 45591029, 45591031] + \
                    list(range(45595795,45595800)) + list(range(45600636,45600643)) + list(range(45605398,45605405))
    
HTN_CODES = ['I10', 'I11', 'I12', 'I13', 'I15']
HTN_CONCEPT_CODES = list(range(35207668,35207679))

LIP_CODES = ['E78.0', 'E78.1', 'E78.2', 'E78.5']
LIP_CONCEPT_CODES = list(range(35207060,35207063))+[35207065,37200312,37200313]

CV_CODES = ['I48.0', 'I48.1', 'I48.2', 'I48.3', 'I48.4', 'I48.9', 'I47', 'I49', 'I21', 'I22', 'I25.2', 'I50', 'I46', 'G45.3', \
            'G45.9', 'I25', 'I63', 'I65', 'I66', 'I67.0', 'I69.3', 'I39.0', 'I39.1', 'I39.2', 'I39.3', 'I39.4', 'I42','R00.0','R00.1']

CV_CONCEPT_CODES = [35207396,35207399,45562340,35207396,35207399,45562340,45572079,35207684,35207685,45576865,1326588,45533436,\
                    45605779,45572080,45557536,1326590,1326591,35207686,45605781,35207702,35207703,35207704,35207705,35207706,\
                    45586572,45557538,45538373,45596199,45596199,45548013,45567168,785999,45605788,45601024,45591456,37402491,\
                    45576866,45596197,45548010,45601027,45605784,45543167,45586574,45548012,45567167,45562344,45605787,35207755,\
                    35207756,35207757,35207758,35207759,35207760,35207761,35207762,35207763,35207764,45572091,45538383,35207779,\
                    35207781,35207782,35207783,786000,786001,786002,37402490,37402503,37402504,35207784,35207785,1569171,1569172,\
                    1569173,1553751,1553752,1553753,1553754,45576876,45572094,45562353,35207786,35207787,35207788,35207789,\
                    35207790,35207791,45591468,35207792,35207793,45586587,45543182,45576878,45567180,45601038,45548022,45533456,\
                    45562355,45533457,45591469,45586588,45567181,1326606,1326607,1326608,1326609,1326601,1326602,1326603,1326604,\
                    1326605,45601041,35207820,45533463,35207821,45567187,45581782,37200496,45562362,45605806,45552802,45538396,\
                    45591474,45562363,45576888,45552803,45601045,45552806,45601047,1595597,1595598,37200498,45596212,37200499,\
                    45576885,45601043,45562365,45543187,45567188,37200502,45557553,45576887,45596213,45586593,45562367,45586594,\
                    37200506,45576889,45543190,37200507,45596214,45548029,45548030,37200508,45581783,45601044,45552805,37200509,\
                    45605809,45586595,37200510,45538397,45586596,45557555,45586597,45581784,45596215,45586598,35207823,45552807,\
                    45601048,45557556,45601049,45601050,45557557,35207827,35207828,45601060,37200539,37200544,37200545,45548047,\
                    45538412,45557562,45533478,45538413,45548048,45586611,45591489,45543202,45533479,45572111,45557563,45557564,\
                    45548049,45605822,45596223]

CAD_CHRONIC_CODES = ['I25.1','I25.4','I25.6','I25.7','I25.8','I25.9']
ARYTHMIA_CHRONIC_CODES = ['I48.0', 'I48.1', 'I48.2','I48.91']
HEART_FAIL_CHRONIC_CODES = ['I25.5','I50','I42']
VALV_CHRONIC_CODES = ['I65','I66','I67.0']

ARYTHMIA_CODES = ['I48.0', 'I48.1', 'I48.2', 'I48.3', 'I48.4', 'I48.92', 'I47', 'I49']
ACS_CODES = ['I21','I22', 'I25.2', 'I46']
CAD_CODES = ['I25.1','I25.4','I25.6','I25.7','I25.8','I25.9']
VALV_CODES = ['I65','I66','I67.0','I39']
CEREBRO_CODES = ['G45.3', 'G45.9','I63','I69.3']
HF_CODES = ['I25.5','I50','I42']

LAB_IDS = {
    'CREATININE': 3016723,
    'ALP': 3035995,
    'ALT': 3006923,
    'CYCLO': 3010375,
    'AST': 3013721,
    'TAC': 3026250,
    'BMI': 3038553
}
CREATININE_ID = 3016723
ALP_ID = 3035995
ALT_ID = 3006923
CYCLO_ID = 3010375
AST_ID = 3013721
TAC_ID = 3026250
BMI_ID = 3038553

LIP_LAB_IDS = {
    'LDL': [3028288,3028437],
    'HDL':[], # TODO: Do we have this?
    'TOTAL_CHOLESTEROL': [3027114],
    
}

# ANTI_PLATELET: keywords: aspirin, asa, clopidogrel, plavix, prasugrel, ticagrelor, dipyridamole. 

# ANTI_HTN: keywords: Amlodipine, caduet, Lecarnidipine, Diltiazem, Verapamil, Doxazosin, Terazosin, Prazosin, Atenolol,
#                     bisoprolol, metoprolol, nadolol, nebivolol, propanolol, Ramipril, Perindopril, Captopril, Lisinopril,
#                     Enalapril, Candesartan, irbesartan, losartan, telmisartan, Hydrochlorthiazide, indapamide

# STATIN: keywords: Atorvastatin, caduet, simvastatin, zocor, rosuvastatin, crestor, fluvastatin, Ezetimibe, Bezafibrate,
#                   fenofibrate, lipitor, lescol, lovastatin, mevacor, altoprev, livalo, zypitamag, pitavastatin, pravachol,
#                   pravastatin, ezallor, vytorin.

MED_IDS = {'ANTI_HTN': [779445,964322,964324,964325,974858,978556,978557,1308221,1308250,1308251,1314006,1314008,1314009,\
                        1314581,1314614,1317675,1328689,1331312,1332419,1332497,1332500,1332525,1332526,1332527,1332528,\
                        1334459,1334460,1334494,1334535,1340161,1341268,1341270,1341302,1350490,1350521,1350552,1351558,\
                        1351559,1351583,1351587,1361519,1363057,1363058,1363059,1363060,1395060,19018811,19019236,19019238,\
                        19019239,19022947,19022948,19022949,19028935,19028936,19029027,19067686,19073093,19073094,19074671,\
                        19074672,19074673,19078080,19078101,19080128,19080129,19101573,19124265,19127430,19127432,19127433,\
                        19133212,19133558,19133562,19133566,19133570,19133574,19133578,19133582,19133585,19133587,19133614,\
                        19133621,19133622,35605001,35605003,40162864,40162867,40162871,40162875,40162878,40165757,40165762,\
                        40165767,40165773,40165785,40165789,40166824,40166826,40166828,40166830,40167087,40167091,40167196,\
                        40167202,40167213,40167218,40167849,40171499,40171510,40171516,40171547,40171550,40171553,40171556,\
                        40171559,40171562,40171849,40171852,40171863,40171884,40171905,40171917,40184184,40184187,40184217,\
                        40185276,40185280,40185304,40221243,40224172,40224175,40224178,42629595,42629596,42629597,42629598,\
                        43560163,46221722,46221724,46287342,46287346],
           'ANTI_PLATELET': [1112841,1112892,1112922,1113346,1322189,1331312,19046742,\
                             19059056,19065472,19066057,19073712,19075601,19076600,19076621,19076623,35605960,40163720,\
                            40163724,40241188,46287538],
           'STATIN': [1526476,1526480,1539407,1539411,1539462,1539463,1545959,1545996,1545997,1551927,1551929,1552015,\
                      19019115,19019116,19019117,19023487,19077244,19077245,19077497,19077498,19098474,19112569,19123592,\
                      19129329,40165245,40165246,40165253,40165257,40165261,40165262,40165638,40165642,40175390,40175394,\
                      40175400,40175404]
          }

# MED_KEYWORDS = {'ANTI_HTN': ['Amlodipine', 'caduet', 'Lercanidipine', 'Diltiazem', 'Verapamil', 'Doxazosin', 'Terazosin',\
#                              'Prazosin', 'Atenolol', 'bisoprolol', 'metoprolol', 'nadolol', 'nebivolol', 'propanolol',\
