#!/usr/bin/env python3
"""Village-focused address synthesis V3.

Combines BOTH gazetteers:
  1. Lands Department "List of Recognized Villages" (New Territories Small House Policy, 2009)
  2. Place Name Gazetteer (May 2026)

Uses the full augmentation suite from V2 / build_hk_address_dataset.py:
- chunk-based rendering + language-dependent order
- importance-weighted omission
- reordering
- floor/unit fusion
- Chinese messy separation + full-width / irregular separators
- English abbreviations + case noise
- Chinese region variants + region abbreviations
- district-suffix noise
- source-backed sub-districts, with synthetic localisation disabled by default
- recoverable name typos (names only)
- richer synthetic 3-D
- optional region-level code-switching

Every output field is derived from the final labelled chunks after all
mutations, so omitted or corrupted input text cannot leave a stale target.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import string
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from opencc import OpenCC
    CC_CONVERTER = OpenCC("t2s")
except ImportError:
    CC_CONVERTER = None

# =============================================================================
# 1. CONSTANTS (ported / adapted from build_hk_address_dataset.py)
# =============================================================================

LABELS = [
    "REGION", "DISTRICT", "SUB_DISTRICT",
    "STREET_NAME", "BUILDING_NUMBER", "VILLAGE_NAME",
    "ESTATE_NAME", "PHASE", "BLOCK", "BUILDING_NAME",
    "FLOOR", "UNIT",
]

OUTPUT_FIELDS = (
    "flat", "floor", "block", "building_name", "phase", "estate_name",
    "building_number", "street_name", "village_name", "sub_district",
    "district", "region",
)

LABEL_TO_FIELD = {
    "UNIT": "flat",
    "FLOOR": "floor",
    "BLOCK": "block",
    "BUILDING_NAME": "building_name",
    "PHASE": "phase",
    "ESTATE_NAME": "estate_name",
    "BUILDING_NUMBER": "building_number",
    "STREET_NAME": "street_name",
    "VILLAGE_NAME": "village_name",
    "SUB_DISTRICT": "sub_district",
    "DISTRICT": "district",
    "REGION": "region",
}

EN_ABBREVIATIONS = {
    "APARTMENT": ("APT",),
    "BUILDING": ("BLDG","BLD",),
    "BLOCK": ("BLK",),
    "DISTRICT": ("DIST",),
    "FLAT": ("FLT", "UNIT", "RM"),
    "FLOOR": ("FL", "F", "/F", "FLR","LVL", "LEVEL"),
    "HOUSE": ("HSE","HS"),
    "ESTATE": ("EST",),
    "ROAD": ("RD",),
    "ROOM": ("RM",),
    "STREET": ("ST",),
    "TOWER": ("TWR",),
    "SUITE": ("STE", "SU"),
    "VILLAGE": ("VIL",),
}

# Village-oriented 3-D weights (heavily favour G/F, whole house, roof)
VILLAGE_3D_WEIGHTS: tuple[tuple[str, int], ...] = (
    ("ground_floor_no_unit", 35),
    ("whole_floor", 25),
    ("roof", 12),
    ("standard_flat", 10),
    ("standard_room", 5),
    ("number_only_unit", 4),
    ("hao_shi_unit", 3),
    ("alphabetic_unit", 3),
    ("leading_zero_unit", 2),
    ("duplex_floor", 2),
    ("combined_units", 2),
    ("special_floor", 1),
    ("basement", 1),
    ("lower_upper_ground", 1),
    ("unit_without_floor", 1),
)

MIX_MODES = (
    "english_3d_chinese_body",
    "chinese_3d_english_body",
    "single_chinese_component",
    "single_english_component",
    "balanced_components",
    "alternating_components",
)

ENGLISH_REGION_CANONICAL = {
    "HK": "Hong Kong", "H.K.": "Hong Kong", "H. K.": "Hong Kong",
    "HONG KONG": "Hong Kong", "HONG KONG ISLAND": "Hong Kong",
    "HK ISLAND": "Hong Kong", "HKI": "Hong Kong",
    "KLN": "Kowloon", "KLN.": "Kowloon", "KOWLOON": "Kowloon",
    "NT": "New Territories", "N.T.": "New Territories",
    "N. T.": "New Territories", "N.T": "New Territories",
    "NEW TERRITORIES": "New Territories",
    "NORTH TERRITORIES": "New Territories",
}

ENGLISH_REGION_ABBREVIATIONS = {
    "Hong Kong": ["HK", "H.K.", "H. K.", "H K", "HK Island", "HKI"],
    "Kowloon": ["KLN", "Kln", "Kln."],
    "New Territories": ["NT", "N.T.", "N. T.", "N T", "N.T"],
}

CHINESE_REGION_VARIATIONS = {
    "zh-Hant": {
        "香港": ["香港島", "港島", "香港區", "香港"],
        "九龍": ["九龍區", "九龍半島", "九龍"],
        "新界": ["新界區", "新界"],
    },
    "zh-Hans": {
        "香港": ["香港岛", "港岛", "香港区", "香港"],
        "九龙": ["九龙区", "九龙半岛", "九龙"],
        "新界": ["新界区", "新界"],
    },
}

COMPONENT_OMISSION_WEIGHTS: dict[str, int] = {
    "region": 100,
    "building_number": 80,
    "district": 68,
    "sub_district": 55,
    "phase": 1,
    "building": 26,
    "street": 24,
    "estate": 18,
    "village": 14,
    "floor": 12,
    "block": 8,
    "unit": 5,
}

QWERTY_NEIGHBOURS = {
    "a": "sqwz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdr",
    "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "i": "ujko", "j": "huikmn",
    "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm", "o": "iklp",
    "p": "ol", "q": "wa", "r": "edft", "s": "awedxz", "t": "rfgy",
    "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu",
    "z": "asx",
}

FULLWIDTH_TRANSLATION = str.maketrans({
    ",": "，", ";": "；", ":": "：", "(": "（", ")": "）",
})

# Full SUB_DISTRICT_MAP (identical to the main builder)
SUB_DISTRICT_MAP = {
    "en": {
        "CENTRAL & WESTERN DISTRICT": [
            "Central", "Admiralty", "Sheung Wan", "Sai Ying Pun", "Shek Tong Tsui",
            "Kennedy Town", "The Peak", "Mid-Levels",
            "Sheungwan", "Saiyingpun", "Shektongtsui", "Kennedytown",
            "Mid Levels", "Midlevels", "Peak", "Sai Wan",
        ],
        "EASTERN DISTRICT": [
            "Taikoo", "North Point", "Quarry Bay", "Chai Wan", "Shau Kei Wan",
            "Fortress Hill", "Taikoo Shing", "Northpoint", "Quarrybay",
            "Chaiwan", "Shaukeiwan", "Fortresshill", "Heng Fa Chuen",
            "Hengfachuen", "Sai Wan Ho", "Saiwanho", "Siu Sai Wan",
            "Siusaiwan", "Braemar Hill", "NP",
        ],
        "SOUTHERN DISTRICT": [
            "Aberdeen", "Ap Lei Chau", "Wong Chuk Hang", "Repulse Bay",
            "Stanley", "Pok Fu Lam", "Cyberport", "Apleichau", "Wongchukhang",
            "Repulsebay", "Pokfulam", "Shek O", "Chung Hom Kok",
            "Deep Water Bay", "Tai Tam",
        ],
        "WAN CHAI DISTRICT": [
            "Wan Chai", "Causeway Bay", "Happy Valley", "Tin Hau", "Tai Hang",
            "Wanchai", "Causewaybay", "Happyvalley", "Tinhau", "Taihang",
            "CWB", "So Kon Po",
        ],
        "KOWLOON CITY DISTRICT": [
            "Kowloon City", "To Kwa Wan", "Hung Hom", "Ho Man Tin",
            "Kowloon Tong", "Kowlooncity", "Tokwawan", "Hunghom", "Homantin",
            "Kowloontong", "Kai Tak", "Kaitak", "Whampoa", "Kowloon Tsai", "KLT",
        ],
        "KWUN TONG DISTRICT": [
            "Kwun Tong", "Ngau Tau Kok", "Kowloon Bay", "Lam Tin", "Yau Tong",
            "Kwuntong", "Ngautaukok", "Kowloonbay", "Lamtin", "Yautong",
            "Sau Mau Ping", "Saumauping", "KLN Bay",
        ],
        "SHAM SHUI PO DISTRICT": [
            "Sham Shui Po", "Cheung Sha Wan", "Lai Chi Kok", "Mei Foo",
            "Shek Kip Mei", "Shamshuipo", "Cheungshawan", "Laichikok",
            "Meifoo", "Shekkipmei", "SSP", "Yau Yat Chuen", "Stonecutters Island",
        ],
        "WONG TAI SIN DISTRICT": [
            "Wong Tai Sin", "Diamond Hill", "Choi Hung", "San Po Kong",
            "Tsz Wan Shan", "Wongtaisin", "Diamondhill", "Choihung",
            "Sanpokong", "Tszwanshan", "Lok Fu", "Lokfu", "Wang Tau Hom", "WTS",
        ],
        "YAU TSIM MONG DISTRICT": [
            "Mong Kok", "Yau Ma Tei", "Tsim Sha Tsui", "Jordan",
            "Prince Edward", "Tai Kok Tsui", "Austin", "Mongkok", "Yaumatei",
            "Tsimshatsui", "Taikoktsui", "TST", "MK", "YMT", "PE", "West Kowloon",
        ],
        "ISLANDS DISTRICT": [
            "Tung Chung", "Discovery Bay", "Chek Lap Kok", "Tai O", "Mui Wo",
            "Cheung Chau", "Lamma Island", "Peng Chau", "Tungchung",
            "Discoverybay", "Cheklapkok", "Taio", "Muiwo", "Cheungchau",
            "Lamma", "Pengchau", "DB", "Pui O", "Tong Fuk", "Lantau",
            "Lantau Island",
        ],
        "KWAI TSING DISTRICT": [
            "Kwai Fong", "Kwai Hing", "Kwai Chung", "Tsing Yi", "Lai King",
            "Kwaifong", "Kwaihing", "Kwaichung", "Tsingyi", "Laiking",
        ],
        "NORTH DISTRICT": [
            "Sheung Shui", "Fanling", "Luen Wo Hui", "Sha Tau Kok",
            "Ta Kwu Ling", "Sheungshui", "Luenwohui", "Shataukok",
            "Takwuling", "Kwu Tung", "Kwutung", "Queen's Hill",
            "Queens Hill", "Ping Che",
        ],
        "SAI KUNG DISTRICT": [
            "Sai Kung", "Tseung Kwan O", "Hang Hau", "Po Lam", "LOHAS Park",
            "Clear Water Bay", "Saikung", "Tseungkwano", "Hanghau", "Polam",
            "Clearwater Bay", "Clearwaterbay", "TKO", "LOHAS",
            "Tiu Keng Leng", "Tiukengleng",
        ],
        "SHA TIN DISTRICT": [
            "Sha Tin", "Tai Wai", "Fo Tan", "Ma On Shan", "Siu Lek Yuen",
            "Shek Mun", "Shatin", "Taiwai", "Fotan", "Maonshan",
            "Siulekyuen", "Shekmun", "MOS", "ST", "Wu Kai Sha",
            "Wukaisha", "Shatin Wai", "City One", "Hin Keng",
        ],
        "TAI PO DISTRICT": [
            "Tai Po Market", "Tai Wo", "Tolo Harbour", "Tai Mei Tuk",
            "Lam Tsuen", "Tai Po", "Taipo", "Taipo Market", "Taiwo",
            "Taimeituk", "Tai Mei Tok", "Lamtsuen", "Pak Shek Kok",
            "Pakshekkok", "Science Park", "TP",
        ],
        "TSUEN WAN DISTRICT": [
            "Tsuen Wan", "Tai Wo Hau", "Sham Tseng", "Ting Kau", "Ma Wan",
            "Tsuenwan", "Taiwohau", "Shamtseng", "Tingkau", "Mawan",
            "Tsing Lung Tau", "TW",
        ],
        "TUEN MUN DISTRICT": [
            "Tuen Mun", "Siu Hong", "Gold Coast", "Lam Tei", "So Kwun Wat",
            "Castle Peak", "Tuenmun", "Siuhong", "Goldcoast", "Lamtei",
            "Sokwunwat", "Castlepeak", "TM",
        ],
        "YUEN LONG DISTRICT": [
            "Yuen Long", "Tin Shui Wai", "Hung Shui Kiu", "Kam Tin",
            "San Tin", "Lau Fau Shan", "Yuenlong", "Tinshuiwai",
            "Hungshuikiu", "Kamtin", "Santin", "Laufaushan", "YL",
            "TSW", "Lok Ma Chau", "Lokmachau", "Fairview Park",
        ],
    },
    "zh-Hant": {
        "中西區": ["中環", "金鐘", "上環", "西營盤", "石塘咀", "堅尼地城", "山頂", "半山", "西環"],
        "東區": ["太古", "北角", "鰂魚涌", "柴灣", "筲箕灣", "炮台山", "太古城", "杏花邨", "西灣河", "小西灣", "寶馬山"],
        "南區": ["香港仔", "鴨脷洲", "黃竹坑", "淺水灣", "赤柱", "薄扶林", "數碼港", "鴨利洲", "石澳", "舂磡角", "深水灣", "大潭"],
        "灣仔區": ["灣仔", "銅鑼灣", "跑馬地", "天后", "大坑", "掃桿埔"],
        "九龍城區": ["九龍城", "土瓜灣", "紅磡", "何文田", "九龍塘", "啟德", "黃埔", "九龍仔"],
        "觀塘區": ["觀塘", "牛頭角", "九龍灣", "藍田", "油塘", "官塘", "秀茂坪"],
        "深水埗區": ["深水埗", "長沙灣", "荔枝角", "美孚", "石硤尾", "深水埔", "石夾尾", "又一村", "昂船洲"],
        "黃大仙區": ["黃大仙", "鑽石山", "彩虹", "新蒲崗", "慈雲山", "樂富", "橫頭磡"],
        "油尖旺區": ["旺角", "油麻地", "尖沙咀", "佐敦", "太子", "大角咀", "柯士甸", "尖沙嘴", "芒角", "西九龍"],
        "離島區": ["東涌", "愉景灣", "赤鱲角", "大澳", "梅窩", "長洲", "南丫島", "坪洲", "赤獵角", "赤臘角", "貝澳", "塘福", "大嶼山"],
        "葵青區": ["葵芳", "葵興", "葵涌", "青衣", "荔景"],
        "北區": ["上水", "粉嶺", "聯和墟", "沙頭角", "打鼓嶺", "古洞", "皇后山", "坪輋"],
        "西貢區": ["西貢", "將軍澳", "坑口", "寶琳", "日出康城", "清水灣", "康城", "調景嶺"],
        "沙田區": ["沙田", "大圍", "火炭", "馬鞍山", "小瀝源", "石門", "烏溪沙", "沙田圍", "第一城", "顯徑"],
        "大埔區": ["大埔墟", "太和", "吐露港", "大尾篤", "林村", "大埔", "大尾督", "大美督", "白石角", "科學園"],
        "荃灣區": ["荃灣", "大窩口", "深井", "汀九", "馬灣", "青龍頭"],
        "屯門區": ["屯門", "兆康", "黃金海岸", "藍地", "掃管笏", "青山", "掃管忽"],
        "元朗區": ["元朗", "天水圍", "洪水橋", "錦田", "新田", "流浮山", "落馬洲", "錦繡花園"],
    },
    "zh-Hans": {
        "中西区": ["中环", "金钟", "上环", "西营盘", "石塘咀", "坚尼地城", "山顶", "半山", "西环"],
        "东区": ["太古", "北角", "鲗鱼涌", "柴湾", "筲湾", "炮台山", "太古城", "杏花邨", "西湾河", "小西湾", "宝马山"],
        "南区": ["香港仔", "鸭脷洲", "黄竹坑", "浅水湾", "赤柱", "薄扶林", "数码港", "鸭利洲", "石澳", "舂磡角", "深水湾", "大潭"],
        "湾仔区": ["湾仔", "铜锣湾", "跑马地", "天后", "大坑", "扫杆埔"],
        "九龙城区": ["九龙城", "土瓜湾", "红磡", "何文田", "九龙塘", "启德", "黄埔", "九龙仔"],
        "观塘区": ["观塘", "牛头角", "九龙湾", "蓝田", "油塘", "官塘", "秀茂坪"],
        "深水埗区": ["深水埗", "长沙湾", "荔枝角", "美孚", "石硖尾", "深水埔", "石夹尾", "又一村", "昂船洲"],
        "黄大仙区": ["黄大仙", "钻石山", "彩虹", "新蒲岗", "慈云山", "乐富", "横头磡"],
        "油尖旺区": ["旺角", "油麻地", "尖沙咀", "佐敦", "太子", "大角咀", "柯士甸", "尖沙嘴", "芒角", "西九龙"],
        "离岛区": ["东涌", "愉景湾", "赤鱲角", "大澳", "梅窝", "长洲", "南丫岛", "坪洲", "赤猎角", "赤腊角", "贝澳", "塘福", "大屿山"],
        "葵青区": ["葵芳", "葵兴", "葵涌", "青衣", "荔景"],
        "北区": ["上水", "粉岭", "联和墟", "沙头角", "打鼓岭", "古洞", "皇后山", "坪輋"],
        "西贡区": ["西贡", "将军澳", "坑口", "宝琳", "日出康城", "清水湾", "康城", "调景岭"],
        "沙田区": ["沙田", "大围", "火炭", "马鞍山", "小沥源", "石门", "乌溪沙", "沙田围", "第一城", "显径"],
        "大埔区": ["大埔墟", "太和", "吐露港", "大尾笃", "林村", "大埔", "大尾督", "大美督", "白石角", "科学园"],
        "荃湾区": ["荃湾", "大窝口", "深井", "汀九", "马湾", "青龙头"],
        "屯门区": ["屯门", "兆康", "黄金海岸", "蓝地", "扫管笏", "青山", "扫管忽"],
        "元朗区": ["元朗", "天水围", "洪水桥", "锦田", "新田", "流浮山", "落马洲", "锦绣花园"],
    },
}

DISTRICT_CODE_TO_EN = {
    "C&W": "Central & Western", "E": "Eastern", "Is": "Islands",
    "K&T": "Kwai Tsing", "KC": "Kowloon City", "KT": "Kwun Tong",
    "N": "North", "S": "Southern", "SK": "Sai Kung", "SSP": "Sham Shui Po",
    "ST": "Sha Tin", "TM": "Tuen Mun", "TP": "Tai Po", "TW": "Tsuen Wan",
    "WC": "Wan Chai", "WTS": "Wong Tai Sin", "YL": "Yuen Long",
    "YTM": "Yau Tsim Mong",
}

DISTRICT_CODE_TO_ZH = {
    "C&W": "中西區", "E": "東區", "Is": "離島", "K&T": "葵青",
    "KC": "九龍城", "KT": "觀塘", "N": "北區", "S": "南區",
    "SK": "西貢", "SSP": "深水埗", "ST": "沙田", "TM": "屯門",
    "TP": "大埔", "TW": "荃灣", "WC": "灣仔", "WTS": "黃大仙",
    "YL": "元朗", "YTM": "油尖旺",
}

# =============================================================================
# 2. GAZETTEER PARSING (unchanged core, minor clean-ups)
# =============================================================================

raw_text = """
地名錄
Place Name Gazetteer
2026 年 5 月
May 2026
備註：
1. 英文／中文地名後括號內所列者為該地點的別名。別名亦包括中文地名的羅馬
拼音。
Notes:
1. The name in the bracket following the English/Chinese name is the alias of that
place. Alias also includes the romanised Chinese name.
May 2026
English Name Chinese Name District* HP5C
A Chau 鴉洲 N 3-NE-C
A Kung Kok 亞公角 ST 7-SE-A
A Kung Ngam 阿公岩 E 11-SE-B
A Kung Tin 亞公田 YL 6-NE-B
A Kung Wan 阿公灣 SK 12-NW-A
A Ma Tsui 亞媽咀 TP 17-NW-C
A Ma Wan 亞媽灣 TP 17-NW-C
A Ma Wat 亞媽笏 N 3-NE-D
A Po Long 亞婆塱 Is 10-SW-A
A Shan 鴉山 TP 3-SE-C
Aberdeen 香港仔 S 11-SW-D
Aberdeen Channel 香港仔海峽 S 15-NW-B
Adamasta Channel 北長洲海峽 Is 14-NW-D
Adamasta Rock 北長洲石 Is 14-NW-B
Ah Kung Au 亞公坳 N 3-NE-B
Ah Kung Kok Fishermen Village 亞公角漁民新村 ST 7-SE-A
Ah Kung Tsui 亞公咀 N 3-NE-B
Ah Kung Wan 亞公灣 N 3-NE-B
Amah Rock 望夫石 ST 7-SW-D
Ap Chau 鴨洲 N 4-NW-A
Ap Chau Hoi 鴨洲海 N 4-NW-A
Ap Chau Mei Pak Tun Pai 鴨洲尾白墩排 N 4-NW-A
Ap Chau Pak Tun Pai 鴨洲白墩排 N 4-NW-A
Ap Lei Chau 鴨脷洲 S 15-NW-B
Ap Lei Pai 鴨脷排 S 15-NW-B
Ap Lo Chun 鴨螺春 N 4-NW-A
Ap Tan Pai 鴨蛋排 N 4-NW-A
Ap Tau Pai 鴨兜排 N 4-NW-C
Assistance Rock 鴨脷咀 S 15-NW-B
Au Ha 凹下 N 3-NE-C
Au Kung Shan 歐公山 TP 17-NW-A
Au Mun 坳門 TP 8-NW-D
Au Pui Leng 坳背嶺 N 3-NE-D
Au Pui Tong 凹背塘 N 4-NW-A
Au Pui Wan 坳背灣 ST 7-SW-B
Au Tau 凹頭 YL 6-NE-A
Au Tau 凹頭 SK 11-NE-B
Au Tsai 滘仔 N 3-NE-D
Au Tsai Tsuen 澳仔村 SK 11-NE-B
Au Yue Tsui 拗魚咀 N 4-SW-A
Basalt Island 火石洲 SK 12-NE-C
Bay Islet ( See Chau ) 匙洲 SK 12-NW-B
Beacon Hill 筆架山 ST 11-NW-B
Beacon Hill 筆架山 KC 11-NW-B
Beaufort Island ( Lo Chau ) 螺洲 Is 15-SE-B
Belcher Bay 卑路乍灣 C&W 11-SW-A
Bennet's Hill 班納山 S 11-SW-D
Big Wave Bay 大浪灣 S 15-NE-B
Biu Tsim Kok 標尖角 SK 8-SE-D
Black Hill ( Ng Kwai Shan ) 五桂山 SK 11-NE-D
Black Point ( Lan Kok Tsui ) 爛角咀 TM 5-SE-A
Blackhead Point ( Tai Pau Mai ) 大包米 YTM 11-SW-B
P. 1 / 53
May 2026
English Name Chinese Name District* HP5C
Bluff Head ( Wong Ma Kok ) 黃麻角 S 15-NE-C
Bluff Island ( Sha Tong Hau Shan ) 沙塘口山 SK 12-NE-C
Boa Vista 野豬徑 S 11-SE-C
Boon Kin Village 半見村 SK 12-NW-C
Braemar Hill 寶馬山 E 11-SE-A
Breaker Reef 浪花排 TP 18
Brick Hill ( Nam Long Shan ) 南朗山 S 15-NW-B
Bride's Pool 新娘潭 TP 3-SE-B
Bridge Hill ( Lin Fa Tseng Shan ) 蓮花井山 S 15-NE-A
Brothers Point ( Tai Lam Kok ) 大欖角 TM 6-SW-D
Buffalo Hill 水牛山 ST 7-SE-D
Buffalo Pass ( Ta She Yau Au ) 打瀉油坳 SK-ST 7-SE-D
Bun Bei Chau 崩鼻洲 SK 12-NE-A
Bun Sha Pai 崩紗排 TP 8-NW-A
Butterfly Hill 蝴蝶山 Is 10-SW-C
Butterfly Valley 蝴蝶谷 K&T 11-NW-A
Camp Cove ( Pak Sha Tau Wan ) 白沙頭灣 N 4-NW-C
Cape Collinson ( Hak Kok Tau ) 黑角頭 E 11-SE-D
Cape D'Aguilar ( Hok Tsui ) 鶴咀 S 15-NE-D
CARE Village 美援新村 TP 7-NW-B
Caroline Hill 加路連山 WC 11-SW-B
Castle Peak 青山 TM 5-SE-B
Castle Peak Bay ( Tsing Shan Wan ) 青山灣 TM 6-SW-C
Cat Hill 雷公坑 N 3-SW-B
Causeway Bay 銅鑼灣 WC 11-SW-B
Central District 中環 C&W 11-SW-B
Centre Island ( A Chau ) 丫洲 TP 7-NE-A
Cha Kwo Chau 茶果洲 Is 13-NE-D
Cha Kwo Ling 茶果嶺 KT 11-NE-D
Cha Kwo Ling Tsuen 茶果嶺村 KT 11-SE-B
Cha Liu Au 茶寮坳 SK 11-NE-B
Cha Yue Pai 炸魚排 SK 8-SW-C
Chai Kek 寨乪 TP 7-NW-A
Chai Wan 柴灣 E 11-SE-D
Chai Wan Au 柴灣坳 E 11-SE-D
Chai Wan Kok 柴灣角 TW 6-SE-D
Cham Keng Chau 斬頸洲 TP 17-NW-A
Cham Pai 沉排 SK 12-NW-D
Cham Pai 杉排 TP 4-SE-C
Cham Shan 杉山 N 3-NW-C
Cham Shuen Wan 沉船灣 SK 8-NE-D
Cham Tau Chau 枕頭洲 SK 8-SW-C
Cham Tin Shan 枕田山 SK 11-NE-B
Chan Uk 陳屋 TP 17-NW-A
Chan Uk Po 陳屋埔 N 2-SE-D
Chan Uk Village 陳屋村 SK 12-NW-C
Chap Mo Chau 執毛洲 N 4-SW-A
Chap Mun Tau 閘門頭 Is 9-SE-C
Chap Wai Kon 插桅杆 ST 7-SE-C
Chap Wai Kon New Village 插桅杆新村 ST 7-SE-C
Chat Wan 獺灣 SK 8-NE-D
Chau Mei 洲尾 TP 17-NW-A
P. 2 / 53
May 2026
English Name Chinese Name District* HP5C
Chau Mei Kok 洲尾角 TP 17-NW-A
Chau Pui 洲背 TP 17-NW-C
Chau Tau 洲頭 YL 2-SE-B
Chau Tau 洲頭 TP 17-NW-C
Chau Tsai 洲仔 SK 12-NE-A
Chau Tsai Kok 洲仔角 TP 4-SE-C
Che Ha 輋下 TP 8-NW-C
Che Keng Tuk 輋徑篤 SK 8-SW-C
Che Keuk Ha 輋腳下 TP 17-NW-C
Che Kung Miu 車公廟 ST 7-SW-D
Che Lei Pai 扯排 TP 4-SW-C
Che Pau Teng 斜炮頂 S 15-NE-C
Che Ting Tsuen 輋頂村 SK 8-SW-C
Che Ting Tsuen 輋頂村 KT 11-SE-B
Che Wan 車灣 TP 4-SE-C
Chek Chue 赤柱 S 15-NE-C
Chek Keng 赤徑 TP 8-NE-C
Chek Keng Hau 赤徑口 TP 8-NE-C
Chek Kok Ngam 赤角岩 SK 12-NE-A
Chek Kok Tau 赤角頭 N 4-NW-A
Chek Lap Kok 赤鱲角 Is 9-NE-C
Chek Lap Kok New Village 赤鱲角新村 Is 9-SE-B
Chek Ma Tau 赤馬頭 N-TP 3-SE-B
Chek Nai Ping 赤泥坪 ST 7-NE-C
Cheung Chau 長洲 Is 14-NW-D
Cheung Chau Lutheran Village 長洲信義村 Is 14-NW-D
Cheung Chau Wan 長洲灣 Is 14-NW-D
Cheung Chun San Tsuen 長春新村 YL 6-NE-A
Cheung Hang Village 長坑村 K&T 11-NW-A
Cheung Kang 長庚 ST 7-NE-D
Cheung Kong Tsuen 長江村 YL 6-NE-A
Cheung Kung Shan 張公山 SK 12-NW-A
Cheung Lek 長瀝 N 2-SE-D
Cheung Lek Mei 長瀝尾 ST 7-NE-C
Cheung Lin Shan 長連山 S 15-NE-A
Cheung Muk Tau 樟木頭 TP 7-NE-D
Cheung Muk Tau 樟木頭 Is 13-SE-A
Cheung Ngam 長岩 N 4-NW-D
Cheung Ngam Teng 長岩頂 SK 8-SE-D
Cheung Ngam Wan 長岩灣 SK 8-SE-D
Cheung Pai Tau 長排頭 N 4-SW-A
Cheung Pai Tau 長排頭 N 4-SW-B
Cheung Pai Tun 長牌墩 TP 4-SW-C
Cheung Po 長莆 YL 6-NE-C
Cheung Po Tau 長甫頭 N 3-NW-C
Cheung Po Tsai Cave 張保仔洞 Is 14-NW-D
Cheung Sha 長沙 Is 13-NE-B
Cheung Sha Ha Tsuen 長沙下村 Is 13-NE-B
Cheung Sha Lan 長沙欄 Is 10-SW-B
Cheung Sha Sheung Tsuen 長沙上村 Is 13-NE-B
Cheung Sha Wan 長沙灣 SSP 11-NW-A
Cheung Sha Wan 長沙灣 Is 14-NW-A
P. 3 / 53
May 2026
English Name Chinese Name District* HP5C
Cheung Sha Wan 長沙灣 TP 17-NW-A
Cheung Shan 長山 N 3-NW-D
Cheung Shan 長山 SK 8-SW-B
Cheung Shan 象山 Is 9-SW-D
Cheung Shek Tsui 長石咀 N 4-NW-A
Cheung Shek Tsui 長石咀 N 3-NE-B
Cheung Sheung 嶂上 TP 8-NW-D
Cheung Shue Tan 樟樹灘 TP 7-NE-C
Cheung Shue Tan Hang 樟樹灘坑 TP 7-NE-C
Cheung Shue Tau 樟樹頭 K&T 6-SE-D
Cheung Sok 長索 TW 10-NW-B
Cheung Sok Tsui 長索咀 TW 10-NW-B
Cheung Ting 長亭 Is 13-NW-B
Cheung Tsui 長咀 TW 10-NE-A
Cheung Tsui 長咀 N 4-SW-A
Cheung Tsui Chau 長咀洲 SK 8-NE-D
Cheung Uk 張屋 TP 3-SW-D
Cheung Uk Tei 張屋地 TP 7-NW-D
Cheung Uk Tsuen 張屋村 YL 6-NE-D
Cheung Uk Wai 張屋圍 SK 8-NE-D
Cheung Wan 長灣 SK 12-NW-B
Cheung Wo 長窩 N 4-SW-B
Chi Ma Hang 芝麻坑 Is 14-NW-D
Chi Ma Lung 芝麻籠 N 3-NE-B
Chi Ma Wan 芝麻灣 Is 14-NW-A
Chi Ma Wan Peninsula 芝麻灣半島 Is 14-NW-A
Chik Mun Tau 直門頭 N 4-NW-D
Chik Mun Tau 直門頭 N 4-NW-D
Ching Chau 青洲 SK 8-SW-B
Chiu Keng Wan 照鏡環 SK 11-SE-B
Chiu Keng Wan Shan 照鏡環山 SK 11-SE-B
Cho Ma Wu 祖麻湖 TP 7-NW-B
Choi Yuen Tsuen 菜園村 TW 6-SW-D
Chong Tsin Leng 倉前嶺 N 3-SW-A
Chow Tin Tsuen 週田村 N 3-NW-C
Chu Mun Tin 珠門田 N 3-NE-D
Chu Wong Ling 豬黃嶺 YL 6-NW-B
Chuen Lo Kok Tsui 串螺角咀 TP 4-SW-D
Chuen Lung 川龍 TW 6-SE-B
Chuen Lung Cha Tau Wo 川龍茶頭窩 TW 6-SE-B
Chuen Lung Chun Ha 川龍圳下 TW 6-SE-B
Chuen Lung Tit Lo Shing 川龍鐵盧城 TW 6-SE-B
Chuen Pei Lung 川背龍 TP 7-NW-A
Chuen Shui Tseng 泉水井 TP 7-NW-A
Chui Tung Au 吹筒坳 SK 8-SE-A
Chuk Hang 竹坑 TP 7-NW-B
Chuk Hang 竹坑 YL 6-NE-B
Chuk Kok 竹角 SK 11-NE-B
Chuk San Tsuen 竹新村 YL 6-NW-B
Chuk U Pai 捉魚排 N 4-SW-B
Chuk Yuen 竹園 SK 11-NE-B
Chuk Yuen Tsuen 竹園村 YL 2-SE-C
P. 4 / 53
May 2026
English Name Chinese Name District* HP5C
Chuk Yuen United Village 竹園聯合村 WTS 11-NE-A
Chuk Yuen Village 竹園村 N 3-NW-A
Chun Fa Lok 春花落 K&T 10-NE-B
Chun Hing San Tsuen 振興新村 YL 6-NW-B
Chung Hau 涌口 Is 13-NW-B
Chung Hau 涌口 Is 10-SW-C
Chung Hau Yu Man San Tsuen 涌口漁民新村 YL 6-NW-B
Chung Hom Kok 舂坎角 S 15-NE-C
Chung Hom Kok 舂坎角 S 15-NE-C
Chung Hom Shan 舂坎山 S 15-NE-C
Chung Hom Wan 舂坎灣 S 15-NE-C
Chung Kan O 中間澳 N 4-NW-A
Chung Kwai Chung Tsuen 中葵涌村 TW 7-SW-C
Chung Mei 涌尾 TP 3-SE-B
Chung Mei Kok 涌尾角 TP 4-SE-C
Chung Mei Lo Uk Village 涌美老屋村 K&T 10-NE-B
Chung Pui 涌背 TP 3-SE-D
Chung Sha Teng 涌沙頂 TP 8-NW-B
Chung Shan 松山 TM 6-NW-C
Chung Shun Lane 忠信里 TP 7-NW-B
Chung Sum Tsuen 中心村 N 3-SW-A
Chung Sum Tsuen 中心村 YL 6-NE-D
Chung Tsai Tsuen 涌仔村 Is 10-SW-B
Chung Uk Tsuen 鍾屋村 TM 6-NW-C
Chung Uk Tsuen 鍾屋村 TP 7-NW-A
Chung Wai 中圍 TP 4-SE-C
Chung Wan 涌灣 N 4-NW-B
Chung Wan 涌灣 N 4-NW-C
Chung Wan Teng 涌灣頂 N 4-NW-B
Chung Wan Tsui 涌灣咀 N 4-NW-C
Chung Wong Toi 頌皇台 TM 6-SW-A
Chung Yan Pei 眾人碑 YL 6-NE-B
Clear Water Bay 清水灣 SK 12-SW-A
Clear Water Bay Peninsula 清水灣半島 SK 12-SW-A
Cloudy Hill ( Kau Lung Hang Shan ) 九龍坑山 TP 3-SW-D
Conic Island ( Fan Tsang Chau ) 飯甑洲 SK 8-SE-D
Cove Hill ( Kau To Shan ) 狗肚山 ST 7-SE-A
Crescent Bay ( Ngo Mei Wan ) 娥眉灣 N 4-NW-D
Crescent Island ( Ngo Mei Chau ) 娥眉洲 N 4-NW-D
Crest Hill ( Tai Shek Mo ) 大石磨 N 2-NE-D
Crocodile Hill 鱷魚山 KT 11-NE-C
Crooked Harbour ( Kat O Hoi ) 吉澳海 N 4-NW-C
Crooked Island ( Kat O ) 吉澳 N 4-NW-C
Crown Point － ST 7-SE-C
Crow's Nest － SSP 11-NW-B
Da Chuen Ping Village 打磚坪村 TW 7-SW-C
D'Aguilar Peak ( Hok Tsui Shan ) 鶴咀山 S 15-NE-D
D'Aguilar Peninsula 鶴咀半島 S 15-NE-B
Deep Bay ( Shenzhen Bay ) 后海灣 ( 深圳灣 ) YL 2-SW-A
Deep Water Bay 深水灣 S 15-NW-B
Denon Terrace 騰龍臺 SK 11-NE-B
Devil's Peak ( Pau Toi Shan ) 炮台山 SK 11-SE-B
P. 5 / 53
May 2026
English Name Chinese Name District* HP5C
Diamond Hill 鑽石山 WTS 11-NE-A
Diamond Hill Stream 鑽石山石澗 WTS 11-NE-A
Discovery Bay ( Tai Pak Wan ) 大白灣 Is 10-NW-D
Double Haven ( Yan Chau Tong ) 印洲塘 N 4-NW-D
Double Island ( Wong Wan Chau ) 往灣洲 N 4-SW-B
Douglas Rock 德己利士礁 Is 10-SE-A
Dragon Beach 青龍灣 TW 6-SE-C
Dragon's Back 龍脊 S 15-NE-B
Duckling Hill 鴨仔山 SK 12-NW-C
Dukes Hill ( Sze Tei Shan ) 獅地山 N 3-SW-B
Eagle's Nest ( Tsim Shan ) 尖山 ST 11-NW-B
East Brother ( Siu Mo To ) 小磨刀 TM 10-NW-A
East Lamma Channel 東博寮海峽 Is 15-NW-A
East Ninepin Island ( Tung Kwo Chau ) 東果洲 SK 12-SE-C
Fa Heung Lo Teng 花香爐頂 TM 5-SE-A
Fa Peng 花坪 Is 14-NW-D
Fa Peng 花坪 TW 10-NE-A
Fa Peng Teng 花瓶頂 TW 10-NE-A
Fa Sam Hang 花心坑 ST 7-SE-C
Fa Shan 花山 SK 8-SE-C
Fan Kei Tok 芬箕托 N 3-SE-B
Fan Kwai Tong 番鬼塘 Is 13-NW-A
Fan Lau 分流 Is 13-NW-C
Fan Lau Kok 分流角 Is 13-NW-C
Fan Lau Miu Wan 分流廟灣 Is 13-NW-C
Fan Lau Sai Wan 分流西灣 Is 13-NW-C
Fan Lau Teng 分流頂 Is 13-NW-C
Fan Lau Tsuen 汾流村 Is 13-NW-C
Fan Lau Tung Wan 分流東灣 Is 13-NW-C
Fan Leng Lau 粉嶺樓 N 3-SW-A
Fan Shui Au 分水坳 Is 13-NW-D
Fan Shui Au 分水凹 N 3-NE-D
Fan Tap Pai 番塔排 SK 12-NE-C
Fan Tin Tsuen 蕃田村 YL 2-SE-A
Fanling 粉嶺 N 3-SW-A
Fanling Wai 粉嶺圍 N 3-SW-A
Fat Tau Chau Village ( Fu Tau Chau ) 佛頭洲村 ( 斧頭洲 ) SK 12-NW-C
Fat Tong Chau 佛堂洲 SK 12-SW-A
Fat Tong Kok 佛堂角 SK 12-SW-D
Fat Tong Mun 佛堂門 SK 12-SW-C
Fat Tong O 佛堂澳 SK 12-SW-C
Fei Kei Teng 飛機頂 Is 13-SE-A
Fei Shue Ngam 飛鼠岩 N 4-NW-B
Finger Hill 手指山 Is 10-SE-A
Flat Island ( Ngan Chau ) 銀洲 TP 4-SW-D
Fo Siu Pai 火燒排 SK 12-SE-A
Fo Tan 火炭 ST 7-SE-A
Fo Tan Kuk San Tsuen 火炭谷新村 ST 7-SW-B
Fo Tan Village 火炭村 ST 7-SE-A
Fong Ma Po 放馬莆 TP 7-NW-A
Fong Yuen 芳園 Is 9-SE-C
Fraser Village 禮修村 YL 6-NW-D
P. 6 / 53
May 2026
English Name Chinese Name District* HP5C
Fu Kong Shan 虎崗山 Is 10-SW-C
Fu Shan 虎山 Is 9-SW-C
Fu Tau Sha 虎頭沙 TP 4-SW-C
Fu Tei Ha Tsuen 虎地下村 TM 6-NW-C
Fu Tei Hau 虎地口 SK 8-SW-A
Fu Tei Pai 虎地排 N 3-SW-B
Fu Tei Sheung Tsuen 虎地上村 TM 6-SW-A
Fu Wong Chau 虎王洲 N 4-NW-C
Fu Yung Pei 芙蓉泌 ST 7-SE-C
Fu Yung Pit 芙蓉別 SK 7-SE-D
Fu Yung Pit 芙蓉別 SK 7-SE-B
Fu Yung Shan 芙蓉山 TW 7-SW-C
Fui Sha Wai 灰沙圍 YL 6-NW-A
Fui Yiu Ha 灰窰下 Is 9-SE-B
Fui Yiu Ha 灰窰下 SK 8-SW-C
Fui Yiu Ha New Village 灰窰下新村 ST 7-SE-C
Fuk Hang Tsuen 福亨村 TM 6-NW-C
Fuk Hing Lei 福興里 YL 2-SE-C
Fuk Tsuen Shan 福全山 N 2-SE-D
Fun Chau 墳洲 N 4-NW-C
Fung Chi Tsuen 鳳池村 YL 6-NW-B
Fung Hang 鳳坑 N 3-NE-C
Fung Ka Wai 馮家圍 YL 6-NW-B
Fung Kat Heung 逢吉鄉 YL 6-NE-A
Fung Kong 鳳崗 N 2-SE-B
Fung Kong Shan 鳳崗山 N 2-SE-B
Fung Kong Tsuen 鳳降村 YL 6-NW-A
Fung Mei Wai 鳳美圍 TP 7-NW-B
Fung Shue Wo 楓樹窩 K&T 6-SE-D
Fung Wong Kai 鳳凰溪 TP 4-SW-D
Fung Wong San Tsuen 鳳凰新村 WTS 11-NE-A
Fung Wong Wat 鳳凰笏 TP 4-SW-D
Fung Wong Wat Teng 鳳凰笏頂 TP 4-SW-B
Fung Wong Wu 鳳凰湖 N 3-NW-C
Fung Yuen 鳳園 TP 7-NW-B
Fung Yuen Lo Tsuen 鳳園老村 TP 3-SW-D
Golden Hill 金山 ST 7-SW-C
Grass Island ( Tap Mun ) 塔門 TP 4-SE-C
Grassy Hill 草山 TW 7-NW-D
Green Island 青洲 C&W 10-SE-B
Ha Che 下輋 YL 6-NE-B
Ha Fa Shan 下花山 TW 6-SE-D
Ha Hang 下坑 TP 7-NW-B
Ha Heung Yuen 下香園 N 3-NW-B
Ha Keng 下徑 Is 14-NW-B
Ha Keng Hau 下徑口 ST 7-SW-D
Ha Ko Po Tsuen 下高埔村 YL 6-NE-A
Ha Ko Tan 下高灘 K&T 10-NE-B
Ha Kok 下角 Is 13-SE-A
Ha Kok Tau 下角頭 SK 12-SW-A
Ha Kok Tsui 下角咀 TW 10-NW-B
Ha Kung Tei 蝦公地 N 3-SW-C
P. 7 / 53
May 2026
English Name Chinese Name District* HP5C
Ha Kwai Chung 下葵涌 K&T 11-NW-A
Ha Kwai Chung Village 下葵涌村 K&T 7-SW-C
Ha Ling Pei 下嶺皮 Is 9-SE-B
Ha Mei San Tsuen 蝦尾新村 YL 6-NW-B
Ha Mei Tsui 下尾咀 Is 14-SE-B
Ha Mei Wan 下尾灣 Is 14-NE-D
Ha Miu Tin 下苗田 N 4-SW-A
Ha Pak Nai 下白泥 YL 5-NE-D
Ha Pak Tsuen 下北村 N 3-SW-A
Ha Shan Kai Wat 下山雞乙 N 3-NW-D
Ha Shan Tuk 蝦山篤 SK 12-SW-A
Ha So Pai 蝦鬚排 Is 14-NW-B
Ha Tam Shui Hang 下担水坑 N 3-NE-A
Ha Tei Ha 蝦地下 TP 3-SE-C
Ha Tin Liu Ha 下田寮下 TP 7-NW-A
Ha Tsat Muk Kiu 下七木橋 N 3-SE-A
Ha Tsuen 廈村 YL 6-NW-A
Ha Tsuen 下村 Is 13-SE-C
Ha Tsuen Long 下村塱 Is 10-SW-C
Ha Tsuen Shi 廈村市 YL 6-NW-A
Ha Wai 下圍 TP 4-SE-C
Ha Wan Fisherman San Tsuen 下灣漁民新村 YL 2-SE-B
Ha Wan Tsuen 下灣村 YL 2-SE-A
Ha Wo Che 下禾輋 ST 7-SE-A
Ha Wo Hang 下禾坑 N 3-NE-C
Ha Wong Yi Au 下黃宜坳 TP 7-NW-B
Ha Wun Yiu 下碗窰 TP 7-NW-B
Ha Yau Tin Tsuen 下攸田村 YL 6-NW-B
Ha Yeung 下洋 SK 12-NW-C
Ha Yeung San Tsuen 下洋新村 SK 12-SW-A
Ha Yeung Shan 下洋山 SK 12-NW-C
Hadden Hill ( Ki Lun Shan ) 麒麟山 N-YL 2-SE-B
Hai Kam Tsui 蟹鉗咀 Is 10-NW-D
Hai Tei Wan 蟹地灣 Is 10-SW-B
Hak Ka Wai 客家圍 N 3-SW-A
Hak Shan Teng 黑山頂 SK 8-SW-C
Ham Tin 鹹田 TW 7-SW-C
Ham Tin 鹹田 SK 8-NE-D
Ham Tin Kau Tsuen 鹹田舊村 Is 14-NW-A
Ham Tin San Tsuen 鹹田新村 Is 14-NW-A
Ham Tin Tsuen 咸田村 TW 7-SW-C
Ham Tin Wan 鹹田灣 SK 8-SE-B
Ham Yue Tsing 鹹魚埕 N 4-SW-A
Hammer Hill 斧山 WTS 11-NE-A
Hang Cho Shui 坑槽水 SK 8-SW-A
Hang Ha Po 坑下莆 TP 7-NW-A
Hang Hau 坑口 ST 7-NE-C
Hang Hau 坑口 SK 12-NW-C
Hang Hau Tsuen 坑口村 YL 2-SW-C
Hang Hau Village 坑口村 SK 12-NW-C
Hang Mei 坑尾 Is 9-SW-D
Hang Mei Tsuen 坑尾村 YL 6-NW-B
P. 8 / 53
May 2026
English Name Chinese Name District* HP5C
Hang Pui 坑背 Is 13-NW-B
Hang Tau 坑頭 N 2-SE-D
Hang Tau Tai Po 坑頭大布 N 2-SE-B
Hang Tau Tsuen 坑頭村 YL 6-NW-B
Hap Mun Bay 廈門灣 SK 8-SW-C
Happy Valley 跑馬地 WC 11-SW-D
Hau Hok Wan 鱟殼灣 Is 9-SE-A
Hau Tong Kai 猴塘溪 TP 8-NW-B
Hau Tsz Kok 孝子角 TP 4-SE-C
Hebe Haven ( Pak Sha Wan ) 白沙灣 SK 8-SW-C
Hebe Hill ( Tsim Fung Shan ) 尖風山 SK 11-NE-B
Hebe Knoll － SK 11-NE-B
Hei Ling Chau 喜靈洲 Is 10-SW-D
Hei Tsz Wan 起子灣 SK 8-SW-B
Heng Mei Deng Village 坑尾頂村 SK 12-NW-C
Heung Chung 響鐘 SK 11-NE-B
Heung Chung Au 響鐘坳 Is 13-NW-C
Heung Fan Liu 香粉寮 ST 7-SW-D
Heung Fan Liu New Village 香粉寮新村 ST 7-SW-D
Heung Lo Kok 響螺角 TP 4-SW-D
Heung Ngam 響巖 Is 15-SE-B
Heung Yuen Wai 香園圍 N 3-NW-B
High Hill 犀牛望月 N 3-SW-A
High Island ( Leung Shuen Wan ) 糧船灣 SK 8-SE-C
High Junk Peak ( Tiu Yue Yung ) 釣魚翁 SK 12-SW-A
High West 西高山 C&W-S 11-SW-C
Hin Pai 蜆排 TP 4-SW-D
Hin Tin 顯田 ST 7-SW-D
Hing Keng Shek 慶徑石 SK 7-SE-D
Hing Yan Tsuen 興仁村 N 3-SW-A
Ho Chung 蠔涌 SK 7-SE-D
Ho Chung New Village 蠔涌新村 SK 11-NE-B
Ho Chung Valley 蠔涌谷 SK 11-NE-B
Ho Hok Shan 蠔殼山 YL 6-NE-A
Ho Lek Pui 河瀝背 N 3-NE-C
Ho Lek Pui 河瀝背 ST 7-NW-D
Ho Lek Pui 河瀝背 TP 3-SE-C
Ho Man Tin 何文田 KC 11-NW-D
Ho Pui 河背 YL 6-NE-C
Ho Pui 河背 N 3-SE-B
Ho Pui Tsuen 河背村 TW 7-SW-C
Ho Sheung Heung 河上鄉 N 2-SE-B
Ho Tin Tsuen 河田村 TM 5-SE-B
Hoi Ha 海下 TP 8-NW-B
Hoi Ha Wan 海下灣 TP 4-SW-D
Hoi Pa Resite Village 海壩村 TW 7-SW-C
Hoi Pa San Tsuen 海壩新村 TW 7-SW-C
Hoi Pa San Tsuen Section Two 海壩新村第二段 TW 7-SW-C
Hoi Pa Village Northeast Terrace 海壩村東北台 TW 7-SW-C
Hoi Pa Village South Terrace 海壩村南台 TW 7-SW-C
Hoi Pui Leng 海背嶺 N 3-NE-C
Hoi Sing Wan 海星灣 SK 8-SW-C
P. 9 / 53
May 2026
English Name Chinese Name District* HP5C
Hoi Tam Hau 海膽口 SK 12-SE-C
Hok Ngam Teng 鶴岩頂 TP 17-NW-C
Hok Tau 鶴藪 N 3-SW-B
Hok Tau Pai 鶴藪排 N 3-SW-B
Hok Tau Wai 鶴藪圍 N 3-SW-B
Hok Tsai Pai 殼仔排 SK 12-SE-C
Hok Tsui Lower Village 鶴咀下村 S 15-NE-D
Hok Tsui Village 鶴咀村 S 15-NE-D
Hok Tsui Wan 鶴咀灣 S 15-NE-D
Hok Wan Tsui 鶴環咀 N 4-NW-D
Hong Kong Island 香港島 - 11
Hong Mei Tsuen 巷尾村 YL 6-NW-A
Hoo Hok Wai 蠔殼圍 N 2-NE-D
Hop Shing Wai 合盛圍 YL 2-SE-A
Horn Hill ( Ngau Kok Shan ) 牛角山 YL 2-NE-D
Hsien Ku Fung 仙姑峰 TP 3-SE-D
Hung Fa Chai 紅花寨 N 3-NE-C
Hung Fa Leng 紅花嶺 N 4-NW-D
Hung Fa Ngan 紅花顏 Is 10-SW-A
Hung Fa Tsuen 紅花村 SK 8-SW-A
Hung Fan Shek 紅粉石 Is 9-SW-D
Hung Hom 紅磡 KC 11-NW-D
Hung Kiu San Tsuen 紅橋新村 ( 洪橋新村 ) N 3-NW-C
Hung Leng 孔嶺 N 3-SW-B
Hung Lung Hang 恐龍坑 N 3-NW-C
Hung Mui Kuk 紅梅谷 ST 7-SW-D
Hung Pai 紅排 N 4-SW-A
Hung Shek Mun ( Wong Chuk Kok Mun ) 紅石門 ( 黃竹角門 ) N 4-SW-A
Hung Shek Mun Au 紅石門坳 N-TP 4-SW-A
Hung Shek Mun Tsuen 紅石門村 N 4-SW-A
Hung Shing Ye 洪聖爺 Is 15-NW-A
Hung Shui Kiu 洪水橋 YL 6-NW-C
Hung Tso Tin Tsuen 紅棗田村 YL 6-NW-D
Hung Uk 洪屋 SK 12-NW-C
Hung Uk Tsuen 洪屋村 YL 6-NW-A
Hung Ying Tsui 紅鷹咀 TP 4-SE-A
Inner Port Shelter ( Sai Kung Hoi ) 西貢海 SK 8-SW-C
Island Bay 香島灣 S 15-NE-B
Jam Pan Wan 砧板灣 SK 12-SW-D
Jardine's Corner 觀龍角 C&W 11-SW-C
Jardine's Lookout 渣甸山 WC 11-SE-C
Jardine's Lookout 渣甸山 WC-E 11-SE-C
Jin Island ( Tiu Chung Chau ) 吊鐘洲 SK 12-NW-B
Jordan Valley 佐敦谷 KT 11-NE-C
Joss House Bay ( Tai Miu Wan ) 大廟灣 SK 12-SW-C
Junk Bay ( Tseung Kwan O ) 將軍澳 SK 11-SE-B
Ka Loon Tsuen 嘉龍村 TW 6-SW-D
Kai Chau 雞洲 SK 8-SW-D
Kai Fong Garden 啟芳園 N 3-NW-D
Kai Ham 界咸 SK 7-SE-D
Kai Kuk Shue Ha 雞谷樹下 N 3-NE-C
Kai Kung Leng 雞公嶺 YL 2-SE-D
P. 10 / 53
May 2026
English Name Chinese Name District* HP5C
Kai Kung Leng 雞公嶺 N 4-NW-B
Kai Kung Pai 雞公排 N 4-NW-B
Kai Kung Shan 雞公山 TP 8-NW-C
Kai Kung Shan 雞公山 Is 13-NW-A
Kai Kung Tau 雞公頭 N 4-NW-B
Kai Leng 雞嶺 N 3-SW-A
Kai Ma Tung 雞麻峒 TP 8-NW-A
Kai Pak Ling 雞伯嶺 YL 6-NW-A
Kai Pei Ngam 雞鼻岩 N 4-NW-D
Kai Shan 髻山 YL 6-NW-B
Kai Tak 啟德 KC 11-NE-C
Kai Yi Wan 雞魚環 SK 12-NE-A
Kai Yue Tam 雞魚氹 SK 16-NW-A
Kak Hang Tun 隔坑墩 SK 8-SW-A
Kak Tin 隔田 ST 7-SW-D
Kak Tin Village Kung Miu 隔田村公廟 ST 7-SW-D
Kak Tin Village Nam Kau 隔田村南滘 ST 7-SW-D
Kam Chuk Kok 金竹角 K&T 10-NE-B
Kam Chuk Pai 金竹排 TP 4-SW-A
Kam Chung Ngam 金鐘岩 SK 12-NW-D
Kam Hing Wai 錦慶圍 YL 6-NE-A
Kam Kai Wan 金雞灣 SK 8-NE-D
Kam Kui Shek Teng 蠄蟝石頂 SK 8-SE-A
Kam Lo Hom 蠄蟧磡 Is 14-NE-B
Kam Lo Wan 蠄蟧灣 SK 12-NE-A
Kam Shan 錦山 TP 7-NW-B
Kam Shek New Village 錦石新村 TP 7-NW-B
Kam Tin 錦田 YL 6-NE-A
Kam Tin River 錦田河 YL 6-NE-A
Kam Tin Shi 錦田市 YL 6-NE-A
Kam Tin Shing Mun San Tsuen 錦田城門新村 YL 6-NE-A
Kam Tsin 金錢 N 2-SE-B
Kam Tsin Wai 金錢圍 YL 6-NE-C
Kan Lung Tsuen 覲龍村 N 3-SW-A
Kan Tau Au 根頭坳 Is 13-NW-A
Kan Tau Tsuen 簡頭村 N 3-SW-B
Kan Tau Wai 簡頭圍 N 3-NW-C
Kang Lau Shek 更樓石 TP 17-NW-C
Kang Mun Tsui 乾門咀 N 4-SW-A
Kap Lo Kok 夾螺角 SK 8-SW-D
Kap Lung 甲龍 YL 6-NE-D
Kap Man Hang 夾萬坑 SK 8-SE-A
Kap Pin Long 甲邊朗 SK 8-SW-A
Kap Pin Long New Village 甲邊朗新村 SK 8-SW-A
Kap Shui Mun 汲水門 TW 10-NE-A
Kar Wo Lei 嘉和里 TM 6-SW-C
Kat Hing Wai 吉慶圍 YL 6-NE-A
Kat O Fisherman Village 吉澳漁民村 N 4-NW-A
Kat O Kok 吉澳角 N 4-NW-B
Kat O Sheung Wai 吉澳上圍 N 4-NW-A
Kat O Wan 吉澳灣 N 4-NW-A
Kat Tsai Shan Au 桔仔山坳 N-TP 3-SW-D
P. 11 / 53
May 2026
English Name Chinese Name District* HP5C
Kat Tsai Wan 桔仔灣 Is 15-NW-C
Kau Chung Wan 滘中灣 SK 12-NW-B
Kau Hui Tsuen 舊墟村 TM 6-SW-A
Kau Keng Shan 九逕山 TM 6-SW-A
Kau Kung Tong 九宮塘 Is 14-NW-D
Kau Lee Uk Tsuen 舊李屋村 YL 6-NW-A
Kau Ling Chung 狗嶺涌 Is 13-NW-D
Kau Liu 較寮 Is 9-SE-A
Kau Liu Ha 較寮下 TP 7-NW-A
Kau Lo Tau 九蘆頭 N 4-NW-C
Kau Lung Hang Lo Wai 九龍坑老圍 TP 3-SW-D
Kau Lung Hang San Wai 九龍坑新圍 TP 3-SW-D
Kau Ma Shek 狗麻石 N 4-NW-C
Kau Nga Ling 狗牙嶺 Is 13-NE-A
Kau Pei Chau 狗髀洲 S 15-NE-D
Kau Po 舊埗 TW 10-NE-A
Kau Sai 滘西 SK 12-NW-B
Kau Sai Chau 滘西洲 SK 8-SW-D
Kau Sai San Tsuen 滘西新村 SK 7-SE-D
Kau Sai Wan 滘西灣 SK 12-NW-B
Kau San Tei 狗伸地 Is 9-SW-D
Kau Shat Wan 狗虱灣 Is 10-SW-D
Kau Shui Liu 較水寮 SK 12-NW-B
Kau Tam Tso 九担租 N 3-SE-B
Kau Tau Shek 狗頭石 N 4-NW-A
Kau To Hang 九肚坑 ST 7-SE-A
Kau To Village 九肚村 ST 7-NE-C
Kau Tsin Uk 較剪屋 SK 11-NE-B
Kau Tung Wan 滘東灣 SK 12-NW-B
Kau Wa Keng 九華徑 K&T 11-NW-A
Kau Wa Keng San Tsuen 九華徑新村 K&T 11-NW-A
Kau Yi Chau 交椅洲 Is 10-SE-A
Kaw Liu Village 較寮村 N 3-NW-C
Kei Kok Tau 企角頭 SK 8-SE-D
Kei Lak Tsai 箕勒仔 N 3-SW-C
Kei Ling 企嶺 YL 6-NE-D
Kei Ling Ha Lo Wai 企嶺下老圍 TP 8-NW-C
Kei Ling Ha San Wai 企嶺下新圍 TP 8-NW-C
Kei Lun Wai 麒麟圍 TM 6-NW-C
Kei Pik Shan 企壁山 SK 7-SE-D
Kei Shan 企山 SK 12-NW-B
Kei Shan Au 企山坳 SK 12-NW-B
Kei Shan Tsui 企山咀 N 3-NE-B
Kei Tau Kok Teng 企頭角頂 SK 12-NE-A
Kei Yan Shek 企人石 SK 8-SW-D
Kellett Island 奇力島 WC 11-SW-B
Keng Pang Ha 徑棚下 SK 7-SE-B
Kennedy Town 堅尼地城 C&W 11-SW-A
Keung Shan 羗山 Is 13-NW-B
Keung Shan 羗山 Is 13-NW-B
Ki Lun Tsuen 麒麟村 YL 2-SE-B
Kim Chu Wan 撿豬灣 SK 8-SE-D
P. 12 / 53
May 2026
English Name Chinese Name District* HP5C
King's Park 京士柏 YTM 11-NW-D
Kiu Tau 橋頭 SK 8-SW-C
Kiu Tau 橋頭 TP 3-SW-D
Kiu Tau Tsuen 橋頭村 YL 6-NE-A
Kiu Tau Wai 橋頭圍 YL 6-NW-A
Kiu Tsui 橋咀 SK 8-SW-C
Ko Hang 高行 YL 2-SE-C
Ko Lau Wan 高流灣 TP 8-NE-A
Ko Lau Wan Tsui 高流灣咀 TP 8-NE-A
Ko Long 高塱 Is 14-NE-B
Ko Pai 高排 Is 13-SE-A
Ko Pai 高排 N 4-NW-C
Ko Pang Teng 高棚頂 N 4-NW-A
Ko Po 高莆 ( 高埔 ) N 3-SW-B
Ko Po North 高埔北 N 3-SW-B
Ko Po San Tsuen 高埔新村 YL 6-NE-A
Ko Po Shan 高埔山 N 3-SW-B
Ko Po Tsuen 高埔村 YL 6-NE-A
Ko Shan Tsuen 高山村 Is 14-NW-D
Ko Tei Teng 高地頂 N 4-NW-A
Ko Tin Hom 高田磡 TP 7-NW-A
Ko Tong 高塘 TP 8-NW-D
Ko Tong Ha Yeung 高塘下洋 TP 8-NW-D
Ko Tong Hau 高塘口 TP 8-NE-C
Kok Tai Pai 角大排 N 4-SW-B
Kon Hang 乾坑 TP 7-NE-C
Kong A Leng 缸瓦嶺 YL 6-NE-A
Kong Ha 崗下 N 3-NE-A
Kong Nga Po 缸瓦甫 N 3-NW-C
Kong Pui Tsuen 崗背村 ST 7-SE-C
Kong Tau Pai 光頭排 SK 12-NE-B
Kong Tau San Tsuen 港頭新村 YL 6-NW-D
Kong Tau Tsuen 港頭村 YL 6-NW-D
Kong Yiu 缸窰 N 3-NW-B
Kop Tong 蛤塘 N 3-SE-B
Kowloon 九龍 - 11
Kowloon Bay 九龍灣 KC 11-NE-C
Kowloon Bay 九龍灣 KT 11-NE-C
Kowloon City 九龍城 KC 11-NE-A
Kowloon Pass 九龍坳 ST 11-NW-B
Kowloon Peak ( Fei Ngo Shan ) 飛鵝山 SK-WTS 11-NE-A
Kowloon Rock 九龍石 KC 11-NE-C
Kowloon Tong 九龍塘 KC 11-NW-B
Kuk Liu 穀寮 ST 7-SW-D
Kuk Po 谷埔 N 3-NE-D
Kuk Po Lo Wai 谷埔老圍 N 3-NE-D
Kuk Po San Uk Ha 谷埔新屋下 N 3-NE-D
Kung Chau 弓洲 TP 4-SE-C
Kung Tsai Wan 公仔灣 TW 10-NE-A
Kwai Au Shan 葵坳山 SK 11-NE-B
Kwai Chung 葵涌 K&T 7-SW-C
Kwai Kiu Kuk 桂橋谷 N 4-NW-A
P. 13 / 53
May 2026
English Name Chinese Name District* HP5C
Kwai Shan 龜山 S 15-NE-A
Kwai Shek 拐石 TW 10-NE-A
Kwai Tau Leng 龜頭嶺 N 3-SE-A
Kwai Tei New Village 桂地新村 ST 7-SW-B
Kwai Tei Village 桂地村 ST 7-SW-B
Kwan Mun Hau Tsuen 關門口村 TW 7-SW-C
Kwan Tei 軍地 N 3-SW-B
Kwan Tei North 軍地北 N 3-SW-B
Kwan Tei River 軍地河 N 3-SW-B
Kwat Tau Tam 掘頭氹 SK 8-SW-D
Kwo Chau Wan 果洲灣 SK 12-SE-C
Kwong Pan Tin San Tsuen 光板田新村 TW 6-SE-D
Kwong Pan Tin Tsuen 光板田村 TW 6-SE-D
Kwong Shan Tsuen 礦山村 TM 5-NE-D
Kwu Hang Village 古坑村 TW 7-SW-C
Kwu Tung 古洞 N 2-SE-B
Kwun Cham Wan 罐杉環 SK 8-SW-D
Kwun Hang 官坑 TP 7-NE-D
Kwun Mun Fishermen Village 官門漁村 SK 8-SW-C
Kwun Tong 觀塘 KT 11-NE-C
Kwun Tong Tsai Wan ( Yau Tong Bay ) 觀塘仔灣 ( 油塘灣 ) KT 11-SE-B
Kwun Tsai 觀仔 SK 12-SW-C
Kwun Tsoi Pai 棺材排 SK 8-SW-C
Kwun Yam Keng 觀音徑 TP 7-NW-C
Kwun Yam Shan 觀音山 S 15-NE-B
Kwun Yam Shan 觀音山 YL 7-NW-C
Kwun Yam Shan 觀音山 Is 13-NW-B
Kwun Yam Shan 觀音山 ST 7-SE-C
Kwun Yam Shan Village 觀音山村 ST 7-SE-C
Kwun Yam Wan 觀音灣 Is 14-NW-D
Lai Chi Chong 荔枝莊 TP 8-NW-B
Lai Chi Hang 荔枝坑 TP 7-NW-D
Lai Chi Kok 荔枝角 SSP 11-NW-A
Lai Chi Shan 荔枝山 TP 7-NW-B
Lai Chi Wo 荔枝窩 N 3-NE-D
Lai Chi Yuen 荔枝園 ST 7-SW-D
Lai Chi Yuen Tsuen 荔枝園村 Is 10-SW-C
Lai Pek Shan 犁壁山 TP-N 3-SE-C
Lai Pek Shan 犁壁山 TP 3-SE-C
Lai Pek Shan San Tsuen 犁壁山新村 TP 3-SE-C
Lai Pik Shan 犁壁山 TW 10-NW-D
Lai Tau Shek 犁頭石 N 4-SW-A
Lai Tau Tsim 泥頭尖 YL 6-NE-B
Lai Uk Tsuen 黎屋村 YL 6-NE-D
Lak Lei Tsai 癩痢仔 SK 12-NW-D
Lam Che 藍輋 Is 9-SE-A
Lam Hau Tsuen 欖口村 YL 6-NW-D
Lam Tei 藍地 TM 6-NW-C
Lam Tin 藍田 KT 11-NE-D
Lam Tin Resite Village 藍田村 K&T 10-NE-B
Lam Tsuen River 林村河 TP 7-NW-A
Lam Tsuen San Tsuen 林村新村 TP 7-NW-A
P. 14 / 53
May 2026
English Name Chinese Name District* HP5C
Lam Tsuen Valley 林村谷 TP 7-NW-A
Lam Uk 林屋 TP 8-NE-A
Lam Uk 林屋 TP 17-NW-C
Lam Uk Wai 林屋圍 SK 8-NE-D
Lam Wan Kok 杬挽角 SK 12-NE-C
Lamb Hill ( Ma Tau Leng ) 馬頭嶺 N 3-SW-A
Lamma Island 南丫島 Is 15-NW-C
Lan Kwo Shui 難過水 TP 17-NW-C
Lan Lo Au 攔路坳 TP 8-NE-A
Lan Nai Wan 爛泥灣 Is 13-SE-A
Lan Nai Wan 爛泥灣 S 15-NE-B
Lan Nai Wan Village 爛泥灣村 S 15-NE-B
Lan Shuen Pai 爛船排 N 4-NW-C
Lan Tau Pai 爛頭排 SK 8-SE-B
Lantau Channel 大嶼海峽 - 13-SW-C
Lantau Island 大嶼山 Is 9-SE-D
Lantau Peak ( Fung Wong Shan ) 鳳凰山 Is 9-SE-C
Lap Ngam Tsui 立岩咀 Is 13-NW-D
Lap Sap Chau 垃圾洲 SK 8-SW-A
Lap Sap Wan 垃圾灣 S 15-NE-D
Lap Wo Tsuen 立和村 N 3-NE-C
Lau Fa Tsuen 柳花村 TW 10-NE-A
Lau Fa Tung 榴花峒 Is 10-NW-C
Lau Fau Shan 流浮山 YL 2-SW-C
Lau Fau Shan 流浮山 YL 2-SW-C
Lau Hang 流坑 TP 7-NW-B
Lau Shui Hang 流水坑 Is 16-SW-A
Lau Shui Hang Tam 流水坑氹 TP 4-SE-A
Lau Shui Heung 流水響 N 3-SW-B
Lau Uk 劉屋 TP 8-NE-A
Lead Mine Pass 鉛鑛坳 TP-TW 7-NW-D
Lecky Pass ( Sheung Ma Lei Yue ) 雙孖鯉魚 N 2-NE-D
Ledge Point ( Cheung Pai Tau ) 長排頭 N 3-NE-B
Lee Uk Village 李屋村 ST 7-SW-D
Lei Uk 李屋 TP 17-NW-A
Lei Uk 李屋 N 3-NW-C
Lei Uk 李屋 TP 3-SW-D
Lei Yue Mun 鯉魚門 E 11-SE-B
Lei Yue Mun Point 鯉魚門咀 KT 11-SE-B
Lei Yue Mun Village 鯉魚門村 KT 11-SE-B
Leighton Hill 禮頓山 WC 11-SW-B
Leng Pei Tsuen 嶺皮村 N 3-SW-B
Leng Pui 嶺背 N 3-SE-B
Leng Tsai 嶺仔 N 3-SW-B
Leung Fai Tin 兩塊田 SK 12-NW-C
Leung Tin Village 良田村 TM 6-SW-A
Leung Uk Tsuen 梁屋村 YL 6-NE-B
Leung Uk Tsuen 梁屋村 Is 9-SW-D
Lin Au 蓮澳 TP 7-NW-A
Lin Barn Tsuen 練板村 YL 2-SE-A
Lin Fa Shan 蓮花山 Is 10-SW-C
Lin Fa Shan 蓮花山 TW 6-SE-B
P. 15 / 53
May 2026
English Name Chinese Name District* HP5C
Lin Fa Tei 蓮花地 YL 6-NE-D
Lin Ma Hang 蓮麻坑 N 3-NW-B
Lin Tong Mei 蓮塘尾 N 2-SE-D
Lin Tong Mei Tsoi Yuen 蓮塘尾菜園 N 2-SE-D
Ling Hill 靈山 N 3-SW-A
Ling Kok Shan 菱角山 Is 15-NW-C
Ling Shan Tsuen 靈山村 N 3-SW-A
Ling Tsui Tau 嶺咀頭 Is 10-SW-C
Ling Wui Shan 靈會山 Is 13-NW-B
Lion Rock 獅子山 ST 11-NW-B
Little Green Island 小青洲 C&W 11-SW-A
Liu Ko Ngam 了哥岩 N 4-NW-C
Liu Pok 料壆 N 2-NE-D
Liu To 寮肚 K&T 10-NE-B
Lo Chau 羅洲 S 15-NE-C
Lo Chau Mun 螺洲門 Is 15-SE-B
Lo Chau Pak Pai 螺洲白排 Is 15-SE-B
Lo Chi Pai 鸕鷀排 SK 8-SW-D
Lo Chi Pai 鸕鷀排 N 4-NW-C
Lo Fu Hang 老虎坑 TM 6-NW-C
Lo Fu Kei Shek 老虎騎石 TP 8-NW-B
Lo Fu Ngam 老虎岩 SK 8-SW-C
Lo Fu Shan 老虎山 S 15-NE-A
Lo Fu Shek Teng 老虎石頂 N 3-NE-D
Lo Fu Tau 老虎頭 Is 10-SW-A
Lo Fu Tiu Pai 老虎吊排 SK 8-SW-D
Lo Fu Wat 老虎笏 TP 4-SW-C
Lo Kei Wan 籮箕灣 Is 13-NE-C
Lo Kei Wan 籮箕灣 N 4-NW-D
Lo Lau Uk 老劉屋 TP 7-NW-D
Lo Lung Tin 老龍田 N 3-SE-A
Lo Sha Tin 老沙田 N 4-SW-B
Lo Shue Ling 老鼠嶺 N 3-NW-C
Lo Shue Pai 老鼠排 E 11-SE-B
Lo Shue Tin 老鼠田 ST 7-SE-C
Lo So Shing 蘆鬚城 Is 15-NW-C
Lo Tei Tun 螺地墩 SK 8-SE-A
Lo Tik Wan 蘆荻灣 Is 15-NW-A
Lo Tsai Shek 爐仔石 TP 8-NW-B
Lo Tsz Tin 蘆慈田 TP 3-SE-D
Lo Uk Tsuen 羅屋村 YL 6-NW-A
Lo Uk Tsuen 羅屋村 Is 14-NW-A
Lo Wai 老圍 N 3-SE-B
Lo Wai 老圍 N 3-SW-A
Lo Wai 老圍 TW 7-SW-C
Lo Wan 螺灣 SK 8-NE-D
Lo Wu 羅湖 N 2-NE-D
Lo Wu 羅湖 N 2-NE-D
Lo Yan Shan 老人山 Is 14-NW-A
Loaf Rock 饅頭石 Is 14-NW-D
Loi Tung 萊洞 N 3-NW-D
Lok Lo Ha 落路下 ST 7-SE-A
P. 16 / 53
May 2026
English Name Chinese Name District* HP5C
Lok Ma Chau 落馬洲 YL 2-SE-B
Lok Ma Chau 落馬洲 YL 2-SE-B
Long Ha 朗廈 YL 2-SE-C
Long Harbour ( Tai Tan Hoi ) 大灘海 TP 8-NE-A
Long Hill ( Tung Sam Kei Shan ) 東心淇山 TP 8-NE-C
Long Ke 浪茄 SK 8-SE-D
Long Ke Tsai 浪茄仔 SK 8-SE-D
Long Ke Wan 浪茄灣 SK 8-SE-D
Long Keng 浪徑 SK 8-SW-A
Long Mei 朗尾 SK 8-SW-A
Long Mong Wan 浪芒灣 SK 8-SW-C
Long Tsai Tsuen 龍仔村 Is 15-NW-A
Long Valley 塱原 N 2-SE-B
Lover's Rock 姻緣石 WC 11-SW-D
Lower Keung Shan 下羗山 Is 9-SW-D
Lower Shing Mun Village 城門下村 K&T 7-SW-C
Luen On San Tsuen 聯安新村 TM 6-SW-D
Luen Wo Hui 聯和墟 N 3-SW-A
Luen Yick Fishermen Village 聯益漁村 TP 7-NE-A
Lui Kung Tin 雷公田 YL 6-NE-D
Lui Ta Shek 雷打石 SK 8-NW-D
Luk Chau 鹿洲 Is 15-NW-A
Luk Chau Au 鹿巢坳 ST 7-SE-B
Luk Chau Shan 鹿巢山 ST 7-SE-B
Luk Chau Shan 鹿洲山 Is 15-NW-C
Luk Chau Village 鹿洲村 Is 15-NW-C
Luk Chau Wan 鹿洲灣 Is 15-NW-A
Luk Keng 鹿頸 N 3-NE-C
Luk Keng 鹿頸 TW 10-NW-B
Luk Keng Bay 鹿頸灣 TW 10-NW-B
Luk Keng Chan Uk 鹿頸陳屋 N 3-NE-C
Luk Keng Lam Uk 鹿頸林屋 N 3-SE-A
Luk Keng Shan 鹿頸山 Is 13-NE-C
Luk Keng Tsuen 鹿頸村 TW 10-NW-B
Luk Keng Wan 鹿頸灣 SK 16-NW-A
Luk Keng Wong Uk 鹿頸黃屋 N 3-NE-C
Luk Mei Tsuen 鹿尾村 YL 2-SE-B
Luk Mei Tsuen 鹿尾村 SK 7-SE-D
Luk Tei Tong 鹿地塘 Is 10-SW-C
Luk Wu 鹿湖 SK 8-SE-A
Luk Wu 鹿湖 Is 9-SW-D
Luk Wu Tung 鹿湖峒 N 4-SW-A
Lung A Pai 龍丫排 TP 7-NW-A
Lung Fu Shan 龍虎山 C&W 11-SW-A
Lung Ha Wan 龍蝦灣 SK 12-NW-D
Lung Ha Wan 龍蝦灣 TW 10-NE-A
Lung Hang 龍坑 SK 8-NW-D
Lung Keng Kan 龍頸筋 TP 4-SE-C
Lung Kwu Chau 龍鼓洲 TM 5-SW-D
Lung Kwu Sheung Tan 龍鼓上灘 TM 5-SE-A
Lung Kwu Tan 龍鼓灘 TM 5-SE-A
Lung Lok Shui 龍落水 TP 17-NW-C
P. 17 / 53
May 2026
English Name Chinese Name District* HP5C
Lung Lun Tsui 龍麟咀 TP 17-NW-C
Lung Mei 龍尾 SK 8-SW-A
Lung Mei 龍尾 Is 14-NW-A
Lung Mei 龍尾 TP 3-SE-D
Lung Mei Hang 龍尾坑 Is 10-SW-C
Lung Mei Tau 龍尾頭 SK 8-NE-D
Lung Mei Teng 龍尾頂 N 3-NW-D
Lung Mei Tsuen 龍尾村 Is 10-SW-C
Lung Ngan Yuen Tau 龍眼圓頭 SK 12-NW-A
Lung Shan 龍山 N 3-SW-B
Lung Shan Pai 龍山排 S 15-NW-A
Lung Shuen Pai 龍船排 SK 12-SE-C
Lung Shuen Pai 龍船排 SK 12-NE-C
Lung Shuen Pai 龍船排 Is 13-SE-A
Lung Tin Tsuen 龍田村 YL 6-NW-B
Lung Tsai 龍仔 TM 5-SE-A
Lung Tsai Ng Yuen 龍仔悟園 Is 13-NW-B
Lung Tsai Tsuen 龍仔村 Is 14-NW-D
Lung Tseng Tau 龍井頭 Is 9-SE-B
Lung Wo Tsuen 龍窩村 SK 11-NE-B
Lung Yeuk Tau 龍躍頭 N 3-SW-A
Lut Chau 甩洲 YL 2-SW-D
Ma Chau 孖洲 Is 13-SE-A
Ma Kok Tsui 媽角咀 TP 17-NW-C
Ma Kok Tsui 馬角咀 TW 10-NE-A
Ma Kwu Lam 馬牯纜 TP 8-NW-C
Ma Lai Hau Hang 馬麗口坑 ST 7-SE-C
Ma Liu Shui 馬料水 ST 7-NE-C
Ma Liu Shui San Tsuen 馬料水新村 N 3-SW-B
Ma Mei Ha 馬尾下 N 3-NW-D
Ma Mei Ha Leng Tsui 馬尾下嶺咀 N 3-SW-B
Ma Nam Wat 麻南笏 SK 8-SW-C
Ma Niu 馬尿 ST 7-SE-A
Ma Niu River 馬尿河 N 4-SW-A
Ma Niu Shui 馬尿水 N 4-SW-A
Ma On Kong 馬鞍崗 YL 6-NE-C
Ma On Shan 馬鞍山 TP-ST 7-SE-B
Ma On Shan 馬鞍山 ST 7-NE-D
Ma On Shan Tsuen 馬鞍山村 ST 7-SE-B
Ma Pau Ling 麻包嶺 YL 6-NE-D
Ma Po Mei 麻布尾 TP 7-NW-A
Ma Po Tsuen 麻布村 ( 麻埔村 ) Is 10-SW-C
Ma Pui Tsuen 馬背村 KT 11-SE-B
Ma Shi Chau 馬屎洲 TP 7-NE-B
Ma Shi Po 馬屎埔 N 3-SW-A
Ma Sim Pai Village 馬閃排村 TW 7-SW-C
Ma Tau Fung 馬頭峰 N-TP 3-SE-B
Ma Tau Kok 馬頭角 KC 11-NE-C
Ma Tau Wai 馬頭圍 KC 11-NW-D
Ma Tau Wan 馬頭環 SK 12-NE-A
Ma Tin Pok 馬田壆 YL 6-NW-B
Ma Tin Tsuen 馬田村 YL 6-NW-B
P. 18 / 53
May 2026
English Name Chinese Name District* HP5C
Ma Tsai Pai 孖仔排 SK 12-NW-D
Ma Tseuk Leng 麻雀嶺 N 3-NE-C
Ma Tseuk Leng San Uk Ha 麻雀嶺新屋下 N 3-NE-C
Ma Tseuk Tong 麻雀塘 TP 3-SE-C
Ma Tso Lung 馬草壟 N 2-NE-D
Ma Tso Lung San Tsuen 馬草壟新村 N 2-NE-D
Ma Tso Lung Shun Yee San Tsuen
(Ma Tso Lung Lutheran New Village) 馬草壟信義新村 N 2-NE-D
Ma Wan 馬灣 TW 10-NE-A
Ma Wan 媽灣 SK 12-SE-C
Ma Wan Channel 馬灣海峽 TW 10-NE-A
Ma Wan Chung 馬灣涌 Is 9-SE-B
Ma Wan Fishermen's Village
( Ma Wan CARE Village )
馬灣漁民新村
( 馬灣美經援村 ) TW 10-NE-A
Ma Wan Main Street Village 馬灣大街村 TW 10-NE-A
Ma Wan Main Street Village Central 馬灣大街村中 TW 10-NE-A
Ma Wan Main Street Village East 馬灣大街村東 TW 10-NE-A
Ma Wan Main Street Village North 馬灣大街村北 TW 10-NE-A
Ma Wan Main Street Village South 馬灣大街村南 TW 10-NE-A
Ma Wan New Village 馬灣新村 Is 9-SE-B
Ma Wan Town 馬灣市 TW 10-NE-A
Ma Wan Tsuen 馬環村 KT 11-SE-B
Ma Wat River 麻笏河 N 3-SW-C
Ma Wat Tsuen 麻笏村 N 3-SW-A
Ma Wat Wai 麻笏圍 N 3-SW-A
Ma Wo 馬窩 TP 7-NW-B
Ma Yau Tong 馬游塘 SK 11-NE-D
Ma Yiu 馬腰 TP 7-NE-B
Magazine Gap 馬己仙峽 C&W 11-SW-D
Magazine Island 火藥洲 S 15-NW-A
Mai Fan Teng 米粉頂 SK 8-NE-D
Mai Fan Tsui 米粉咀 SK 8-NE-B
Mai Po 米埔 YL 2-SE-A
Mai Po Lo Wai 米埔老圍 YL 2-SE-A
Mai Po Lung Tsuen 米埔隴村 YL 2-SE-A
Mai Po San Tsuen 米埔新村 YL 2-SE-A
Mak Uk 麥屋 TP 3-SW-D
Man Cheung Po 萬丈布 Is 13-NW-B
Man Hang 蚊坑 ST 7-SE-A
Man Kam To 文錦渡 N 3-NW-C
Man King Terrace 萬景台 SK 11-NE-B
Man Kok 萬角 Is 10-SW-D
Man Kok Tsui 萬角咀 Is 10-SW-D
Man Kok Village 文閣村 N 3-SW-A
Man Sau Sun Tsuen 萬壽新村 SK 8-SW-C
Man Tau Tsui 萬頭咀 SK 8-SW-C
Man Tau Tun 饅頭墩 ST 7-SE-C
Man Uk Pin 萬屋邊 N 3-NW-D
Man Wo 蠻窩 SK 11-NE-B
Man Yee Wan New Village 萬宜灣新村 SK 8-SW-C
Man Yuen Chuen 文苑村 YL 2-SE-C
Mang Kung Uk 孟公屋 SK 12-NW-C
P. 19 / 53
May 2026
English Name Chinese Name District* HP5C
Mat Chau 墨洲 Is 15-SE-D
Mat Chau Mun 墨洲門 Is 15-SE-D
Mat Chau Pai 墨洲排 Is 15-SE-D
Mau Ping 茅坪 ST 7-SE-B
Mau Ping Lo Uk 茅坪老屋 ST 7-SE-B
Mau Ping New Village 茅坪新村 SK 8-SW-A
Mau Ping San Uk 茅坪新屋 ST 7-SE-B
Mau Ping Shan 茅平山 TP 4-SE-C
Mau Po 茅莆 SK 12-NW-C
Mau Tat 茅笪 ST 7-SE-C
Mau Tin 苗田 SK 12-NW-A
Mau Tso Ngam 茂草岩 ST 7-SE-C
Mau Wu Shan 茅湖山 SK 11-NE-D
Mau Wu Tsai 茅湖仔 SK 11-NE-D
Mau Yuen 茅園 Is 10-SW-C
Mei Pai 尾排 Is 14-NW-C
Middle Bay 中灣 S 15-NE-A
Middle Channel ( Chek Chau Hau ) 赤洲口 TP 4-SE-A
Middle Gap 中峽 WC 11-SW-D
Middle Hill ( Cheung Shan ) 象山 SK-WTS 11-NE-A
Middle Island ( Tong Po Chau ) 熨波洲 S 15-NW-B
Mid-Levels 半山區 C&W 11-SW-A
Mirror Pool 照鏡潭 N-TP 3-SE-B
Mirs Bay ( Dapeng Wan ) 大鵬灣 TP-N 4-NE-D
Mit Kok Tsui 滅角咀 N 4-SW-B
Miu Keng 廟徑 N 3-NW-D
Miu Kok 廟角 Is 15-SE-D
Miu Pui 廟背 Is 15-SE-B
Miu Tsai 廟仔 SK 12-NW-C
Miu Tsai Tun 廟仔墩 SK 12-SW-A
Mo Chau 磨洲 TP 4-SW-D
Mo Fan Heung 模範鄉 YL 6-NE-A
Mo Tat New Village 模達新村 Is 15-NW-C
Mo Tat Old Village 模達舊村 Is 15-NW-C
Mo Tat Wan 模達灣 Is 15-NW-C
Mo To Hang 磨刀坑 N 4-NW-C
Mo Uk 巫屋 TP 8-NE-A
Mok Ka 莫家 Is 9-SE-C
Mok Tse Che 莫遮輋 SK 11-NE-B
Mong Chau Tsai 芒洲仔 SK 8-SW-D
Mong Kok 旺角 YTM 11-NW-D
Mong To Au 望渡坳 Is 10-SW-A
Mong Tseng Tsuen 輞井村 YL 2-SW-C
Mong Tseng Wai 輞井圍 YL 2-SW-C
Mong Tung Hang 望東坑 TW 10-NW-D
Mong Tung Wan 望東灣 Is 14-NW-A
Mong Tung Wan 望東灣 Is 14-NW-A
Mong Yue Kok 望魚角 SK 8-NE-D
Morrison Hill 摩理臣山 WC 11-SW-B
Mount Butler 畢拿山 E 11-SE-C
Mount Cameron 金馬倫山 WC 11-SW-D
Mount Collinson 歌連臣山 S 11-SE-D
P. 20 / 53
May 2026
English Name Chinese Name District* HP5C
Mount Davis 摩星嶺 C&W 11-SW-A
Mount Gough 歌賦山 C&W 11-SW-D
Mount Hallowes ( Tam Chai Shan ) 担柴山 TP 8-NW-B
Mount Kellett 奇力山 C&W 11-SW-C
Mount Newland ( Kwun Yam Tung ) 觀音峒 TP 4-SW-A
Mount Nicholson 聶高信山 WC 11-SW-D
Mount Parker 柏架山 E 11-SE-C
Mount Stenhouse ( Shan Tei Tong ) 山地塘 Is 15-SW-A
Mui Shue Hang 梅樹坑 TP 7-NW-A
Mui Tsz Lam 梅子林 N 3-NE-D
Mui Tsz Lam 梅子林 ST 7-SE-B
Mui Wo 梅窩 Is 10-SW-C
Mui Wo Kau Tsuen 梅窩舊村 Is 10-SW-C
Muk Fat Teng 木筏頂 N 4-NW-D
Muk Kiu Tau Tsuen 木橋頭村 YL 6-NW-D
Muk Min Ha Tsuen 木棉下村 TW 7-SW-C
Muk Min Shan 木棉山 SK 8-SW-A
Muk Min Tau 木棉頭 N 3-NE-C
Muk Wu 木湖 N 3-NW-C
Muk Wu Nga Yiu 木湖瓦窰 N 3-NW-C
Muk Yue Chau 木魚洲 SK 8-SW-B
Muk Yue Shan 木魚山 Is 9-SE-C
Muk Yue Tau 木魚頭 SK 8-SW-B
Mun Hau Tsai 門口仔 YL 6-NE-D
Mun Hau Tsuen 門口村 N 3-SW-A
Mun Tsai 門仔 N 4-NW-D
Mun Tsai Wan 門仔灣 N 4-NW-D
Mun Wan 蚊灣 SK 8-NE-B
Nai Chung 泥涌 TP 7-NE-D
Nai Mang Po 泥鯭埔 SK 12-NW-B
Nai Tau 奶頭 TP 17-NW-C
Nai Tong Kok 泥塘角 TP 3-SE-B
Nai Wai 泥圍 TM 6-NW-C
Nam A 南丫 SK 8-SW-A
Nam Chung 南涌 N 3-NE-C
Nam Chung Cheng Uk 南涌鄭屋 N 3-NE-C
Nam Chung Cheung Uk 南涌張屋 N 3-NE-C
Nam Chung Lei Uk 南涌李屋 N 3-NE-C
Nam Chung Lo Uk 南涌羅屋 N 3-NE-C
Nam Chung River 南涌河 N 3-NE-C
Nam Chung Tsuen 南涌村 Is 9-SW-D
Nam Chung Yeung Uk 南涌楊屋 N 3-NE-C
Nam Fung Chau 南風洲 SK 12-NE-A
Nam Fung Kok 南風角 TP 8-NE-A
Nam Fung Shan 南風山 TP 8-NE-A
Nam Fung Wan 南風灣 SK 8-SW-D
Nam Fung Wan 南風灣 TP 8-NE-A
Nam Hang 南坑 N 3-NW-C
Nam Hang 南坑 TP 7-NW-B
Nam Hang Mei 南坑尾 N 3-NE-C
Nam Hang Pai 南坑排 YL 6-NW-D
Nam Hang Tsuen 南坑村 YL 6-NW-D
P. 21 / 53
May 2026
English Name Chinese Name District* HP5C
Nam Hing Lei 南慶里 YL 6-NE-D
Nam Kok Tsui 南角咀 Is 13-SE-C
Nam Kok Tsui 南角咀 Is 15-SE-D
Nam Long 南朗 TM 5-SE-A
Nam Mun Hau 南門口 YL 6-NW-B
Nam Pin Wai 南邊圍 SK 11-NE-B
Nam Pin Wai 南邊圍 YL 6-NW-B
Nam Sang Wai 南生圍 YL 6-NW-B
Nam Sha Po 南沙莆 YL 2-SW-C
Nam Shan 南山 Is 10-SW-C
Nam Shan 南山 Is 13-SE-C
Nam Shan 南山 ST 7-SE-A
Nam Shan 南山 SK 8-SW-A
Nam Shan 南山 N 3-SW-B
Nam Shan Mei 南山尾 WTS 11-NE-A
Nam Shan Mei 南山尾 TP 3-SE-C
Nam Shan Tung 南山洞 TP 8-NW-B
Nam She 蚺蛇 SK 8-NE-C
Nam She Au 蚺蛇坳 SK 8-NE-C
Nam She Tong 南蛇塘 Is 14-NW-D
Nam She Wan 蚺蛇灣 SK 8-NE-B
Nam Tam 南氹 Is 14-NW-D
Nam Tam Wan 南氹灣 Is 15-SE-D
Nam Tam Wan 南氹灣 Is 14-NW-D
Nam Tin 南田 Is 9-SW-D
Nam Tong 南塘 TP 17-NW-C
Nam Tong 南堂 SK 12-SW-C
Nam Tsui 南咀 Is 15-NW-A
Nam Wa Po 南華莆 TP 3-SW-C
Nam Wai 南圍 SK 11-NE-B
Nam Wan 南灣 TW 10-NE-A
Nam Wan 南灣 Is 10-SW-B
Nam Wan 南灣 K&T 10-NE-B
Nam Wan 南環 K&T 10-NE-B
Nam Wan Kok 南灣角 K&T 10-NE-B
Nam Wan San Tsuen 南灣新村 Is 10-SW-B
Nam Wan Shan Ting San Tsuen 南灣山頂新村 Is 10-SW-B
Needle Hill 針山 ST 7-SW-B
Nei Lak Shan 彌勒山 Is 9-SE-C
New Tung Chung Hang 新東涌坑 Is 9-SE-B
Ng Fai Tin 五塊田 SK 12-NW-C
Ng Fan Chau 五分洲 S 15-NE-B
Ng Ka Tsuen 吳家村 YL 6-NE-A
Ng Kwu Leng 五鼓嶺 TW 10-NE-A
Ng To 五肚 N 3-NE-D
Ng Tung Chai 梧桐寨 TP 7-NW-A
Ng Tung River 梧桐河 N 3-SW-A
Ng Uk Tsuen 吳屋村 TP 3-SE-D
Ng Uk Tsuen 吳屋村 N 3-SW-B
Ng Uk Tsuen 吳屋村 YL 2-SW-D
Ng Uk Tsuen 吳屋村 N 3-SW-A
Nga Kau Wan 牙較灣 Is 14-NE-B
P. 22 / 53
May 2026
English Name Chinese Name District* HP5C
Nga Tsin Wai 衙前圍 WTS 11-NE-A
Nga Ying Chau 牙鷹洲 SK 8-SW-B
Nga Ying Kok 牙鷹角 Is 13-NW-A
Nga Ying Pai 牙鷹排 SK 16-NW-A
Nga Ying Shan 牙鷹山 Is 13-NW-A
Nga Yiu Ha 瓦窰下 N 3-NW-D
Nga Yiu Tau 瓦窰頭 TP 8-NW-C
Nga Yiu Tau 瓦窰頭 N 3-NE-C
Nga Yiu Tau 瓦窰頭 YL 6-NW-D
Ngai Kok 崖角 TP 17-NW-A
Ngai Kong 矮崗 TP 7-NW-A
Ngai Tau 崖頭 Is 15-NW-D
Ngam Ha Tong 岩下堂 SK 12-SW-A
Ngam Hau Shek 岩口石 TW 10-NE-A
Ngam Pin 菴邊 N 2-NE-D
Ngam Tau 岩頭 SK 8-SE-C
Ngam Tau Sha 岩頭沙 SK 12-NW-C
Ngam Tau Shan 岩頭山 SK-TP 8-NW-D
Ngan Chau 銀洲 Is 10-SE-A
Ngan Chau Ngam Pai 銀洲岩排 S 15-NW-D
Ngan Chau Tau Pai 銀洲頭排 S 15-NW-D
Ngan Hang Village 銀坑村 S 15-NE-B
Ngan Peng Keng 銀瓶頸 SK 12-SE-C
Ngan Peng Tau 銀瓶頭 SK 12-SE-C
Ngan Wan 銀灣 E 11-SE-D
Ngau Au 牛凹 ST 7-SE-B
Ngau Au 牛坳 TP 3-SE-D
Ngau Au 牛凹 Is 9-SE-A
Ngau Au Shan 牛坳山 ST 7-SE-C
Ngau Chi Wan 牛池灣 WTS 11-NE-A
Ngau Chi Wan Village 牛池灣村 WTS 11-NE-A
Ngau Hom 鰲磡 YL 2-SW-C
Ngau Hom Sha 鰲磡沙 YL 5-NE-B
Ngau Hom Shek 鰲磡石 YL 6-NW-A
Ngau Keng 牛徑 YL 6-NE-D
Ngau Kok Chung 牛角涌 N 4-SW-A
Ngau Kok Wan 牛角灣 K&T 6-SE-D
Ngau Kwo Lo 牛過路 TP 4-SW-D
Ngau Kwo Tin 牛過田 Is 13-NW-B
Ngau Kwu Kok 牛牯角 YL 6-NE-B
Ngau Kwu Leng 牛牯嶺 TP 7-NW-A
Ngau Kwu Long 牛牯塱 Is 10-SW-A
Ngau Kwu Wan 牛牯灣 Is 10-SW-C
Ngau Lan Tsui 牛欄咀 TW 6-SE-C
Ngau Liu 牛寮 TW 6-SE-B
Ngau Liu 牛寮 SK 8-SW-A
Ngau Liu 牛寮 SK 11-NE-B
Ngau Liu Ha 牛寮下 TP 7-NE-B
Ngau Pei Sha 牛皮沙 ST 7-SE-C
Ngau Pei Sha New Village 牛皮沙新村 ST 7-SE-C
Ngau Pui Wo 牛背窩 SK 11-NE-B
Ngau Shi Pui 牛屎缽 SK 8-NE-D
P. 23 / 53
May 2026
English Name Chinese Name District* HP5C
Ngau Shi Shan 牛屎山 N 4-SW-B
Ngau Shi Wu 牛屎湖 N 4-NW-C
Ngau Shi Wu Shan 牛屎湖山 N 4-NW-C
Ngau Shi Wu Wan 牛屎湖灣 N 4-NW-C
Ngau Tam Mei 牛潭尾 YL 2-SE-C
Ngau Tau Kok 牛頭角 N 4-SW-A
Ngau Tau Kok 牛頭角 KT 11-NE-C
Ngau Tau Pai 牛頭排 SK 12-NW-D
Ngau Tau Wan 牛頭灣 Is 10-NW-C
Ngau Tei 牛地 N 2-SE-B
Ngau Wu 牛湖 Is 15-SE-B
Ngau Wu Teng 牛湖頂 Is 15-SE-B
Ngau Wu Tok 牛湖托 ST 7-SW-B
Ngau Wu Tun 牛湖墩 TP 8-NE-C
Ngau Yee Shek Shan 牛耳石山 TP 8-NW-D
Ngau Yue Tau 鰲魚頭 TP 8-NW-C
Ngo Keng Tsui 鵝頸咀 TP 8-NW-B
Ngong Chong 昂裝 Is 15-SE-D
Ngong Chong Shan 昂莊山 TP 4-SW-B
Ngong Ping 昂平 ST 7-SE-B
Ngong Ping 昂坪 Is 9-SE-C
Ngong Ping Tsuen 昂坪村 Is 9-SE-C
Ngong Shuen Au 昂船凹 TW 10-NW-B
Ngong Tong 昂塘 N 3-NW-D
Ngong Tong Shan 昂堂山 ST 7-SW-B
Ngong Wo 昂窩 SK 8-SW-A
Ngor Kai Teng 鵝髻頂 TP 4-SW-C
Ngor Tau Tsui 鵝頭咀 N 4-SW-A
Nim Au 稔凹 ST 7-NE-C
Nim Po Tsuen 稔埔村 Is 10-SW-C
Nim Shue Wan 稔樹灣 Is 10-SW-B
Nim Shue Wan Village 稔樹灣村 Is 10-SW-B
Nim Wan 稔灣 TM 6-SW-C
Nim Yuen 稔園 Is 9-SE-C
Ninepin Group ( Kwo Chau Islands ) 果洲群島 SK 12-SE-C
North Channel ( Tai Chek Mun ) 大赤門 TP 4-SE-A
North Ninepin Island ( Pak Kwo Chau ) 北果洲 SK 12-SE-C
North Point 北角 E 11-SE-A
O Long Village 澳朗村 SK 8-SW-C
O Mun Village 澳門村 SK 12-NW-C
O Pui Tong 澳背塘 N 4-NW-D
O Pui Village 澳貝村 SK 12-NW-C
O Shi Kok 奧士角 N 4-NW-A
O Shi Kok Tsui 奧士角咀 N 4-NW-D
O Tau 澳頭 SK 8-SW-A
O Tsai 澳仔 Is 14-NE-B
Obelisk Hill 石碑山 S 11-SE-D
Ocean Point ( Kwun Tsoi Kok ) 棺材角 TP 4-SE-C
On Li Sai Tsuen 安里西村 KT 11-SE-B
On Lung Tsuen 安龍村 YL 2-SE-A
On Po 安圃 N 3-SW-C
Pa Tau Kwu 扒頭鼓 TW 10-NE-C
P. 24 / 53
May 2026
English Name Chinese Name District* HP5C
Pa Tau Kwu Nam Wan 扒頭鼓南灣 TW 10-NE-C
Pa Tau Kwu Pak Wan 扒頭鼓北灣 TW 10-NE-C
Paak Kap Hang 白鴿坑 Is 15-NW-A
Pai Min Kok Village 排棉角村 TW 6-SE-C
Pai Mun 排門 TP 7-NE-C
Pai Ngak Shan 牌額山 TP 8-NE-C
Pai Tau 排頭 ST 7-SW-B
Pai Tau Hang 排頭坑 ST 7-SW-B
Pai Tau Tun 擺頭墩 SK 8-SW-B
Pak A 北丫 SK 12-NE-A
Pak A Teng 北丫頂 SK 8-SE-C
Pak Chau 白洲 TM 5-SW-D
Pak Fa Lam 百花林 SK 11-NE-B
Pak Fa Tsuen 白花村 YL 6-NW-B
Pak Fu Shan 白虎山 N 3-NW-B
Pak Fu Shan 白虎山 SK 12-NE-A
Pak Fu Tin 白富田 Is 10-SW-C
Pak Hang 北坑 TP 8-NE-C
Pak Hoi Tuk 北海篤 N 4-SW-A
Pak Hok Chau 白鶴洲 YL 2-SE-C
Pak Hok Lam 白鶴林 N 3-NE-C
Pak Hok Shan 白鶴山 N 3-NW-D
Pak Ka Chau 筆架洲 N 4-NW-C
Pak Kan 八間 S 15-NE-C
Pak Kiu Tsai 白橋仔 TP 7-NW-B
Pak Kok 北角 Is 13-SE-A
Pak Kok 北角 Is 15-NW-A
Pak Kok 北角 Is 15-NW-A
Pak Kok 白角 Is 13-NW-D
Pak Kok Chai 白角仔 TP 8-NW-A
Pak Kok Kau Tsuen 北角舊村 Is 14-NE-B
Pak Kok San Tsuen 北角新村 Is 14-NE-B
Pak Kok Shan 白角山 TP 4-SW-B
Pak Kok Shan 北角山 Is 15-NW-A
Pak Kok Tsui 北角咀 Is 14-NW-D
Pak Kok Tsui 北角咀 Is 15-NW-A
Pak Kok Wan 白角灣 N 4-NW-C
Pak Kong 北港 SK 7-SE-D
Pak Kong Au 北港㘭 SK 7-SE-B
Pak Kung Au 伯公坳 N 3-NE-A
Pak Kung Au 伯公坳 SK 11-NE-B
Pak Kung Au 伯公坳 Is 9-SE-D
Pak Kung Tsui 伯公咀 TP 7-NE-B
Pak Lap 白腊 SK 12-NE-A
Pak Lap Tsai 白腊仔 SK 12-NE-A
Pak Lap Wan 白腊灣 SK 12-NE-A
Pak Lap Wan 白鱲灣 TP 17-NW-A
Pak Lau Kok 北流角 Is 16-SW-A
Pak Lau Tsai 北流仔 Is 15-SE-B
Pak Lau Tsai 北流仔 Is 16-SW-A
Pak Lau Wan 北流灣 Is 16-SW-A
Pak Long 北朗 TM 5-SE-A
P. 25 / 53
May 2026
English Name Chinese Name District* HP5C
Pak Ma Tsui 白馬咀 SK 12-NW-A
Pak Ma Tsui Pai 白馬咀排 SK 12-NW-A
Pak Min Kok 北面角 SK 8-SE-C
Pak Mong 白芒 Is 10-SW-A
Pak Nai Shan 白泥山 TW 10-NE-A
Pak Ngan Heung 白銀鄉 Is 10-SW-C
Pak Ngau Shek 白牛石 TP 7-NW-A
Pak Ngau Shek Ha Tsuen 白牛石下村 TP 7-NW-A
Pak Ngau Shek Sheung Tsuen 白牛石上村 TP 7-NW-A
Pak Pai 白排 SK 12-NW-D
Pak Pai Kok 白排角 TP 7-NE-B
Pak Pin Tsuen 北邊村 YL 6-NE-D
Pak Sha Chau 白沙洲 SK 8-SW-C
Pak Sha O 白沙澳 TP 8-NW-B
Pak Sha O Ha Yeung 白沙澳下洋 TP 8-NW-B
Pak Sha Tau 白沙頭 N 4-NW-C
Pak Sha Tau 白沙頭 TP 7-NE-B
Pak Sha Tau Tsui 白沙頭咀 N 4-NW-C
Pak Sha Tau Tsui 白沙頭咀 TP 7-NE-B
Pak Sha Tsuen 白沙村 YL 6-NW-D
Pak Sha Tsui 白沙咀 Is 9-SE-A
Pak Sha Tsui 白沙咀 SK 12-NW-B
Pak Sha Wan 白沙灣 SK 7-SE-D
Pak Sha Wan 白沙灣 E 11-SE-B
Pak Sha Wan 白沙灣 Is 13-SE-A
Pak Sha Wan 白沙灣 S 15-NE-C
Pak She San Tsuen 北社新村 Is 14-NW-D
Pak Shek 白石 ST 7-SW-D
Pak Shek Au 白石凹 N 2-SE-B
Pak Shek Hang 白石坑 TM 6-SW-D
Pak Shek Kiu 白石橋 TW 6-SE-B
Pak Shek Kok 白石角 TP 7-NE-C
Pak Shek Terrace 白石臺 SK 11-NE-B
Pak Shek Wo 白石窩 SK 11-NE-B
Pak Shek Wo San Tsuen 白石窩新村 SK 11-NE-B
Pak Shing Kok 百勝角 SK 12-NW-C
Pak Shui Wun 白水碗 SK 12-NW-A
Pak Tai To Yan 北大刀屻 YL-N-TP 3-SW-C
Pak Tam 北潭 SK 8-NW-D
Pak Tam Au 北潭凹 TP 8-NW-D
Pak Tam Chung 北潭涌 SK 8-SW-B
Pak Tin 白田 ST 7-SW-D
Pak Tin Kong 白田崗 TP 7-NW-A
Pak Tin New Village 白田新村 N 3-NW-D
Pak Tin Pa San Tsuen 白田壩新村 TW 6-SE-D
Pak Tin Pa Tsuen 白田壩村 TW 7-SW-C
Pak Tong 北塘 TP 17-NW-A
Pak Tso Wan 白曹灣 Is 13-SE-C
Pak Tso Wan 白鰽灣 Is 14-NW-D
Pak Wai 北圍 SK 7-SE-D
Pak Wai Tsuen 北圍村 YL 6-NE-A
Pak Wan 北灣 TP 4-SE-C
P. 26 / 53
May 2026
English Name Chinese Name District* HP5C
Pak Wan 北灣 TW 10-NE-A
Pak Wan 北灣 TW 10-NE-A
Pak Wan Teng 北灣頂 TW 10-NE-A
Pan Chung 泮涌 TP 7-NW-B
Pan Chung San Tsuen 泮涌新村 TP 7-NW-B
Pan Long Wan 檳榔灣 SK 12-NW-C
Pan Long Wan 檳榔灣 SK 12-NW-C
Pan Pui Teng 攀背頂 N 3-NE-D
Pang Ka Tsuen 彭家村 YL 6-NE-A
Pang Loon Tei 彭龍地 YL 2-SE-D
Pat Heung 八鄉 YL 6-NE-B
Pat Ka Chau 筆架洲 N 4-SW-A
Pat Sin Leng 八仙嶺 TP 3-SE-D
Pat Tsz Wo Village 拔子窩村 ST 7-SE-A
Peaked Hill ( Kai Yet Kok ) 雞翼角 Is 13-NW-C
Pearl Island 龍珠島 TM 6-SW-C
Pei Tau 陂頭 SK 11-NE-B
Pei Tau Ling Kok 碑頭嶺角 N 3-SW-A
Peng Chau 坪洲 Is 10-SW-B
Penny's Bay ( Chok Ko Wan ) 竹篙灣 TW 10-NW-D
Picnic Bay ( Sok Kwu Wan ) 索罟灣 Is 15-NW-C
Pik Shui Sun Tsuen 碧水新村 SK 11-NE-B
Pik Uk 壁屋 SK 11-NE-B
Pik Uk Au 壁屋凹 SK 11-NE-B
Pik Uk San Tsuen 壁屋新村 SK 11-NE-B
Pillar Point ( Mong Hau Shek ) 望后石 TM 5-SE-D
Pin Chau 扁洲 SK 12-NE-C
Pinehill Village 松嶺 TP 7-NW-B
Ping Chau 平洲 TP 17-NW-C
Ping Chau Hoi 平洲海 TP 17-NW-A
Ping Che 坪輋 N 3-NW-D
Ping Che New Village 坪輋新村 N 3-NW-D
Ping Fung Shan 屏風山 TP-N 3-SE-A
Ping Hang 坪坑 YL 2-NE-D
Ping Kong 丙崗 N 3-SW-C
Ping Long 坪朗 TP 7-NW-A
Ping Min Chau 平面洲 SK 12-NW-D
Ping Pai 平排 SK 12-NE-A
Ping Shan 屏山 YL 6-NW-B
Ping Shan Chai 平山仔 TP 3-SE-C
Ping Shan San Tsuen 屏山新村 YL 6-NW-B
Ping Teng Au 平頂坳 N 3-SE-A
Ping Tok Hang Shan 平托坑山 SK 12-NW-D
Ping Tun 坪墩 SK 8-SW-B
Ping Yeung 坪洋 N 3-NW-D
Ping Yuen River 平原河 N 3-NW-D
Piper's Hill 琵琶山 ST-SSP 11-NW-B
Plover Cove ( Shuen Wan Hoi ) 船灣海 TP 7-NE-A
Po Chong Wan 布廠灣 S 15-NW-B
Po Chue Tam 寶珠潭 Is 9-SW-D
Po Kat Tsai 布吉仔 N 3-SW-B
Po Keng Teng 寶鏡頂 SK 12-SW-D
P. 27 / 53
May 2026
English Name Chinese Name District* HP5C
Po Kwu Wan 曝罟灣 SK 8-SW-B
Po Lam 寶林 SK 11-NE-D
Po Leng 蒲嶺 N 3-SW-C
Po Leng Au 蒲嶺坳 N 3-SW-A
Po Lo Che 菠蘿輋 SK 8-SW-C
Po Lo Shan 波羅山 YL 6-NE-D
Po Lo Tsui 波羅咀 Is 14-NE-D
Po Min 坡面 TP 7-NE-C
Po Pin Chau 破邊洲 SK 8-SE-D
Po Sam Pai 布心排 TP 3-SE-C
Po Sheung Tsuen 莆上村 N 3-SW-A
Po Toi 蒲台 Is 16-SW-A
Po Toi Islands 蒲台群島 Is 16-SW-A
Po Toi O 布袋澳 SK 12-SW-A
Po Toi O 布袋澳 SK 12-SW-A
Po Tong Ha 寶塘下 TM 6-NW-C
Po Wah Yuen 寶華園 Is 14-NE-B
Po Yue Pai 蒲魚排 N 4-NW-A
Po Yue Pai 蒲魚排 SK 12-NE-C
Po Yue Wan 鯆魚灣 Is 14-NW-D
Pok Fu Lam 薄扶林 S 11-SW-C
Pok Fu Lam Village 薄扶林村 S 11-SW-C
Pok Tau Ha 膊頭下 N 3-NE-C
Pok To Yan 萡刀屻 Is 9-SE-B
Pok Wai 壆圍 YL 2-SE-C
Por Kai Shan 婆髻山 Is 9-SE-B
Por Lo Shan 菠蘿山 TM 5-SE-B
Port Island ( Chek Chau ) 赤洲 TP 4-SE-A
Port Shelter ( Ngau Mei Hoi ) 牛尾海 SK 12-NW-A
Pottinger Gap ( Ma Tong Au ) 馬塘坳 S-E 11-SE-D
Pottinger Peak 砵甸乍山 E 11-SE-D
Princess Hill 公主山 N 3-NW-D
Pui O Au 貝澳坳 Is 10-SW-C
Pui O Lo Wai Tsuen 貝澳老圍村 Is 14-NW-A
Pui O Lo Wai Tsuen Pui O Au 貝澳老圍村貝澳坳 Is 10-SW-C
Pui O San Wai Tsuen 貝澳新圍村 Is 14-NW-A
Pui O Wan 貝澳灣 Is 14-NW-A
Pun Chun Yuen 半春園 TP 7-NW-B
Pun Shan Chau 半山洲 TP 7-NW-D
Pun Shan Shek 半山石 TW 10-NE-C
Pun Shan Tsuen 半山村 TW 6-SE-D
Pun Uk Tsuen 潘屋村 YL 2-SE-B
Pyramid Hill ( Tai Kam Chung ) 大金鐘 TP-ST 7-SE-B
Pyramid Rock 尖柱石 SK 12-NE-C
Quarry Bay 鰂魚涌 E 11-SE-A
Quarry Gap ( Tai Fung Au ) 大風坳 E 11-SE-C
Queen's Hill 皇后山 N 3-SW-B
Rambler Channel 藍巴勒海峽 K&T 10-NE-B
Razor Hill ( Che Kwu Shan ) 鷓鴣山 SK 11-NE-B
Red Hill ( Pak Pat Shan ) 白筆山 S 15-NE-B
Red Incense Burner Summit ( Hung Heung Lo Fung ) 紅香爐峰 E 11-SE-A
Repulse Bay 淺水灣 S 15-NE-A
P. 28 / 53
May 2026
English Name Chinese Name District* HP5C
Repulse Bay 淺水灣 S 15-NE-A
River Silver 銀河 Is 10-SW-C
Robin's Nest ( Hung Fa Leng ) 紅花嶺 N 3-NE-C
Rocky Harbour ( Leung Shuen Wan Hoi ) 糧船灣海 SK 12-NE-A
Round Island ( Ngan Chau ) 銀洲 S 15-NW-D
Round Island ( Pak Sha Chau ) 白沙洲 N 4-NW-D
Round Table First Village 圓桌第一村 Is 14-NW-D
Round Table Second Village 圓桌第二村 Is 14-NW-D
Round Table Third Village 圓桌第三村 Is 14-NW-D
Round Table Village 圓桌村 Is 10-SW-C
Saddle Pass ( Ki Lun Shan Au ) 麒麟山坳 YL-N 2-SE-B
Sai Ap Chau 細鴨洲 N 3-NE-B
Sai Chau Mei 細洲尾 SK 12-SE-C
Sai Keng 西徑 TP 8-NW-C
Sai Kung 西貢 SK 8-SW-C
Sai Kung Tuk 西貢篤 SK 8-SW-C
Sai Lau Kok Tsuen 西樓角村 TW 7-SW-C
Sai Lau Kong 西流江 N 4-NW-C
Sai O 西澳 N 4-NW-A
Sai O 西澳 TP 7-NE-D
Sai Pai 細排 Is 16-SW-A
Sai Pin Wai 西邊圍 YL 6-NW-B
Sai Shan 細山 K&T 10-NE-B
Sai Tso Wan 茜草灣 Is 9-SW-D
Sai Tso Wan 西草灣 K&T 10-NE-B
Sai Tso Wan 晒草灣 KT 11-NE-D
Sai Wan 西灣 SK 8-SE-A
Sai Wan 西灣 SK 8-SE-B
Sai Wan 西環 C&W 11-SW-A
Sai Wan 西灣 Is 13-SE-A
Sai Wan 西灣 Is 14-NW-D
Sai Wan CARE Village 西灣美經援村 Is 14-NW-D
Sai Wan Ho 西灣河 E 11-SE-A
Sai Wan Shan 西灣山 SK 8-SE-B
Sai Ying Pun 西營盤 C&W 11-SW-A
Sam A Chung 三椏涌 N 4-SW-A
Sam A Tsuen 三椏村 N 4-NW-C
Sam A Wan 三椏灣 N 4-SW-A
Sam Chau Mun 三洲門 SK 12-NE-C
Sam Chuen 三轉 TW 10-NE-A
Sam Dip Tam 三疊潭 TW 7-SW-C
Sam Fai Tin 三塊田 SK 7-SE-D
Sam Ka Tsuen 三家村 N 3-SE-B
Sam Ka Tsuen 三家村 KT 11-SE-B
Sam Kok Tsui 三角咀 N 3-NE-B
Sam Long 心朗 SK 11-NE-B
Sam Mun Shan 三門山 TP 4-SW-C
Sam Mun Tsai New Village 三門仔新村 TP 7-NE-A
Sam Mun Tun 三門墩 TP 7-NE-B
Sam Nga Hau 三丫口 SK 8-SW-D
Sam Pai 三排 SK 12-NE-D
Sam Pak 三白 Is 10-NW-D
P. 29 / 53
May 2026
English Name Chinese Name District* HP5C
Sam Pak Au 三白坳 TW 10-NW-D
Sam Pak Wan 三白灣 Is 10-NW-D
Sam Po Shek 三抱石 TP 4-SE-C
Sam Po Shue 三寶樹 YL 2-SE-A
Sam Pui Chau 三杯酒 TP 8-NW-C
Sam Shing Hui 三聖墟 TM 6-SW-C
Sam Sing Wan 三星灣 SK 8-SW-C
Sam Tam Lo 三担籮 N 3-SE-B
Sam To 三肚 N 3-NE-D
Sam Tung Uk Resite Village 三棟屋村 TW 7-SW-C
San Chau 新洲 Is 9-SW-D
San Hei Tsuen 新起村 YL 6-NW-B
San Hing Tsuen 新慶村 YL 2-SW-C
San Hing Tsuen 新慶村 TM 6-NW-C
San Hui Village 新墟村 TM 6-SW-A
San Kau Po Kok 新舅埔角 N 3-NE-B
San Keng 新徑 Is 9-SE-C
San Kwai Tin 新桂田 N 3-NE-A
San Lee Uk Tsuen 新李屋村 YL 6-NW-A
San Lung Tsuen 新龍村 YL 2-SE-A
San Lung Wai 新隆圍 YL 6-NE-B
San Pai 散排 Is 16-SW-C
San Po Kong 新蒲崗 WTS 11-NE-A
San Po Tsui 新舖咀 TW 10-NE-A
San Sang San Tsuen 新生新村 YL 6-NW-A
San Sang Tsuen 新生村 YL 6-NW-A
San Shek Wan 䃟石灣 Is 9-SW-B
San Shek Wan 䃟石灣 Is 13-NE-B
San Sin Tseng Shan 神仙井山 SK 12-NW-B
San Sin Tso 神仙灶 N 4-NW-D
San Tau 䃟頭 Is 9-SE-A
San Tau Kok 䃟頭角 TP 3-SE-C
San Tau Yiu 䃟頭窰 TP 3-SE-B
San Tin 新田 YL 2-SE-A
San Tin Hang 䃟田坑 SK 8-SW-A
San Tin Village 新田村 ST 7-SW-D
San Tong 新塘 TP 7-NW-A
San Tong Po 新塘莆 N 3-SW-B
San Tsuen 新村 N 3-NE-C
San Tsuen 新村 TW 7-SW-C
San Tsuen 新村 Is 9-SW-D
San Uk 新屋 SK 8-SW-A
San Uk Ha 新屋下 N 3-SE-B
San Uk Ka 新屋家 TP 7-NW-D
San Uk Ling 新屋嶺 N 3-NW-C
San Uk Resite Village 新屋村 K&T 10-NE-B
San Uk Tsai 新屋仔 N 3-SW-B
San Uk Tsai 新屋仔 TP 7-NW-A
San Uk Tsuen 新屋村 N 3-SE-B
San Uk Tsuen 新屋村 N 3-SW-A
San Uk Tsuen 新屋村 YL 6-NW-A
San Wai 新圍 N 3-SW-A
P. 30 / 53
May 2026
English Name Chinese Name District* HP5C
San Wai 新圍 YL 6-NW-A
San Wai Tsai 新圍仔 TP 7-NW-B
San Wai Tsai 新圍仔 TM 6-SW-A
San Wai Tsuen 新圍村 YL 2-SE-C
Sandy Bay 沙灣 S 11-SW-C
Sandy Ridge 沙嶺 N 3-NW-C
Sau Mau Ping 秀茂坪 KT 11-NE-D
Scenic Hill 觀景山 Is 9-SE-B
See Chau Mun 匙洲門 SK 12-NW-B
Self Help CARE Village 自助美經援村 Is 14-NW-D
Sha Chau 沙洲 TM 9-NW-B
Sha Chau Lei 沙洲里 YL 6-NW-A
Sha Ha 沙下 SK 8-SW-A
Sha Kiu Tau 沙橋頭 SK 12-NE-A
Sha Kiu Tsuen 沙橋村 YL 2-SW-C
Sha Kok Mei 沙角尾 SK 8-SW-A
Sha Kong Tsuen 沙江村 YL 6-NW-A
Sha Kong Wai 沙江圍 YL 2-SW-C
Sha Kong Wai Tsai 沙江圍仔 YL 6-NW-A
Sha Lan 沙欄 TP 7-NE-A
Sha Ling 沙嶺 N 3-NW-C
Sha Lo Tung 沙羅洞 TP 3-SW-D
Sha Lo Wan 沙螺灣 Is 9-SE-A
Sha Lo Wan 沙螺灣 Is 9-SE-A
Sha Lo Wan Chung Hau 沙螺灣涌口 Is 9-SE-A
Sha Lo Wan San Tsuen 沙螺灣新村 Is 9-SE-A
Sha Lo Wan Tsuen 沙螺灣村 Is 9-SE-A
Sha Ngam Tau 沙岩頭 N 4-NW-D
Sha Pa 沙壩 TP 7-NW-A
Sha Pai 沙排 N 4-NW-C
Sha Po Kong 沙埔崗 TM 5-SE-A
Sha Po New Village 沙埔新村 Is 14-NE-B
Sha Po Old Village 沙埔舊村 Is 14-NE-B
Sha Po Tsuen 沙埔村 YL 6-NE-A
Sha Shek Tan 沙石灘 S 15-NE-C
Sha Tau 沙頭 TP 8-NE-C
Sha Tau 沙頭 TP 17-NW-C
Sha Tau Kok 沙頭角 N 3-NE-A
Sha Tau Kok River 沙頭角河 N 3-NE-A
Sha Tin 沙田 ST 7-SE-C
Sha Tin Heights 沙田嶺 ST 7-SW-D
Sha Tin Hoi 沙田海 ST 7-NE-C
Sha Tin Pass 沙田坳 ST 7-SE-C
Sha Tin Tau 沙田頭 ST 7-SW-D
Sha Tin Tau New Village 沙田頭新村 ST 7-SE-C
Sha Tin Wai 沙田圍 ST 7-SE-C
Sha Tin Wai New Village 沙田圍新村 ST 7-SE-C
Sha Tong Hau Mei 沙塘口尾 SK 12-NE-C
Sha Tseng Tsuen 沙井村 YL 6-NW-B
Sha Tsui 沙咀 SK 8-SW-C
Sha Tsui 沙咀 Is 13-NE-B
Sha Tsui New Village 沙咀新村 SK 8-SW-C
P. 31 / 53
May 2026
English Name Chinese Name District* HP5C
Sha Tsui Tau 沙咀頭 Is 9-SE-A
Sham Chung 深涌 N 4-NW-C
Sham Chung 深涌 TP 8-NW-A
Sham Chung Kok 深涌角 TP 8-NW-A
Sham Chung Tsuen 深涌村 YL 6-NW-D
Sham Chung Wan 深涌灣 TP 8-NW-A
Sham Hang Lek 深坑瀝 Is 13-NW-C
Sham Shek Tsuen 深石村 Is 9-SW-B
Sham Shui Kok 深水角 TW 10-NW-C
Sham Shui Kok 深水角 S 15-NW-B
Sham Shui Pai 深水排 Is 14-NE-C
Sham Shui Po 深水埗 SSP 11-NW-B
Sham Tseng 深井 TW 6-SE-C
Sham Tseng Commercial New Village 深井商業新村 TW 6-SE-C
Sham Tseng East Village 深井東村 TW 6-SE-C
Sham Tseng Kau Tsuen 深井舊村 TW 6-SE-C
Sham Tseng San Tsuen 深井新村 TW 6-SE-C
Sham Tseng Village 深井村 TW 6-SE-C
Sham Tseng West Village 深井西村 TW 6-SE-C
Sham Tuk 深篤 SK 8-SE-C
Sham Tuk Mun 深篤門 SK 8-SW-D
Sham Wan 深灣 SK 8-NE-B
Sham Wan 深灣 S 15-NW-B
Sham Wan 深灣 Is 15-SW-A
Sham Wat 深屈 Is 9-SW-D
Sham Wat Wan 深屈灣 Is 9-SW-D
Shan Ha ( Pa Mei ) 山下 ( 壩尾 ) Is 9-SE-B
Shan Ha Tsuen 山下村 ( 山廈村 ) YL 6-NW-D
Shan Ha Wai ( Tsang Tai Uk ) 山下圍 ( 曾大屋 ) ST 7-SE-C
Shan Leng Kok 山嶺角 SK 12-SE-C
Shan Liu 山寮 TP 3-SE-C
Shan Liu 山寮 SK 8-SW-A
Shan Liu 山寮 Is 15-SE-B
Shan Mei 山尾 ST 7-SW-B
Shan Mei Au 山尾坳 N 4-NW-C
Shan O 山塢 N 3-NE-D
Shan Pin Tsuen 山邊村 YL 6-NW-B
Shan Pui 山貝 YL 6-NW-B
Shan Pui Chung Hau Tsuen 山貝涌口村 YL 6-NW-B
Shan Pui Hung Tin Tsuen 山貝洪田村 YL 6-NW-B
Shan Pui River 山貝河 YL 6-NW-B
Shan Ting Tsuen 山頂村 Is 10-SW-B
Shan Tong 山塘 N 3-NW-D
Shan Tong New Village 山塘新村 TP 7-NW-B
Shan Tsui 山咀 N 3-NE-A
Shap Long Chung Hau 十塱涌口 Is 14-NW-A
Shap Long Kau Tsuen 十塱舊村 Is 14-NW-A
Shap Long San Tsuen 十塱新村 Is 14-NW-A
Shap Pat Heung 十八鄉 YL 6-NW-D
Shap Sze Heung 十四鄉 TP 8-NW-C
Shap Yi Wat 十二笏 ST 7-SE-C
Sharp Island ( Kiu Tsui Chau ) 橋咀洲 SK 8-SW-C
P. 32 / 53
May 2026
English Name Chinese Name District* HP5C
Sharp Peak ( Nam She Tsim ) 蚺蛇尖 SK 8-NE-D
Shau Kei Pai 筲箕排 N 4-NW-C
Shau Kei Wan 筲箕灣 E 11-SE-B
She Shan River 社山河 TP 7-NW-A
She Shan Tsuen 社山村 TP 7-NW-A
She Shek Au 蛇石坳 TP 8-NW-B
She Tau 蛇頭 SK 8-SW-B
She Tei Hang 蛇地坑 SK 8-SW-B
She Wan Kok 蛇灣角 SK 12-NE-A
She Wan Shan 蛇灣山 SK 12-NE-A
Shek Au Shan 石坳山 N-TP 3-SW-B
Shek Chau 石洲 SK 8-SW-D
Shek Chau 石洲 Is 13-SE-A
Shek Chung Au 石涌凹 N 3-NE-C
Shek Chung Kok 石涌角 SK 16-NW-A
Shek Hang 石坑 SK 8-SW-A
Shek Kip Mei 石硤尾 SSP 11-NW-B
Shek Kiu Tau 石橋頭 N 3-NE-C
Shek Kok Tsui 石角咀 TM 5-SE-D
Shek Kok Tsui 石角咀 Is 14-NE-B
Shek Kong 石崗 YL 6-NE-D
Shek Kong San Tsuen 石崗新村 YL 6-NE-A
Shek Kwu Chau 石鼓洲 Is 14-NW-C
Shek Kwu Lung 石古壟 TP 7-NW-B
Shek Kwu Lung 石古壟 ST 7-SE-A
Shek Kwu Wan 石鼓環 SK 8-SW-C
Shek Lam Chau 石欖洲 Is 13-NE-C
Shek Lau Po 石榴埔 Is 9-SE-C
Shek Lau Tung 石榴洞 ST 7-SW-B
Shek Lei Tau 石梨頭 K&T 11-NW-A
Shek Li 石梨 Is 14-NE-B
Shek Li Ka Nam 石梨嘉南 Is 14-NE-B
Shek Lung Kung 石龍拱 TW 6-SE-D
Shek Lung Tsai 石壟仔 ST 7-SE-B
Shek Lung Tsai New Village 石壟仔新村 SK 7-SE-B
Shek Ma 石馬 N 2-NE-D
Shek Mei Tau 石尾頭 SK 12-SW-B
Shek Mun 石門 ST 7-SE-A
Shek Mun Kap 石門甲 Is 9-SE-C
Shek Mun Shan 石門山 Is 13-NE-C
Shek Nga Pui 石芽背 ST-SK 7-SE-D
Shek Nga Shan 石芽山 ST 7-SE-D
Shek Nga Tau 石芽頭 N 3-NE-D
Shek Nga Tau 石芽頭 TP-N 4-SW-A
Shek Nga Tau 石芽頭 TP 8-NW-A
Shek Ngau Chau 石牛洲 TP 17-SW-C
Shek O 石澳 N 3-NW-D
Shek O 石澳 S 15-NE-B
Shek O Headland 石澳山仔 S 15-NE-B
Shek O Peak 打爛埕頂山 S 15-NE-B
Shek O Village 石澳村 S 15-NE-B
Shek O Wan 石澳灣 S 15-NE-B
P. 33 / 53
May 2026
English Name Chinese Name District* HP5C
Shek Pai Wan 石排灣 S 15-NW-B
Shek Pai Wan 石排灣 Is 15-NW-C
Shek Pan Tam 石板潭 N 3-SE-A
Shek Pik 石壁 Is 13-NW-B
Shek Pik Au 石壁凹 Is 9-SE-C
Shek Pik San Tsuen 石碧新村 TW 7-SW-C
Shek Po Tsuen 石埗村 YL 6-NW-A
Shek Pok Wai 石壆圍 SK 11-NE-B
Shek Shan 石山 YL 2-SW-B
Shek Sheung River 石上河 N 3-SW-A
Shek Shui Kan 石水澗 N 3-SE-B
Shek Sze Shan 石獅山 Is 9-SE-D
Shek Tau Wai 石頭圍 YL 6-NE-D
Shek Tong Tsuen 石塘村 YL 6-NE-A
Shek Tong Tsui 石塘咀 C&W 11-SW-A
Shek Tsai Ha 石寨下 N 3-NW-D
Shek Tsai Leng 石仔嶺 N 2-SE-B
Shek Tsai Po 石仔埗 Is 9-SW-C
Shek Tsai Wan 石仔灣 N 4-SW-B
Shek Tsai Wan 石仔灣 TW 10-NE-A
Shek Tsai Wan 石仔灣 SK 12-NE-A
Shek Uk Shan 石屋山 TP 8-NW-D
Shek Wai Kok New Village 石圍角新村 TW 7-SW-C
Shek Wan 石環 K&T 10-NE-B
Shek Wu Hui 石湖墟 N 3-SW-A
Shek Wu San Tsuen 石湖新村 N 3-SW-A
Shek Wu Tong 石湖塘 YL 6-NE-C
Shek Wu Wai 石湖圍 YL 2-SE-A
Shek Wu Wai San Tsuen 石湖圍新村 YL 2-SE-C
Shelter Island ( Ngau Mei Chau ) 牛尾洲 SK 12-NW-C
Shenzhen River 深圳河 N-YL 3-NW-C
Sheung Che 上輋 YL 6-NE-B
Sheung Cheung Wai 上章圍 YL 6-NW-A
Sheung Chuk Yuen 上竹園 YL 2-SE-C
Sheung Fa Shan 上花山 TW 6-SE-B
Sheung Keng Hau 上徑口 ST 7-SW-D
Sheung Ko Tan 上高灘 K&T 10-NE-B
Sheung Kok 上角 Is 13-SE-A
Sheung Kok Tsui 上角咀 N 3-NE-D
Sheung Kwai Chung 上葵涌 TW 7-SW-C
Sheung Kwai Chung Village 上葵涌村 TW 7-SW-C
Sheung Ling Pei 上嶺皮 Is 9-SE-B
Sheung Ma Shek 上馬石 TP 7-NW-A
Sheung Miu Tin 上苗田 N 3-SE-B
Sheung Pai 雙排 N 4-NW-A
Sheung Pak Nai 上白泥 YL 5-NE-B
Sheung Pak Tsuen 上北村 N 3-SW-A
Sheung Shan Kai Wat 上山雞乙 N 3-NW-C
Sheung Shui 上水 N 3-SW-A
Sheung Shui Heung 上水鄉 N 3-SW-A
Sheung Shui Wa Shan 上水華山 N 3-SW-A
Sheung Sze Mun 雙四門 S 15-NE-D
P. 34 / 53
May 2026
English Name Chinese Name District* HP5C
Sheung Sze Mun 雙四門 S 15-NE-D
Sheung Sze Wan 相思灣 SK 12-NW-C
Sheung Tam Shui Hang 上担水坑 N 3-NE-A
Sheung Tin Liu Ha 上田寮下 TP 7-NW-A
Sheung Tong 上塘 TW 6-SE-B
Sheung Tsat Muk Kiu 上七木橋 N 3-SE-A
Sheung Tsuen 上村 YL 6-NE-D
Sheung Tsuen 上村 Is 13-SE-C
Sheung Tsuen San Tsuen 上村新村 YL 6-NE-D
Sheung Tung Au 雙東坳 Is 9-SE-D
Sheung Wai 上圍 TP 4-SE-C
Sheung Wan 上環 C&W 11-SW-A
Sheung Wo Che 上禾輋 ST 7-SE-A
Sheung Wo Hang 上禾坑 N 3-NE-C
Sheung Wong Yi Au 上黃宜坳 TP 7-NW-B
Sheung Wun Yiu 上碗窰 TP 7-NW-D
Sheung Yat Tsuen 上一村 K&T 7-SW-C
Sheung Yau Tin Tsuen 上攸田村 YL 6-NW-B
Sheung Yeung 上洋 SK 12-NW-C
Sheung Yeung Shan 上洋山 SK 12-NW-C
Sheung Yiu 上窰 SK 8-SW-B
Sheung Yue River 雙魚河 N 2-SE-B
Shing Mun River Channel 城門河道 ST 7-SE-A
Shing Uk Tsuen 盛屋村 YL 6-NW-B
Shouson Hill 壽臣山 S 11-SW-D
Shouson Hill 壽臣山 S 11-SW-D
Shu On Terrace 舒安台 TW 6-SE-C
Shue Long Chau 薯莨洲 SK 12-SE-C
Shuen Wan 船灣 TP 3-SE-C
Shuen Wan Chan Uk 船灣陳屋 TP 7-NE-A
Shuen Wan Chim Uk 船灣詹屋 TP 3-SE-C
Shuen Wan Lei Uk 船灣李屋 TP 7-NE-A
Shui Bin Village 水邊村 SK 12-NW-C
Shui Cham Tsui 水站咀 N 3-NE-B
Shui Cham Tsui Pai 水浸咀排 N 3-NE-D
Shui Chuen O 水泉澳 ST 7-SE-C
Shui Hang 水坑 Is 14-NW-D
Shui Hau 水口 N 3-NW-D
Shui Hau 水口 SK 11-NE-B
Shui Hau 水口 Is 13-NE-A
Shui Hau Wan 水口灣 Is 13-NE-A
Shui Kan Shek 水澗石 YL 6-NE-B
Shui Keng Teng 水徑頂 SK 8-SE-C
Shui Lau Hang 水流坑 N 3-NW-D
Shui Lau Tin 水流田 YL 6-NE-D
Shui Lo Cho 水澇漕 Is 13-NW-A
Shui Long Wo 水浪窩 TP 8-SW-A
Shui Mei 水尾 YL 6-NE-A
Shui Mong Tin 水茫田 TP 7-NE-A
Shui Ngau Tso 水牛槽 N 3-NW-D
Shui Pai 水排 Is 14-NW-C
Shui Pin Tsuen 水邊村 YL 6-NW-B
P. 35 / 53
May 2026
English Name Chinese Name District* HP5C
Shui Pin Wai 水邊圍 YL 6-NW-B
Shui Tau 水頭 YL 6-NE-A
Shui Tin Tsuen 水田村 YL 6-NW-B
Shui Tsan Tin 水盞田 YL 6-NE-D
Shui Tseng Wan 水井灣 Is 10-SW-C
Shui Tsiu Lo Wai 水蕉老圍 YL 6-NW-D
Shui Tsiu San Tsuen 水蕉新村 YL 6-NW-D
Shui Wo 水窩 TP 7-NW-A
Shun Yeung Fung 純陽峰 TP 3-SE-C
Shung Ching San Tsuen 崇正新村 YL 6-NW-D
Shung Him Tong 崇謙堂 N 3-SW-A
Sik Kong Tsuen 錫降村 YL 6-NW-A
Sik Kong Wai 錫降圍 YL 6-NW-A
Silver Mine Bay 銀鑛灣 Is 10-SW-C
Silverstrand 銀線灣 SK 12-NW-C
Sin Kung Tung 仙公洞 Is 14-NW-D
Sin Yan Tseng 仙人井 Is 14-NW-D
Sing Ping Village 昇平村 N 3-NW-D
Siu A Chau 小鴉洲 Is 13-SE-A
Siu A Chau Shan 小鴉洲山 Is 13-SE-A
Siu A Chau Tsuen 小鴉洲村 Is 13-SE-A
Siu A Chau Wan 小鴉洲灣 Is 13-SE-A
Siu Chik Sha 小赤沙 SK 12-SW-A
Siu Hang Hau 小坑口 SK 12-NW-C
Siu Hang San Tsuen 小坑新村 N 3-SW-A
Siu Hang Tsuen 小坑村 N 3-SW-A
Siu Hang Tsuen 小坑村 TM 6-NW-C
Siu Ho 小蠔 Is 10-NW-C
Siu Ho Wan 小蠔灣 Is 10-NW-C
Siu Hum Tsuen 小磡村 YL 2-SE-C
Siu Kau 小滘 TP 4-SW-C
Siu Kau Yi Chau 小交椅洲 Is 10-SE-A
Siu Kwai Wan 小貴灣 Is 14-NW-D
Siu Lam 小欖 TM 6-SW-D
Siu Lam 小欖 TM 6-SW-D
Siu Lam San Tsuen 小欖新村 TM 6-SW-C
Siu Lang Shui 小冷水 TM 5-SE-D
Siu Lek Yuen 小瀝源 ST 7-SE-C
Siu Ma Shan 小馬山 WC-E 11-SE-C
Siu Nim Chau 小稔洲 N 4-NW-C
Siu Om Shan 小菴山 TP 7-NW-A
Siu Sai Wan 小西灣 E 11-SE-D
Siu Sau 小秀 TM 6-SW-C
Siu Sau Village 小秀村 TM 6-SW-C
Siu Tan 小灘 N 4-NW-C
Siu To Yuen Village 小桃源村 SK 11-NE-B
Siu Tsan Chau 小鏟洲 SK 8-SW-C
Small Traders New Village 小商新村 YL 6-NW-B
Smugglers' Pass 走私坳 ST 7-SW-C
Smugglers' Ridge ( Ma Tsz Keng ) 孖指徑 ST 7-SW-C
So Kon Po 掃桿埔 WC 11-SW-D
So Kwun Po 掃管埔 N 3-SW-A
P. 36 / 53
May 2026
English Name Chinese Name District* HP5C
So Kwun Tan 掃管灘 TM 6-SW-C
So Kwun Wat 掃管笏 TM 6-SW-C
So Kwun Wat San Tsuen 掃管笏新村 TM 6-SW-D
So Kwun Wat Tsuen 掃管笏村 TM 6-SW-C
So Lo Pun 鎖羅盆 N 3-NE-D
So Shi Tau 鎖匙頭 SK 12-SW-A
So Uk 蘇屋 SSP 11-NW-B
Sok Kwu Wan 索罟灣 Is 15-NW-C
Soko Islands 索罟群島 Is 13-SE-A
Sor Lung Ngam 鎖龍岩 SK 12-SE-C
Sor See Kau 鎖匙扣 SK 12-NE-A
Sor See Mun 鎖匙門 SK 12-NE-A
South Bay 南灣 S 15-NE-A
South Channel ( Tap Mun Hau ) 塔門口 TP 8-NE-A
South Ninepin Island ( Nam Kwo Chau ) 南果洲 SK 12-SE-C
St. Paul's Village 聖保祿村 K&T 6-SE-D
St. Peter's Village 伯多祿村 SK 8-SW-C
Stag Hill 貓嶺 N 3-NW-D
Stanley ( Chek Chue ) 赤柱 S 15-NE-C
Stanley Bay 赤柱灣 S 15-NE-C
Stanley Peninsula 赤柱半島 S 15-NE-C
Starling Inlet ( Sha Tau Kok Hoi ) 沙頭角海 N 3-NE-C
Steep Island ( Ching Chau ) 青洲 SK 12-SW-B
Stone Hill ( Ma Hang Shan ) 馬坑山 S 15-NE-A
Stonecutters Island ( Ngong Shuen Chau ) 昂船洲 SSP 11-NW-C
Sulphur Channel 硫磺海峽 C&W 11-SW-A
Sum Wan 深灣 Is 13-SE-A
Sun Fung Wai 順風圍 TM 6-NW-C
Sun Hoi Tin 新開田 TW 6-SE-B
Sun King Terrace 新景台 SK 8-SW-C
Sun Lung Wai 新龍圍 Is 10-SW-C
Sun Tei Village 新地村 SK 11-NE-B
Sung Kong 宋崗 Is 16-SW-A
Sung Shan New Village 崇山新村 YL 6-NW-D
Sunny Bay 欣澳 TW 10-NW-B
Sunset Peak ( Tai Tung Shan ) 大東山 Is 9-SE-D
Sunshine Island ( Chau Kung To ) 周公島 Is 10-SE-C
Sze Fong Shan 四方山 TP 7-NW-C
Sze Pak 四白 Is 10-NW-D
Sze Pak Au 四白坳 TW 10-NW-D
Sze Pak Tsui 四白咀 Is 10-NW-D
Sze Pak Wan 四白灣 Is 10-NW-D
Sze Shan 獅山 Is 9-SW-D
Sze Tau Leng 獅頭嶺 N 3-SW-B
Sze Tei 獅地 TP 8-NE-A
Sze To 四肚 N 3-NE-D
Sze Tsz Tau Shan 獅子頭山 Is 9-SE-C
Ta Ho Pai 打蠔排 N 4-SW-B
Ta Ho Tun Ha Wai 打蠔墩下圍 SK 8-SW-C
Ta Ho Tun Sheung Wai 打蠔墩上圍 SK 8-SW-C
Ta Ku Ling 打鼓嶺 SK 11-NE-B
Ta Ku Ling San Tsuen 打鼓嶺新村 SK 11-NE-B
P. 37 / 53
May 2026
English Name Chinese Name District* HP5C
Ta Kwu Ling 打鼓嶺 N 3-NW-D
Ta Kwu Ling Village 打鼓嶺村 N 3-NW-C
Ta Pang Po 打棚埔 TW 10-NW-D
Ta Sha Lok 大沙落 N 2-NE-D
Ta Shek Wu 打石湖 YL 6-NE-B
Ta Shek Wu Shek Tong 打石湖石塘 YL 6-NE-B
Ta Shui Wan 打水灣 Is 15-NW-C
Ta Tit Yan 打鐵屻 TP 7-NW-D
Tai A Chau 大鴉洲 Is 13-SE-C
Tai Au Mun 大坳門 SK 12-SW-A
Tai Cham Koi 大枕蓋 SK 8-SE-A
Tai Chau 大洲 SK 8-SE-B
Tai Chau 大洲 SK 8-SW-A
Tai Chau 大洲 SK 12-SE-C
Tai Chau Mei 大洲尾 SK 12-SE-C
Tai Chau Mei Teng 大洲尾頂 Is 13-SE-C
Tai Chau To 大洲渡 N 4-SW-B
Tai Che 大輋 ST 7-SE-C
Tai Che Leng Tun 大輋嶺墩 SK-TP 8-NW-D
Tai Che Tei 大輋地 TP 7-NW-A
Tai Che Tung 大輋峒 Is 10-NW-D
Tai Chik Sha 大赤沙 SK 12-SW-A
Tai Ching Cheung 大蒸場 K&T 11-NW-A
Tai Chung Hau 大涌口 SK 7-SE-D
Tai Fung Au 大風坳 Is 9-SW-D
Tai Hang 泰亨 TP 3-SW-C
Tai Hang 大坑 WC 11-SE-A
Tai Hang Chung Sum Wai 泰亨中心圍 TP 3-SW-C
Tai Hang Fui Sha Wai 泰亨灰沙圍 TP 3-SW-C
Tai Hang Hau 大坑口 SK 12-NW-C
Tai Hang Mei 大坑尾 Is 15-SE-B
Tai Hang Tsz Tong Tsuen 泰亨祠堂村 TP 3-SW-C
Tai Hang Tuk 大坑篤 SK 12-NW-D
Tai Hang Tun 大坑墩 SK 12-SW-B
Tai Ho 大蠔 Is 10-SW-A
Tai Ho San Tsuen 大蠔新村 Is 10-SW-A
Tai Ho Wan 大蠔灣 Is 10-SW-A
Tai Hom 大磡 TP 8-NW-D
Tai Hom Sham 大磡森 Is 13-NW-A
Tai Hom Tuk 大砍篤 N 3-NW-D
Tai Hong Tsuen 泰康村 YL 6-NE-A
Tai Hong Wai 泰康圍 YL 6-NE-A
Tai Kau 大滘 TP 4-SW-A
Tai Kei Leng 大旗嶺 YL 6-NW-B
Tai Kek 大乪 YL 6-NE-C
Tai Kiu 大橋 YL 6-NW-B
Tai Kok 大角 Is 15-SW-A
Tai Kok Tau 大角頭 N 4-NW-D
Tai Kok Tau 大角頭 Is 16-SW-C
Tai Kok Tsui 大角咀 YTM 11-NW-D
Tai Kong Po 大江埔 YL 6-NE-A
Tai Kwai Wan 大貴灣 Is 14-NW-D
P. 38 / 53
May 2026
English Name Chinese Name District* HP5C
Tai Kwai Wan San Tsuen 大貴灣新村 Is 10-NW-D
Tai Lam Chung 大欖涌 TM 6-SW-D
Tai Lam Chung Tsuen 大欖涌村 TM 6-SW-D
Tai Lam Chung Wong Uk 大欖涌黃屋 TM 6-SW-D
Tai Lam Chung Wu Uk 大欖涌胡屋 TM 6-SW-D
Tai Lam Liu 大藍寮 ST 7-SE-A
Tai Lam Wu 大藍湖 SK 11-NE-B
Tai Lang Shui 大冷水 TM 5-SE-B
Tai Law Hau 大羅口 YL 2-SE-B
Tai Lei 大利 Is 10-SW-B
Tai Leng 大嶺 TP-N 4-SW-B
Tai Leng 大嶺 SK 12-NW-B
Tai Leng Pei 大嶺皮 N 3-SW-B
Tai Leng Tau 大嶺頭 TW 10-NE-A
Tai Leng Tun 大嶺墩 TP 4-SE-C
Tai Leng Tung 大嶺峒 SK 12-SW-B
Tai Ling 大嶺 YL 6-NE-D
Tai Ling Tsuen 大嶺村 Is 15-NW-A
Tai Long 大浪 SK 8-NE-D
Tai Long 大浪 SK 8-SE-A
Tai Long 大浪 Is 14-NW-A
Tai Long Au 大浪坳 TP 8-NE-C
Tai Long Kei 大榔基 YL 2-SW-D
Tai Long Pai 大浪排 S 16-NW-A
Tai Long Tsui 大浪咀 SK 8-NE-D
Tai Long Wan 大浪灣 SK 8-SE-B
Tai Long Wan 大浪灣 S 11-SE-D
Tai Long Wan 大浪灣 Is 13-NW-B
Tai Long Wan 大浪灣 Is 14-NW-C
Tai Long Wan Tsuen 大浪灣村 Is 13-NW-B
Tai Lung 大隴 N 3-SW-C
Tai Lung Tsuen 大龍村 Is 10-SW-B
Tai Mei Tuk 大美督 TP 3-SE-D
Tai Miu Au 大廟坳 SK 12-SW-C
Tai Mo Shan 大帽山 TW 7-NW-C
Tai Mong Tsai 大網仔 SK 8-SW-B
Tai Mong Tsai 大網仔 SK 8-SW-B
Tai Mun Shan 大蚊山 TP 8-NE-C
Tai Ngam Hau 大岩口 SK 12-NW-A
Tai Ngam Hau 大岩口 SK 8-NE-D
Tai Ngam Teng 大岩頂 SK 12-NE-A
Tai Ngam Tsui 大岩咀 SK 12-NW-A
Tai Ngau Wu 大牛湖 SK 11-NE-B
Tai Nim Chau 大稔洲 N 4-NW-C
Tai No 大腦 SK 7-SE-D
Tai No Sheung Yeung 大腦上洋 SK 7-SE-D
Tai O 大澳 Is 9-SW-D
Tai O Sam Chung 大澳三涌 Is 9-SW-D
Tai O San Ki 大澳新基 Is 9-SW-D
Tai O San Sha 大澳新沙 Is 9-SW-D
Tai O Sha Chai Min 大澳沙仔面 Is 9-SW-D
Tai O Tai Chung 大澳大涌 Is 9-SW-D
P. 39 / 53
May 2026
English Name Chinese Name District* HP5C
Tai O Yee Chung 大澳二涌 Is 9-SW-D
Tai Om 大菴 TP 7-NW-A
Tai Om Shan 大菴山 TP 7-NW-C
Tai Pai 大排 SK 12-NW-A
Tai Pai 大排 SK 12-NE-C
Tai Pai 大排 Is 16-SW-A
Tai Pai Tong Teng 大排塘頂 Is 16-SW-A
Tai Pai Tsui 大排咀 TW 10-NE-A
Tai Pai Wan 大排灣 Is 16-SW-A
Tai Pak Kok 大白角 TP 8-NW-B
Tai Pak Tin Village 大白田村 TW 7-SW-C
Tai Pak Tsui 大白咀 Is 10-SW-B
Tai Peng 大坪 Is 14-NE-B
Tai Ping Village 太平村 SK 8-SW-C
Tai Po 大埔 TP 7-NW-B
Tai Po Kau 大埔滘 TP 7-NW-B
Tai Po Kau Lo Wai 大埔滘老圍 TP 7-NE-C
Tai Po Kau San Wai 大埔滘新圍 TP 7-NE-C
Tai Po Market 大埔墟 TP 7-NW-B
Tai Po Mei 大埔尾 TP 7-NE-C
Tai Po Mei Hang 大埔尾坑 TP 7-NE-C
Tai Po Old Market 大埔舊墟 TP 7-NW-B
Tai Po River 大埔河 TP 7-NW-B
Tai Po Tau 大埔頭 TP 7-NW-B
Tai Po Tau Shui Wai 大埔頭水圍 TP 7-NW-B
Tai Po Tin 大埔田 N 3-NW-C
Tai Po Tsai 大埔仔 SK 11-NE-B
Tai Po Tsai 大埗仔 SK 8-SW-B
Tai Sang Wai 大生圍 YL 2-SW-D
Tai Sha Kok 大沙角 SK 12-SW-D
Tai Sham Chung 大深涌 N 3-NE-B
Tai Shan 大山 TW 10-NW-D
Tai Shan Central 大山中 Is 14-NE-B
Tai Shan East 大山東 Is 14-NE-B
Tai Shan West 大山西 Is 14-NE-B
Tai She Teng 大蛇頂 SK 8-SE-C
Tai She Wan 大蛇灣 SK 8-SE-C
Tai She Wan 大蛇灣 SK 8-SE-C
Tai Shek Ha 大石下 N 4-SW-B
Tai Shek Ha Shan 大石下山 Is 13-SE-A
Tai Shek Hau 大石口 Is 14-NW-D
Tai Shek Kwu 大石鼓 ST 7-SE-A
Tai Sheung Tok 大上托 SK 11-NE-D
Tai Shing Stream 大城石澗 TW 7-SW-A
Tai Shue Wan 大樹灣 S 15-NW-B
Tai Shui Hang 大水坑 TM 5-NE-D
Tai Shui Hang 大水坑 ST 7-SE-A
Tai Shui Hang 大水坑 Is 10-SW-B
Tai Shui Tseng 大水井 SK 7-SE-B
Tai Shui Wu 大水湖 N 4-SW-A
Tai Tam 大氹 SK 12-SW-D
Tai Tam 大潭 S 15-NE-C
P. 40 / 53
May 2026
English Name Chinese Name District* HP5C
Tai Tam 大氹 Is 16-SW-C
Tai Tam Bay 大潭灣 S 15-NE-D
Tai Tam Gap 大潭峽 E-S 11-SE-D
Tai Tam Harbour 大潭港 S 15-NE-B
Tai Tam Mound 大潭崗 S 11-SE-C
Tai Tam Tau 大潭頭 S 15-NE-C
Tai Tam Wan 大氹灣 Is 16-SW-C
Tai Tan 大灘 TP 8-NW-B
Tai Tao Tsuen 大道村 YL 6-NW-C
Tai Tau Chau 大頭洲 SK 8-SW-D
Tai Tau Chau 大頭洲 S 15-NE-B
Tai Tau Chau Tsui 大頭洲咀 SK 8-SW-D
Tai Tau Leng 大頭嶺 N 3-SW-A
Tai Tei Tong 大地塘 Is 10-SW-C
Tai To Yan 大刀屻 TP-YL 7-NW-A
Tai Tong 大塘 TP 17-NW-A
Tai Tong Lek 大塘瀝 N 4-SW-B
Tai Tong Tsuen 大棠村 YL 6-NW-D
Tai Tong Wan 大塘灣 TP 17-NW-A
Tai Tong Wu 大塘湖 N 3-NW-D
Tai Tsan Chau 大鏟洲 SK 8-SW-C
Tai Tseng Wai 大井圍 YL 6-NW-B
Tai Tsing Chau 大青洲 TW 10-NE-A
Tai Tsoi Yuen Kui 大菜園區 Is 14-NW-D
Tai Tun 太墩 SK 8-SW-B
Tai Tung 大峒 TP-N 4-SW-A
Tai Tung 大洞 TP 8-NW-C
Tai Tung Wo Liu 大洞禾寮 TP 8-NW-C
Tai Wai 大圍 ST 7-SW-D
Tai Wai New Village 大圍新村 ST 7-SW-D
Tai Wai Tsuen 大圍村 YL 6-NW-B
Tai Wan 大灣 N 3-NE-C
Tai Wan 大環 N 3-NE-D
Tai Wan 大環 TP 4-SE-A
Tai Wan 大灣 SK 8-NE-D
Tai Wan 大灣 SK 8-NE-D
Tai Wan 大環 SK 8-SW-A
Tai Wan 大灣 Is 15-SE-D
Tai Wan 大灣 Is 15-SE-D
Tai Wan Kau Tsuen 大灣舊村 Is 15-NW-A
Tai Wan Nam 大灣南 Is 14-NE-B
Tai Wan San Tsuen 大灣新村 Is 14-NE-B
Tai Wan Tau 大環頭 SK 12-SW-A
Tai Wan Tau Kok 大環頭角 SK 12-SW-A
Tai Wan To 大灣肚 Is 15-NW-A
Tai Wo 大窩 N 3-SW-B
Tai Wo 大窩 TP 3-SW-D
Tai Wo 大窩 YL 6-NE-C
Tai Wo 大窩 Is 10-SE-A
Tai Wo Ping 大窩坪 SSP 11-NW-B
Tai Wong Ha Resite Village 大王下村 K&T 10-NE-B
Tai Wong Kung 大王公 SK 12-SW-A
P. 41 / 53
May 2026
English Name Chinese Name District* HP5C
Tai Wong Wan 大王灣 SK 12-NW-A
Tai Wong Wan 大往灣 SK 12-NW-B
Tai Yam Teng 大陰頂 TW 10-NW-B
Tai Yat San Tsuen 第一新村 Is 10-SW-B
Tai Yeung Che 大陽輋 TP 7-NW-A
Tai Yue Ngam Teng 睇魚岩頂 SK 8-SE-B
Tai Yuen New Village 大園新村 Is 14-NE-B
Tai Yuen Tsuen 大元村 N 3-SW-A
Tai Yuen Village 大園村 Is 14-NE-B
Tai Yuk Road 體育路 TW 10-NE-A
Tak Yuet Lau 得月樓 N 2-NE-D
Tam Kon Chau 担竿洲 YL 2-SE-A
Tam Shui Hang 担水坑 N 2-SE-D
Tam Shui Keng 担水徑 TP 7-NW-A
Tam Shui Wan 淡水灣 SK 8-SE-C
Tam Shui Wan 淡水灣 TW 10-NE-A
Tam Wat 氹笏 SK 8-SW-B
Tan Cheung 躉場 SK 8-SW-A
Tan Chuk Hang 丹竹坑 N 3-SW-B
Tan Chuk Hang Lo Wai 丹竹坑老圍 N 3-SW-B
Tan Ka Wan 蛋家灣 TP 8-NE-A
Tan Kwai Tsuen 丹桂村 YL 6-NW-C
Tan Shan 炭山 SK 11-NE-B
Tan Shan River 丹山河 N 3-SW-B
Tang Chau 燈洲 TP 7-NE-B
Tang Lung Chau 燈籠洲 TW 10-NE-A
Tap Mun 塔門 TP 4-SE-C
Tap Mun New Fishermen's Village 塔門漁民新村 TP 4-SE-C
Tap Shek Kok 踏石角 TM 5-SE-C
Target Valley 牛山 N 3-SW-C
Tate's Cairn ( Tai Lo Shan ) 大老山 WTS-ST 7-SE-C
Tate's Pass ( Tai Lo Au ) 大老坳 WTS-SK 7-SE-C
Tate's Ridge － WTS-ST 7-SE-C
Tathong Channel 藍塘海峽 SK-S 16-NW-A
Tathong Point ( Nam Tong Mei ) 南堂尾 SK 16-NW-A
Tau Chau 頭洲 S 15-NE-A
Tau Lo Chau 頭顱洲 Is 13-SE-C
Tau Tun 頭墩 N 4-NW-B
Tei Lung Hau 地龍口 ST 7-SW-D
Tei Po New Village 低埔新村 Is 9-SE-B
Tei Tong Tsai 地塘仔 Is 9-SE-C
Tei Tong Tsui 地堂咀 SK 12-SW-C
Temple Hill ( Tsz Wan Shan ) 慈雲山 ST 7-SE-C
The Brothers 大小磨刀 TM 10-NW-A
The Hunch Backs ( Ngau Ngak Shan ) 牛押山 TP-ST 7-NE-D
The Peak 山頂 C&W 11-SW-C
The Twins 孖崗山 S 15-NE-A
Three Fathoms Cove ( Kei Ling Ha Hoi ) 企嶺下海 TP 8-NW-C
Tin Fu Tsai 田夫仔 TM 6-SE-A
Tin Ha Au 田下坳 SK 12-SW-A
Tin Ha Shan 田下山 SK 12-SW-A
Tin Ha Wan Village 田下灣村 SK 12-NW-C
P. 42 / 53
May 2026
English Name Chinese Name District* HP5C
Tin Liu 田寮 SK 7-SE-D
Tin Liu 田寮 ST 7-SW-D
Tin Liu 田寮 TW 10-NE-A
Tin Liu 田寮 Is 10-SW-A
Tin Liu New Village 田寮新村 TW 10-NE-A
Tin Liu Tsuen 田寮村 YL 6-NW-D
Tin Mei Shan 田尾山 SK 8-SE-A
Tin Ping Shan Tsuen 天平山村 N 3-SW-A
Tin Sam 田心 N 3-SE-B
Tin Sam 田心 YL 6-NW-C
Tin Sam 田心 TP 7-NW-B
Tin Sam 田心 ST 7-SW-D
Tin Sam 田心 Is 9-SE-A
Tin Sam San Tsuen 田心新村 YL 6-NE-C
Tin Sam Tsuen 田心村 YL 6-NE-C
Tin Shui Wai 天水圍 YL 6-NW-A
Tin Wan Shan 田灣山 S-C&W 11-SW-D
Ting Kau 汀九 TW 6-SE-C
Ting Kok 汀角 TP 3-SE-C
Tit Cham Chau 鐵篸洲 SK 12-SW-C
Tit Chi Shan 鐵矢山 SK 8-SW-A
Tit Hang 鐵坑 YL 2-SE-B
Tit Kim Hang 鐵鉗坑 SK 8-SW-B
Tit Mei Tsai 鐵尾仔 TP 7-NW-B
Tit Sha Long 鐵砂塱 Is 15-NW-C
Tit Tak Shue 鐵德樹 Is 9-SW-D
Tiu Chung Pai 吊鐘排 SK 12-NW-D
Tiu Chung Tai Shan 吊鐘大山 SK 12-NW-B
Tiu Keng Leng 調景嶺 SK 11-SE-B
Tiu Shau Ngam 吊手岩 ST 7-NE-D
Tiu Tang Lung 吊燈籠 N 3-SE-B
Tiu Tso Ngam 吊草岩 ST 7-SE-C
Tiu Yu Toi 釣魚台 SK 12-SW-A
To Fung Shan 道風山 ST 7-SW-B
To Hang Tung 桃坑峒 TM 6-SW-D
To Kau Wan 倒扣灣 TW 10-NE-A
To Kwa Peng 土瓜坪 TP 8-NE-C
To Kwa Wan 土瓜灣 KC 11-NE-C
To Shek 多石 ST 7-SE-C
To Tau 渡頭 ST 7-NE-D
To Tau Tsui 刀頭咀 TP 4-SW-C
To Tei Wan 土地灣 S 15-NE-B
To Tei Wan Village 土地灣村 S 15-NE-B
To Uk Tsuen 杜屋村 YL 6-NE-D
To Yuen Tung 桃源洞 TP 7-NW-B
To Yuen Wai 桃園圍 TM 6-NW-C
Tolo Channel ( Chek Mun ) 赤門 TP 4-SW-D
Tolo Harbour 吐露港 TP 7-NE-A
Tong Chai Wan 塘仔灣 Is 13-SE-A
Tong Fong 塘坊 N 3-NW-C
Tong Fong Tsuen 塘坊村 YL 6-NW-B
Tong Fuk 塘福 Is 13-NE-A
P. 43 / 53
May 2026
English Name Chinese Name District* HP5C
Tong Fuk Miu Wan 塘福廟灣 Is 13-NE-A
Tong Hang 塘坑 N 3-SW-A
Tong Hang Tung Chuen 塘坑東村 TP 3-SW-D
Tong Hau Pai 塘口排 SK 12-NE-A
Tong Kai Tseng 劏雞井 TP 7-NE-A
Tong Kok 塘角 N 2-SE-B
Tong Kung Leng 唐公嶺 N 2-SE-D
Tong Lek Tsai 塘瀝仔 N 4-SW-B
Tong Pai Tau 盪排頭 N 4-SW-B
Tong Sheung Tsuen ( Tong Min Tsuen ) 塘上村 ( 塘面村 ) TP 7-NW-A
Tong Tau Po Tsuen 塘頭埔村 YL 6-NW-D
Tong To 塘肚 N 3-NE-C
Tong To Ping Tsuen 塘肚坪村 N 3-NE-C
Tong To Shan Tsuen 塘肚山村 N 3-NW-B
Tong Yan Pai 劏人排 S 15-NE-C
Tong Yan San Tsuen 唐人新村 YL 6-NW-D
Town Island ( Fo Tau Fan Chau ) 伙頭墳洲 SK 12-NE-A
Trio Island ( Tai Lak Lei ) 大癩痢 SK 12-SW-B
Tsak Yue Wu 鯽魚湖 SK 8-SW-B
Tsam Chuk Wan 斬竹灣 SK 8-SW-B
Tsam Chuk Wan 斬竹灣 SK 8-SW-B
Tsang Pang Kok 罾棚角 SK 8-SE-D
Tsang Pang Kok Teng 罾棚角頂 SK 8-SE-D
Tsang Tai Uk New Village 曾大屋新村 ST 7-SE-C
Tsang Tsai Au 曾仔坳 Is 15-NW-A
Tsang Tsui 曾咀 TM 5-NE-C
Tsang Uk Tsuen 曾屋村 YL 6-NE-D
Tsat Shue Wan 漆樹灣 N 4-SW-A
Tsat Sing Kong 七星崗 YL 6-NE-B
Tsau Uk 鄒屋 TP 17-NW-A
Tsau Wan 酒灣 KT 11-SE-B
Tse Koo Hang 鷓鴣坑 N 2-NE-D
Tse Uk 謝屋 TP 8-NE-A
Tse Uk Tsuen 謝屋村 YL 6-NE-D
Tse Uk Village 謝屋村 ST 7-SE-C
Tseng Lan Shue 井欄樹 SK 11-NE-B
Tseng Tau 井頭 TP 8-NW-C
Tseng Tau 井頭 TP 3-SE-C
Tseng Tau Chung Tsuen 井頭中村 TM 6-SW-A
Tseng Tau Ha Tsuen 井頭下村 TM 6-SW-A
Tseng Tau San Tsuen 井頭新村 Is 10-SW-C
Tseng Tau Sheung Tsuen North 井頭上村北 TM 6-SW-A
Tseng Tau Sheung Tsuen South 井頭上村南 TM 6-SW-A
Tseung Kong Wai 祥降圍 YL 6-NW-A
Tseung Kwan O 將軍澳 SK 11-NE-D
Tseung Kwan O Town Centre 將軍澳市中心 SK 11-NE-D
Tseung Kwan O Village 將軍澳村 SK 11-NE-D
Tsim Bei Tsui 尖鼻咀 YL 2-SW-D
Tsim Chau 尖洲 SK 8-SE-B
Tsim Fung Shan 尖峰山 Is 13-NW-B
Tsim Kong Tung 尖光峒 N 3-NE-D
Tsim Kong Tung Au 尖光峒坳 N 3-NE-D
P. 44 / 53
May 2026
English Name Chinese Name District* HP5C
Tsim Mei Fung 尖尾峰 ST-SK 7-SE-D
Tsim Sha Tsui 尖沙咀 YTM 11-SW-B
Tsin Shui Wan Au 淺水灣坳 S 15-NE-A
Tsin Yue Wan 煎魚灣 Is 13-NW-C
Tsing Chau 青洲 N 4-NW-C
Tsing Chau Lek 青洲瀝 N 4-NW-C
Tsing Chau Tsai 青洲仔 TW 10-NW-B
Tsing Chau Wan 青洲灣 TW 10-NW-B
Tsing Chuen Wai 青磚圍 TM 6-NW-C
Tsing Fai San Tsuen 青輝新村 K&T 10-NE-B
Tsing Fai Tong 清快塘 TW 6-SE-C
Tsing Fai Tong New Village 清快塘新村 TW 6-SE-C
Tsing Kei Hang 程其坑 N 3-SW-C
Tsing Lam Kok 青林角 Is 13-NW-A
Tsing Lung Tau 青龍頭 TW 6-SE-C
Tsing Lung Tau New Village 青龍頭新村 TW 6-SE-C
Tsing Lung Tau Tsuen 青龍頭村 TW 6-SE-C
Tsing Lung Tsuen 青龍村 YL 2-SE-A
Tsing Shan Tsuen 青山村 TM 5-SE-B
Tsing Shan Tsuen San Shek Wan North 青山村散石灣北 TM 5-SE-B
Tsing Shan Tsuen San Shek Wan South 青山村散石灣南 TM 5-SE-D
Tsing Tam 清潭 YL 6-NE-D
Tsing Yi 青衣 K&T 10-NE-B
Tsing Yi Hui 青衣墟 K&T 6-SE-D
Tsing Yi Lutheran New Village 青衣信義新村 K&T 10-NE-B
Tsing Yu New Village 青裕新村 K&T 6-SE-D
Tsiu Hang 蕉坑 N 3-NE-A
Tsiu Hang 蕉坑 TP 7-NE-C
Tsiu Hang 蕉坑 SK 8-SW-C
Tsiu Hang Hau 蕉坑口 SK 8-SW-C
Tsiu Keng 蕉徑 N 2-SE-D
Tsiu Keng Lo Wai 蕉徑老圍 N 2-SE-D
Tsiu Keng Pang Uk 蕉徑彭屋 N 2-SE-D
Tsiu Keng San Wai 蕉徑新圍 N 2-SE-D
Tsiu Lam 蕉林 TP 3-SE-C
Tsiu Lan Shue 蕉欄樹 SK 11-NE-B
Tsiu Wo 蕉窩 SK 12-SW-A
Tso Kung Tam 曹公潭 TW 6-SE-D
Tso Tui Ha 草堆下 ST 7-SE-C
Tso Tui Wan 草堆灣 E 11-SE-D
Tso Wan 草灣 TW 10-NE-A
Tso Wo Hang 早禾坑 SK 8-SW-A
Tsoi Uk 蔡屋 TP 17-NW-C
Tsoi Uk Tsuen 蔡屋村 YL 6-NW-B
Tsoi Yuen Kok 菜園角 N 3-NE-B
Tsoi Yuen Tsuen 菜園村 TM 6-NW-C
Tsoi Yuen Tsuen 菜園村 Is 10-SW-C
Tsoi Yuen Wan 菜園灣 Is 10-SW-B
Tsok Pok Hang San Tsuen 作壆坑新村 ST 7-SE-C
Tsuen Kam Au 荃錦坳 YL-TW 6-SE-B
Tsuen Wan 荃灣 TW 6-SE-D
Tsui Lam 翠林 SK 11-NE-D
P. 45 / 53
May 2026
English Name Chinese Name District* HP5C
Tsui Pai 咀排 Is 14-NW-C
Tsung Pak Long 松柏塱 N 3-SW-A
Tsung Shan 松山 N 3-SW-B
Tsung Tsai Yuen 松仔園 TP 7-NW-D
Tsung Yuen 松園 N 2-SE-B
Tsung Yuen Ha 松園下 N 3-NW-B
Tsz Kan Chau 匙羹洲 TM 10-NW-C
Tsz Tin Tsuen 紫田村 TM 6-NW-C
Tsz Tong Tsuen 祠堂村 N 3-SW-B
Tsz Tong Tsuen 祠塘村 YL 6-NE-A
Tsz Tong Tsuen 祠堂村 YL 6-NE-D
Tsz Wan Shan 慈雲山 WTS 11-NE-A
Tuen Chau Tsai 短洲仔 SK 12-SE-C
Tuen Keng 斷頸 SK 12-SE-C
Tuen Mun 屯門 TM 6-SW-A
Tuen Mun Kau Hui 屯門舊墟 TM 5-SE-B
Tuen Mun River Channel 屯門河道 TM 6-SW-A
Tuen Mun San Hui 屯門新墟 TM 6-SW-A
Tuen Mun San Tsuen 屯門新村 TM 6-NW-C
Tuen Tau Chau 斷頭洲 SK 8-SW-C
Tuen Tsui 短咀 SK 8-NE-D
Tuen Tsz Wai 屯子圍 TM 6-NW-C
Tui Min Hoi 對面海 SK 8-SW-C
Tuk Mei Chung 篤尾涌 TM 5-SE-A
Tuk Ngu Shan 獨孤山 SK 8-SE-C
Tung A 東丫 SK 12-NE-A
Tung Ah Pui Village 東丫背村 S 15-NE-B
Tung Ah Village 東丫村 S 15-NE-B
Tung Chan Wai 東鎮圍 YL 2-SE-A
Tung Chung 東涌 Is 9-SE-B
Tung Chung Bay 東涌灣 Is 9-SE-A
Tung Fong 東方 N 2-SE-B
Tung Fung Au 東風坳 N 3-NW-D
Tung Hing 東慶 Is 9-SE-A
Tung Kok Wai 東閣圍 N 3-SW-B
Tung Lau 東樓 Is 15-SE-B
Tung Lo Hang 銅鑼坑 N 3-NW-D
Tung Lo Wan 銅鑼灣 ST 7-SW-D
Tung Lung Chau 東龍洲 SK 12-SW-C
Tung Mun Hau 東門口 YL 6-NW-B
Tung O 東澳 N 4-NW-A
Tung O 東澳 Is 15-NW-C
Tung O Wan 東澳灣 N 4-NW-A
Tung O Wan 東澳灣 Is 15-NW-C
Tung Sam Chau 棟心洲 SK 12-NE-C
Tung Sam Kei 東心淇 TP 8-NE-A
Tung Sam Kei Tsui 東心淇咀 TP 8-NE-A
Tung Shan 東山 SK 11-NE-A
Tung Shan Ha 東山下 N 3-SW-B
Tung Shing Lei 東成里 YL 6-NE-A
Tung Tau Chau 東頭洲 TP 7-NE-B
Tung Tau Teng 東頭頂 Is 16-SW-A
P. 46 / 53
May 2026
English Name Chinese Name District* HP5C
Tung Tau Tsuen 東頭村 YL 6-NW-A
Tung Tau Tsuen 東頭村 YL 6-NW-B
Tung Tau Wai San Tsuen 東頭圍新村 YL 6-NW-B
Tung Tau Wan 東頭灣 S 15-NE-C
Tung Tau Yuen 東頭園 YL 6-NW-B
Tung Tsui Teng 東咀頂 Is 16-SW-C
Tung Tsz 洞梓 TP 3-SE-C
Tung Wan 東灣 N 4-NW-D
Tung Wan 東灣 N 4-SW-B
Tung Wan 東灣 SK 8-NE-D
Tung Wan 東灣 SK 8-NE-D
Tung Wan 東灣 TW 10-NE-A
Tung Wan 東灣 TW 10-NE-A
Tung Wan 東灣 Is 10-SW-B
Tung Wan 東灣 Is 13-NE-A
Tung Wan 東灣 Is 13-NW-B
Tung Wan 東灣 Is 13-SE-A
Tung Wan 東灣 Is 14-NW-D
Tung Wan Hang 東灣坑 N 4-SW-B
Tung Wan Mei 東灣尾 Is 13-NE-C
Tung Wan Shan 東灣山 SK 8-NE-D
Tung Wan Tau 東灣頭 Is 10-SW-C
Tung Wan Tsai 東灣仔 TW 10-NE-A
Tung Wan Tsai 東灣仔 Is 14-NW-D
Tung Wan Tsui 東灣咀 N 4-NW-D
Tung Yeung Shan 東洋山 ST-SK 7-SE-C
Tung Yip Hang 東葉坑 TW 10-NW-C
Turret Hill ( Nui Po Shan ) 女婆山 ST 7-SE-A
Turret Pass ( Nui Po Au ) 女婆坳 ST 7-SE-B
Turtle Cove 龜背灣 S 15-NE-A
Uk Cheung 屋場 SK 7-SE-D
Uk Tau 屋頭 TP 8-NW-D
Uk Tau Tsuen 屋頭村 YL 6-NE-D
Ung Kong Wan 甕缸灣 SK 12-NE-C
Unicorn Ridge 雞胸山 ST 7-SE-C
Upper Keung Shan 上羗山 Is 9-SW-D
Urmston Road 龍鼓水道 TM 5-SE-A
Vernon Pass ( Pai Tau Lo ) 排頭路 N 2-NE-D
Victoria Gap 爐峰峽 C&W 11-SW-C
Victoria Harbour 維多利亞港 - 11-SW-B
Victoria Peak 扯旗山 C&W 11-SW-A
Violet Hill 紫羅蘭山 S 11-SE-C
Wa Mei Shan 畫眉山 N 3-SW-C
Wa Mei Shan 畫眉山 TP 8-NW-D
Wa Shan 華山 N 3-SW-A
Waglan Island 橫瀾島 Is 16-SW-B
Wah Fu 華富 S 11-SW-C
Wah Shing Tsuen 華盛村 YL 6-NE-A
Wah Yuen 華苑 YL 6-NE-D
Wai Ha 圍下 TP 3-SE-C
Wai Kap Pai 桅夾排 SK 12-NE-A
Wai Kap Shek Leng 桅夾石嶺 SK 11-NE-D
P. 47 / 53
May 2026
English Name Chinese Name District* HP5C
Wai Loi Tsuen 圍內村 N 3-SW-A
Wai Sum Village 圍心村 SK 12-NW-C
Wai Tau Tsuen 圍頭村 TP 7-NW-A
Wai Tau Tsuen Che Tei 圍頭村輋地 TP 3-SW-C
Wai Tsai 圍仔 YL 2-SE-C
Wai Tsai Tseng San Tsuen 圍仔井新村 Is 10-SW-B
Wai Tsai Tsuen 圍仔村 Is 10-SW-B
Wan Chai 灣仔 WC 11-SW-B
Wan Chai Gap 灣仔峽 WC 11-SW-D
Wan Cham Shan 雲枕山 S 15-NE-B
Wan Hau Chau 灣口洲 Is 13-SE-A
Wan Tam Shan 穩氹山 SK 12-NE-C
Wan Tsai 環仔 N 4-NW-D
Wan Tsai 灣仔 TP 4-SE-C
Wan Tsai 灣仔 Is 10-SW-C
Wan Tsai 灣仔 Is 15-SE-D
Wan Tuk 灣篤 TW 10-NW-D
Wang Chau 橫洲 YL 6-NW-B
Wang Chau 橫洲 SK 12-NE-B
Wang Chau Chung Sam Wai 橫洲忠心圍 YL 6-NW-B
Wang Chau Fuk Hing Tsuen 橫洲福慶村 YL 6-NW-B
Wang Chau Kok 橫洲角 SK 12-NE-D
Wang Chau Lam Uk Tsuen 橫洲林屋村 YL 6-NW-B
Wang Chau Sai Tau Wai 橫洲西頭圍 YL 6-NW-B
Wang Chau Tung Tau Wai 橫洲東頭圍 YL 6-NW-B
Wang Chau Yeung Uk Tsuen 橫洲楊屋村 YL 6-NW-B
Wang Che 橫輋 SK 7-SE-D
Wang Hang Village 橫坑村 Is 9-SW-D
Wang Kong Tsuen 橫江村 SK 8-SW-A
Wang Lek 橫瀝 N 3-NW-B
Wang Leng 橫嶺 N-TP 3-SE-B
Wang Leng 橫嶺 N 3-SW-B
Wang Leng Au 橫嶺坳 N-TP 4-SW-A
Wang Leng Pui 橫嶺背 TP 4-SW-C
Wang Leng Tau 橫嶺頭 TP 3-SE-D
Wang Long 橫塱 Is 14-NE-B
Wang Mun Hoi 橫門海 N 4-NW-D
Wang Pai 橫排 SK 12-SE-A
Wang Shan Keuk Ha Tsuen 橫山腳下村 N 3-SE-B
Wang Shan Keuk San Tsuen 橫山腳新村 N 3-NW-D
Wang Shan Keuk Sheung Tsuen 橫山腳上村 N 3-SE-A
Wang Tau Hom 橫頭磡 WTS 11-NW-B
Wang Tau Tun 橫頭墩 SK 8-SE-A
Wang Toi Shan 橫台山 YL 6-NE-B
Wang Toi Shan Ho Lik Pui 橫台山河瀝背 YL 6-NE-B
Wang Toi Shan Hung Mo Tam 橫台山紅毛潭 YL 6-NE-B
Wang Toi Shan Lo Uk Tsuen 橫台山羅屋村 YL 6-NE-B
Wang Toi Shan San Tsuen 橫台山新村 YL 6-NE-B
Wang Toi Shan Shan Tsuen 橫台山散村 YL 6-NE-B
Wang Toi Shan Tsoi Yuen Tsuen (North) 橫台山菜園村 (北) YL 6-NE-B
Wang Toi Shan Tsoi Yuen Tsuen (South) 橫台山菜園村 (南) YL 6-NE-B
Wang Toi Shan Wing Ning Lei 橫台山永寧里 YL 6-NE-B
P. 48 / 53
May 2026
English Name Chinese Name District* HP5C
Wang Toi Shan Yau Uk Tsuen 橫台山邱屋村 YL 6-NE-B
Wang Tong 橫塘 Is 10-SW-C
Wang Tong 橫塘 Is 14-NW-A
Wang Tong River 橫塘河 Is 10-SW-A
Waterfall Bay 瀑布灣 S 11-SW-C
West Brother ( Tai Mo To ) 大磨刀 TM 9-NE-B
West Buffalo Hill 黃牛山 ST 7-SE-D
West Lamma Channel 西博寮海峽 Is 14-NE-A
Whiskey Beach 白環 SK 12-NW-B
Windy Gap 大風坳 S 15-NE-B
Wing Kei Tsuen 榮基村 YL 6-NE-A
Wing Lung Wai 永隆圍 YL 6-NE-A
Wing Ning Tsuen 永寧村 N 3-SW-A
Wing Ning Tsuen 永寧村 YL 6-NW-B
Wing Ning Wai 永寧圍 N 3-SW-A
Wing Ping Tsuen 永平村 YL 2-SE-A
Wo Hang Tai Long 禾坑大朗 N 3-NE-C
Wo Hing Tsuen 和興村 N 3-SW-C
Wo Hop Shek 和合石 N 3-SW-C
Wo Hop Shek San Tsuen 和合石新村 N 3-SW-C
Wo Hop Shek Village 和合石村 N 3-SW-C
Wo Keng Shan 禾徑山 N 3-NW-D
Wo Keng Shan 禾徑山 N 3-NW-D
Wo Liu 禾寮 TP 7-NW-A
Wo Liu 禾寮 SK 8-SW-A
Wo Liu Hang 禾寮坑 ST 7-SE-A
Wo Liu Tun 禾寮墩 Is 9-SE-D
Wo Mei 窩美 SK 11-NE-B
Wo Ping San Tsuen 和平新村 TM 6-NW-C
Wo Shang Wai 和生圍 YL 2-SE-C
Wo Sheung Au 禾上坳 Is 10-SW-A
Wo Sheung Chau 和尚洲 SK 8-NE-D
Wo Sheung Tun 禾上墩 ST 7-SW-B
Wo Tin 窩田 Is 10-SW-A
Wo Tong Kong 禾塘崗 N 3-NE-C
Wo Tong Kong 禾塘江 SK 8-SW-A
Wo Tong Kong 禾塘崗 SK 12-NW-C
Wo Tong Pui 禾堂背 TP 7-NW-A
Wo Yi Hop 和宜合 TW 7-SW-A
Wok Tai Wan 鑊底灣 K&T 6-SE-D
Wong Chuk Chung 黃竹涌 N 4-SW-A
Wong Chuk Hang 黃竹坑 S 11-SW-D
Wong Chuk Hang San Wai Village 黃竹坑新圍村 S 11-SW-D
Wong Chuk Kok 黃竹角 Is 15-NW-D
Wong Chuk Kok Hoi 黃竹角海 N 4-SW-B
Wong Chuk Kok Tsui 黃竹角咀 N 4-SW-B
Wong Chuk Long 黃竹塱 TP 8-NW-D
Wong Chuk Shan 黃竹山 ST 7-SE-D
Wong Chuk Shan New Village 黃竹山新村 SK 7-SE-B
Wong Chuk Tsuen 黃竹村 TP 3-SE-D
Wong Chuk Wan 黃竹灣 SK 8-SW-A
Wong Chuk Yeung 黃竹洋 ST 7-SW-B
P. 49 / 53
May 2026
English Name Chinese Name District* HP5C
Wong Chuk Yeung 黃竹洋 TP 8-SW-A
Wong Chuk Yuen 黃竹園 YL 6-NE-D
Wong Fa Pai 黃花排 Is 13-NW-A
Wong Fong Shan 黃幌山 N 4-NW-C
Wong Ka Wai 皇家圍 TM 6-SW-A
Wong Ka Wai 黃家圍 Is 9-SE-B
Wong Keng Tei 黃麖地 SK 8-SW-B
Wong Keng Tsai 黃麖仔 SK 11-NE-B
Wong Kok Mei 王角尾 N 4-NW-D
Wong Kok Tau 王角頭 N 4-NW-D
Wong Kok Teng 王角頂 N 4-NW-D
Wong Kong Shan 黃崗山 N 3-SW-C
Wong Kung Tin 黃公田 Is 10-SW-A
Wong Leng 黃嶺 N-TP 3-SE-C
Wong Lung Hang 黃龍坑 Is 9-SE-D
Wong Ma Tei 黃麻地 TP 8-NE-C
Wong Mau Chau 黃茅洲 SK 8-NE-B
Wong Mau Hang Shan 黃茅坑山 N 3-NW-B
Wong Mau Kok 黃茅角 TP 8-NE-A
Wong Mo Ying 黃毛應 SK 8-SW-A
Wong Nai Chau 黃泥洲 N 4-NW-C
Wong Nai Chau 黃泥洲 N 4-NW-D
Wong Nai Chau 黃泥洲 SK 12-NE-A
Wong Nai Chau Tsai 黃泥洲仔 SK 8-SW-B
Wong Nai Chung Gap 黃泥涌峽 WC 11-SE-C
Wong Nai Fai 黃泥塊 TP 7-NE-C
Wong Nai Tau 黃泥頭 ST 7-SE-C
Wong Nai Tun Tsuen 黃泥墩村 YL 6-NW-D
Wong Nai Uk 黃泥屋 Is 9-SE-B
Wong Tai Sin 黃大仙 WTS 11-NE-A
Wong Tei Tung 黃地峒 TP 8-NW-C
Wong Uk Tsuen 黃屋村 YL 6-NW-B
Wong Uk Village 王屋村 ST 7-SE-C
Wong Wan 往灣 N 4-SW-B
Wong Wan 往灣 N 4-SW-B
Wong Wan Pai 往灣排 SK 12-NW-D
Wong Wan Pak Teng 往灣北頂 N 4-NW-D
Wong Wan Sai Teng 往灣西頂 N 4-SW-B
Wong Wan Tsai 往灣仔 TP 4-SW-C
Wong Wan Tsui 往灣咀 N 4-SW-B
Wong Ye Kok 王爺角 TP 17-NW-A
Wong Yi Chau 黃宜洲 SK 8-SW-B
Wong Yi Chau 黃宜洲 SK 8-SW-B
Wong Yue Tan 黃魚灘 TP 7-NE-A
Wu Chau 烏洲 N 4-SW-B
Wu Chau 烏洲 TP 8-NW-C
Wu Chau Tong 烏洲塘 N 4-SW-B
Wu Kai Sha 烏溪沙 ST 7-NE-D
Wu Kai Sha Village 烏溪沙村 ST 7-NE-D
Wu Kau Tang 烏蛟騰 N 3-SE-B
Wu Kwai Sha Tsui 烏龜沙咀 ST 7-NE-B
Wu Lei Kiu 狐狸叫 TP 8-NE-A
P. 50 / 53
May 2026
English Name Chinese Name District* HP5C
Wu Lei Tau 狐狸頭 SK 7-SE-D
Wu Nga Lok Yeung 烏鴉落陽 N 3-SW-A
Wu Pai 烏排 N 4-NW-C
Wu Pai 烏排 N 4-NW-D
Wu Pai 烏排 N 4-SW-B
Wu Shek Kok 烏石角 N 3-NE-C
Wu Tip Shan 蝴蝶山 N 3-SW-C
Wu Tip Shan Village 蝴蝶山村 N 3-SW-C
Wu Yeung Chau Pai 湖洋洲排 N 4-NW-C
Wu Ying Pai 烏蠅排 Is 10-NW-D
Yam Tsai 陰仔 TW 10-NW-B
Yam Tsai Wan 陰仔灣 TW 10-NW-B
Yan Chau 印洲 N 4-NW-C
Yan O Tuk 欣澳篤 TW 10-NW-D
Yan O Wan 欣澳灣 TW 10-NW-D
Yan Shau Wai 仁壽圍 YL 2-SE-A
Yau Cha Po 油渣埔 YL 6-NW-D
Yau Kom Tau 油柑頭 TW 6-SE-D
Yau Kom Tau 油柑頭 K&T 6-SE-D
Yau Kom Tau Village 油柑頭村 TW 6-SE-D
Yau Lung Kok 游龍角 SK 8-SW-C
Yau Ma Hom Resite Village 油麻磡村 TW 7-SW-C
Yau Ma Po 油麻莆 SK 8-SW-C
Yau Ma Tei 油麻地 YTM 11-NW-D
Yau Mei San Tsuen 攸美新村 YL 2-SE-C
Yau Oi Tsuen 友愛村 ST 7-SW-B
Yau Tam Mei Tsuen 攸潭美村 YL 2-SE-C
Yau Tong 油塘 KT 11-SE-B
Yau Yat Tsuen 又一村 SSP 11-NW-B
Yau Yue Tong 魷魚塘 N 4-SW-B
Yau Yue Wan Village 魷魚灣村 SK 11-NE-D
Yeung Chau 洋洲 N 4-NW-D
Yeung Chau 洋洲 TP 7-NE-A
Yeung Chau 羊洲 SK 8-SW-C
Yeung Ka Tsuen 楊家村 YL 6-NW-D
Yeung Kok Tau 羊角頭 TP 4-SE-A
Yeung Siu Hang 楊小坑 TM 5-SE-B
Yeung Uk San Tsuen 楊屋新村 YL 6-NW-B
Yeung Uk Tsuen 楊屋村 YL 6-NW-B
Yeung Uk Tsuen 楊屋村 TW 7-SW-C
Yi Chuen 二轉 TW 10-NE-A
Yi Leng 二嶺 SK 12-NW-B
Yi Long 二浪 Is 14-NW-C
Yi Long Pai 二浪排 Is 14-NW-C
Yi Long Wan 二浪灣 Is 14-NW-C
Yi O 二澳 Is 13-NW-A
Yi O Hau 二澳口 Is 13-NW-A
Yi O Kau Tsuen 二澳舊村 Is 13-NW-A
Yi O San Tsuen 二澳新村 Is 13-NW-A
Yi Pai 二排 SK 12-NE-D
Yi Pak Au 二白坳 Is 10-NW-C
Yi Pak Wan 二白灣 Is 10-NW-D
P. 51 / 53
May 2026
English Name Chinese Name District* HP5C
Yi Pei Chun 二陂圳 TW 7-SW-C
Yi Pei Chun New Village 二陂圳新村 TW 7-SW-C
Yi To 二肚 N 3-NE-D
Yi Tung Shan 二東山 Is 9-SE-D
Yick Yuen Tsuen 亦園村 TM 6-NW-C
Yim Liu Ha 鹽寮下 N 3-NE-C
Yim Tin 鹽田 Is 9-SW-D
Yim Tin Kok Resite Village 鹽田角村 K&T 10-NE-B
Yim Tin Pai 鹽田排 SK 8-SW-D
Yim Tin Tsai 鹽田仔 TP 7-NE-A
Yim Tin Tsai 鹽田仔 SK 8-SW-D
Yim Tso Ha 鹽灶下 N 3-NE-C
Yin Kong 燕崗 N 2-SE-B
Yin Ngam 燕岩 TP 7-NW-C
Yin Tsz Ngam 燕子岩 SK 8-SE-B
Ying Lung Wai 英龍圍 YL 6-NW-B
Ying Pun 營盤 N 2-SE-D
Ying Pun Ha 營盤下 TP 7-NW-B
Ying Sin Leung CARE Village 應善良美經援村 Is 14-NW-D
Yiu Dau Ping 搖斗坪 ST 7-SW-B
Yu Uk Village 俞屋村 SK 12-NW-C
Yue Kok 魚角 TP 7-NW-B
Yuen Chau 圓洲 Is 13-SE-A
Yuen Chau Kok 圓洲角 ST 7-SE-A
Yuen Chau Tsai 元洲仔 TP 7-NW-B
Yuen Kok 圓角 Is 15-SW-A
Yuen Kong 元崗 YL 6-NE-C
Yuen Kong Chau 圓崗洲 SK 12-NE-C
Yuen Kong Chau 圓崗洲 Is 13-SE-C
Yuen Kong San Tsuen 元崗新村 YL 6-NE-C
Yuen Leng 元嶺 TP 3-SW-D
Yuen Leng Chai 圓嶺仔 N 3-NW-C
Yuen Ling 元嶺 SK 11-NE-B
Yuen Ling Tsai 元嶺仔 Is 10-SW-B
Yuen Long 元朗 YL 6-NW-B
Yuen Long Kau Hui 元朗舊墟 YL 6-NW-B
Yuen Ng Fan 元五墳 SK 8-SE-C
Yuen Shan 園山 YL 6-NE-A
Yuen Tau Shan 圓頭山 YL 6-NW-C
Yuen Tuen Shan 元墩山 N 3-NE-A
Yuen Tun 圓墩 TW 6-SE-C
Yuen Tun Ha 元墩下 TP 7-NW-D
Yuen Tun Village 圓墩村 TW 6-SE-C
Yuk Kwai Shan 玉桂山 S 15-NW-B
Yuk Sau Fung 玉秀峰 TP 3-SW-D
Yung Kok 榕角 N 4-NW-B
Yung Shu Village 榕樹村 TP 4-SE-C
Yung Shue Au 榕樹凹 N 3-NE-D
Yung Shue Au Wan 榕樹凹灣 N 3-NE-B
Yung Shue Ha 榕樹下 Is 15-NW-C
Yung Shue Ling 榕樹嶺 Is 14-NE-B
Yung Shue Long New Village 榕樹塱新村 Is 14-NE-B
P. 52 / 53
May 2026
English Name Chinese Name District* HP5C
Yung Shue Long Old Village 榕樹塱舊村 Is 14-NE-B
Yung Shue O 榕樹澳 TP 8-NW-C
Yung Shue Wan 榕樹灣 Is 14-NE-B
Yung Shue Wan 榕樹灣 Is 14-NE-B
P. 53 / 53
*地區代號
 District Code
District Code English District Name Chinese District Name
C&W Central & Western 中⻄區
E Eastern 東區
Is Islands 離島
K&T Kwai Tsing 葵青
KC Kowloon City 九龍城
KT Kwun Tong 觀塘
N North 北區
S Southern 南區
SK Sai Kung ⻄貢
SSP Sham Shui Po 深水埗
ST Sha Tin 沙田
TM Tuen Mun 屯門
TP Tai Po 大埔
TW Tsuen Wan 荃灣
WC Wan Chai 灣仔
WTS Wong Tai Sin 黃大仙
YL Yuen Long 元朗
YTM Yau Tsim Mong 油尖旺
"""

# =============================================================================
# 1b. RECOGNIZED VILLAGES RAW TEXT (Lands Dept Small House Policy list)
# =============================================================================

RECOGNIZED_VILLAGES_RAW = """
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
位於南區的認可/歷史鄉村
VILLAGES IN SOUTHERN DISTRICT
Village Name 村名 District 地區
1 POK FU LAM VILLAGE 薄扶林村 POK FU LAM 薄扶林
2 SHEK O VILLAGE 石澳村 SHEK O 石澳
3 STANLEY VILLAGE 赤柱村 STANLEY 赤柱
4 TAI TAM TUK VILLAGE 大潭篤村 TAI TAM 大潭
5 HOK TSUI VILLAGE 鶴咀村 CAPE D'AGUILAR 鶴咀
6 WONG CHUK HANG OLD VILLAGE 黃竹坑舊圍 WONG CHUK HANG 黃竹坑
7 KAU PUI LUNG VILLAGE 較盃龍村 POK FU LAM 薄扶林
8 TUNG TAU WAN VILLAGE 東頭灣村 STANLEY 赤柱
9 AP LEI CHAU OLD VILLAGE 鴨脷洲舊村 AP LEI CHAU 鴨脷洲
位於灣仔區及東區的鄉村
VILLAGES IN WAN CHAI & EASTERN DISTRICT
Village Name 村名 District 地區
1 TAI HANG VILLAGE 大坑村 TAI HANG 大坑
2 MOUNT DAVIS VILLAGE 摩星嶺村 MOUNT DAVIS 摩星嶺
位於觀塘區的鄉村
VILLAGES IN KWUN TONG DISTRICT
Village Name 村名 District 地區
1 LEI YUE MUN VILLAGE 鯉魚門村 LEI YUE MUN 鯉魚門
2 SAM KA TSUEN 三家村 LEI YUE MUN 鯉魚門
3 CHA KWO LING VILLAGE 茶果嶺村 CHA KWO LING 茶果嶺
位於黃大仙區及九龍城區的鄉村
VILLAGES IN WONG TAI SIN & KOWLOON CITY DISTRICT
Village Name 村名 District 地區
1 NGA TSIN WAI TSUEN 衙前圍村 SAN PO KONG 新蒲崗
2 CHUK YUEN VILLAGE 竹園村 CHUK YUEN 竹園
3 NGAU CHI WAN VILLAGE 牛池灣村 NGAU CHI WAN 牛池灣
4 TAI HOM VILLAGE 大磡村 DIAMOND HILL 鑽石山
位於新界及離島區的非丁屋/寮屋/非原居民鄉村
NON-INDIGENOUS & URBANIZED VILLAGES IN NEW TERRITORIES
Village Name 村名 District 地區
1 KWU TUNG VILLAGE 古洞村 KWU TUNG 古洞
2 MA SHI PO VILLAGE 馬寶寶村 FANLING 粉嶺
3 TIU KENG LENG VILLAGE 調景嶺村 TIU KENG LENG 調景嶺
4 CHEUNG CHAU MAIN VILLAGE 長洲舊村 CHEUNG CHAU 長洲
5 PENG CHAU MAIN VILLAGE 坪洲舊村 PENG CHAU 坪洲
"""

KNOWN_DISTRICTS_EN = [
  "MA WAN & NORTH EAST LANTAU", "SAI KUNG NORTH", "SHAP PAT HEUNG",
  "LAMMA NORTH", "LAMMA SOUTH", "SOUTH LANTAU", "SHA TAU KOK",
  "SHEUNG SHUI", "TA KWU LING", "TUNG CHUNG", "TUEN MUN",
  "HANG HAU", "SAI KUNG", "SHA TIN", "TAI PO", "TSUEN WAN",
  "KWAI CHUNG", "TSING YI", "HA TSUEN", "KAM TIN", "PAT HEUNG",
  "PING SHAN", "SAN TIN", "MUI WO", "TAI O", "FANLING", "ISLANDS", "NORTH", "YUEN LONG",
  # Additional sub-districts represented by the embedded source text.
  "POK FU LAM", "SHEK O", "STANLEY", "TAI TAM", "CAPE D'AGUILAR", "WONG CHUK HANG",
  "AP LEI CHAU", "TAI HANG", "MOUNT DAVIS", "LEI YUE MUN", "CHA KWO LING",
  "SAN PO KONG", "CHUK YUEN", "NGAU CHI WAN", "DIAMOND HILL", "KWU TUNG",
  "TIU KENG LENG", "CHEUNG CHAU", "PENG CHAU"
]

KNOWN_DISTRICTS_ZH = [
  "馬灣及大嶼山東北", "西貢北", "十八鄉", "南丫島北", "南丫島南",
  "大嶼山南", "沙頭角", "上水", "打鼓嶺", "東涌", "屯門",
  "坑口", "西貢", "沙田", "大埔", "荃灣", "葵涌", "青衣",
  "廈村", "錦田", "八鄉", "屏山", "新田", "梅窩", "大澳", "粉嶺", "離島", "北區", "元朗",
  # Additional sub-districts represented by the embedded source text.
  "薄扶林", "石澳", "赤柱", "大潭", "鶴咀", "黃竹坑", "鴨脷洲", "大坑", "摩星嶺",
  "鯉魚門", "茶果嶺", "新蒲崗", "竹園", "牛池灣", "鑽石山", "古洞", "調景嶺",
  "長洲", "坪洲"
]

SUB_DISTRICT_TO_DISTRICT = {
  "LAMMA NORTH": "ISLANDS", "LAMMA SOUTH": "ISLANDS", "MUI WO": "ISLANDS",
  "SOUTH LANTAU": "ISLANDS", "TAI O": "ISLANDS", "TUNG CHUNG": "ISLANDS",
  "FANLING": "NORTH", "SHA TAU KOK": "NORTH", "SHEUNG SHUI": "NORTH", "TA KWU LING": "NORTH",
  "HANG HAU": "SAI KUNG", "SAI KUNG": "SAI KUNG", "SHA TIN": "SHA TIN",
  "TUEN MUN": "TUEN MUN", "SAI KUNG NORTH": "TAI PO", "TAI PO": "TAI PO",
  "MA WAN & NORTH EAST LANTAU": "TSUEN WAN", "TSUEN WAN": "TSUEN WAN",
  "KWAI CHUNG": "KWAI TSING", "TSING YI": "KWAI TSING",
  "HA TSUEN": "YUEN LONG", "KAM TIN": "YUEN LONG", "PAT HEUNG": "YUEN LONG",
  "PING SHAN": "YUEN LONG", "SAN TIN": "YUEN LONG", "SHAP PAT HEUNG": "YUEN LONG",
  "南丫島北": "離島", "南丫島南": "離島", "梅窩": "離島", "大嶼山南": "離島",
  "大澳": "離島", "東涌": "離島", "粉嶺": "北區", "沙頭角": "北區",
  "上水": "北區", "打鼓嶺": "北區", "坑口": "西貢", "西貢": "西貢",
  "沙田": "沙田", "屯門": "屯門", "西貢北": "大埔", "大埔": "大埔",
  "馬灣及大嶼山東北": "荃灣", "荃灣": "荃灣", "葵涌": "葵青", "青衣": "葵青",
  "廈村": "元朗", "錦田": "元朗", "八鄉": "元朗", "屏山": "元朗",
  "新田": "元朗", "十八鄉": "元朗",
  # Additional urban-village mappings.
  "POK FU LAM": "SOUTHERN", "SHEK O": "SOUTHERN", "STANLEY": "SOUTHERN",
  "TAI TAM": "SOUTHERN", "CAPE D'AGUILAR": "SOUTHERN", "WONG CHUK HANG": "SOUTHERN",
  "AP LEI CHAU": "SOUTHERN", "TAI HANG": "WAN CHAI", "MOUNT DAVIS": "CENTRAL & WESTERN",
  "LEI YUE MUN": "KWUN TONG", "CHA KWO LING": "KWUN TONG", "SAN PO KONG": "WONG TAI SIN",
  "CHUK YUEN": "WONG TAI SIN", "NGAU CHI WAN": "WONG TAI SIN", "DIAMOND HILL": "WONG TAI SIN",
  "KWU TUNG": "NORTH", "TIU KENG LENG": "SAI KUNG", "CHEUNG CHAU": "ISLANDS", "PENG CHAU": "ISLANDS",

  "薄扶林": "南區", "石澳": "南區", "赤柱": "南區", "大潭": "南區", "鶴咀": "南區",
  "黃竹坑": "南區", "鴨脷洲": "南區", "大坑": "灣仔", "摩星嶺": "中西區",
  "鯉魚門": "觀塘", "茶果嶺": "觀塘", "新蒲崗": "黃大仙", "竹園": "黃大仙",
  "牛池灣": "黃大仙", "鑽石山": "黃大仙", "古洞": "北區", "調景嶺": "西貢",
  "長洲": "離島", "坪洲": "離島",
}







def normalize_text(text: str) -> str:
    char_map = {
        "\u03a4": "T", "\u0391": "A", "\u0399": "I", "\u03a1": "P",
        "\u039f": "O", "\u039d": "N", "\u039c": "M", "\u039a": "K",
        "\u0395": "E", "\u0397": "H", "\u03a5": "Y", "\u03a7": "X",
        "\u0392": "B", "\u03a6": "F", "\u03a0": "P",
    }
    for greek, ascii_char in char_map.items():
        text = text.replace(greek, ascii_char)
    return unicodedata.normalize("NFKC", text)


def extract_bilingual_pairs(raw_text: str) -> list[dict[str, str]]:
    normalized = normalize_text(raw_text)
    lines = normalized.split("\n")
    results: list[dict[str, str]] = []

    ignore_phrases = [
        "地名錄", "Place Name Gazetteer", "May 2026", "2026 年", "備註", "Notes:",
        "English Name", "Chinese Name", "District*", "HP5C", "地區代號",
        "District Code", "Central & Western", "Eastern", "Islands", "Kwai Tsing",
        "Kowloon City", "Kwun Tong", "North", "Southern", "Sai Kung",
        "Sham Shui Po", "Sha Tin", "Tuen Mun", "Tai Po", "Tsuen Wan",
        "Wan Chai", "Wong Tai Sin", "Yuen Long", "Yau Tsim Mong",
        "中⻄區", "東區", "離島", "葵青", "九龍城", "觀塘", "北區",
        "南區", "⻄貢", "深水埗", "沙田", "屯門", "大埔", "荃灣",
        "灣仔", "黃大仙", "元朗", "油尖旺",
    ]

    line_pattern = re.compile(
        r"^([a-zA-Z0-9\s\(\)\.,;&\-\'\/\"]+?)\s+"
        r"([\u4e00-\u9fa5\w\(\)\.,;&\-\'\/]+)\s+"
        r"([A-Z&]{1,3}(?:-[A-Z&]{1,3})?)\s*"
        r"([0-9]+-[A-Z]{2}-[A-Z]|\d+)?$"
    )

    for line in lines:
        line = line.replace("|", "").strip()
        if not line or any(p in line for p in ignore_phrases) or line.startswith("P."):
            continue
        match = line_pattern.match(line)
        if match:
            eng_village = match.group(1).strip()
            chi_village = match.group(2).strip()
            dist_code = match.group(3).strip()
            results.append({
                "eng_village": eng_village,
                "zh_village": chi_village,
                "eng_sub_district": "",
                "zh_sub_district": "",
                "eng_district": DISTRICT_CODE_TO_EN.get(dist_code, ""),
                "zh_district": DISTRICT_CODE_TO_ZH.get(dist_code, ""),
            })
    return results


# =============================================================================
# 2b. RECOGNIZED VILLAGES PARSER (from Lands Dept list)
# =============================================================================

def extract_district_from_end(text, district_list):
    """Removes the matching district strictly from the tail end of the string."""
    for d in sorted(district_list, key=len, reverse=True):
        if re.search(r'[a-zA-Z]', d):
            d_pattern = r'\s+'.join(re.escape(word) for word in d.split())
        else:
            d_pattern = re.escape(d)

        pattern = r'^(.*?)\s*' + d_pattern + r'[\s\(\)\.,;&\-]*$'
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(), d

    return text, ""


def parse_blocks(raw_text: str) -> list[tuple[list[int], list[str]]]:
    """Split numbered records without leaking section headers into a village."""

    blocks: list[tuple[list[int], list[str]]] = []
    current_indices: list[int] = []
    current_block_lines: list[str] = []
    header_markers = (
        "list of recognized villages",
        "under the new territories",
        "village improvement section",
        "lands department",
        "september 2009 edition",
        "recognized villages in",
        "villages in",
        "non-indigenous",
        "village name",
        "district",
        "地政總署",
        "鄉村改善組",
        "二o一九年九月版",
        "二oo九年九月版",
        "位於",
        "村名",
        "地區",
    )
    district_header = re.compile(
        r"^(islands|north|sai kung|sha tin|tuen mun|tai po|tsuen wan|"
        r"kwai tsing|yuen long)\b",
        flags=re.IGNORECASE,
    )

    def flush() -> None:
        nonlocal current_indices, current_block_lines
        if current_indices and current_block_lines:
            blocks.append((current_indices, current_block_lines))
        current_indices = []
        current_block_lines = []

    for raw_line in raw_text.splitlines():
        line = raw_line.replace("|", "").strip()
        if not line:
            continue
        folded = line.casefold().replace("ｏ", "o").replace("Ｏ", "o")
        if any(marker in folded for marker in header_markers) or district_header.match(line):
            flush()
            continue

        match = re.match(r"^(\d+)\s+(.*)", line)
        if match:
            flush()
            current_indices = [int(match.group(1))]
            current_block_lines = [match.group(2).strip()]
        elif current_indices:
            current_block_lines.append(line)

    flush()
    return blocks


def extract_recognized_villages(raw_text: str) -> list[dict[str, str]]:
    normalized = normalize_text(raw_text)
    blocks = parse_blocks(normalized)
    results = []

    for indices, lines in blocks:
        combined_text = " ".join(lines).strip()
        if not combined_text:
            continue

        combined_text, chi_sub_district = extract_district_from_end(combined_text, KNOWN_DISTRICTS_ZH)
        combined_text, eng_sub_district = extract_district_from_end(combined_text, KNOWN_DISTRICTS_EN)

        eng_village = ""
        chi_village = ""

        match = re.match(r'^([a-zA-Z0-9\s\(\)\.,;&\-\'\/"]+)\s*(.*)$', combined_text)

        if match:
            eng_village = match.group(1).strip()
            chi_village = match.group(2).strip()
        else:
            eng_village = combined_text

        results.append({
            "eng_village": eng_village,
            "zh_village": chi_village,
            "eng_sub_district": eng_sub_district,
            "zh_sub_district": chi_sub_district,
            "eng_district": SUB_DISTRICT_TO_DISTRICT.get(eng_sub_district, ""),
            "zh_district": SUB_DISTRICT_TO_DISTRICT.get(chi_sub_district, "")
        })

    return results





# =============================================================================
# 3. HELPERS (stable RNG, weighted choice, clean, etc.)
# =============================================================================

def clean_text(value: Any) -> str:
    if value is None or isinstance(value, (dict, list, tuple)):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"[\r\n\t]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def stable_digest(*parts: Any) -> str:
    payload = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_rng(*parts: Any) -> random.Random:
    return random.Random(int(stable_digest(*parts)[:16], 16))


def weighted_choice(rng: random.Random, choices: Sequence[tuple[str, float | int]]) -> str:
    total = sum(w for _, w in choices)
    draw = rng.uniform(0, total)
    running = 0.0
    for value, weight in choices:
        running += weight
        if draw <= running:
            return value
    return choices[-1][0]


@dataclass
class Atom:
    text: str
    label: str | None = None


@dataclass
class Chunk:
    kind: str
    atoms: list[Atom]
    source_language: str | None = None


# =============================================================================
# 4. AUGMENTATION PRIMITIVES (ported from the main builder)
# =============================================================================

def apply_english_abbreviations(chunks: Sequence[Chunk], rng: random.Random) -> None:
    replacements = {w: rng.choice(opts) for w, opts in EN_ABBREVIATIONS.items()}
    for chunk in chunks:
        for atom in chunk.atoms:
            if not atom.text:
                continue
            original = atom.text
            for src, tgt in sorted(replacements.items(), key=lambda p: -len(p[0])):
                atom.text = re.sub(
                    rf"\b{re.escape(src)}\b", tgt, atom.text, flags=re.IGNORECASE
                )
            # If something was abbreviated, maybe add a trailing "." on the short form
            if atom.text != original and rng.random() < 0.30:
                for abbr in set(replacements.values()):
                    if abbr.endswith("."):
                        continue
                    atom.text = re.sub(
                        rf"\b{re.escape(abbr)}\b(?!\.)",
                        abbr + ".",
                        atom.text,
                        count=1,
                        flags=re.IGNORECASE,
                    )


def apply_english_case_noise(
    chunks: Sequence[Chunk],
    rng: random.Random,
    upper_rate: float,
    title_rate: float,
    lower_rate: float,
    mixed_rate: float,
) -> str:
    mode = weighted_choice(rng, [
        ("upper", upper_rate), ("title", title_rate),
        ("lower", lower_rate), ("mixed", mixed_rate),
    ])
    for chunk in chunks:
        if chunk.source_language != "en":
            continue
        for atom in chunk.atoms:
            if not atom.text or atom.label in {"FLOOR", "UNIT", "BUILDING_NUMBER"}:
                continue
            if mode == "lower":
                atom.text = atom.text.lower()
            elif mode == "upper":
                atom.text = atom.text.upper()
            elif mode == "title":
                atom.text = atom.text.upper() if len(atom.text) <= 2 and atom.text.isalpha() else atom.text.title()
            else:
                atom.text = "".join(
                    c.upper() if c.isalpha() and rng.random() < 0.35 else c.lower()
                    for c in atom.text
                )
    return mode


def apply_chinese_region_variation(
    chunks: Sequence[Chunk], fallback_language: str, rng: random.Random
) -> str | None:
    scenario = None
    for chunk in chunks:
        if chunk.kind != "region":
            continue
        lang = chunk.source_language or fallback_language
        if not lang.startswith("zh-"):
            continue
        var_map = CHINESE_REGION_VARIATIONS.get(lang, CHINESE_REGION_VARIATIONS["zh-Hant"])
        for atom in chunk.atoms:
            if atom.label == "REGION" and atom.text in var_map:
                new = rng.choice(var_map[atom.text])
                if new != atom.text:
                    atom.text = new
                    scenario = "chinese_region_variation"
    return scenario


def apply_region_abbreviation(chunks: Sequence[Chunk], rng: random.Random) -> bool:
    for chunk in chunks:
        if chunk.kind != "region" or chunk.source_language != "en":
            continue
        for atom in chunk.atoms:
            canonical = ENGLISH_REGION_CANONICAL.get(atom.text.upper(), atom.text)
            opts = ENGLISH_REGION_ABBREVIATIONS.get(canonical)
            if atom.label == "REGION" and opts:
                atom.text = rng.choice(opts)
                return True
    return False


def apply_district_suffix_noise(
    chunks: Sequence[Chunk], fallback_language: str, rng: random.Random
) -> str | None:
    scenario = None
    for chunk in chunks:
        if chunk.kind != "district":
            continue
        for atom in chunk.atoms:
            if atom.label != "DISTRICT" or not atom.text:
                continue
            text = atom.text
            lang = chunk.source_language or fallback_language
            is_en = lang == "en"
            if is_en:
                if re.search(r"(?i)\s+DISTRICT$", text):
                    atom.text = re.sub(r"(?i)\s+DISTRICT$", "", text)
                    scenario = "removed_district_suffix"
                else:
                    suffix = " DISTRICT" if text.isupper() else " District"
                    atom.text = text + suffix
                    scenario = "added_district_suffix"
            else:
                if text.endswith(("區", "区")):
                    atom.text = text[:-1]
                    scenario = "removed_district_suffix"
                else:
                    suffix = "区" if lang == "zh-Hans" else "區"
                    atom.text = text + suffix
                    scenario = "added_district_suffix"
    return scenario


def apply_localized_district(
    chunks: list[Chunk], effective_language: str, rng: random.Random
) -> str | None:
    lang_key = effective_language
    if lang_key.startswith("mixed-"):
        lang_key = lang_key.replace("mixed-", "").replace("-en", "")
    lang_map = SUB_DISTRICT_MAP.get(lang_key)
    if not lang_map:
        lang_map = (
            SUB_DISTRICT_MAP.get("en")
            if "en" in effective_language
            else SUB_DISTRICT_MAP.get("zh-Hant")
        )
    if not lang_map:
        return None

    district_idx = next((i for i, c in enumerate(chunks) if c.kind == "district"), -1)
    if district_idx < 0 or not chunks[district_idx].atoms:
        return None

    current = chunks[district_idx].atoms[0].text
    matched = None
    for official in lang_map:
        if official.upper() in current.upper() or current.upper() in official.upper():
            matched = official
            break
    if not matched:
        # also try without "District" / "區"
        clean = re.sub(r"(?i)\s*district$", "", current).strip()
        clean = clean.replace("區", "").replace("区", "")
        for official in lang_map:
            if clean.upper() in official.upper() or official.upper().startswith(clean.upper()):
                matched = official
                break
    if not matched:
        return None

    assigned = rng.choice(lang_map[matched])
    new_chunk = Chunk(
        kind="sub_district",
        atoms=[Atom(assigned, "SUB_DISTRICT")],
        source_language=chunks[district_idx].source_language,
    )
    chunks.append(new_chunk)
    return "synthetic_sub_district"


def trim_unlabelled_edge_atoms(atoms: Sequence[Atom]) -> list[Atom]:
    trimmed = list(atoms)
    while trimmed and trimmed[0].label is None and not trimmed[0].text.strip(" ,，、;/／.-"):
        trimmed.pop(0)
    while trimmed and trimmed[-1].label is None and not trimmed[-1].text.strip(" ,，、;/／.-"):
        trimmed.pop()
    return trimmed


def importance_weighted_omission(
    chunks: Sequence[Chunk], rng: random.Random
) -> tuple[list[Chunk], list[str]]:
    kinds_present = {c.kind for c in chunks}
    has_bn = any(
        any(a.label == "BUILDING_NUMBER" for a in c.atoms) for c in chunks
    )
    candidates = {
        k: w for k, w in COMPONENT_OMISSION_WEIGHTS.items()
        if k in kinds_present or (k == "building_number" and has_bn)
    }
    if not candidates:
        return list(chunks), []

    target = min(len(candidates), rng.choice([1, 1, 1, 1, 2, 2, 3]))
    omitted: list[str] = []
    anchors = {"street", "village", "estate", "block", "building"}

    while candidates and len(omitted) < target:
        sel = weighted_choice(rng, list(candidates.items()))
        candidates.pop(sel, None)
        proposed = {k for k in (set(omitted) | {sel}) if k != "building_number"}
        remaining = {c.kind for c in chunks if c.kind not in proposed}
        if not (remaining & anchors):
            continue
        omitted.append(sel)

    if not omitted:
        return list(chunks), []

    omitted_set = set(omitted)
    filtered: list[Chunk] = []
    for chunk in chunks:
        if chunk.kind in omitted_set:
            continue
        if "building_number" in omitted_set:
            new_atoms = trim_unlabelled_edge_atoms(
                [a for a in chunk.atoms if a.label != "BUILDING_NUMBER"]
            )
            if new_atoms:
                nc = copy.deepcopy(chunk)
                nc.atoms = new_atoms
                filtered.append(nc)
        else:
            filtered.append(chunk)
    return filtered, omitted


def reorder_chunks(chunks: list[Chunk], rng: random.Random) -> tuple[list[Chunk], str]:
    if len(chunks) < 2:
        return chunks, "reorder_not_applicable"
    mode = rng.choice(["district_last", "adjacent_swap", "section_reverse", "light_shuffle"])
    reordered = list(chunks)
    if mode == "district_last":
        tail = [c for c in reordered if c.kind in {"district", "region", "sub_district"}]
        head = [c for c in reordered if c.kind not in {"district", "region", "sub_district"}]
        if tail and head:
            reordered = head + tail
        else:
            mode = "adjacent_swap"
    if mode == "adjacent_swap":
        i = rng.randrange(len(reordered) - 1)
        reordered[i], reordered[i + 1] = reordered[i + 1], reordered[i]
    elif mode == "section_reverse":
        start = rng.randrange(0, len(reordered) - 1)
        end = rng.randrange(start + 2, len(reordered) + 1)
        reordered[start:end] = reversed(reordered[start:end])
    elif mode == "light_shuffle":
        for _ in range(min(2, len(reordered) - 1)):
            l, r = rng.randrange(len(reordered)), rng.randrange(len(reordered))
            reordered[l], reordered[r] = reordered[r], reordered[l]
    return reordered, mode


def apply_floor_unit_fusion(
    chunks: list[Chunk],
    three_d: Mapping[str, Any] | None,
    language: str,
    rng: random.Random,
) -> str | None:
    if not three_d:
        return None
    floor_idx = next((i for i, c in enumerate(chunks) if c.kind == "floor"), -1)
    unit_idx = next((i for i, c in enumerate(chunks) if c.kind == "unit"), -1)
    if floor_idx < 0 or unit_idx < 0:
        return None

    floor_chunk, unit_chunk = chunks[floor_idx], chunks[unit_idx]
    if (
        floor_chunk.source_language
        and unit_chunk.source_language
        and floor_chunk.source_language != unit_chunk.source_language
    ):
        return None

    chunk_lang = floor_chunk.source_language or unit_chunk.source_language or language
    is_en = chunk_lang == "en" or (chunk_lang and not str(chunk_lang).startswith("zh"))

    floor_num = clean_text(three_d.get("floor_num"))
    unit_no = clean_text(three_d.get("unit_no"))
    if not floor_num or not unit_no:
        return None

    floor_atoms = list(floor_chunk.atoms)
    unit_atoms = list(unit_chunk.atoms)
    if not floor_atoms or not unit_atoms:
        return None

    def is_safe(f: str, u: str) -> bool:
        if re.search(r"[A-Za-z]", f) or re.search(r"[A-Za-z]", u):
            return True
        return len(f) <= 2 and len(u) <= 2

    modes = [
        ("floor_then_unit_compact", 22), ("unit_then_floor_compact", 18),
        ("floor_then_unit_marker", 15), ("unit_then_floor_marker", 12),
        ("floor_then_unit_spaced", 10), ("unit_then_floor_spaced", 8),
    ]
    mode = weighted_choice(rng, modes)
    if mode in {"floor_then_unit_compact", "unit_then_floor_compact"} and not is_safe(floor_num, unit_no):
        mode = "floor_then_unit_spaced" if mode.startswith("floor") else "unit_then_floor_spaced"

    atoms: list[Atom] = []
    if mode in {"floor_then_unit_compact", "unit_then_floor_compact"}:
        # Force clear unit prefix on inverted order or pure-numeric units
        unit_is_numeric = not re.search(r"[A-Za-z]", unit_no)
        force_prefix = (mode == "unit_then_floor_compact") or unit_is_numeric

        if force_prefix:
            if is_en:
                prefix = rng.choice([
                    "Flat ", "Rm ", "Unit ", "FLT ", "RM ", "Apt ", "Room ",
                    "Flat", "Rm", "Unit", "FLT", "RM", "Apt",
                ])
                unit_text = prefix + unit_no
            else:
                is_hans = (chunk_lang or language or "").startswith("zh-Hans")
                desc = rng.choice(
                    ["室", "號室", "單位", "房"] if not is_hans
                    else ["室", "号室", "单位", "房"]
                )
                unit_text = unit_no + desc
            unit_atom = Atom(unit_text, "UNIT")
        else:
            unit_atom = Atom(unit_no, "UNIT")

        if mode == "floor_then_unit_compact":
            atoms = [Atom(floor_num, "FLOOR"), unit_atom]
        else:
            atoms = [unit_atom, Atom(floor_num, "FLOOR")]
    else:
        first = floor_atoms if mode.startswith("floor") else unit_atoms
        second = unit_atoms if mode.startswith("floor") else floor_atoms
        atoms = list(first)
        need_space = mode.endswith("_spaced") or (is_en and mode.endswith("_marker"))
        if need_space:
            ends_sp = atoms and atoms[-1].text and atoms[-1].text[-1].isspace()
            starts_sp = second and second[0].text and second[0].text[0].isspace()
            if not ends_sp and not starts_sp:
                atoms.append(Atom(" ", None))
        atoms.extend(second)

    if not atoms:
        return None

    new_chunk = Chunk(kind="unit", atoms=atoms, source_language=chunk_lang)
    for idx in sorted((floor_idx, unit_idx), reverse=True):
        del chunks[idx]
    chunks.insert(min(floor_idx, unit_idx), new_chunk)
    return f"floor_unit_fusion_{mode}"


def _latin_typo(text: str, rng: random.Random) -> tuple[str, str] | None:
    matches = list(re.finditer(r"\b[A-Z]?[a-z]{3,}\b", text))
    if not matches:
        return None
    m = rng.choice(matches)
    word = m.group(0)
    op = rng.choice(["delete", "transpose", "repeat", "keyboard_neighbour"])
    if op == "transpose" and len(word) >= 4:
        i = rng.randrange(1, len(word) - 1)
        changed = word[:i] + word[i + 1] + word[i] + word[i + 2 :]
    elif op == "repeat":
        i = rng.randrange(1, len(word))
        changed = word[:i] + word[i] + word[i:]
    elif op == "keyboard_neighbour":
        eligible = [i for i, c in enumerate(word) if c.casefold() in QWERTY_NEIGHBOURS]
        if not eligible:
            return None
        i = rng.choice(eligible)
        orig = word[i]
        repl = rng.choice(QWERTY_NEIGHBOURS[orig.casefold()])
        if orig.isupper():
            repl = repl.upper()
        changed = word[:i] + repl + word[i + 1 :]
    else:
        op = "delete"
        i = rng.randrange(1, len(word))
        changed = word[:i] + word[i + 1 :]
    return text[: m.start()] + changed + text[m.end() :], op


def _han_typo(text: str, rng: random.Random) -> tuple[str, str] | None:
    positions = [i for i, c in enumerate(text) if "\u3400" <= c <= "\u9fff"]
    if len(positions) < 2:
        return None
    i = rng.choice(positions)
    if rng.random() < 0.72:
        return text[:i] + text[i + 1 :], "delete_character"
    return text[:i] + text[i] + text[i:], "repeat_character"


def introduce_minor_typo(
    chunks: Sequence[Chunk], rng: random.Random
) -> dict[str, str] | None:
    eligible = {
        "DISTRICT", "STREET_NAME", "VILLAGE_NAME",
        "ESTATE_NAME", "BLOCK", "BUILDING_NAME", "SUB_DISTRICT",
    }
    candidates: list[tuple[Chunk, Atom]] = []
    for chunk in chunks:
        for atom in chunk.atoms:
            if atom.label not in eligible:
                continue
            if re.search(r"[A-Za-z]{4,}", atom.text) or len(re.findall(r"[\u3400-\u9fff]", atom.text)) >= 2:
                candidates.append((chunk, atom))
    if not candidates:
        return None
    chunk, atom = rng.choice(candidates)
    original = atom.text
    has_latin = bool(re.search(r"[A-Za-z]{4,}", atom.text))
    has_han = bool(re.search(r"[\u3400-\u9fff]", atom.text))
    use_latin = has_latin if not (has_latin and has_han) else rng.random() < 0.5
    changed = _latin_typo(atom.text, rng) if use_latin else _han_typo(atom.text, rng)
    if changed is None:
        return None
    atom.text, operation = changed
    return {
        "chunk": chunk.kind,
        "label": atom.label or "",
        "operation": operation,
        "original": original,
        "corrupted": atom.text,
    }


def canonical_order(chunks: Sequence[Chunk], language: str) -> list[Chunk]:
    priority_zh = {
        "region": 10, "district": 20, "sub_district": 25,
        "street": 40, "village": 45, "estate": 50, "phase": 55,
        "block": 60, "building": 65, "floor": 70, "unit": 80,
    }
    priority_en = {
        "unit": 10, "floor": 20, "block": 30, "building": 35, "phase": 40,
        "estate": 45, "village": 50, "street": 55,
        "sub_district": 65, "district": 70, "region": 80,
    }
    pri = priority_en if language == "en" else priority_zh
    return sorted(chunks, key=lambda c: pri.get(c.kind, 999))


def separator_for(style: str, language: str, rng: random.Random) -> str:
    if style == "canonical":
        return ", " if language == "en" else ""
    if style == "comma":
        return rng.choice([", ", ",", "; "]) if language == "en" else rng.choice(["，", "、", "， "])
    if style == "space":
        return rng.choice([" ", "  "])
    if style == "mixed":
        return rng.choice([", ", "，", "; ", " / ", "  "])
    if style == "compact":
        return " " if language == "en" else ""
    return " "


def render_chunks(
    chunks: Sequence[Chunk], separator: str
) -> tuple[str, list[dict[str, Any]]]:
    parts: list[str] = []
    entities: list[dict[str, Any]] = []
    cursor = 0
    for index, chunk in enumerate(chunks):
        if index:
            parts.append(separator)
            cursor += len(separator)
        for atom in chunk.atoms:
            start = cursor
            parts.append(atom.text)
            cursor += len(atom.text)
            if atom.label and atom.text:
                entities.append(
                    {
                        "start": start,
                        "end": cursor,
                        "label": atom.label,
                        "text": atom.text,
                        "component_kind": chunk.kind,
                    }
                )
    return "".join(parts), entities


def target_from_entities(entities: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Derive the target after every mutation, never from the pre-noise source."""

    values: dict[str, list[str]] = {field: [] for field in OUTPUT_FIELDS}
    seen: dict[str, set[str]] = {field: set() for field in OUTPUT_FIELDS}
    for entity in entities:
        field = LABEL_TO_FIELD.get(clean_text(entity.get("label")))
        value = entity.get("text")
        if not field or not isinstance(value, str) or not value:
            continue
        marker = value.casefold()
        if marker not in seen[field]:
            values[field].append(value)
            seen[field].add(marker)
    return {field: " / ".join(values[field]) for field in OUTPUT_FIELDS}


def format_output(values: Mapping[str, str], layout: str) -> dict[str, Any]:
    if layout == "flat":
        return {field: values.get(field, "") for field in OUTPUT_FIELDS}
    return {
        "line1": {
            field: values.get(field, "")
            for field in (
                "flat", "floor", "block", "building_name", "phase", "estate_name"
            )
        },
        "line2": {
            field: values.get(field, "")
            for field in (
                "building_number", "street_name", "village_name",
                "sub_district", "district", "region",
            )
        },
    }


# =============================================================================
# 5. SYNTHETIC 3-D (village-oriented, richer than original)
# =============================================================================

def synthetic_three_d(group_id: str, language: str, seed: int, key: Any) -> dict[str, str]:
    rng = stable_rng(seed, group_id, key, "synthetic_3d")
    floor_num_int = rng.randint(1, 45)
    floor_number = str(floor_num_int)

    # Weighted unit types – pure small numbers become rare
    unit_type = weighted_choice(rng, [
        ("letter", 40),
        ("number", 10),
        ("leading_zero", 10),
        ("combo", 20),
        ("floor_room", 20),  # realistic 4-digit
    ])
    if unit_type == "letter":
        letter = chr(rng.randint(65, 76))  # A–L
        # optional: ~40 % lower-case
        unit_number = letter.lower() if rng.random() < 0.4 else letter
    elif unit_type == "number":
        unit_number = str(rng.randint(1, 30))
    elif unit_type == "leading_zero":
        unit_number = f"{rng.randint(1, 15):02d}"
    elif unit_type == "combo":
        letter = chr(rng.randint(65, 68))
        unit_number = f"{rng.randint(1, 15)}{letter.lower() if rng.random() < 0.3 else letter}"
    else:  # floor_room
        unit_number = f"{floor_num_int}{rng.randint(1, 12):02d}"

    if rng.random() < 0.85:
        length = rng.choice([1, 2])
        alphabetic_unit = "".join(rng.choice(string.ascii_uppercase) for _ in range(length))
    else:
        alphabetic_unit = rng.choice([
            "LA", "LB", "LC", "LD", "LE", "AA", "AB", "BA", "PA", "PH",
            "ZZ", "XX", "QQ", "ME", "RF", "SK", "CP", "UG", "LG",
        ])

    comb_style = rng.choice(["letter_amp", "letter_dash", "num_dash", "num_slash"])
    if comb_style == "letter_amp":
        l1 = chr(rng.randint(65, 74))
        l2 = chr(ord(l1) + 1)
        combined_en, combined_zh = f"{l1} & {l2}", f"{l1}及{l2}"
    elif comb_style == "letter_dash":
        l1 = chr(rng.randint(65, 74))
        l2 = chr(ord(l1) + 1)
        combined_en = combined_zh = f"{l1}-{l2}"
    elif comb_style == "num_dash":
        n1 = rng.randint(1, 15)
        combined_en, combined_zh = f"{n1:02d}-{n1+1:02d}", f"{n1:02d}至{n1+1:02d}"
    else:
        n1 = rng.randint(1, 15)
        combined_en = combined_zh = f"{n1}/{n1+1}"

    leading_zero_unit = rng.choice(["01", "02", "03", "05", "06", "08", "09"])
    dyn_room = (
        str(rng.randint(101, 3508))
        if rng.random() < 0.88
        else str(rng.randint(1, 50))
    )
    case = weighted_choice(rng, VILLAGE_3D_WEIGHTS)

    if language == "en":
        unit_desc_pool = [
            "FLAT", "Flat", "flat", "UNIT", "Unit", "unit",
            "ROOM", "Room", "room", "RM", "Rm", "rm", "FLT", "APT", "Apt",
        ]
        floor_desc_pool = ["/F", "/f", "F", "FL", "FLOOR", "Floor", "floor", "FLR"]
        profile = {
            "floor_num": floor_number,
            "floor_description": rng.choice(floor_desc_pool),
            "unit_descriptor": rng.choice(unit_desc_pool),
            "unit_no": unit_number,
            "unit_portion": "",
        }
        overrides = {
            "number_only_unit": {"unit_descriptor": "", "unit_no": dyn_room},
            "hao_shi_unit": {
                "unit_descriptor": rng.choice(["RM", "Rm", "ROOM", "Room", "NO.", "No.", "UNIT", "FLAT", "#"]),
                "unit_no": dyn_room,
            },
            "standard_room": {"unit_descriptor": rng.choice(["RM", "Rm", "ROOM", "Room"])},
            "ground_floor_no_unit": {
                "floor_num": "", "floor_description": rng.choice(["G/F", "GF", "GROUND FLOOR", "G.F."]),
                "unit_descriptor": "", "unit_no": "",
            },
            "whole_floor": {"unit_descriptor": "", "unit_no": ""},
            "unit_without_floor": {"floor_num": "", "floor_description": ""},
            "roof": {
                "floor_num": "", "floor_description": rng.choice(["ROOF", "R/F", "ROOFTOP"]),
                "unit_descriptor": "", "unit_no": "",
            },
            "alphabetic_unit": {
                "unit_descriptor": rng.choice(["FLAT", "Flat", "UNIT", "Unit", ""]),
                "unit_no": alphabetic_unit,
            },
            "leading_zero_unit": {
                "unit_descriptor": rng.choice(["FLAT", "Flat", "UNIT", "ROOM", ""]),
                "unit_no": leading_zero_unit,
            },
            "duplex_floor": {
                "floor_num": "",
                "floor_description": rng.choice([
                    f"{floor_number}-{floor_num_int+1}/F",
                    f"{floor_number} & {floor_num_int+1}/F",
                    f"{floor_number}/F-{floor_num_int+1}/F",
                ]),
            },
            "combined_units": {
                "unit_descriptor": rng.choice(["FLATS", "Flats", "UNITS", "Units", "ROOMS", "FLAT"]),
                "unit_no": combined_en,
            },
            "special_floor": {
                "floor_num": "",
                "floor_description": rng.choice([
                    f"{floor_number}A/F", f"LEVEL {rng.randint(1,5)}", "PH/F", "PENTHOUSE",
                ]),
            },
            "basement": {
                "floor_num": "",
                "floor_description": rng.choice(["B1/F", "B2/F", "BASEMENT", "LG/F"]),
                "unit_descriptor": rng.choice(["SHOP", "UNIT", "ROOM", ""]),
                "unit_no": str(rng.randint(1, 30)),
            },
            "lower_upper_ground": {
                "floor_num": "",
                "floor_description": rng.choice(["LG/F", "UG/F", "M/F", "L1/F"]),
            },
        }
    else:
        is_simp = language == "zh-Hans"
        floor_suffix = "层" if is_simp else "樓"
        profile = {
            "floor_num": floor_number,
            "floor_description": floor_suffix,
            "unit_descriptor": rng.choice(["室", "單位" if not is_simp else "单位", "房"]),
            "unit_no": unit_number,
            "unit_portion": "",
        }
        overrides = {
            "number_only_unit": {"unit_descriptor": "", "unit_no": dyn_room},
            "hao_shi_unit": {
                "unit_descriptor": rng.choice([
                    "號室" if not is_simp else "号室",
                    "號單位" if not is_simp else "号单位",
                    "號房" if not is_simp else "号房",
                    "號" if not is_simp else "号",
                ]),
                "unit_no": dyn_room,
            },
            "standard_room": {"unit_descriptor": "室"},
            "ground_floor_no_unit": {
                "floor_num": "", "floor_description": "地下",
                "unit_descriptor": "", "unit_no": "",
            },
            "whole_floor": {"unit_descriptor": "", "unit_no": ""},
            "unit_without_floor": {"floor_num": "", "floor_description": ""},
            "roof": {
                "floor_num": "", "floor_description": "天台",
                "unit_descriptor": "", "unit_no": "",
            },
            "alphabetic_unit": {
                "unit_descriptor": rng.choice(["室", "單位" if not is_simp else "单位", ""]),
                "unit_no": alphabetic_unit,
            },
            "leading_zero_unit": {
                "unit_descriptor": rng.choice(["室", "單位" if not is_simp else "单位", ""]),
                "unit_no": leading_zero_unit,
            },
            "duplex_floor": {
                "floor_num": "",
                "floor_description": rng.choice([
                    f"{floor_number}-{floor_num_int+1}/F",
                    f"{floor_number}及{floor_num_int+1}{floor_suffix}",
                ]),
            },
            "combined_units": {"unit_descriptor": "室", "unit_no": combined_zh},
            "special_floor": {
                "floor_num": "",
                "floor_description": rng.choice([
                    f"{floor_number}A{floor_suffix}",
                    "頂層" if not is_simp else "顶层",
                    "閣樓" if not is_simp else "阁楼",
                ]),
            },
            "basement": {
                "floor_num": "",
                "floor_description": rng.choice(["地庫" if not is_simp else "地库", "B1/F"]),
                "unit_descriptor": rng.choice(["室", "舖" if not is_simp else "铺", ""]),
                "unit_no": str(rng.randint(1, 30)),
            },
            "lower_upper_ground": {
                "floor_num": "",
                "floor_description": (
                    ["低層地下", "高層地下", "閣樓"] if not is_simp
                    else ["低层地下", "高层地下", "阁楼"]
                )[rng.randrange(3)],
            },
        }
    profile.update(overrides.get(case, {}))
    profile["case"] = case
    return profile


def floor_text(three_d: Mapping[str, Any] | None, language: str) -> str:
    if not three_d:
        return ""
    number = clean_text(three_d.get("floor_num"))
    description = clean_text(three_d.get("floor_description"))
    if not number:
        return description
    if number.casefold() in description.casefold():
        return description
    if language == "en":
        if not description:
            return f"{number}/F"
        desc_u = description.upper()
        if desc_u in {"F", "/F", "FL", "FLOOR"}:
            suffix = "/f" if description.endswith(("f", "/f")) else "/F"
            if desc_u in {"F", "/F"}:
                return f"{number}{suffix}"
            return f"{number} {description}"
        return f"{number} {description}"
    if not description:
        description = "层" if language == "zh-Hans" else "樓"
    return f"{number}{description}"


def unit_text(three_d: Mapping[str, Any] | None, language: str) -> str:
    if not three_d:
        return ""
    descriptor = clean_text(three_d.get("unit_descriptor"))
    number = clean_text(three_d.get("unit_no"))
    portion = clean_text(three_d.get("unit_portion"))
    if language == "en":
        return " ".join(p for p in (descriptor, number, portion) if p)
    return "".join(p for p in (number, descriptor, portion) if p)


def infer_region(district: str, language: str) -> str:
    """Infer a canonical region from the official district without false noise."""

    normalized = clean_text(district).upper()
    normalized = re.sub(r"\s+DISTRICT$", "", normalized).strip()
    normalized = normalized.removesuffix("區").removesuffix("区").strip()
    hk_island = {
        "SOUTHERN", "WAN CHAI", "EASTERN", "CENTRAL & WESTERN",
        "南", "灣仔", "湾仔", "東", "东", "中西",
    }
    kowloon = {
        "KWUN TONG", "WONG TAI SIN", "KOWLOON CITY", "SHAM SHUI PO",
        "YAU TSIM MONG", "觀塘", "观塘", "黃大仙", "黄大仙", "九龍城",
        "九龙城", "深水埗", "油尖旺",
    }
    if normalized in hk_island:
        return "Hong Kong" if language == "en" else "香港"
    if normalized in kowloon:
        if language == "zh-Hans":
            return "九龙"
        return "Kowloon" if language == "en" else "九龍"
    return "New Territories" if language == "en" else "新界"


def canonical_english_region(value: str) -> str:
    normalized = clean_text(value).upper()
    canonical = ENGLISH_REGION_CANONICAL.get(normalized)
    if canonical:
        return canonical
    if any(marker in value for marker in ("新界",)):
        return "New Territories"
    if any(marker in value for marker in ("九龍", "九龙")):
        return "Kowloon"
    if any(marker in value for marker in ("香港", "港島", "港岛")):
        return "Hong Kong"
    return ""


# =============================================================================
# 6. VILLAGE AUGMENTER (now with full logic)
# =============================================================================

class VillageAugmenter:
    def __init__(self, seed: int, config: argparse.Namespace):
        self.seed = seed
        self.rng = random.Random(seed)
        self.config = config

    def _convert(self, text: str, lang: str) -> str:
        if lang == "zh-Hans" and text:
            if CC_CONVERTER:
                return CC_CONVERTER.convert(text)
            return (
                text.replace("樓", "楼").replace("約", "约").replace("號", "号")
                .replace("區", "区").replace("離", "离").replace("灣", "湾")
                .replace("餘", "余").replace("鄉", "乡").replace("圍", "围")
            )
        return text

    def generate_building(self, lang: str) -> tuple[str, str]:
        """Returns (building_name, building_number)."""
        is_en = lang == "en"
        if self.rng.random() < self.config.dd_lot_rate:
            dd = self.rng.randint(1, 400)
            lot = self.rng.randint(1, 6000)
            suffix_data = self.rng.choices(
                [("", ""), ("A", "A"), ("B", "B"), ("C", "C"),
                 (" S.A", "A分段"), (" Sec. A", "A段"), (" RP", "餘段"),
                 (" R.P.", "餘段"), (" SS1", "第1小分段")],
                weights=[50, 10, 10, 5, 10, 5, 5, 3, 2],
            )[0]
            s_en, s_zh = suffix_data
            if is_en:
                templates = [
                    f"D.D. {dd} Lot {lot}{s_en}", f"DD {dd} Lot {lot}{s_en}",
                    f"Lot {lot}{s_en} in D.D. {dd}", f"Lot No. {lot}{s_en}, D.D. {dd}",
                    f"DD{dd}LOT{lot}{s_en.strip()}", f"Lot {lot}{s_en}",
                ]
            else:
                templates = [
                    f"丈量約份第{dd}約地段第{lot}{s_zh}號",
                    f"丈量約份第{dd}約地段{lot}{s_zh}號",
                    f"第{dd}約地段{lot}{s_zh}號", f"DD{dd}地段{lot}{s_zh}號",
                    f"地段第{lot}號{s_zh}", f"地段{lot}{s_zh}號",
                ]
            name = self.rng.choice(templates)
            return "", self._convert(name, lang)
        else:
            num = self.rng.randint(1, 400)
            letter = (
                self.rng.choice(string.ascii_uppercase[:6])
                if self.rng.random() < 0.25 else ""
            )
            if is_en:
                templates = [
                    f"House {num}{letter}", f"House No. {num}{letter}",
                    f"Hse {num}{letter}", f"No. {num}{letter}", f"{num}{letter}",
                ]
                return "", self.rng.choice(templates)
            else:
                templates = [
                    f"{num}{letter}號屋", f"{num}{letter}號村屋",
                    f"村屋{num}{letter}號", f"屋{num}{letter}號",
                    f"{num}號{letter}", f"{num}{letter}號",
                ]
                return "", self._convert(self.rng.choice(templates), lang)

    def build_chunks_from_pair(
        self,
        pair: dict[str, str],
        language: str,
        three_d: Mapping[str, Any] | None,
        group_id: str,
        variant_index: int,
    ) -> list[Chunk]:
        """Build source-backed chunks; the target is derived only after noise."""

        prefix = "eng" if language == "en" else "zh"
        raw_village = self._convert(pair[f"{prefix}_village"], language)
        raw_district = self._convert(pair[f"{prefix}_district"], language)
        raw_sub = self._convert(pair.get(f"{prefix}_sub_district", "") or "", language)

        if variant_index > 0 and self.rng.random() < self.config.village_suffix_rate:
            if language == "en" and not raw_village.upper().endswith(("VILLAGE", "TSUEN")):
                raw_village = f"{raw_village} {self.rng.choice(['Village', 'Tsuen'])}"
            elif language != "en" and not raw_village.endswith(("村", "鄉", "乡", "围", "圍")):
                raw_village = f"{raw_village}村"

        village = raw_village
        district = raw_district
        sub_dist = raw_sub
        region = infer_region(district, language) if district else ""

        b_name, b_number = self.generate_building(language)
        floor = floor_text(three_d, language)
        unit = unit_text(three_d, language)

        phase = ""
        block = ""
        if self.rng.random() < self.config.block_rate:
            blk = self.rng.choice(list(string.ascii_uppercase[:5]) + [str(i) for i in range(1, 6)])
            block = f"Block {blk}" if language == "en" else f"第{blk}座"
        if self.rng.random() < self.config.phase_rate:
            ph = self.rng.choice(["1", "2", "3", "I", "II"])
            phase = f"Phase {ph}" if language == "en" else f"第{ph}期"

        def add(kind: str, text: str, label: str) -> Chunk | None:
            if not text:
                return None
            return Chunk(kind, [Atom(text, label)], language)

        chunks: list[Chunk] = []
        for kind, text, label in [
            ("region", region, "REGION"),
            ("district", district, "DISTRICT"),
            ("sub_district", sub_dist, "SUB_DISTRICT"),
            ("building", b_name, "BUILDING_NAME"),
            ("phase", phase, "PHASE"),
            ("block", block, "BLOCK"),
            ("floor", floor, "FLOOR"),
            ("unit", unit, "UNIT"),
        ]:
            c = add(kind, text, label)
            if c:
                chunks.append(c)

        if language == "en":
            village_atoms = []
            if b_number:
                village_atoms.append(Atom(b_number, "BUILDING_NUMBER"))
            if b_number and village:
                village_atoms.append(Atom(" "))
            if village:
                village_atoms.append(Atom(village, "VILLAGE_NAME"))
        else:
            village_atoms = []
            if village:
                village_atoms.append(Atom(village, "VILLAGE_NAME"))
            if b_number:
                village_atoms.append(Atom(b_number, "BUILDING_NUMBER"))
        if village_atoms:
            chunks.append(Chunk("village", village_atoms, language))

        return canonical_order(chunks, language)

    def build_variant(self, pair: dict[str, str], group_id: str, variant_index: int) -> dict[str, Any]:
        # Reset per record so a row is reproducible even if source order changes.
        self.rng = stable_rng(self.seed, group_id, variant_index, "source_generation")
        language_options = ["en", "zh-Hant"]
        language_weights = [
            self.config.english_weight,
            self.config.traditional_weight,
        ]
        if self.config.simplified:
            language_options.append("zh-Hans")
            language_weights.append(self.config.simplified_weight)
        language = self.rng.choices(language_options, weights=language_weights)[0]

        three_d = None
        if self.rng.random() < self.config.synthetic_3d_rate:
            three_d = synthetic_three_d(group_id, language, self.seed, (variant_index,))

        chunks = self.build_chunks_from_pair(
            pair, language, three_d, group_id, variant_index
        )

        scenarios: list[str] = []
        rng = stable_rng(self.seed, group_id, variant_index, language, "variant")

        already_has_sub = any(c.kind == "sub_district" for c in chunks)
        if (
            variant_index > 0
            and not already_has_sub
            and rng.random() < self.config.synthetic_subdistrict_rate
        ):
            local_sc = apply_localized_district(chunks, language, rng)
            if local_sc:
                scenarios.append(local_sc)
        chunks = canonical_order(chunks, language)

        if (
            variant_index > 0
            and rng.random() < self.config.district_suffix_noise_rate
        ):
            suf_sc = apply_district_suffix_noise(chunks, language, rng)
            if suf_sc:
                scenarios.append(suf_sc)

        if (
            variant_index > 0
            and language.startswith("zh")
            and rng.random() < self.config.chinese_separator_noise_rate
        ):
            for chunk in chunks:
                if chunk.kind == "unit" and chunk.atoms:
                    messy = rng.choice([" ", "  ", " / ", " - ", "，"])
                    chunk.atoms.insert(0, Atom(messy, None))
                if chunk.kind in {"floor", "unit"}:
                    for atom in chunk.atoms:
                        if atom.label and rng.random() < 0.5:
                            atom.text = re.sub(
                                r"(\d+)([\u4e00-\u9fa5]+)", r"\1 \2", atom.text
                            )
            scenarios.append("chinese_messy_separation")

        # --- Structural mess / reorder / abbreviations ---
        is_messy = False
        if variant_index > 0 and rng.random() < self.config.messy_rate:
            is_messy = True
            ops = ["reorder", "mixed_punctuation", "space_noise"]
            if language == "en":
                ops.append("abbreviation")
            for op in rng.sample(ops, min(rng.choice([1, 2, 2, 3]), len(ops))):
                if op == "reorder":
                    chunks, mode = reorder_chunks(chunks, rng)
                    scenarios.extend(["reordered_components", mode])
                elif op == "mixed_punctuation":
                    scenarios.append("mixed_punctuation")
                elif op == "space_noise":
                    scenarios.append("irregular_whitespace")
                elif op == "abbreviation":
                    apply_english_abbreviations(chunks, rng)
                    scenarios.append("english_abbreviation")
            sep_style = "mixed" if "mixed_punctuation" in scenarios else "space"
        else:
            sep_style = "canonical" if variant_index == 0 else rng.choice(["comma", "space", "compact"])
            scenarios.append({
                "canonical": "canonical_order",
                "comma": "comma_separated",
                "space": "no_comma_space_separated",
                "compact": "compact_no_comma",
            }.get(sep_style, "canonical_order"))

        # --- English case noise ---
        if variant_index > 0 and any(c.source_language == "en" for c in chunks):
            case_mode = apply_english_case_noise(
                chunks, rng,
                self.config.english_upper_rate,
                self.config.english_title_rate,
                self.config.english_lower_rate,
                self.config.english_mixed_rate,
            )
            scenarios.append(f"english_case_{case_mode}")

        # --- Importance-weighted omission ---
        omitted: list[str] = []
        if variant_index > 0 and rng.random() < self.config.component_drop_rate:
            chunks, omitted = importance_weighted_omission(chunks, rng)
            if omitted:
                scenarios.append("importance_weighted_component_omission")
                scenarios.extend(f"missing_{k}" for k in omitted)

        # --- Region abbreviation ---
        if variant_index > 0 and rng.random() < self.config.region_abbreviation_rate:
            if apply_region_abbreviation(chunks, rng):
                scenarios.append("english_region_abbreviation")

        if (
            variant_index > 0
            and rng.random() < self.config.chinese_region_variation_rate
        ):
            cr = apply_chinese_region_variation(chunks, language, rng)
            if cr:
                scenarios.append(cr)

        fusion_rate = (
            self.config.floor_unit_fusion_rate
            if variant_index > 0
            else self.config.canonical_floor_unit_fusion_rate
        )
        if rng.random() < fusion_rate:
            fus = apply_floor_unit_fusion(chunks, three_d, language, rng)
            if fus:
                scenarios.append(fus)
                scenarios.append("floor_unit_compact_or_inverted")

        # --- Full-width punctuation ---
        separator = separator_for(sep_style, language, rng)
        if "irregular_whitespace" in scenarios:
            separator = rng.choice(["  ", " ,", ",  ", "， ", " / "])
        if variant_index > 0 and rng.random() < self.config.fullwidth_punctuation_rate:
            separator = separator.translate(FULLWIDTH_TRANSLATION)
            for chunk in chunks:
                for atom in chunk.atoms:
                    atom.text = atom.text.translate(FULLWIDTH_TRANSLATION)
            scenarios.append("fullwidth_punctuation")

        # --- Minor typo (names only) ---
        corruption = None
        if variant_index > 0 and rng.random() < self.config.typo_rate:
            corruption = introduce_minor_typo(chunks, rng)
            if corruption:
                scenarios.extend(["minor_typo", f"typo_{corruption['operation']}"])

        output_language = language
        if (
            variant_index > 0
            and self.config.mixed_rate > 0
            and rng.random() < self.config.mixed_rate
            and language.startswith("zh")
        ):
            for chunk in chunks:
                if chunk.kind == "region" and chunk.atoms:
                    english_region = canonical_english_region(chunk.atoms[0].text)
                    options = ENGLISH_REGION_ABBREVIATIONS.get(
                        english_region
                    )
                    if options:
                        chunk.atoms[0].text = rng.choice(options)
                        chunk.source_language = "en"
                        scenarios.append("code_switched_region")
                        output_language = f"mixed-{language}-en"
                        break

        text, entities = render_chunks(chunks, separator)
        text = text.strip()
        target = target_from_entities(entities)
        for field, value in target.items():
            if value and any(part not in text for part in value.split(" / ")):
                raise ValueError(
                    f"Output field {field!r} contains text absent from input: {value!r}"
                )

        return {
            "input": text,
            "output": format_output(target, self.config.output_layout),
            "language": output_language,
            "scenario": list(dict.fromkeys(scenarios)),
            "is_messy": (
                is_messy
                or bool(omitted)
                or corruption is not None
                or any(
                    scenario
                    in {
                        "chinese_messy_separation",
                        "added_district_suffix",
                        "removed_district_suffix",
                        "chinese_region_variation",
                        "code_switched_region",
                        "floor_unit_compact_or_inverted",
                    }
                    for scenario in scenarios
                )
            ),
            "has_typo": corruption is not None,
            "omitted_component_kinds": omitted,
            "three_d_case": (three_d or {}).get("case", "no_floor_no_unit"),
        }


# =============================================================================
# 7. MAIN
# =============================================================================


def _dedup_pairs(pairs: list[dict[str, str]]) -> list[dict[str, str]]:
    """Deduplicate names within a district without merging same-name villages."""

    seen: dict[tuple[str, str, str], dict[str, str]] = {}

    def normalized(value: str) -> str:
        return re.sub(r"[^0-9a-z\u3400-\u9fff]", "", value.casefold())

    for p in pairs:
        eng_name = normalized(p.get("eng_village") or "")
        zh_name = normalized(p.get("zh_village") or "")
        district = normalized(
            p.get("eng_district")
            or p.get("zh_district")
            or p.get("eng_sub_district")
            or p.get("zh_sub_district")
            or ""
        )
        if not eng_name and not zh_name:
            continue
        key = (eng_name, zh_name, district)
        existing = seen.get(key)
        if existing is None:
            seen[key] = p
            continue
        # Prefer the record that already has a real sub-district
        has_sub = bool(p.get("eng_sub_district") or p.get("zh_sub_district"))
        had_sub = bool(existing.get("eng_sub_district") or existing.get("zh_sub_district"))
        if has_sub and not had_sub:
            seen[key] = p
        elif has_sub == had_sub:
            # Prefer the one with a district filled
            if (p.get("eng_district") or p.get("zh_district")) and not (
                existing.get("eng_district") or existing.get("zh_district")
            ):
                seen[key] = p
    return list(seen.values())


def main(argv: Sequence[str] | None = None) -> int:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Village address synthesis V3 – dual gazetteer with exact post-noise targets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "processed_data" / "auto_train_villageV3.jsonl",
    )
    parser.add_argument(
        "--gazetteer-text",
        type=Path,
        help="Optional UTF-8 text extracted from Place_Name_Gazetteer.pdf",
    )
    parser.add_argument(
        "--recognized-villages-text",
        type=Path,
        help="Optional UTF-8 text extracted from rv0909.pdf",
    )
    parser.add_argument("--variants-per-village", type=int, default=15)
    parser.add_argument("--max-villages", type=int, help="Optional development limit")
    parser.add_argument("--dd-lot-rate", type=float, default=0.35)
    parser.add_argument("--drop-rate", type=float, default=None,
                        help="Legacy alias; prefer --component-drop-rate")
    parser.add_argument("--component-drop-rate", type=float, default=0.08)
    parser.add_argument("--messy-rate", type=float, default=0.04)
    parser.add_argument("--typo-rate", type=float, default=0.02)
    parser.add_argument("--region-abbreviation-rate", type=float, default=0.22)
    parser.add_argument("--fullwidth-punctuation-rate", type=float, default=0.1)
    parser.add_argument("--chinese-separator-noise-rate", type=float, default=0.10)
    parser.add_argument("--synthetic-subdistrict-rate", type=float, default=0.0)
    parser.add_argument("--district-suffix-noise-rate", type=float, default=0.10)
    parser.add_argument("--chinese-region-variation-rate", type=float, default=0.10)
    parser.add_argument("--floor-unit-fusion-rate", type=float, default=0.20)
    parser.add_argument("--canonical-floor-unit-fusion-rate", type=float, default=0.0)
    parser.add_argument("--synthetic-3d-rate", type=float, default=0.85)
    parser.add_argument("--village-suffix-rate", type=float, default=0.50)
    parser.add_argument("--block-rate", type=float, default=0.08)
    parser.add_argument("--phase-rate", type=float, default=0.02)
    parser.add_argument("--english-upper-rate", type=float, default=0.15)
    parser.add_argument("--english-title-rate", type=float, default=0.25)
    parser.add_argument("--english-lower-rate", type=float, default=0.55)
    parser.add_argument("--english-mixed-rate", type=float, default=0.05)
    parser.add_argument("--mixed-rate", type=float, default=0.05,
                        help="Probability of switching a visible Chinese region to English")
    parser.add_argument("--english-weight", type=float, default=1.0)
    parser.add_argument("--traditional-weight", type=float, default=1.0)
    parser.add_argument("--simplified-weight", type=float, default=1.0)
    parser.add_argument(
        "--simplified",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate Simplified Chinese with OpenCC",
    )
    parser.add_argument(
        "--output-layout", choices=("nested", "flat"), default="nested"
    )
    parser.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)

    if args.drop_rate is not None:
        args.component_drop_rate = args.drop_rate

    probability_options = {
        "--dd-lot-rate": args.dd_lot_rate,
        "--component-drop-rate": args.component_drop_rate,
        "--messy-rate": args.messy_rate,
        "--typo-rate": args.typo_rate,
        "--region-abbreviation-rate": args.region_abbreviation_rate,
        "--fullwidth-punctuation-rate": args.fullwidth_punctuation_rate,
        "--chinese-separator-noise-rate": args.chinese_separator_noise_rate,
        "--synthetic-subdistrict-rate": args.synthetic_subdistrict_rate,
        "--district-suffix-noise-rate": args.district_suffix_noise_rate,
        "--chinese-region-variation-rate": args.chinese_region_variation_rate,
        "--floor-unit-fusion-rate": args.floor_unit_fusion_rate,
        "--canonical-floor-unit-fusion-rate": args.canonical_floor_unit_fusion_rate,
        "--synthetic-3d-rate": args.synthetic_3d_rate,
        "--village-suffix-rate": args.village_suffix_rate,
        "--block-rate": args.block_rate,
        "--phase-rate": args.phase_rate,
        "--mixed-rate": args.mixed_rate,
    }
    for option, value in probability_options.items():
        if not (0.0 <= value <= 1.0):
            parser.error(f"{option} must be between 0 and 1")
    if args.variants_per_village < 1:
        parser.error("--variants-per-village must be at least 1")
    if args.max_villages is not None and args.max_villages < 1:
        parser.error("--max-villages must be at least 1")
    case_weights = (
        args.english_upper_rate,
        args.english_title_rate,
        args.english_lower_rate,
        args.english_mixed_rate,
    )
    if any(weight < 0 for weight in case_weights) or sum(case_weights) <= 0:
        parser.error("English case weights must be non-negative with a positive sum")
    language_weights = (
        args.english_weight,
        args.traditional_weight,
        args.simplified_weight if args.simplified else 0.0,
    )
    if any(weight < 0 for weight in language_weights) or sum(language_weights) <= 0:
        parser.error("Language weights must be non-negative with a positive enabled sum")
    if args.simplified and args.simplified_weight > 0 and CC_CONVERTER is None:
        parser.error(
            "Simplified Chinese requires opencc-python-reimplemented; install it "
            "or pass --no-simplified"
        )

    def read_source(path: Path | None, embedded: str, option: str) -> str:
        if path is None:
            return embedded
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            parser.error(f"{option} does not exist: {resolved}")
        return resolved.read_text(encoding="utf-8")

    gazetteer_source = read_source(args.gazetteer_text, raw_text, "--gazetteer-text")
    recognized_source = read_source(
        args.recognized_villages_text,
        RECOGNIZED_VILLAGES_RAW,
        "--recognized-villages-text",
    )

    print("Parsing Place Name Gazetteer…")
    gaz_pairs = extract_bilingual_pairs(gazetteer_source)
    print(f"  → {len(gaz_pairs)} pairs from gazetteer")

    print("Parsing Recognized Villages list (Lands Dept / Small House Policy)…")
    recog_pairs = extract_recognized_villages(recognized_source)
    print(f"  → {len(recog_pairs)} pairs from recognized-villages list")

    base_pairs = _dedup_pairs(gaz_pairs + recog_pairs)
    if args.max_villages is not None:
        base_pairs = base_pairs[: args.max_villages]
    if not base_pairs:
        parser.error("No village/place pairs could be parsed from the supplied sources")
    print(f"Merged & deduplicated: {len(base_pairs)} unique base village/place pairs.")

    augmenter = VillageAugmenter(args.seed, args)
    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        parser.error(f"Output exists: {output_path}. Use --overwrite to replace it")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tqdm_factory = None
    if args.progress:
        try:
            from tqdm.auto import tqdm as tqdm_factory  # type: ignore
        except ImportError:
            parser.error("Progress display needs tqdm; install it or pass --no-progress")
    progress = (
        tqdm_factory(
            total=len(base_pairs) * args.variants_per_village,
            desc="Generating village addresses",
            unit=" rows",
            dynamic_ncols=True,
        )
        if tqdm_factory is not None
        else None
    )
    total = 0
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            for pair in base_pairs:
                group_id = f"village-{stable_digest(pair['eng_village'], pair['zh_village'])[:16]}"
                for variant_index in range(args.variants_per_village):
                    record = augmenter.build_variant(pair, group_id, variant_index)
                    handle.write(
                        json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    )
                    total += 1
                    if progress is not None:
                        progress.update(1)
        temporary_path.replace(output_path)
    finally:
        if progress is not None:
            progress.close()
        if temporary_path.exists():
            temporary_path.unlink()

    print(
        f"\nSuccess! Wrote {total} augmented village addresses "
        f"→ {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
