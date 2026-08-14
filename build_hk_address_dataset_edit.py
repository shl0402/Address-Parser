#!/usr/bin/env python3
"""Build an auditable multilingual Hong Kong address-component dataset.

The reader follows the Hong Kong Address Lookup Service (ALS) GeoJSON schema,
but deliberately tolerates small schema differences seen between releases.

Outputs are JSON Lines files suitable for a Hugging Face token-classification
pipeline.  Entity offsets are Unicode character offsets into ``text``; align
them to the chosen model tokenizer during model preprocessing.

Parallel government fields are also used to create separately identified
Chinese-English code-switched rows.  Sparse typos are confined to address
names, while floor/unit identifiers remain exact.

Government-provided 3D data and grammar-generated 3D examples are always
marked separately.  Synthetic floor/unit details teach address *parsing* only;
they must never be treated as verified premises or used as geocoding truth.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import logging
import random
import re
import shutil
import unicodedata
import zipfile
import string
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Mapping, Sequence

LOGGER = logging.getLogger("hk_address_dataset")
DATASET_SCHEMA_VERSION = "1.3.0"

LABELS = [
    "REGION",
    "DISTRICT",
    "SUB_DISTRICT",
    "STREET_NAME",
    "BUILDING_NUMBER",
    "VILLAGE_NAME",
    "ESTATE_NAME",
    "PHASE",
    "BLOCK",
    "BUILDING_NAME",
    "FLOOR",
    "UNIT",
]

LANGUAGE_ORDER = ("en", "zh-Hant", "zh-Hans")

KNOWN_PREMISES_KEYS = {
    "buildingcsuinformation",
    "chipremisesaddress",
    "engpremisesaddress",
    "geoaddress",
}

EN_ABBREVIATIONS = {
    "APARTMENT": ("APT",),
    "BUILDING": ("BLDG", "BLD",),
    "BLOCK": ("BLK",),
    "DISTRICT": ("DIST",),
    "FLAT": ("FLT", "UNIT", "RM"),
    "FLOOR": ("FL", "F", "/F", "FLR", "LVL", "LEVEL"),
    "HOUSE": ("HSE", "HS"),
    "ESTATE": ("EST",),
    "ROAD": ("RD",),
    "ROOM": ("RM",),
    "STREET": ("ST",),
    "TOWER": ("TWR",),
    "SUITE": ("STE", "SU"),
    "VILLAGE": ("VIL",),
}

# Synthetic patterns are deliberately weighted.  Rare identifiers and unusual
# floor forms are represented, but ordinary flats still remain the largest
# synthetic family.  These weights are not claims about Hong Kong prevalence.
STANDARD_3D_WEIGHTS: tuple[tuple[str, int], ...] = (
    ("standard_flat", 35),  # Heavily favor standard flats
    ("standard_room", 10),
    ("number_only_unit", 3),
    ("hao_shi_unit", 10),
    ("alphabetic_unit", 8),
    ("leading_zero_unit", 7),
    ("unit_without_floor", 2),
    ("office_suite", 3),
    ("whole_floor", 1),
    ("unit_portion", 1),
    ("duplex_floor", 1),
    ("combined_units", 3),
    ("special_floor", 2),
    ("shop", 1),
    ("basement", 2),
    ("podium", 1),
    ("lower_upper_ground", 2),
    ("car_park_space", 1),
)

VILLAGE_3D_WEIGHTS: tuple[tuple[str, int], ...] = (
    ("ground_floor_no_unit", 40),  # Heavily favor G/F for villages
    ("whole_floor", 30),  # Heavily favor whole floors
    ("roof", 15),  # Favor rooftops
    ("duplex_floor", 10),
    ("standard_flat", 5),
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
    "HK": "Hong Kong",
    "H.K.": "Hong Kong",
    "H. K.": "Hong Kong",
    "HONG KONG": "Hong Kong",
    "HONG KONG ISLAND": "Hong Kong",
    "HK ISLAND": "Hong Kong",
    "HKI": "Hong Kong",
    "KLN": "Kowloon",
    "KLN.": "Kowloon",
    "KOWLOON": "Kowloon",
    "NT": "New Territories",
    "N.T.": "New Territories",
    "N. T.": "New Territories",
    "N.T": "New Territories",
    "NEW TERRITORIES": "New Territories",
    # Common typo for normalization
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
        "新界": ["新界區", "新界"]
    },
    "zh-Hans": {
        "香港": ["香港岛", "港岛", "香港区", "香港"],
        "九龙": ["九龙区", "九龙半岛", "九龙"],
        "新界": ["新界区", "新界"]
    }
}

# Larger numbers mean a component is more likely to be absent from a user
# query.  Location-defining anchors receive lower weights and are only removed
# when another useful anchor remains.
COMPONENT_OMISSION_WEIGHTS: dict[str, int] = {
    "region": 100,
    "building_number": 50,  # High chance of dropping building numbers
    "sub_district": 35,
    "district": 36,
    "phase": 1,
    "building": 26,
    "street": 24,
    "estate": 18,
    "village": 1,
    "floor": 12,
    "block": 8,
    "unit": 5,
}

QWERTY_NEIGHBOURS = {
    "a": "sqwz",
    "b": "vghn",
    "c": "xdfv",
    "d": "serfcx",
    "e": "wsdr",
    "f": "drtgvc",
    "g": "ftyhbv",
    "h": "gyujnb",
    "i": "ujko",
    "j": "huikmn",
    "k": "jiolm",
    "l": "kop",
    "m": "njk",
    "n": "bhjm",
    "o": "iklp",
    "p": "ol",
    "q": "wa",
    "r": "edft",
    "s": "awedxz",
    "t": "rfgy",
    "u": "yhji",
    "v": "cfgb",
    "w": "qase",
    "x": "zsdc",
    "y": "tghu",
    "z": "asx",
}

SUB_DISTRICT_MAP = {
    "en": {
        # Hong Kong Island
        "CENTRAL & WESTERN DISTRICT": [
            "Central", "Admiralty", "Sheung Wan", "Sai Ying Pun", "Shek Tong Tsui",
            "Kennedy Town", "The Peak", "Mid-Levels",
            # Variations & Additions
            "Sheungwan", "Saiyingpun", "Shektongtsui", "Kennedytown", "Mid Levels", "Midlevels", "Peak", "Sai Wan",
            "SYP", "KT",
        ],
        "EASTERN DISTRICT": [
            "Taikoo", "North Point", "Quarry Bay", "Chai Wan", "Shau Kei Wan", "Fortress Hill",
            # Variations & Additions
            "Taikoo Shing", "Northpoint", "Quarrybay", "Chaiwan", "Shaukeiwan", "Fortresshill", "SWH", "SKW",
            "Heng Fa Chuen", "Hengfachuen", "Sai Wan Ho", "Saiwanho", "Siu Sai Wan", "Siusaiwan", "Braemar Hill", "NP"
        ],
        "SOUTHERN DISTRICT": [
            "Aberdeen", "Ap Lei Chau", "Wong Chuk Hang", "Repulse Bay", "Stanley", "Pok Fu Lam", "Cyberport",
            # Variations & Additions
            "Apleichau", "Wongchukhang", "Repulsebay", "Pokfulam", "Shek O", "Chung Hom Kok", "Deep Water Bay",
            "Tai Tam"
        ],
        "WAN CHAI DISTRICT": [
            "Wan Chai", "Causeway Bay", "Happy Valley", "Tin Hau", "Tai Hang",
            # Variations & Additions
            "Wanchai", "Causewaybay", "Happyvalley", "Tinhau", "Taihang", "CWB", "So Kon Po"
        ],

        # Kowloon
        "KOWLOON CITY DISTRICT": [
            "Kowloon City", "To Kwa Wan", "Hung Hom", "Ho Man Tin", "Kowloon Tong",
            # Variations & Additions
            "Kowlooncity", "Tokwawan", "Hunghom", "Homantin", "Kowloontong", "Kai Tak", "Kaitak", "Whampoa",
            "Kowloon Tsai", "KLT"
        ],
        "KWUN TONG DISTRICT": [
            "Kwun Tong", "Ngau Tau Kok", "Kowloon Bay", "Lam Tin", "Yau Tong",
            # Variations & Additions
            "Kwuntong", "Ngautaukok", "Kowloonbay", "Lamtin", "Yautong", "Sau Mau Ping", "Saumauping", "KLN Bay"
        ],
        "SHAM SHUI PO DISTRICT": [
            "Sham Shui Po", "Cheung Sha Wan", "Lai Chi Kok", "Mei Foo", "Shek Kip Mei",
            # Variations & Additions
            "Shamshuipo", "Cheungshawan", "Laichikok", "Meifoo", "Shekkipmei", "SSP", "Yau Yat Chuen",
            "Stonecutters Island"
        ],
        "WONG TAI SIN DISTRICT": [
            "Wong Tai Sin", "Diamond Hill", "Choi Hung", "San Po Kong", "Tsz Wan Shan",
            # Variations & Additions
            "Wongtaisin", "Diamondhill", "Choihung", "Sanpokong", "Tszwanshan", "Lok Fu", "Lokfu", "Wang Tau Hom", "WTS"
        ],
        "YAU TSIM MONG DISTRICT": [
            "Mong Kok", "Yau Ma Tei", "Tsim Sha Tsui", "Jordan", "Prince Edward", "Tai Kok Tsui", "Austin",
            # Variations & Additions
            "Mongkok", "Yaumatei", "Tsimshatsui", "Taikoktsui", "TST", "MK", "YMT", "PE", "West Kowloon"
        ],

        # New Territories
        "ISLANDS DISTRICT": [
            "Tung Chung", "Discovery Bay", "Chek Lap Kok", "Tai O", "Mui Wo", "Cheung Chau", "Lamma Island",
            "Peng Chau",
            # Variations & Additions
            "Tungchung", "Discoverybay", "Cheklapkok", "Taio", "Muiwo", "Cheungchau", "Lamma", "Pengchau", "DB",
            "Pui O", "Tong Fuk", "Lantau", "Lantau Island"
        ],
        "KWAI TSING DISTRICT": [
            "Kwai Fong", "Kwai Hing", "Kwai Chung", "Tsing Yi", "Lai King",
            # Variations & Additions
            "Kwaifong", "Kwaihing", "Kwaichung", "Tsingyi", "Laiking"
        ],
        "NORTH DISTRICT": [
            "Sheung Shui", "Fanling", "Luen Wo Hui", "Sha Tau Kok", "Ta Kwu Ling",
            # Variations & Additions
            "Sheungshui", "Luenwohui", "Shataukok", "Takwuling", "Kwu Tung", "Kwutung", "Queen's Hill", "Queens Hill",
            "Ping Che"
        ],
        "SAI KUNG DISTRICT": [
            "Sai Kung", "Tseung Kwan O", "Hang Hau", "Po Lam", "LOHAS Park", "Clear Water Bay", "Hang hau",
            # Variations & Additions
            "Saikung", "Tseungkwano", "Hanghau", "Polam", "Clearwater Bay", "Clearwaterbay", "TKO", "LOHAS",
            "Tiu Keng Leng", "Tiukengleng", "TKL", "HH"
        ],
        "SHA TIN DISTRICT": [
            "Sha Tin", "Tai Wai", "Fo Tan", "Ma On Shan", "Siu Lek Yuen", "Shek Mun",
            # Variations & Additions
            "Shatin", "Taiwai", "Fotan", "Maonshan", "Siulekyuen", "Shekmun", "MOS", "ST", "Wu Kai Sha", "Wukaisha",
            "Shatin Wai", "City One", "Hin Keng"
        ],
        "TAI PO DISTRICT": [
            "Tai Po Market", "Tai Wo", "Tolo Harbour", "Tai Mei Tuk", "Lam Tsuen",
            # Variations & Additions
            "Tai Po", "Taipo", "Taipo Market", "Taiwo", "Taimeituk", "Tai Mei Tok", "Lamtsuen", "Pak Shek Kok",
            "Pakshekkok", "Science Park", "TP"
        ],
        "TSUEN WAN DISTRICT": [
            "Tsuen Wan", "Tai Wo Hau", "Sham Tseng", "Ting Kau", "Ma Wan",
            # Variations & Additions
            "Tsuenwan", "Taiwohau", "Shamtseng", "Tingkau", "Mawan", "Tsing Lung Tau", "TW"
        ],
        "TUEN MUN DISTRICT": [
            "Tuen Mun", "Siu Hong", "Gold Coast", "Lam Tei", "So Kwun Wat", "Castle Peak",
            # Variations & Additions
            "Tuenmun", "Siuhong", "Goldcoast", "Lamtei", "Sokwunwat", "Castlepeak", "TM"
        ],
        "YUEN LONG DISTRICT": [
            "Yuen Long", "Tin Shui Wai", "Hung Shui Kiu", "Kam Tin", "San Tin", "Lau Fau Shan",
            # Variations & Additions
            "Yuenlong", "Tinshuiwai", "Hungshuikiu", "Kamtin", "Santin", "Laufaushan", "YL", "TSW", "Lok Ma Chau",
            "Lokmachau", "Fairview Park"
        ]
    },
    "zh-Hant": {
        # 香港島
        "中西區": ["中環", "金鐘", "上環", "西營盤", "石塘咀", "堅尼地城", "山頂", "半山", "西環"],
        "東區": ["太古", "北角", "鰂魚涌", "柴灣", "筲箕灣", "炮台山", "太古城", "杏花邨", "西灣河", "小西灣",
                 "寶馬山"],
        "南區": ["香港仔", "鴨脷洲", "黃竹坑", "淺水灣", "赤柱", "薄扶林", "數碼港", "鴨利洲", "石澳", "舂磡角",
                 "深水灣", "大潭"],
        "灣仔區": ["灣仔", "銅鑼灣", "跑馬地", "天后", "大坑", "掃桿埔"],

        # 九龍
        "九龍城區": ["九龍城", "土瓜灣", "紅磡", "何文田", "九龍塘", "啟德", "黃埔", "九龍仔"],
        "觀塘區": ["觀塘", "牛頭角", "九龍灣", "藍田", "油塘", "官塘", "秀茂坪"],
        "深水埗區": ["深水埗", "長沙灣", "荔枝角", "美孚", "石硤尾", "深水埔", "石夾尾", "又一村", "昂船洲"],
        "黃大仙區": ["黃大仙", "鑽石山", "彩虹", "新蒲崗", "慈雲山", "樂富", "橫頭磡"],
        "油尖旺區": ["旺角", "油麻地", "尖沙咀", "佐敦", "太子", "大角咀", "柯士甸", "尖沙嘴", "芒角", "西九龍"],

        # 新界
        "離島區": ["東涌", "愉景灣", "赤鱲角", "大澳", "梅窩", "長洲", "南丫島", "坪洲", "赤獵角", "赤臘角", "貝澳",
                   "塘福", "大嶼山"],
        "葵青區": ["葵芳", "葵興", "葵涌", "青衣", "荔景"],
        "北區": ["上水", "粉嶺", "聯和墟", "沙頭角", "打鼓嶺", "古洞", "皇后山", "坪輋"],
        "西貢區": ["西貢", "將軍澳", "坑口", "寶琳", "日出康城", "清水灣", "康城", "調景嶺"],
        "沙田區": ["沙田", "大圍", "火炭", "馬鞍山", "小瀝源", "石門", "烏溪沙", "沙田圍", "第一城", "顯徑"],
        "大埔區": ["大埔墟", "太和", "吐露港", "大尾篤", "林村", "大埔", "大尾督", "大美督", "白石角", "科學園"],
        "荃灣區": ["荃灣", "大窩口", "深井", "汀九", "馬灣", "青龍頭"],
        "屯門區": ["屯門", "兆康", "黃金海岸", "藍地", "掃管笏", "青山", "掃管忽"],
        "元朗區": ["元朗", "天水圍", "洪水橋", "錦田", "新田", "流浮山", "落馬洲", "錦繡花園"]
    },
    "zh-Hans": {
        # 香港岛
        "中西区": ["中环", "金钟", "上环", "西营盘", "石塘咀", "坚尼地城", "山顶", "半山", "西环"],
        "东区": ["太古", "北角", "鲗鱼涌", "柴湾", "筲湾", "炮台山", "太古城", "杏花邨", "西湾河", "小西湾", "宝马山"],
        "南区": ["香港仔", "鸭脷洲", "黄竹坑", "浅水湾", "赤柱", "薄扶林", "数码港", "鸭利洲", "石澳", "舂磡角",
                 "深水湾", "大潭"],
        "湾仔区": ["湾仔", "铜锣湾", "跑马地", "天后", "大坑", "扫杆埔"],

        # 九龙
        "九龙城区": ["九龙城", "土瓜湾", "红磡", "何文田", "九龙塘", "启德", "黄埔", "九龙仔"],
        "观塘区": ["观塘", "牛头角", "九龙湾", "蓝田", "油塘", "官塘", "秀茂坪"],
        "深水埗区": ["深水埗", "长沙湾", "荔枝角", "美孚", "石硖尾", "深水埔", "石夹尾", "又一村", "昂船洲"],
        "黄大仙区": ["黄大仙", "钻石山", "彩虹", "新蒲岗", "慈云山", "乐富", "横头磡"],
        "油尖旺区": ["旺角", "油麻地", "尖沙咀", "佐敦", "太子", "大角咀", "柯士甸", "尖沙嘴", "芒角", "西九龙"],

        # 新界
        "离岛区": ["东涌", "愉景湾", "赤鱲角", "大澳", "梅窝", "长洲", "南丫岛", "坪洲", "赤猎角", "赤腊角", "贝澳",
                   "塘福", "大屿山"],
        "葵青区": ["葵芳", "葵兴", "葵涌", "青衣", "荔景"],
        "北区": ["上水", "粉岭", "联和墟", "沙头角", "打鼓岭", "古洞", "皇后山", "坪輋"],
        "西贡区": ["西贡", "将军澳", "坑口", "宝琳", "日出康城", "清水湾", "康城", "调景岭"],
        "沙田区": ["沙田", "大围", "火炭", "马鞍山", "小沥源", "石门", "乌溪沙", "沙田围", "第一城", "显径"],
        "大埔区": ["大埔墟", "太和", "吐露港", "大尾笃", "林村", "大埔", "大尾督", "大美督", "白石角", "科学园"],
        "荃湾区": ["荃湾", "大窝口", "深井", "汀九", "马湾", "青龙头"],
        "屯门区": ["屯门", "兆康", "黄金海岸", "蓝地", "扫管笏", "青山", "扫管忽"],
        "元朗区": ["元朗", "天水围", "洪水桥", "锦田", "新田", "流浮山", "落马洲", "锦绣花园"]
    }
}

FULLWIDTH_TRANSLATION = str.maketrans(
    {
        ",": "，",
        ";": "；",
        ":": "：",
        "(": "（",
        ")": "）",
    }
)


@dataclass
class Atom:
    """A rendered text atom with an optional entity label."""

    text: str
    label: str | None = None


@dataclass
class Chunk:
    """One address component group that can be reordered as a unit."""

    kind: str
    atoms: list[Atom]
    source_language: str | None = None


@dataclass(frozen=True)
class SourceRef:
    display_name: str
    member_name: str | None = None

    def as_string(self) -> str:
        if self.member_name:
            return f"{self.display_name}!{self.member_name}"
        return self.display_name


class DatasetBuildError(RuntimeError):
    pass


def clean_text(value: Any) -> str:
    """Return a stable one-line string while preserving meaningful symbols."""

    if value is None or isinstance(value, (dict, list, tuple)):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"[\r\n\t]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def canonical_region(value: Any, language: str) -> str:
    raw = clean_text(value)
    if language == "en":
        normalized = re.sub(r"\s+", " ", raw.upper()).strip()
        return ENGLISH_REGION_CANONICAL.get(normalized, normalized)
    return raw


def key_normal_form(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def get_ci(mapping: Any, *names: str, default: Any = None) -> Any:
    """Case/punctuation-insensitive dictionary lookup."""

    if not isinstance(mapping, Mapping):
        return default
    wanted = {key_normal_form(name) for name in names}
    for key, value in mapping.items():
        if key_normal_form(key) in wanted:
            return value
    return default


def ensure_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Mapping):
        return [value]
    return []


def object_text(value: Any, *preferred_keys: str) -> str:
    """Extract a scalar from either a schema object or a scalar field."""

    if isinstance(value, Mapping):
        for key in preferred_keys:
            candidate = clean_text(get_ci(value, key))
            if candidate:
                return candidate
        scalar_values = [clean_text(v) for v in value.values()]
        scalar_values = [v for v in scalar_values if v]
        return scalar_values[0] if len(scalar_values) == 1 else ""
    return clean_text(value)


def stable_digest(*parts: Any) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_unit_interval(*parts: Any) -> float:
    return int(stable_digest(*parts)[:16], 16) / float(16 ** 16)


def stable_rng(*parts: Any) -> random.Random:
    return random.Random(int(stable_digest(*parts)[:16], 16))


def weighted_choice(rng: random.Random, choices: Sequence[tuple[str, float | int]]) -> str:
    total = sum(weight for _, weight in choices)
    draw = rng.uniform(0, total)
    running = 0.0
    for value, weight in choices:
        running += weight
        if draw <= running:
            return value
    return choices[-1][0]


def find_premises_address(feature: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Locate PremisesAddress, including across minor wrapper differences."""

    properties = ensure_mapping(get_ci(feature, "properties"))
    address = ensure_mapping(get_ci(properties, "Address"))
    premises = get_ci(address, "PremisesAddress")
    if isinstance(premises, Mapping):
        return premises

    direct = get_ci(properties, "PremisesAddress")
    if isinstance(direct, Mapping):
        return direct

    # Last-resort breadth-first search.  It is bounded to avoid walking a
    # malformed document forever and makes the reader tolerant of wrappers.
    queue: list[tuple[Any, int]] = [(properties, 0)]
    seen: set[int] = set()
    while queue:
        current, depth = queue.pop(0)
        if depth > 6 or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, Mapping):
            normalized = {key_normal_form(k) for k in current}
            if normalized & {"chipremisesaddress", "engpremisesaddress"}:
                return current
            for key, value in current.items():
                if key_normal_form(key) == "premisesaddress" and isinstance(
                        value, Mapping
                ):
                    return value
                if isinstance(value, (Mapping, list)):
                    queue.append((value, depth + 1))
        elif isinstance(current, list):
            for value in current[:20]:
                if isinstance(value, (Mapping, list)):
                    queue.append((value, depth + 1))
    return None


def extract_coordinates(feature: Mapping[str, Any]) -> dict[str, Any]:
    properties = ensure_mapping(get_ci(feature, "properties"))
    geometry = ensure_mapping(get_ci(feature, "geometry"))
    coords = get_ci(geometry, "coordinates")
    longitude: float | None = None
    latitude: float | None = None
    if (
            isinstance(coords, Sequence)
            and not isinstance(coords, (str, bytes))
            and len(coords) >= 2
    ):
        try:
            longitude = float(coords[0])
            latitude = float(coords[1])
        except (TypeError, ValueError):
            pass

    def numeric_or_none(value: Any) -> float | int | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return int(number) if number.is_integer() else number

    return {
        "longitude_wgs84": longitude,
        "latitude_wgs84": latitude,
        "easting_hk1980": numeric_or_none(get_ci(properties, "Easting")),
        "northing_hk1980": numeric_or_none(get_ci(properties, "Northing")),
    }


def district_text(address: Mapping[str, Any], field_name: str) -> str:
    return object_text(get_ci(address, field_name), "DcDistrict", "District")


def normalize_three_d_item(item: Any, language: str) -> dict[str, str]:
    item_map = ensure_mapping(item)
    prefix = "Eng" if language == "en" else "Chi"
    floor = ensure_mapping(get_ci(item_map, f"{prefix}Floor", "Floor"))
    unit = ensure_mapping(get_ci(item_map, f"{prefix}Unit", "Unit"))
    return {
        "floor_num": clean_text(get_ci(floor, "FloorNum", "FloorNumber")),
        "floor_description": clean_text(
            get_ci(floor, "FloorDescription", "FloorDescriptor")
        ),
        "unit_descriptor": clean_text(get_ci(unit, "UnitDescriptor")),
        "unit_no": clean_text(get_ci(unit, "UnitNo", "UnitNumber")),
        "unit_portion": clean_text(get_ci(unit, "UnitPortion")),
    }


def normalize_address_components(
        premises: Mapping[str, Any], language: str
) -> dict[str, Any]:
    """Flatten one ALS language object without losing its real 3D array."""

    if language == "en":
        address = ensure_mapping(get_ci(premises, "EngPremisesAddress"))
        district_field = "EngDistrict"
        street_field = "EngStreet"
        village_field = "EngVillage"
        estate_field = "EngEstate"
        phase_field = "EngPhase"
        block_field = "EngBlock"
        three_d_field = "Eng3dAddress"
    else:
        address = ensure_mapping(get_ci(premises, "ChiPremisesAddress"))
        district_field = "ChiDistrict"
        street_field = "ChiStreet"
        village_field = "ChiVillage"
        estate_field = "ChiEstate"
        phase_field = "ChiPhase"
        block_field = "ChiBlock"
        three_d_field = "Chi3dAddress"

    street = ensure_mapping(get_ci(address, street_field))
    village = ensure_mapping(get_ci(address, village_field))
    estate = ensure_mapping(get_ci(address, estate_field))
    phase = ensure_mapping(get_ci(estate, phase_field, "Phase"))
    block = ensure_mapping(get_ci(address, block_field))

    three_d_addresses = [
        normalize_three_d_item(item, language)
        for item in ensure_list(get_ci(address, three_d_field))
    ]
    three_d_addresses = [
        item for item in three_d_addresses if any(clean_text(v) for v in item.values())
    ]

    region_source = clean_text(get_ci(address, "Region"))
    return {
        "region": canonical_region(region_source, language),
        "region_source": region_source,
        "district": district_text(address, district_field),
        "street_location": clean_text(get_ci(street, "LocationName")),
        "street_name": clean_text(get_ci(street, "StreetName")),
        "street_no_from": clean_text(get_ci(street, "BuildingNoFrom")),
        "street_no_to": clean_text(get_ci(street, "BuildingNoTo")),
        "village_location": clean_text(get_ci(village, "LocationName")),
        "village_name": clean_text(get_ci(village, "VillageName")),
        "village_no_from": clean_text(get_ci(village, "BuildingNoFrom")),
        "village_no_to": clean_text(get_ci(village, "BuildingNoTo")),
        "estate_name": clean_text(get_ci(estate, "EstateName")),
        "phase_name": clean_text(get_ci(phase, "PhaseName")),
        "phase_no": clean_text(get_ci(phase, "PhaseNo")),
        "block_no": clean_text(get_ci(block, "BlockNo")),
        "block_descriptor": clean_text(get_ci(block, "BlockDescriptor")),
        "block_descriptor_precedes": clean_text(
            get_ci(block, "BlockDescriptorPrecedenceIndicator")
        ),
        "building_name": clean_text(get_ci(address, "BuildingName")),
        "three_d_addresses": three_d_addresses,
    }


def address_has_content(components: Mapping[str, Any]) -> bool:
    keys = (
        "region",
        "district",
        "street_name",
        "village_name",
        "estate_name",
        "block_no",
        "block_descriptor",
        "building_name",
    )
    return any(clean_text(components.get(key)) for key in keys)


def load_opencc() -> Any:
    try:
        from opencc import OpenCC  # type: ignore
    except ImportError as exc:
        raise DatasetBuildError(
            "Simplified Chinese output needs opencc-python-reimplemented. "
            "Install dependencies with: pip install -r requirements.txt, or "
            "run with --no-simplified."
        ) from exc
    return OpenCC("t2s")


def convert_nested_strings(value: Any, converter: Any) -> Any:
    if isinstance(value, str):
        return converter.convert(value)
    if isinstance(value, list):
        return [convert_nested_strings(item, converter) for item in value]
    if isinstance(value, Mapping):
        return {
            key: convert_nested_strings(item, converter) for key, item in value.items()
        }
    return value


def iter_leaf_paths(value: Any, prefix: str = "") -> Iterator[str]:
    if isinstance(value, Mapping):
        if not value:
            yield prefix or "<root>"
        for key, child in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_leaf_paths(child, next_prefix)
    elif isinstance(value, list):
        marker = f"{prefix}[]"
        if not value:
            yield marker
        else:
            for child in value[:3]:
                yield from iter_leaf_paths(child, marker)
    else:
        yield prefix or "<root>"


def import_ijson() -> Any | None:
    try:
        import ijson  # type: ignore

        return ijson
    except ImportError:
        return None


def load_tqdm() -> Any:
    try:
        from tqdm.auto import tqdm  # type: ignore
    except ImportError as exc:
        raise DatasetBuildError(
            "Progress display needs tqdm. Install dependencies with: "
            "pip install -r requirements.txt, or run with --no-progress."
        ) from exc
    return tqdm


def iter_features_from_binary(
        stream: BinaryIO,
        source_ref: SourceRef,
        *,
        size_hint: int | None,
        ijson_module: Any | None,
) -> Iterator[Mapping[str, Any]]:
    """Yield Feature objects from one FeatureCollection."""

    if ijson_module is not None:
        found_any = False
        try:
            for feature in ijson_module.items(stream, "features.item"):
                found_any = True
                if isinstance(feature, Mapping):
                    yield feature
            if found_any:
                return
        except Exception as exc:
            raise DatasetBuildError(
                f"Could not stream {source_ref.as_string()}: {exc}"
            ) from exc
        # Empty FeatureCollections are valid.  We do not rewind arbitrary ZIP
        # streams, so an empty result simply completes here.
        return

    if size_hint and size_hint >= 100 * 1024 * 1024:
        LOGGER.warning(
            "%s is %.1f MiB and ijson is not installed; built-in JSON loading may use "
            "several times the file size in RAM",
            source_ref.as_string(),
            size_hint / (1024 * 1024),
        )
    try:
        wrapper = io.TextIOWrapper(stream, encoding="utf-8-sig")
        root = json.load(wrapper)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetBuildError(
            f"Invalid JSON in {source_ref.as_string()}: {exc}"
        ) from exc

    if (
            isinstance(root, Mapping)
            and key_normal_form(get_ci(root, "type")) == "featurecollection"
    ):
        features = get_ci(root, "features", default=[])
    elif (
            isinstance(root, Mapping) and key_normal_form(get_ci(root, "type")) == "feature"
    ):
        features = [root]
    elif isinstance(root, list):
        features = root
    else:
        raise DatasetBuildError(
            f"{source_ref.as_string()} is JSON but not a GeoJSON FeatureCollection"
        )

    for feature in features:
        if isinstance(feature, Mapping):
            yield feature


def discover_inputs(input_dir: Path, recursive: bool, output_dir: Path) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    allowed = {".geojson", ".json", ".zip"}
    files: list[Path] = []
    output_resolved = output_dir.resolve()
    for path in iterator:
        if not path.is_file() or path.suffix.casefold() not in allowed:
            continue
        try:
            path.resolve().relative_to(output_resolved)
            continue
        except ValueError:
            pass
        files.append(path)
    return sorted(files, key=lambda path: str(path).casefold())


def iter_all_features(
        paths: Sequence[Path], *, input_dir: Path, max_features: int | None
) -> Iterator[tuple[Mapping[str, Any], SourceRef, int]]:
    ijson_module = import_ijson()
    emitted = 0
    for path in paths:
        display_name = str(path.relative_to(input_dir))
        if path.suffix.casefold() == ".zip":
            try:
                archive = zipfile.ZipFile(path)
            except zipfile.BadZipFile as exc:
                raise DatasetBuildError(f"Invalid ZIP archive: {display_name}") from exc
            with archive:
                members = sorted(
                    (
                        info
                        for info in archive.infolist()
                        if not info.is_dir()
                           and Path(info.filename).suffix.casefold()
                           in {".geojson", ".json"}
                    ),
                    key=lambda info: info.filename.casefold(),
                )
                if not members:
                    LOGGER.warning("No GeoJSON/JSON members found in %s", display_name)
                for info in members:
                    source_ref = SourceRef(display_name, info.filename)
                    with archive.open(info, "r") as stream:
                        for index, feature in enumerate(
                                iter_features_from_binary(
                                    stream,
                                    source_ref,
                                    size_hint=info.file_size,
                                    ijson_module=ijson_module,
                                )
                        ):
                            yield feature, source_ref, index
                            emitted += 1
                            if max_features is not None and emitted >= max_features:
                                return
        else:
            source_ref = SourceRef(display_name)
            with path.open("rb") as stream:
                for index, feature in enumerate(
                        iter_features_from_binary(
                            stream,
                            source_ref,
                            size_hint=path.stat().st_size,
                            ijson_module=ijson_module,
                        )
                ):
                    yield feature, source_ref, index
                    emitted += 1
                    if max_features is not None and emitted >= max_features:
                        return


def count_all_features(
        paths: Sequence[Path],
        *,
        input_dir: Path,
        max_features: int | None,
        tqdm_factory: Any,
) -> int:
    """First pass: count source features so generation can show a percentage."""

    count = 0
    progress = tqdm_factory(
        desc="Counting GeoJSON features",
        unit=" features",
        dynamic_ncols=True,
    )
    try:
        for _feature, _source_ref, _feature_index in iter_all_features(
                paths, input_dir=input_dir, max_features=max_features
        ):
            count += 1
            progress.update(1)
    finally:
        progress.close()
    return count


def number_range(number_from: str, number_to: str, language: str) -> str:
    if not number_from and not number_to:
        return ""
    if not number_from:
        number_from = number_to
    if number_to and number_to != number_from:
        base = f"{number_from}-{number_to}"
    else:
        base = number_from
    if language == "zh-Hans":
        return f"{base}号"
    if language == "zh-Hant":
        return f"{base}號"
    return base


def phase_text(components: Mapping[str, Any], language: str) -> str:
    name = clean_text(components.get("phase_name"))
    number = clean_text(components.get("phase_no"))
    if name:
        # ALS PhaseName is already the display form (for example "PHASE I" or
        # "第一期").  PhaseNo is a parallel normalized value, not text that
        # should be appended; doing so would produce "PHASE I 1".
        return name
    if not number:
        return ""
    return f"Phase {number}" if language == "en" else f"第{number}期"


def block_text(components: Mapping[str, Any], language: str) -> str:
    number = clean_text(components.get("block_no"))
    descriptor = clean_text(components.get("block_descriptor"))

    # Normalize common ALS English abbreviations before rendering.
    if language == "en":
        desc_upper = descriptor.upper()
        if desc_upper in {"BLK", "BLK."}:
            descriptor = "Block"
        elif desc_upper in {"TWR", "TWR."}:
            descriptor = "Tower"

    if not number:
        return descriptor
    if not descriptor:
        return number
    if language == "en":
        if clean_text(components.get("block_descriptor_precedes")).upper() == "N":
            return f"{number} {descriptor}"
        return f"{descriptor} {number}"
    return f"{number}{descriptor}"


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
        desc_upper = description.upper()
        if desc_upper in {"F", "/F", "FL", "FLOOR"}:
            if desc_upper in {"F", "/F"}:
                # Preserve the case of the original description's "F"
                if description.endswith("f") or description.endswith("/f"):
                    suffix = "/f"
                else:
                    suffix = "/F"
            else:
                suffix = f" {description}"
            return f"{number}{suffix}"
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
        return " ".join(part for part in (descriptor, number, portion) if part)
    return "".join(part for part in (number, descriptor, portion) if part)


def atoms_with_space(*items: tuple[str, str | None]) -> list[Atom]:
    atoms: list[Atom] = []
    for text, label in items:
        if not text:
            continue
        if atoms:
            atoms.append(Atom(" "))
        atoms.append(Atom(text, label))
    return atoms


def build_chunks(
        components: Mapping[str, Any], language: str, three_d: Mapping[str, Any] | None
) -> list[Chunk]:
    chunks: list[Chunk] = []

    def add(kind: str, text: str, label: str) -> None:
        if text:
            chunks.append(Chunk(kind, [Atom(text, label)], language))

    add("region", clean_text(components.get("region")), "REGION")
    district_val = clean_text(components.get("district"))
    add("district", district_val, "DISTRICT")

    # Prefer a real sub-district when present; otherwise fall back to street/village
    # location — but never emit a SUB_DISTRICT that is the same text as DISTRICT
    # (common in ALS for places like 九龍城 where LocationName == District).
    def _norm_place(s: str) -> str:
        s = s.casefold().strip()
        for suffix in (" district", "區", "区"):
            if s.endswith(suffix):
                s = s[: -len(suffix)].strip()
        return s

    sub = (
            clean_text(components.get("sub_district"))
            or clean_text(components.get("street_location"))
            or clean_text(components.get("village_location"))
    )
    if sub and (not district_val or _norm_place(sub) != _norm_place(district_val)):
        add("sub_district", sub, "SUB_DISTRICT")

    street_name = clean_text(components.get("street_name"))
    street_number = number_range(
        clean_text(components.get("street_no_from")),
        clean_text(components.get("street_no_to")),
        language,
    )
    if street_name or street_number:
        if language == "en":
            street_atoms = atoms_with_space(
                (street_number, "BUILDING_NUMBER"),
                (street_name, "STREET_NAME"),
            )
        else:
            street_atoms = [
                Atom(text, label)
                for text, label in (
                    (street_name, "STREET_NAME"),
                    (street_number, "BUILDING_NUMBER"),
                )
                if text
            ]
        chunks.append(Chunk("street", street_atoms, language))

    village_name = clean_text(components.get("village_name"))
    village_number = number_range(
        clean_text(components.get("village_no_from")),
        clean_text(components.get("village_no_to")),
        language,
    )
    if village_name or village_number:
        if language == "en":
            village_atoms = atoms_with_space(
                (village_number, "BUILDING_NUMBER"),
                (village_name, "VILLAGE_NAME"),
            )
        else:
            village_atoms = [
                Atom(text, label)
                for text, label in (
                    (village_name, "VILLAGE_NAME"),
                    (village_number, "BUILDING_NUMBER"),
                )
                if text
            ]
        chunks.append(Chunk("village", village_atoms, language))

    add("estate", clean_text(components.get("estate_name")), "ESTATE_NAME")
    add("phase", phase_text(components, language), "PHASE")
    add("block", block_text(components, language), "BLOCK")
    add("building", clean_text(components.get("building_name")), "BUILDING_NAME")
    add("floor", floor_text(three_d, language), "FLOOR")
    add("unit", unit_text(three_d, language), "UNIT")
    return chunks


def canonical_order(chunks: Sequence[Chunk], language: str) -> list[Chunk]:
    priority_zh = {
        "region": 10,
        "district": 20,
        "sub_district": 25,
        "street": 40,
        "village": 45,
        "estate": 50,
        "phase": 55,
        "block": 60,
        "building": 65,
        "floor": 70,
        "unit": 80,
    }
    priority_en = {
        "unit": 10,
        "floor": 20,
        "block": 30,
        "building": 35,
        "phase": 40,
        "estate": 45,
        "village": 50,
        "street": 55,
        "sub_district": 65,
        "district": 70,
        "region": 80,
    }
    priorities = priority_en if language == "en" else priority_zh
    return sorted(chunks, key=lambda chunk: priorities.get(chunk.kind, 999))


def separator_for(style: str, language: str, rng: random.Random) -> str:
    if style == "canonical":
        return ", " if language == "en" else ""
    if style == "comma":
        if language == "en":
            return rng.choice([", ", ",", "; "])
        return rng.choice(["，", "、", "， "])
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
    text_parts: list[str] = []
    entities: list[dict[str, Any]] = []
    cursor = 0
    for chunk_index, chunk in enumerate(chunks):
        if chunk_index:
            text_parts.append(separator)
            cursor += len(separator)
        for atom in chunk.atoms:
            start = cursor
            text_parts.append(atom.text)
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
    return "".join(text_parts), entities


def apply_localized_district(chunks: list[Chunk], effective_language: str, rng: random.Random) -> str | None:
    """Add a plausible sub-district drawn from SUB_DISTRICT_MAP when none exists.

    Used so the model sees the full range of real Hong Kong sub-district names
    (including ones that ALS rarely supplies as LocationName).  The value is
    synthetic and is never treated as verified locality truth.
    """
    # Already have a sub-district (from ALS location fallback or prior pass) — do not double up.
    if any(chunk.kind == "sub_district" for chunk in chunks):
        return None

    lang_key = effective_language
    if lang_key.startswith("mixed-"):
        lang_key = lang_key.replace("mixed-", "").replace("-en", "")

    lang_map = SUB_DISTRICT_MAP.get(lang_key)
    if not lang_map:
        lang_map = SUB_DISTRICT_MAP.get("en") if "en" in effective_language else SUB_DISTRICT_MAP.get("zh-Hant")
        if not lang_map:
            return None

    # Find the district chunk to figure out which sub-districts are valid
    district_idx = next((i for i, chunk in enumerate(chunks) if chunk.kind == "district"), -1)
    if district_idx == -1 or not chunks[district_idx].atoms:
        return None

    current_district_text = chunks[district_idx].atoms[0].text
    matched_official = None

    # Prefer exact / longest match so "九龍城區" does not accidentally match a
    # shorter key that is a substring of another district name.
    for official_dist in sorted(lang_map.keys(), key=len, reverse=True):
        if official_dist.upper() in current_district_text.upper():
            matched_official = official_dist
            break

    if not matched_official:
        return None

    def _norm_place(s: str) -> str:
        s = s.casefold().strip()
        for suffix in (" district", "區", "区"):
            if s.endswith(suffix):
                s = s[: -len(suffix)].strip()
        return s

    district_norm = _norm_place(current_district_text)
    # Drop candidates that are just the district name again (e.g. "九龍城" under 九龍城區)
    sub_districts = [
        s for s in lang_map[matched_official]
        if _norm_place(s) != district_norm
    ]
    if not sub_districts:
        return None

    # Shuffle then pick so every listed sub-district has equal chance and
    # the fixed map order does not create a bias across many examples.
    candidates = list(sub_districts)
    rng.shuffle(candidates)
    assigned_sub = candidates[0]

    new_chunk = Chunk(
        kind="sub_district",
        atoms=[Atom(assigned_sub, "SUB_DISTRICT")],
        source_language=chunks[district_idx].source_language,
    )
    chunks.append(new_chunk)
    chunks[:] = canonical_order(chunks, effective_language)
    return "synthetic_sub_district"


def apply_district_suffix_noise(chunks: Sequence[Chunk], fallback_language: str, rng: random.Random) -> str | None:
    """Randomly add or remove '區'/'区' or ' District' from the district name."""
    scenario_added = None
    for chunk in chunks:
        if chunk.kind != "district":
            continue
        for atom in chunk.atoms:
            if atom.label != "DISTRICT" or not atom.text:
                continue

            text = atom.text
            # Identify language for mixed/code-switched scenarios
            chunk_lang = chunk.source_language or fallback_language
            is_english = chunk_lang == "en"

            if is_english:
                if re.search(r"(?i)\s+DISTRICT$", text):
                    atom.text = re.sub(r"(?i)\s+DISTRICT$", "", text)
                    scenario_added = "removed_district_suffix"
                else:
                    suffix = " DISTRICT" if text.isupper() else " District"
                    atom.text = text + suffix
                    scenario_added = "added_district_suffix"
            else:
                if text.endswith(("區", "区")):
                    atom.text = text[:-1]
                    scenario_added = "removed_district_suffix"
                else:
                    suffix = "区" if chunk_lang == "zh-Hans" else "區"
                    atom.text = text + suffix
                    scenario_added = "added_district_suffix"

    return scenario_added


def validate_entities(text: str, entities: Sequence[Mapping[str, Any]]) -> None:
    previous_end = -1
    for entity in entities:
        start = int(entity["start"])
        end = int(entity["end"])
        if not (0 <= start < end <= len(text)):
            raise DatasetBuildError(f"Invalid entity range {start}:{end} for {text!r}")
        if start < previous_end:
            raise DatasetBuildError(f"Overlapping entity spans in {text!r}")
        if text[start:end] != entity["text"]:
            raise DatasetBuildError(f"Entity text mismatch in {text!r}")
        if entity["label"] not in LABELS:
            raise DatasetBuildError(f"Unknown label: {entity['label']}")
        previous_end = end


def replace_words(text: str, replacements: Mapping[str, str]) -> str:
    for source, target in sorted(replacements.items(), key=lambda pair: -len(pair[0])):
        text = re.sub(rf"\b{re.escape(source)}\b", target, text, flags=re.IGNORECASE)
    return text


def apply_english_abbreviations(chunks: Sequence[Chunk], rng: random.Random) -> None:
    replacements = {
        word: rng.choice(options) for word, options in EN_ABBREVIATIONS.items()
    }
    for chunk in chunks:
        for atom in chunk.atoms:
            if not atom.text:
                continue
            new_text = replace_words(atom.text, replacements)
            # If a replacement actually happened, maybe append "." to the short form
            if new_text != atom.text and rng.random() < 0.40:
                # Only add a period after the abbreviated token(s), not the whole string
                for abbr in replacements.values():
                    # Match whole-word abbr that does not already end with "."
                    pattern = rf"\b{re.escape(abbr)}\b(?!\.)"
                    new_text = re.sub(
                        pattern,
                        abbr + ".",
                        new_text,
                        count=1,
                        flags=re.IGNORECASE,
                    )
            atom.text = new_text


def apply_english_case_noise(
        chunks: Sequence[Chunk],
        rng: random.Random,
        upper_rate: float,
        title_rate: float,
        lower_rate: float,
        mixed_rate: float
) -> str:
    choices = (
        ("upper", upper_rate),
        ("title", title_rate),
        ("lower", lower_rate),
        ("mixed", mixed_rate),
    )
    mode = weighted_choice(rng, choices)

    for chunk in chunks:
        # Only apply casing to English components
        if chunk.source_language == "en":
            for atom in chunk.atoms:
                if not atom.text:
                    continue

                if atom.label in {"FLOOR", "UNIT", "BUILDING_NUMBER"}:
                    continue

                if mode == "lower":
                    atom.text = atom.text.lower()
                elif mode == "upper":
                    atom.text = atom.text.upper()
                elif mode == "title":
                    # Keep small identifiers like 'A' or 'LD' uppercase
                    if len(atom.text) <= 2 and atom.text.isalpha():
                        atom.text = atom.text.upper()
                    else:
                        atom.text = atom.text.title()
                else:  # mixed
                    atom.text = "".join(
                        char.upper() if char.isalpha() and rng.random() < 0.35 else char.lower()
                        for char in atom.text
                    )
    return mode


def apply_chinese_region_variation(chunks: Sequence[Chunk], fallback_language: str, rng: random.Random) -> str | None:
    """Randomly apply colloquial and localized variations for Chinese regions."""
    scenario_added = None
    for chunk in chunks:
        if chunk.kind != "region":
            continue

        chunk_lang = chunk.source_language or fallback_language
        if not chunk_lang.startswith("zh-"):
            continue

        var_map = CHINESE_REGION_VARIATIONS.get(chunk_lang, CHINESE_REGION_VARIATIONS["zh-Hant"])
        for atom in chunk.atoms:
            if atom.label == "REGION" and atom.text in var_map:
                new_text = rng.choice(var_map[atom.text])
                if new_text != atom.text:
                    atom.text = new_text
                    scenario_added = "chinese_region_variation"
    return scenario_added


def apply_region_abbreviation(chunks: Sequence[Chunk], rng: random.Random) -> bool:
    """Use a short English region code in a minority of augmented inputs."""
    for chunk in chunks:
        if chunk.kind != "region" or chunk.source_language != "en":
            continue
        for atom in chunk.atoms:
            canonical = ENGLISH_REGION_CANONICAL.get(atom.text.upper(), atom.text)
            abbreviation_options = ENGLISH_REGION_ABBREVIATIONS.get(canonical)
            if atom.label == "REGION" and abbreviation_options:
                atom.text = rng.choice(abbreviation_options)
                return True
    return False


def apply_floor_unit_fusion(
        chunks: list[Chunk],
        three_d: Mapping[str, Any] | None,
        language: str,
        rng: random.Random,
) -> str | None:
    """Sometimes fuse floor + unit into compact / inverted forms.

    - Pure-compact modes strip descriptors and use clean cores,
      EXCEPT when the order is inverted or the unit is purely numeric.
      In those cases a clear prefix (Rm / Flat / Unit / 室 …) is forced
      so the model still has a lexical cue after positional clues are lost.
    - Marker / spaced modes re-use the original atom lists so any leading
      spaces or messy separators that already exist stay consistent with
      the recorded entity offsets.
    """
    if not three_d:
        return None

    floor_idx = next((i for i, c in enumerate(chunks) if c.kind == "floor"), -1)
    unit_idx = next((i for i, c in enumerate(chunks) if c.kind == "unit"), -1)
    if floor_idx < 0 or unit_idx < 0:
        return None

    floor_chunk = chunks[floor_idx]
    unit_chunk = chunks[unit_idx]

    # Do not fuse across a code-switch boundary
    if (
            floor_chunk.source_language
            and unit_chunk.source_language
            and floor_chunk.source_language != unit_chunk.source_language
    ):
        return None

    chunk_lang = (
            floor_chunk.source_language
            or unit_chunk.source_language
            or language
    )
    is_en = chunk_lang == "en" or (
            chunk_lang is not None and not str(chunk_lang).startswith("zh")
    )

    floor_num = clean_text(three_d.get("floor_num"))
    unit_no = clean_text(three_d.get("unit_no"))

    # Need clear numeric floor + unit id for the pure-compact forms
    if not floor_num or not unit_no:
        return None

    # Original atom lists (may already contain leading unlabeled spaces
    # from the Chinese-messy-separation noise – we must keep them intact)
    floor_atoms = list(floor_chunk.atoms)
    unit_atoms = list(unit_chunk.atoms)
    if not floor_atoms or not unit_atoms:
        return None

    def is_safe_for_compact(floor: str, unit: str) -> bool:
        # Letter on either side → always safe (27LD, LD27, 5A12, …)
        if re.search(r"[A-Za-z]", floor) or re.search(r"[A-Za-z]", unit):
            return True
        # Both very short → low collision risk
        if len(floor) <= 2 and len(unit) <= 2:
            return True
        # Otherwise too ambiguous (30+10 → 3010, 30+3010 → 301030, …)
        return False

    # ------------------------------------------------------------------
    modes: list[tuple[str, int]] = [
        ("floor_then_unit_compact", 22),
        ("unit_then_floor_compact", 18),
        ("floor_then_unit_marker", 15),
        ("unit_then_floor_marker", 12),
        ("floor_then_unit_spaced", 10),
        ("unit_then_floor_spaced", 8),
    ]
    mode = weighted_choice(rng, modes)

    # If the chosen compact mode would be ambiguous, force a safer mode
    if mode in {"floor_then_unit_compact", "unit_then_floor_compact"}:
        if not is_safe_for_compact(floor_num, unit_no):
            mode = (
                "floor_then_unit_spaced"
                if mode == "floor_then_unit_compact"
                else "unit_then_floor_spaced"
            )

    atoms: list[Atom] = []

    if mode in {"floor_then_unit_compact", "unit_then_floor_compact"}:
        # Pure compact – only reached when it is safe.
        # Force a clear unit prefix when:
        #   1. order is inverted (unit then floor), OR
        #   2. the unit itself is purely numeric
        # Letter-only units (LD, A, 5A …) stay bare – the letters are distinctive.
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
        # Re-use the original atom lists (preserves any internal / leading
        # spaces that were already present and already matched their entities)
        if mode in {"floor_then_unit_marker", "floor_then_unit_spaced"}:
            first_atoms = floor_atoms
            second_atoms = unit_atoms
        else:
            first_atoms = unit_atoms
            second_atoms = floor_atoms

        atoms = list(first_atoms)

        # Decide whether to insert a separator between the two groups
        need_space = mode.endswith("_spaced") or (is_en and mode.endswith("_marker"))
        if need_space:
            # Only add a space when the first group does not already end
            # with whitespace and the second group does not already start
            # with whitespace – avoids double spaces that shift offsets.
            first_ends_space = atoms and atoms[-1].text and atoms[-1].text[-1].isspace()
            second_starts_space = (
                    second_atoms
                    and second_atoms[0].text
                    and second_atoms[0].text[0].isspace()
            )
            if not first_ends_space and not second_starts_space:
                atoms.append(Atom(" ", None))

        atoms.extend(second_atoms)

    if not atoms:
        return None

    new_chunk = Chunk(
        kind="unit",
        atoms=atoms,
        source_language=chunk_lang,
    )

    # Remove the two original chunks and insert the fused one
    for idx in sorted((floor_idx, unit_idx), reverse=True):
        del chunks[idx]
    chunks.insert(min(floor_idx, unit_idx), new_chunk)

    return f"floor_unit_fusion_{mode}"


def trim_unlabelled_edge_atoms(atoms: Sequence[Atom]) -> list[Atom]:
    """Remove separator atoms left at a chunk edge after atom-level omission."""

    trimmed = list(atoms)
    while trimmed and trimmed[0].label is None and not trimmed[0].text.strip(" ,，、;/／.-"):
        trimmed.pop(0)
    while trimmed and trimmed[-1].label is None and not trimmed[-1].text.strip(" ,，、;/／.-"):
        trimmed.pop()
    return trimmed


def importance_weighted_omission(
        chunks: Sequence[Chunk], rng: random.Random
) -> tuple[list[Chunk], list[str]]:
    kinds_present = {chunk.kind for chunk in chunks}

    # Check if any chunk contains a BUILDING_NUMBER atom
    has_building_number = any(
        any(atom.label == "BUILDING_NUMBER" for atom in chunk.atoms)
        for chunk in chunks
    )

    candidates = {
        kind: weight
        for kind, weight in COMPONENT_OMISSION_WEIGHTS.items()
        if kind in kinds_present or (kind == "building_number" and has_building_number)
    }
    if not candidates:
        return list(chunks), []

    target_count = min(len(candidates), rng.choice([1, 1, 1, 1, 2, 2, 3]))
    omitted: list[str] = []
    anchor_kinds = {"street", "village", "estate", "block", "building"}

    while candidates and len(omitted) < target_count:
        selected = weighted_choice(rng, list(candidates.items()))
        candidates.pop(selected, None)

        # Ensure at least one anchor remains
        proposed_omitted_chunks = {k for k in (set(omitted) | {selected}) if k != "building_number"}
        remaining_kinds = {chunk.kind for chunk in chunks if chunk.kind not in proposed_omitted_chunks}
        if not (remaining_kinds & anchor_kinds):
            continue

        omitted.append(selected)

    if not omitted:
        return list(chunks), []

    omitted_set = set(omitted)
    filtered_chunks: list[Chunk] = []

    for chunk in chunks:
        if chunk.kind in omitted_set:
            continue

        # Handle atom-level removal (building numbers)
        if "building_number" in omitted_set:
            new_atoms = trim_unlabelled_edge_atoms(
                [atom for atom in chunk.atoms if atom.label != "BUILDING_NUMBER"]
            )
            if new_atoms:
                new_chunk = copy.deepcopy(chunk)
                new_chunk.atoms = new_atoms
                filtered_chunks.append(new_chunk)
        else:
            filtered_chunks.append(chunk)

    return filtered_chunks, omitted


def reorder_chunks(chunks: list[Chunk], rng: random.Random) -> tuple[list[Chunk], str]:
    if len(chunks) < 2:
        return chunks, "reorder_not_applicable"
    mode = rng.choice(
        ["district_last", "adjacent_swap", "section_reverse", "light_shuffle"]
    )
    reordered = list(chunks)
    if mode == "district_last":
        tail = [
            chunk
            for chunk in reordered
            if chunk.kind in {"district", "region"}
        ]
        head = [
            chunk
            for chunk in reordered
            if chunk.kind not in {"district", "region"}
        ]
        if tail and head:
            reordered = head + tail
        else:
            mode = "adjacent_swap"
    if mode == "adjacent_swap":
        index = rng.randrange(len(reordered) - 1)
        reordered[index], reordered[index + 1] = reordered[index + 1], reordered[index]
    elif mode == "section_reverse":
        start = rng.randrange(0, len(reordered) - 1)
        end = rng.randrange(start + 2, len(reordered) + 1)
        reordered[start:end] = reversed(reordered[start:end])
    elif mode == "light_shuffle":
        swaps = min(2, len(reordered) - 1)
        for _ in range(swaps):
            left = rng.randrange(len(reordered))
            right = rng.randrange(len(reordered))
            reordered[left], reordered[right] = reordered[right], reordered[left]
    return reordered, mode


def mix_language_chunks(
        base_chunks: Sequence[Chunk],
        alternate_chunks: Sequence[Chunk],
        *,
        base_language: str,
        alternate_language: str,
        mode: str,
        rng: random.Random,
) -> tuple[list[Chunk], list[dict[str, str]], str]:
    """Replace whole semantic chunks with their parallel-language versions."""

    alternate_by_kind: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in alternate_chunks:
        alternate_by_kind[chunk.kind].append(copy.deepcopy(chunk))

    occurrence: Counter[str] = Counter()
    candidates: dict[int, Chunk] = {}
    for index, chunk in enumerate(base_chunks):
        position = occurrence[chunk.kind]
        occurrence[chunk.kind] += 1
        matches = alternate_by_kind.get(chunk.kind, [])
        if position < len(matches):
            candidates[index] = matches[position]

    available = sorted(candidates)
    if not available:
        result = copy.deepcopy(list(base_chunks))
        assignments = [
            {"kind": chunk.kind, "language": base_language} for chunk in result
        ]
        return result, assignments, "code_switch_not_applicable"

    if mode in {"english_3d_chinese_body", "chinese_3d_english_body"}:
        selected = {
            index for index in available if base_chunks[index].kind in {"floor", "unit"}
        }
        if not selected:
            selected = {rng.choice(available)}
    elif mode in {"single_chinese_component", "single_english_component"}:
        informative = [
            index
            for index in available
            if base_chunks[index].kind not in {"region", "floor"}
        ]
        selected = {rng.choice(informative or available)}
    elif mode == "alternating_components":
        selected = set(available[::2])
    else:  # balanced_components
        target = max(1, len(available) // 2)
        selected = set(rng.sample(available, target))

    # A mixed row must contain both languages when at least two chunks exist.
    if len(base_chunks) > 1 and len(selected) == len(base_chunks):
        selected.remove(rng.choice(sorted(selected)))
    if not selected:
        selected = {rng.choice(available)}

    result: list[Chunk] = []
    assignments: list[dict[str, str]] = []
    for index, base_chunk in enumerate(base_chunks):
        if index in selected:
            result.append(copy.deepcopy(candidates[index]))
            source_language = alternate_language
        else:
            result.append(copy.deepcopy(base_chunk))
            source_language = base_language
        assignments.append({"kind": result[-1].kind, "language": source_language})
    return result, assignments, mode


def _latin_typo(text: str, rng: random.Random) -> tuple[str, str] | None:
    # Requires at least 3 lowercase letters to avoid mutating all-caps acronyms (e.g., BLDG, BLK)
    matches = list(re.finditer(r"\b[A-Z]?[a-z]{3,}\b", text))
    if not matches:
        return None
    match = rng.choice(matches)
    word = match.group(0)
    operation = rng.choice(["delete", "transpose", "repeat", "keyboard_neighbour"])
    if operation == "transpose" and len(word) >= 4:
        index = rng.randrange(1, len(word) - 1)
        changed = word[:index] + word[index + 1] + word[index] + word[index + 2:]
    elif operation == "repeat":
        index = rng.randrange(1, len(word))
        changed = word[:index] + word[index] + word[index:]
    elif operation == "keyboard_neighbour":
        eligible = [
            index
            for index, char in enumerate(word)
            if char.casefold() in QWERTY_NEIGHBOURS
        ]
        if not eligible:
            return None
        index = rng.choice(eligible)
        original = word[index]
        replacement = rng.choice(QWERTY_NEIGHBOURS[original.casefold()])
        if original.isupper():
            replacement = replacement.upper()
        changed = word[:index] + replacement + word[index + 1:]
    else:
        operation = "delete"
        index = rng.randrange(1, len(word))
        changed = word[:index] + word[index + 1:]
    return text[: match.start()] + changed + text[match.end():], operation


def _han_typo(text: str, rng: random.Random) -> tuple[str, str] | None:
    positions = [
        index for index, char in enumerate(text) if "\u3400" <= char <= "\u9fff"
    ]
    if len(positions) < 2:
        return None
    index = rng.choice(positions)
    if rng.random() < 0.72:
        return text[:index] + text[index + 1:], "delete_character"
    return text[:index] + text[index] + text[index:], "repeat_character"


def introduce_minor_typo(
        chunks: Sequence[Chunk], rng: random.Random
) -> dict[str, str] | None:
    """Apply one recoverable-looking typo to a name, never to floor/unit truth."""

    eligible_labels = {
        "DISTRICT",
        "STREET_NAME",
        "VILLAGE_NAME",
        "ESTATE_NAME",
        "BLOCK",
        "BUILDING_NAME",
    }
    candidates: list[tuple[Chunk, Atom]] = []
    for chunk in chunks:
        for atom in chunk.atoms:
            if atom.label not in eligible_labels:
                continue
            if (
                    re.search(r"[A-Za-z]{4,}", atom.text)
                    or len(re.findall(r"[\u3400-\u9fff]", atom.text)) >= 2
            ):
                candidates.append((chunk, atom))
    if not candidates:
        return None

    chunk, atom = rng.choice(candidates)
    original = atom.text
    has_latin = bool(re.search(r"[A-Za-z]{4,}", atom.text))
    has_han = bool(re.search(r"[\u3400-\u9fff]", atom.text))
    if has_latin and has_han:
        use_latin = rng.random() < 0.5
    else:
        use_latin = has_latin
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


def synthetic_three_d(
        group_id: str, language: str, seed: int, key: Any, is_village: bool = False
) -> dict[str, str]:
    # The random case/number decision is language-neutral.  Only rendering
    # terms differ, preventing one language from receiving easier 3D cases.
    rng = stable_rng(seed, group_id, key, "synthetic_3d")
    # 1. Dynamic Floor Number (1 to 65, high-entropy integer sampling)
    floor_num_int = rng.randint(1, 65)
    floor_number = str(floor_num_int)
    # 2. Dynamic Unit Number Generator (Prevents token memorization)
    unit_type = weighted_choice(rng, [
        ("letter", 40),
        ("number", 5),
        ("leading_zero", 5),
        ("combo", 25),
        ("floor_room", 15),  # realistic 4-digit
    ])
    if unit_type == "letter":
        letter = chr(rng.randint(65, 76))  # A–L
        unit_number = letter.lower() if rng.random() < 0.3 else letter
    elif unit_type == "number":
        unit_number = str(rng.randint(1, 30))
    elif unit_type == "leading_zero":
        unit_number = f"{rng.randint(1, 15):02d}"
    elif unit_type == "combo":
        letter = f"{rng.randint(1, 15)}{chr(rng.randint(65, 68))}"  # e.g., 5A
        unit_number = letter.lower() if rng.random() < 0.3 else letter
    else:  # floor_room
        unit_number = f"{floor_num_int}{rng.randint(1, 12):02d}"  # e.g., 1203, 2107
    # Repetition intentionally makes LD visible without making every unusual
    # alphabetic-unit example look like one development.
    # 85 chance to generate a completely random 1-2 letter unit (e.g. "AC", "G", "XZ")
    # 15% chance to use common Hong Kong alphabetic unit formats
    random_letter_prob = 0.85

    if rng.random() < random_letter_prob:
        length = rng.choice([1, 2])
        alphabetic_unit = "".join(rng.choice(string.ascii_uppercase) for _ in range(length))
    else:
        alphabetic_unit = rng.choice([
            # Standard / previous set
            "LA", "LB", "LC", "LD", "LE", "AA", "AB", "BA", "PA", "PH",
            # Rare double letters
            "ZZ", "XX", "QQ", "WW", "YY", "KK",
            # High-entropy pairs
            "QX", "XZ", "JQ", "KZ", "VZ", "MX",
            # Functional / structural units
            "ME", "RF", "SK", "CP", "SU", "RU", "UG", "LG"
        ])

    # Dynamic Combined / Range Units (e.g., A & B, A-B, 01-02, 1/2)
    comb_style = rng.choice(["letter_amp", "letter_dash", "num_dash", "num_slash"])
    if comb_style == "letter_amp":
        l1 = chr(rng.randint(65, 74))
        l2 = chr(ord(l1) + 1)
        combined_unit_en = f"{l1} & {l2}"
        combined_unit_zh = f"{l1}及{l2}"
    elif comb_style == "letter_dash":
        l1 = chr(rng.randint(65, 74))
        l2 = chr(ord(l1) + 1)
        combined_unit_en = f"{l1}-{l2}"
        combined_unit_zh = f"{l1}-{l2}"
    elif comb_style == "num_dash":
        n1 = rng.randint(1, 15)
        combined_unit_en = f"{n1:02d}-{n1 + 1:02d}"
        combined_unit_zh = f"{n1:02d}至{n1 + 1:02d}"
    else:
        n1 = rng.randint(1, 15)
        combined_unit_en = f"{n1}/{n1 + 1}"
        combined_unit_zh = f"{n1}/{n1 + 1}"

    leading_zero_unit = rng.choice(["01", "02", "03", "05", "06", "08", "09"])
    portion_index = rng.randrange(4)
    # special_floor_index = rng.randrange(4)
    # ground_floor_index = rng.randrange(4)
    descriptor_index = rng.randrange(3)

    weights = VILLAGE_3D_WEIGHTS if is_village else STANDARD_3D_WEIGHTS
    case = weighted_choice(rng, weights)

    # Dynamic Carpark Space Generator
    cp_prefix = rng.choice(["P", "C", "L", "B", "CP", ""])
    cp_number = f"{cp_prefix}{rng.randint(1, 350)}"

    # Dynamic Room/Flat Generator for number_only & hao_shi (3-4 digit rooms or 1-2 digit flats)
    dyn_room_num = str(rng.randint(101, 3508)) if rng.random() < 0.9 else str(rng.randint(1, 50))

    if language == "en":
        # Mix of forms users actually type
        unit_descriptor_pool = [
            "FLAT", "Flat", "flat",
            "UNIT", "Unit", "unit",
            "ROOM", "Room", "room",
            "RM", "Rm", "rm",
            "FLT", "APT", "Apt",
        ]
        descriptor = unit_descriptor_pool[descriptor_index % 3]  # or rng.choice(...)
        # Better: always randomize
        descriptor = rng.choice(unit_descriptor_pool)

        floor_desc_pool = ["/F", "/f", "F", "FL", "FLOOR", "Floor", "floor", "FLR"]
        profile = {
            "floor_num": floor_number,
            "floor_description": rng.choice(floor_desc_pool),
            "unit_descriptor": descriptor,
            "unit_no": unit_number,
            "unit_portion": "",
        }
        overrides = {
            "number_only_unit": {
                "unit_descriptor": "",
                "unit_no": dyn_room_num,
            },
            "hao_shi_unit": {
                "unit_descriptor": rng.choice([
                    "RM", "Rm", "rm", "ROOM", "Room", "room",
                    "NO.", "No.", "UNIT", "Unit", "#", "FLAT", "Flat", "flat",
                ]),
                "unit_no": dyn_room_num,
            },
            "standard_room": {
                "unit_descriptor": rng.choice(["RM", "Rm", "rm", "ROOM", "Room", "room"]),
            },
            "ground_floor_no_unit": {
                "floor_num": "",
                "floor_description": rng.choice(["G/F", "GF", "GROUND FLOOR"]),
                "unit_descriptor": "",
                "unit_no": "",
            },
            "whole_floor": {"unit_descriptor": "", "unit_no": ""},
            "unit_without_floor": {"floor_num": "", "floor_description": ""},
            "shop": {
                "floor_num": "",
                "floor_description": rng.choice(["G/F", "GF", "GROUND FLOOR"]),
                "unit_descriptor": rng.choice(["SHOP", "STORE", "NO."]),
                "unit_no": str(rng.randint(1, 50)),
            },
            "basement": {
                "floor_num": "",
                "floor_description": rng.choice(["B1/F", "B2/F", "BASEMENT", "LG/F"]),
                "unit_descriptor": rng.choice(["SHOP", "UNIT", "ROOM", "STORE", ""]),
                "unit_no": str(rng.randint(1, 50)),
            },
            "podium": {
                "floor_num": "",
                "floor_description": rng.choice(["PODIUM", "P/F", f"P{rng.randint(1, 3)}/F"]),
            },
            "roof": {
                "floor_num": "",
                "floor_description": rng.choice(["ROOF", "R/F", "ROOFTOP"]),
                "unit_descriptor": "",
                "unit_no": "",
            },
            "unit_portion": {"unit_portion": ["E", "FR", "LF", "RF"][portion_index]},
            "alphabetic_unit": {
                "unit_descriptor": rng.choice([
                    "FLAT", "Flat", "flat", "UNIT", "Unit", "unit", "",
                ]),
                "unit_no": alphabetic_unit,
            },
            "leading_zero_unit": {
                "unit_descriptor": rng.choice([
                    "FLAT", "Flat", "flat", "UNIT", "Unit", "ROOM", "Room", "",
                ]),
                "unit_no": leading_zero_unit,
            },
            "duplex_floor": {
                "floor_num": "",
                "floor_description": rng.choice([
                    f"{floor_number}-{floor_num_int + 1}/F",
                    f"{floor_number} & {floor_num_int + 1}/F",
                    f"{floor_number}/F-{floor_num_int + 1}/F"
                ]),
            },
            "combined_units": {
                "unit_descriptor": rng.choice([
                    "FLATS", "Flats", "UNITS", "Units", "ROOMS", "Rooms", "FLAT", "Flat",
                ]),
                "unit_no": combined_unit_en,
            },
            "special_floor": {
                "floor_num": "",
                "floor_description": rng.choice([
                    f"{floor_number}A/F",
                    f"LEVEL {rng.randint(1, 5)}",
                    f"P{rng.randint(1, 3)}/F",
                    "PH/F",
                    "PENTHOUSE",
                    f"{floor_number}/F REAR",
                ]),
            },
            "lower_upper_ground": {
                "floor_num": "",
                "floor_description": rng.choice([
                    "LG/F", "UG/F", "M/F", "L1/F", "LG1/F", "LG2/F", "MZZ/F"
                ]),
            },
            "office_suite": {
                "unit_descriptor": rng.choice([
                    "OFFICE", "Office", "SUITE", "Suite", "ROOM", "Room",
                    "RM", "Rm", "STE", "Ste", "UNIT", "Unit",
                ]),
                "unit_no": rng.choice([
                    f"{rng.randint(1, 12):02d}",
                    chr(rng.randint(65, 72)),
                    f"{floor_num_int}{rng.randint(1, 12):02d}"
                ]),
            },
            "car_park_space": {
                "floor_num": "",
                "floor_description": rng.choice(["B1/F", "B2/F", "B3/F", "PODIUM", "G/F", "P1/F"]),
                "unit_descriptor": rng.choice(["CAR PARK SPACE", "CARPARK", "CP", "SPACE", "PARKING NO."]),
                "unit_no": cp_number,
            },
        }
    else:
        is_simplified = language == "zh-Hans"
        floor_suffix = "层" if is_simplified else "樓"
        descriptor = [
            "室",
            "單位" if not is_simplified else "单位",
            "房",
        ][descriptor_index]
        profile = {
            "floor_num": floor_number,
            "floor_description": floor_suffix,
            "unit_descriptor": descriptor,
            "unit_no": unit_number,
            "unit_portion": "",
        }
        overrides = {
            "number_only_unit": {
                "unit_descriptor": "",
                "unit_no": dyn_room_num,
            },
            "hao_shi_unit": {
                "unit_descriptor": rng.choice([
                    "號室" if not is_simplified else "号室",
                    "號單位" if not is_simplified else "号单位",
                    "號房" if not is_simplified else "号房",
                    "號" if not is_simplified else "号",
                ]),
                "unit_no": dyn_room_num,
            },
            "standard_room": {"unit_descriptor": "室"},
            "ground_floor_no_unit": {
                "floor_num": "",
                "floor_description": rng.choice([
                    "G/F", "G/f", "GF", "Gf", "gf",
                    "GROUND FLOOR", "Ground Floor", "ground floor",
                ]),
                "unit_descriptor": "",
                "unit_no": "",
            },
            "whole_floor": {"unit_descriptor": "", "unit_no": ""},
            "unit_without_floor": {"floor_num": "", "floor_description": ""},
            "shop": {
                "floor_num": "",
                "floor_description": "地下",
                "unit_descriptor": rng.choice(["鋪", "號鋪"] if is_simplified else ["舖", "鋪", "號舖"]),
                "unit_no": str(rng.randint(1, 50)),
            },
            "basement": {
                "floor_num": "",
                "floor_description": rng.choice([
                    "B1/F", "B1/f", "B2/F", "B2/f", "BASEMENT", "Basement", "LG/F", "LG/f",
                ]),
                "unit_descriptor": rng.choice(["鋪", "室", "单位"] if is_simplified else ["舖", "鋪", "室", "單位"]),
                "unit_no": str(rng.randint(1, 50)),
            },
            "podium": {
                "floor_num": "",
                "floor_description": "平台",
            },
            "roof": {
                "floor_num": "",
                "floor_description": "天台",
                "unit_descriptor": "",
                "unit_no": "",
            },
            "unit_portion": {
                "unit_portion": (
                    ["東部", "前部", "左部", "連天台"]
                    if not is_simplified
                    else ["东部", "前部", "左部", "连天台"]
                )[portion_index]
            },
            "alphabetic_unit": {
                "unit_descriptor": rng.choice(["室", "單位", ""] if not is_simplified else ["室", "单位", ""]),
                "unit_no": alphabetic_unit,
            },
            "leading_zero_unit": {
                "unit_descriptor": rng.choice(["室", "單位", ""] if not is_simplified else ["室", "单位", ""]),
                "unit_no": leading_zero_unit,
            },
            "duplex_floor": {
                "floor_num": "",
                "floor_description": rng.choice([
                    f"{floor_number}-{floor_num_int + 1}/F",
                    f"{floor_number}-{floor_num_int + 1}/f",
                    f"{floor_number} & {floor_num_int + 1}/F",
                    f"{floor_number}/F-{floor_num_int + 1}/F",
                    f"{floor_number}/f-{floor_num_int + 1}/f",
                ]),
            },
            "combined_units": {
                "unit_descriptor": "室",
                "unit_no": combined_unit_zh,
            },
            "special_floor": {
                "floor_num": "",
                "floor_description": rng.choice([
                    f"{floor_number}A{floor_suffix}",
                    f"{floor_number}{floor_suffix}上層" if not is_simplified else f"{floor_number}{floor_suffix}上层",
                    f"平台{rng.randint(1, 3)}{floor_suffix}",
                    "頂層" if not is_simplified else "顶层",
                    "閣樓" if not is_simplified else "阁楼",
                ]),
            },
            "lower_upper_ground": {
                "floor_num": "",
                "floor_description": (
                    ["低層地下", "高層地下", "閣樓", "一樓低層", "地庫一層"]
                    if not is_simplified
                    else ["低层地下", "高层地下", "阁楼", "一楼低层", "地库一层"]
                )[rng.randrange(5)],
            },
            "office_suite": {
                "unit_descriptor": "辦公室" if not is_simplified else "办公室",
                "unit_no": rng.choice([
                    f"{rng.randint(1, 12):02d}",
                    chr(rng.randint(65, 72)),
                    f"{floor_num_int}{rng.randint(1, 12):02d}"
                ]),
            },
            "car_park_space": {
                "floor_num": "",
                "floor_description": rng.choice(
                    ["地庫一層", "地庫二層", "平台", "地下"]
                    if not is_simplified
                    else ["地库一层", "地库二层", "平台", "地下"]
                ),
                "unit_descriptor": "車位" if not is_simplified else "车位",
                "unit_no": cp_number,
            },
        }
    profile.update(overrides.get(case, {}))
    profile["case"] = case
    return profile


def three_d_case(profile: Mapping[str, Any] | None) -> str:
    if not profile:
        return "no_floor_no_unit"
    if clean_text(profile.get("case")):
        return clean_text(profile.get("case"))
    has_floor = bool(
        clean_text(profile.get("floor_num"))
        or clean_text(profile.get("floor_description"))
    )
    has_unit = bool(
        clean_text(profile.get("unit_no"))
        or clean_text(profile.get("unit_descriptor"))
        or clean_text(profile.get("unit_portion"))
    )
    if has_floor and has_unit:
        return "source_floor_and_unit"
    if has_floor:
        return "source_floor_only"
    if has_unit:
        return "source_unit_only"
    return "no_floor_no_unit"


def floor_shape(profile: Mapping[str, Any] | None) -> str:
    if not profile:
        return "missing"
    value = " ".join(
        part
        for part in (
            clean_text(profile.get("floor_num")),
            clean_text(profile.get("floor_description")),
        )
        if part
    ).upper()
    if not value:
        return "missing"
    if any(token in value for token in ("GROUND", "G/F", "地下")):
        return "ground"
    if any(token in value for token in ("BASEMENT", "B1/F", "B2/F", "地庫", "地库")):
        return "basement"
    if any(token in value for token in ("PODIUM", "P1/F", "平台")):
        return "podium"
    if any(token in value for token in ("ROOF", "PH/F", "天台", "頂層", "顶层")):
        return "roof_or_penthouse"
    if any(token in value for token in ("M/F", "閣樓", "阁楼")):
        return "mezzanine"
    if re.search(r"\d+\s*(?:-|至)\s*\d+", value):
        return "range_or_duplex"
    if re.search(r"\d+[A-Z]", value):
        return "alphanumeric"
    if re.search(r"\d", value):
        return "numeric"
    return "descriptive_other"


def unit_shape(profile: Mapping[str, Any] | None) -> str:
    if not profile:
        return "missing"
    number = clean_text(profile.get("unit_no")).upper()
    portion = clean_text(profile.get("unit_portion"))
    descriptor = clean_text(profile.get("unit_descriptor"))
    if not (number or portion or descriptor):
        return "missing"
    if portion:
        return "with_portion"
    if re.search(r"\s(?:&|AND)\s|及|/|-", number):
        return "combined_or_range"
    if number.isalpha() and len(number) >= 2:
        return "multi_letter"
    if number.isalpha():
        return "single_letter"
    if number.isdigit() and len(number) > 1 and number.startswith("0"):
        return "leading_zero_numeric"
    if number.isdigit():
        return "numeric"
    if re.fullmatch(r"[A-Z]+\d+|\d+[A-Z]+", number):
        return "alphanumeric"
    if not number:
        return "descriptor_only"
    return "other"


def address_shapes(
        components: Mapping[str, Any], three_d: Mapping[str, Any] | None
) -> list[str]:
    shapes: list[str] = []
    if clean_text(components.get("village_name")):
        shapes.append("rural_village")
    if clean_text(components.get("street_name")):
        shapes.append("street")
    if clean_text(components.get("estate_name")):
        shapes.append("estate")
    if clean_text(components.get("block_no")) or clean_text(
            components.get("block_descriptor")
    ):
        shapes.append("block")
    if clean_text(components.get("building_name")):
        shapes.append("named_building")
    else:
        shapes.append("no_building_name")
    if clean_text(components.get("phase_name")) or clean_text(
            components.get("phase_no")
    ):
        shapes.append("phase")
    case = three_d_case(three_d)
    shapes.append(case)
    return shapes


def choose_split(
        group_id: str,
        seed: int,
        train_ratio: float,
        validation_ratio: float,
) -> str:
    value = stable_unit_interval(seed, group_id, "split")
    if value < train_ratio:
        return "train"
    if value < train_ratio + validation_ratio:
        return "validation"
    return "test"


def component_payload(
        components: Mapping[str, Any], three_d: Mapping[str, Any] | None
) -> dict[str, Any]:
    payload = {
        key: value for key, value in components.items() if key != "three_d_addresses"
    }
    payload["floor_num"] = clean_text((three_d or {}).get("floor_num"))
    payload["floor_description"] = clean_text((three_d or {}).get("floor_description"))
    payload["unit_descriptor"] = clean_text((three_d or {}).get("unit_descriptor"))
    payload["unit_no"] = clean_text((three_d or {}).get("unit_no"))
    payload["unit_portion"] = clean_text((three_d or {}).get("unit_portion"))
    return payload


def sanitize_component_payload(
        payload: dict[str, Any], chunks: Sequence[Chunk]
) -> dict[str, Any]:
    """Clear source fields whose semantic chunk/label is absent from the input.

    Label-only checks are insufficient because street and village numbers both
    use ``BUILDING_NUMBER``.  Checking the owning chunk prevents a visible
    village number from accidentally keeping an omitted street number (and the
    reverse) in metadata.
    """

    sanitized = dict(payload)

    visibility_rules = (
        ("region", "REGION", ("region",)),
        ("district", "DISTRICT", ("district",)),
        ("sub_district", "SUB_DISTRICT", ("sub_district",)),
        ("street", "STREET_NAME", ("street_name",)),
        ("street", "BUILDING_NUMBER", ("street_no_from", "street_no_to")),
        ("village", "VILLAGE_NAME", ("village_name",)),
        ("village", "BUILDING_NUMBER", ("village_no_from", "village_no_to")),
        ("estate", "ESTATE_NAME", ("estate_name",)),
        ("phase", "PHASE", ("phase_name", "phase_no")),
        ("block", "BLOCK", ("block_no", "block_descriptor")),
        ("building", "BUILDING_NAME", ("building_name",)),
        (("floor", "unit"), "FLOOR", ("floor_num", "floor_description")),
        ("unit", "UNIT", ("unit_descriptor", "unit_no", "unit_portion")),
    )

    for kind, label, keys in visibility_rules:
        allowed_kinds = (kind,) if isinstance(kind, str) else kind
        visible = any(
            chunk.kind in allowed_kinds
            and any(atom.label == label and atom.text for atom in chunk.atoms)
            for chunk in chunks
        )
        if visible:
            continue
        for key in keys:
            if key in sanitized:
                sanitized[key] = ""

    return sanitized


def build_training_example(
        *,
        components: Mapping[str, Any],
        language: str,
        three_d: Mapping[str, Any] | None,
        three_d_source: str,
        split: str,
        group_id: str,
        geo_address: str,
        csu_id: str,
        coordinates: Mapping[str, Any],
        source_ref: SourceRef,
        feature_index: int,
        variant_index: int,
        profile_index: int,
        variants_per_address: int,
        messy_rate: float,
        typo_rate: float,
        component_drop_rate: float,
        region_abbreviation_rate: float,
        fullwidth_punctuation_rate: float,
        chinese_separator_noise_rate: float,
        synthetic_subdistrict_rate: float,
        district_suffix_noise_rate: float,
        chinese_region_variation_rate: float,
        floor_unit_fusion_rate: float,
        canonical_floor_unit_fusion_rate: float,
        english_upper_rate: float,
        english_title_rate: float,
        english_lower_rate: float,
        english_mixed_rate: float,
        seed: int,
        alternate_components: Mapping[str, Any] | None = None,
        alternate_language: str | None = None,
        alternate_three_d: Mapping[str, Any] | None = None,
        output_language: str | None = None,
        mix_mode: str | None = None,
) -> dict[str, Any]:
    effective_language = output_language or language
    rng = stable_rng(
        seed,
        group_id,
        profile_index,
        variant_index,
        "mixed" if alternate_components is not None else "single_language",
        mix_mode or "",
        "variant",
    )
    base_chunks = canonical_order(build_chunks(components, language, three_d), language)

    scenarios: list[str] = []
    if alternate_components is not None and alternate_language:
        alternate_chunks = canonical_order(
            build_chunks(alternate_components, alternate_language, alternate_three_d),
            alternate_language,
        )
        base_chunks, _, applied_mix_mode = mix_language_chunks(
            base_chunks,
            alternate_chunks,
            base_language=language,
            alternate_language=alternate_language,
            mode=mix_mode or "balanced_components",
            rng=rng,
        )
        scenarios.extend(["mixed_chinese_english", applied_mix_mode])

    separator_language = "en" if alternate_components is not None else language
    canonical_separator = separator_for("canonical", separator_language, rng)
    canonical_text, _ = render_chunks(base_chunks, canonical_separator)

    chunks = copy.deepcopy(base_chunks)
    is_structurally_messy = False

    if (
            variant_index > 0
            and rng.random() < chinese_separator_noise_rate
    ):
        for chunk in chunks:
            # Only apply this to Chinese components
            if chunk.source_language and chunk.source_language.startswith("zh-"):

                # 1. Insert wrong separators BEFORE the Unit (e.g., "5樓 / 12室")
                if chunk.kind == "unit" and len(chunk.atoms) > 0:
                    messy_sep = rng.choice([" ", "  ", " / ", " - ", "，"])
                    # Insert as an unlabeled Atom so the NER model learns to ignore the symbol
                    chunk.atoms.insert(0, Atom(messy_sep, None))

                # 2. Inject space INSIDE the Floor or Unit itself (e.g., "12 室" instead of "12室")
                if chunk.kind in {"floor", "unit"}:
                    for atom in chunk.atoms:
                        if atom.label and rng.random() < 0.5:
                            # Safely insert a space between numbers and Chinese characters using Regex
                            atom.text = re.sub(r"(\d+)([\u4e00-\u9fa5]+)", r"\1 \2", atom.text)

        scenarios.append("chinese_messy_separation")
    # Synthetic sub-districts teach the model the real vocabulary of
    # Hong Kong sub-districts.  Applied whenever the address has a
    # recognisable district and no existing sub-district chunk.
    # Rate is controlled by --synthetic-subdistrict-rate (default 0.30).
    local_district_rng = stable_rng(
        seed,
        group_id,
        effective_language,
        profile_index,
        variant_index,
        mix_mode or "",
        "local_district"
    )
    if local_district_rng.random() < synthetic_subdistrict_rate:
        local_scenario = apply_localized_district(
            chunks, effective_language, local_district_rng
        )
        if local_scenario:
            scenarios.append(local_scenario)

    if variant_index > 0:
        # Create a stable RNG specific to this operation to maintain reproducibility
        district_suffix_rng = stable_rng(
            seed,
            group_id,
            effective_language,
            profile_index,
            variant_index,
            mix_mode or "",
            "district_suffix"
        )

        if district_suffix_rng.random() < district_suffix_noise_rate:
            suffix_scenario = apply_district_suffix_noise(
                chunks, effective_language, district_suffix_rng
            )
            if suffix_scenario:
                scenarios.append(suffix_scenario)

    if variant_index == 0:
        separator_style = "canonical"
        scenarios.append("canonical_order")
    else:
        adjusted_rate = (
            min(1.0, messy_rate * variants_per_address / (variants_per_address - 1))
            if variants_per_address > 1
            else 0.0
        )
        is_structurally_messy = rng.random() < adjusted_rate
        if is_structurally_messy:
            operation_count = rng.choice([1, 1, 2, 2, 3])
            operations = ["reorder", "mixed_punctuation", "space_noise"]
            has_english = language == "en" or alternate_language == "en"
            if has_english:
                operations.extend(["abbreviation"])
            for operation in rng.sample(
                    operations, min(operation_count, len(operations))
            ):
                if operation == "reorder":
                    chunks, mode = reorder_chunks(chunks, rng)
                    scenarios.extend(["reordered_components", mode])
                elif operation == "mixed_punctuation":
                    scenarios.append("mixed_punctuation")
                elif operation == "space_noise":
                    scenarios.append("irregular_whitespace")
                elif operation == "abbreviation":
                    apply_english_abbreviations(chunks, rng)
                    scenarios.append("english_abbreviation")
            separator_style = "mixed" if "mixed_punctuation" in scenarios else "space"
        else:
            separator_style = rng.choice(["comma", "space", "compact"])
            scenarios.append(
                {
                    "comma": "comma_separated",
                    "space": "no_comma_space_separated",
                    "compact": "compact_no_comma",
                }[separator_style]
            )

    if variant_index > 0:
        casing_rng = stable_rng(
            seed, group_id, effective_language, profile_index, variant_index, mix_mode or "", "casing"
        )
        has_english_chunks = any(chunk.source_language == "en" for chunk in chunks)
        if has_english_chunks:
            applied_case = apply_english_case_noise(
                chunks,
                casing_rng,
                english_upper_rate,
                english_title_rate,
                english_lower_rate,
                english_mixed_rate
            )
            scenarios.append(f"english_case_{applied_case}")

    omission_rng = stable_rng(
        seed,
        group_id,
        effective_language,
        profile_index,
        variant_index,
        mix_mode or "",
        "component_omission",
    )
    if alternate_components is not None:
        adjusted_component_drop_rate = component_drop_rate
        omission_eligible = True
    else:
        adjusted_component_drop_rate = (
            min(
                1.0,
                component_drop_rate * variants_per_address / (variants_per_address - 1),
            )
            if variants_per_address > 1
            else 0.0
        )
        omission_eligible = variant_index > 0

    omitted_component_kinds: list[str] = []
    if omission_eligible and omission_rng.random() < adjusted_component_drop_rate:
        chunks_before_omission = copy.deepcopy(chunks)
        chunks, omitted_component_kinds = importance_weighted_omission(
            chunks, omission_rng
        )
        rendered_after_omission = {
            chunk.source_language for chunk in chunks if chunk.source_language
        }
        if alternate_components is not None and len(rendered_after_omission) < 2:
            chunks = chunks_before_omission
            omitted_component_kinds = []
        if omitted_component_kinds:
            scenarios.append("importance_weighted_component_omission")
            scenarios.extend(f"missing_{kind}" for kind in omitted_component_kinds)

    abbreviation_rng = stable_rng(
        seed,
        group_id,
        effective_language,
        profile_index,
        variant_index,
        mix_mode or "",
        "region_abbreviation",
    )
    if alternate_components is not None:
        adjusted_region_abbreviation_rate = region_abbreviation_rate
        abbreviation_eligible = True
    else:
        adjusted_region_abbreviation_rate = (
            min(
                1.0,
                region_abbreviation_rate
                * variants_per_address
                / (variants_per_address - 1),
            )
            if variants_per_address > 1
            else 0.0
        )
        abbreviation_eligible = variant_index > 0
    if (
            abbreviation_eligible
            and abbreviation_rng.random() < adjusted_region_abbreviation_rate
            and apply_region_abbreviation(chunks, abbreviation_rng)
    ):
        scenarios.append("english_region_abbreviation")

    if (
            abbreviation_eligible
            and abbreviation_rng.random() < chinese_region_variation_rate
    ):
        chinese_scenario = apply_chinese_region_variation(
            chunks, effective_language, abbreviation_rng
        )
        if chinese_scenario:
            scenarios.append(chinese_scenario)

    separator = separator_for(separator_style, separator_language, rng)
    if "irregular_whitespace" in scenarios:
        separator = rng.choice(["  ", " ,", ",  ", "， ", " / "])
    adjusted_fullwidth_rate = (
        min(1.0, fullwidth_punctuation_rate * variants_per_address / (variants_per_address - 1))
        if variants_per_address > 1 else 0.0
    )
    if variant_index > 0 and rng.random() < adjusted_fullwidth_rate:
        # Translate the separators between chunks
        separator = separator.translate(FULLWIDTH_TRANSLATION)
        # Translate internal punctuation (like parentheses) inside the text atoms
        for chunk in chunks:
            for atom in chunk.atoms:
                atom.text = atom.text.translate(FULLWIDTH_TRANSLATION)
        scenarios.append("fullwidth_punctuation")
    # Compact / inverted floor+unit fusion (27LD, LD27, 27樓LD, ...).
    fusion_rng = stable_rng(
        seed,
        group_id,
        effective_language,
        profile_index,
        variant_index,
        mix_mode or "",
        "floor_unit_fusion",
    )
    fusion_rate = (
        floor_unit_fusion_rate
        if variant_index > 0
        else canonical_floor_unit_fusion_rate
    )
    if fusion_rng.random() < fusion_rate:
        fusion_scenario = apply_floor_unit_fusion(
            chunks, three_d, language, fusion_rng
        )
        if fusion_scenario:
            scenarios.append(fusion_scenario)
            scenarios.append("floor_unit_compact_or_inverted")
    adjusted_typo_rate = (
        min(1.0, typo_rate * variants_per_address / (variants_per_address - 1))
        if variants_per_address > 1
        else 0.0
    )
    typo_rng = stable_rng(
        seed,
        group_id,
        effective_language,
        profile_index,
        variant_index,
        mix_mode or "",
        "minor_typo",
    )
    corruption: dict[str, str] | None = None
    if variant_index > 0 and typo_rng.random() < adjusted_typo_rate:
        corruption = introduce_minor_typo(chunks, typo_rng)
        if corruption:
            scenarios.extend(["minor_typo", f"typo_{corruption['operation']}"])

    text, entities = render_chunks(chunks, separator)
    validate_entities(text, entities)
    observed_components: dict[str, list[str]] = defaultdict(list)
    for entity in entities:
        observed_components[str(entity["label"])].append(str(entity["text"]))

    active_component_payload = sanitize_component_payload(
        component_payload(components, three_d), chunks
    )

    rendered_languages = {
        chunk.source_language for chunk in chunks if chunk.source_language
    }
    is_code_switched = alternate_components is not None and len(rendered_languages) >= 2

    case = three_d_case(three_d)
    scenarios.append(case)
    if effective_language.startswith("mixed-"):
        scenarios.append("code_switched_input")
        if "zh-Hans" in effective_language:
            scenarios.append("mixed_simplified_english")
        else:
            scenarios.append("mixed_traditional_english")
    elif language == "zh-Hans":
        scenarios.append("simplified_conversion")
    elif language == "zh-Hant":
        scenarios.append("traditional_chinese")
    else:
        scenarios.append("english")

    example_id = stable_digest(
        DATASET_SCHEMA_VERSION,
        group_id,
        effective_language,
        source_ref.as_string(),
        feature_index,
        profile_index,
        variant_index,
        text,
    )[:24]
    return {
        "id": example_id,
        "group_id": group_id,
        "split": split,
        "language": effective_language,
        "text": text,
        "canonical_text": canonical_text,
        "entities": entities,
        "observed_components": dict(observed_components),
        "components": active_component_payload,
        "parallel_components": (
            {
                language: component_payload(components, three_d),
                alternate_language: component_payload(
                    alternate_components, alternate_three_d
                ),
            }
            if alternate_components is not None and alternate_language
            else None
        ),
        "component_languages": [
            {
                "kind": chunk.kind,
                "language": chunk.source_language or language,
            }
            for chunk in chunks
        ],
        "address_shapes": address_shapes(components, three_d),
        "scenario": list(dict.fromkeys(scenarios)),
        "is_messy": (
                is_structurally_messy
                or corruption is not None
                or bool(omitted_component_kinds)
        ),
        "is_structurally_messy": is_structurally_messy,
        "has_omission": bool(omitted_component_kinds),
        "omitted_component_kinds": omitted_component_kinds,
        "has_typo": corruption is not None,
        "corruptions": [corruption] if corruption else [],
        "three_d_source": three_d_source,
        "three_d_case": case,
        "floor_shape": floor_shape(three_d),
        "unit_shape": unit_shape(three_d),
        "mix_mode": mix_mode,
        "geo_address": geo_address,
        "csu_id": csu_id,
        "coordinates": dict(coordinates),
        "provenance": {
            "source_file": source_ref.as_string(),
            "feature_index": feature_index,
            "profile_index": profile_index,
            "variant_index": variant_index,
            "government_record": True,
            "floor_unit_verified_by_als": three_d_source == "als",
            "code_switched": is_code_switched,
        },
    }


def select_real_three_d(
        profiles: Sequence[Mapping[str, Any]],
        maximum: int,
        *,
        seed: int,
        group_id: str,
) -> list[Mapping[str, Any]]:
    if maximum <= 0 or len(profiles) <= maximum:
        return list(profiles)
    ranked = sorted(
        enumerate(profiles),
        key=lambda pair: stable_digest(seed, group_id, pair[0], "real_3d_sample"),
    )
    selected_indices = sorted(index for index, _ in ranked[:maximum])
    return [profiles[index] for index in selected_indices]


def select_real_three_d_indices(
        profile_count: int,
        maximum: int,
        *,
        seed: int,
        group_id: str,
) -> list[int]:
    indices = list(range(profile_count))
    if maximum <= 0 or profile_count <= maximum:
        return indices
    chosen = sorted(
        indices,
        key=lambda index: stable_digest(seed, group_id, index, "real_3d_sample"),
    )[:maximum]
    return sorted(chosen)


def derive_group_id(
        premises: Mapping[str, Any],
        components_by_language: Mapping[str, Mapping[str, Any]],
        coordinates: Mapping[str, Any],
        source_ref: SourceRef,
        feature_index: int,
) -> tuple[str, str, str]:
    geo_address = clean_text(get_ci(premises, "GeoAddress"))
    csu_info = ensure_mapping(get_ci(premises, "BuildingCsuInformation"))
    csu_id = clean_text(get_ci(csu_info, "CsuId", "BuildingCsuId"))
    identity = csu_id or geo_address
    if not identity:
        names: list[str] = []
        for language in ("en", "zh-Hant"):
            components = components_by_language.get(language, {})
            names.extend(
                clean_text(components.get(key))
                for key in (
                    "building_name",
                    "estate_name",
                    "block_no",
                    "street_name",
                    "street_no_from",
                    "village_name",
                    "village_no_from",
                )
            )
        coordinate_key = (
            coordinates.get("easting_hk1980"),
            coordinates.get("northing_hk1980"),
            coordinates.get("longitude_wgs84"),
            coordinates.get("latitude_wgs84"),
        )
        useful_names = [name for name in names if name]
        useful_coordinates = [value for value in coordinate_key if value is not None]
        if useful_names or useful_coordinates:
            # Excluding source/index here makes duplicate records without an
            # identifier land in the same split instead of leaking across it.
            identity = stable_digest(
                "fallback-address-content", *useful_names, *useful_coordinates
            )[:24]
        else:
            identity = stable_digest(
                "fallback-source-position", source_ref.as_string(), feature_index
            )[:24]
    return f"address-{stable_digest(identity)[:24]}", geo_address, csu_id


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise DatasetBuildError(
                f"Output directory is not empty: {output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def write_json_line(handle: Any, value: Mapping[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


class Audit:
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.by_split: Counter[str] = Counter()
        self.by_language: Counter[str] = Counter()
        self.by_scenario: Counter[str] = Counter()
        self.by_shape: Counter[str] = Counter()
        self.by_district: Counter[str] = Counter()
        self.by_three_d_source: Counter[str] = Counter()
        self.by_three_d_case: Counter[str] = Counter()
        self.by_floor_shape: Counter[str] = Counter()
        self.by_unit_shape: Counter[str] = Counter()
        self.by_mix_mode: Counter[str] = Counter()
        self.by_typo_operation: Counter[str] = Counter()
        self.by_omitted_component: Counter[str] = Counter()
        self.by_omission_profile: Counter[str] = Counter()
        self.by_observed_region: Counter[str] = Counter()
        self.by_source: Counter[str] = Counter()
        self.schema_paths: Counter[str] = Counter()
        self.unknown_premises_keys: Counter[str] = Counter()
        self.groups_by_split: dict[str, set[str]] = defaultdict(set)

    def observe_example(self, example: Mapping[str, Any]) -> None:
        self.counts["examples"] += 1
        self.by_split[clean_text(example.get("split"))] += 1
        self.by_language[clean_text(example.get("language"))] += 1
        self.by_three_d_source[clean_text(example.get("three_d_source"))] += 1
        self.by_three_d_case[clean_text(example.get("three_d_case"))] += 1
        self.by_floor_shape[clean_text(example.get("floor_shape"))] += 1
        self.by_unit_shape[clean_text(example.get("unit_shape"))] += 1
        mix_mode = clean_text(example.get("mix_mode"))
        if mix_mode:
            self.by_mix_mode[mix_mode] += 1
        self.groups_by_split[clean_text(example.get("split"))].add(
            clean_text(example.get("group_id"))
        )
        components = ensure_mapping(example.get("components"))
        self.by_district[clean_text(components.get("district")) or "<missing>"] += 1
        for scenario in ensure_list(example.get("scenario")):
            self.by_scenario[clean_text(scenario)] += 1
        for shape in ensure_list(example.get("address_shapes")):
            self.by_shape[clean_text(shape)] += 1
        if bool(example.get("is_messy")):
            self.counts["messy_examples"] += 1
        if bool(example.get("is_structurally_messy")):
            self.counts["structurally_messy_examples"] += 1
        if bool(example.get("has_typo")):
            self.counts["typo_examples"] += 1
            for corruption in ensure_list(example.get("corruptions")):
                operation = clean_text(ensure_mapping(corruption).get("operation"))
                if operation:
                    self.by_typo_operation[operation] += 1
        if bool(ensure_mapping(example.get("provenance")).get("code_switched")):
            self.counts["code_switched_examples"] += 1
        omitted = sorted(
            clean_text(kind)
            for kind in ensure_list(example.get("omitted_component_kinds"))
            if clean_text(kind)
        )
        if omitted:
            self.counts["incomplete_examples"] += 1
            self.by_omission_profile["+".join(omitted)] += 1
            for kind in omitted:
                self.by_omitted_component[kind] += 1
        observed = ensure_mapping(example.get("observed_components"))
        observed_regions = [
            clean_text(value)
            for value in ensure_list(observed.get("REGION"))
            if clean_text(value)
        ]
        if observed_regions:
            for region in observed_regions:
                self.by_observed_region[region] += 1
            if any(
                    canonical_region(region, "en") in ENGLISH_REGION_ABBREVIATIONS
                    for region in observed_regions
            ):
                self.counts["examples_with_observed_english_region"] += 1
        else:
            self.by_observed_region["<omitted>"] += 1
        if "english_region_abbreviation" in ensure_list(example.get("scenario")):
            self.counts["region_abbreviation_examples"] += 1


def counter_rows(
        counter: Counter[str], limit: int | None = None
) -> list[tuple[str, int]]:
    rows = counter.most_common(limit)
    return [(key or "<missing>", count) for key, count in rows]


def markdown_table(rows: Sequence[tuple[str, int]], left_title: str) -> str:
    lines = [f"| {left_title} | Count |", "| --- | ---: |"]
    for key, count in rows:
        safe = str(key).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {safe} | {count:,} |")
    if not rows:
        lines.append("| (none) | 0 |")
    return "\n".join(lines)


def write_audit_files(
        output_dir: Path,
        audit: Audit,
        *,
        input_paths: Sequence[Path],
        args: argparse.Namespace,
) -> None:
    overlap = (
            (audit.groups_by_split["train"] & audit.groups_by_split["validation"])
            | (audit.groups_by_split["train"] & audit.groups_by_split["test"])
            | (audit.groups_by_split["validation"] & audit.groups_by_split["test"])
    )
    total_examples = audit.counts["examples"]
    messy_rate = (
        audit.counts["messy_examples"] / total_examples if total_examples else 0.0
    )
    structural_rate = (
        audit.counts["structurally_messy_examples"] / total_examples
        if total_examples
        else 0.0
    )
    typo_rate = (
        audit.counts["typo_examples"] / total_examples if total_examples else 0.0
    )
    mixed_rate = (
        audit.counts["code_switched_examples"] / total_examples
        if total_examples
        else 0.0
    )
    omission_rate = (
        audit.counts["incomplete_examples"] / total_examples if total_examples else 0.0
    )
    region_abbreviation_rate = (
        audit.counts["region_abbreviation_examples"] / total_examples
        if total_examples
        else 0.0
    )
    observed_english_region_examples = audit.counts[
        "examples_with_observed_english_region"
    ]
    region_abbreviation_rate_eligible = (
        audit.counts["region_abbreviation_examples"] / observed_english_region_examples
        if observed_english_region_examples
        else 0.0
    )
    summary = {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "input_files": [str(path) for path in input_paths],
        "configuration": {
            "seed": args.seed,
            "variants_per_address": args.variants_per_address,
            "mixed_variants_per_address": args.mixed_variants_per_address,
            "messy_rate_requested": args.messy_rate,
            "typo_rate_requested": args.typo_rate,
            "component_drop_rate_requested": args.component_drop_rate,
            "region_abbreviation_rate_requested": args.region_abbreviation_rate,
            "synthetic_3d_rate": args.synthetic_3d_rate,
            "fullwidth_punctuation_rate": args.fullwidth_punctuation_rate,
            "chinese_separator_noise_rate": args.chinese_separator_noise_rate,
            "synthetic_subdistrict_rate": args.synthetic_subdistrict_rate,
            "district_suffix_noise_rate": args.district_suffix_noise_rate,
            "chinese_region_variation_rate": args.chinese_region_variation_rate,
            "floor_unit_fusion_rate": args.floor_unit_fusion_rate,
            "canonical_floor_unit_fusion_rate": args.canonical_floor_unit_fusion_rate,
            "max_real_3d_per_address": args.max_real_3d_per_address,
            "simplified": args.simplified,
            "progress": args.progress,
            "train_ratio": args.train_ratio,
            "validation_ratio": args.validation_ratio,
            "test_ratio": 1.0 - args.train_ratio - args.validation_ratio,
        },
        "counts": dict(audit.counts),
        "actual_messy_rate": messy_rate,
        "actual_structural_messy_rate": structural_rate,
        "actual_typo_rate": typo_rate,
        "actual_code_switched_rate": mixed_rate,
        "actual_component_omission_rate": omission_rate,
        "actual_region_abbreviation_rate_over_all_examples": region_abbreviation_rate,
        "actual_region_abbreviation_rate_among_observed_english_regions": (
            region_abbreviation_rate_eligible
        ),
        "by_split": dict(audit.by_split),
        "groups_by_split": {
            split: len(groups) for split, groups in audit.groups_by_split.items()
        },
        "split_group_overlap_count": len(overlap),
        "by_language": dict(audit.by_language),
        "by_three_d_source": dict(audit.by_three_d_source),
        "by_three_d_case": dict(audit.by_three_d_case),
        "by_floor_shape": dict(audit.by_floor_shape),
        "by_unit_shape": dict(audit.by_unit_shape),
        "by_mix_mode": dict(audit.by_mix_mode),
        "by_typo_operation": dict(audit.by_typo_operation),
        "by_omitted_component": dict(audit.by_omitted_component),
        "by_omission_profile": dict(audit.by_omission_profile),
        "by_observed_region": dict(audit.by_observed_region),
        "by_shape": dict(audit.by_shape),
        "by_scenario": dict(audit.by_scenario),
        "by_district": dict(audit.by_district),
        "by_source": dict(audit.by_source),
        "unknown_premises_keys": dict(audit.unknown_premises_keys),
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "schema_paths.json").write_text(
        json.dumps(dict(audit.schema_paths.most_common()), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "labels.json").write_text(
        json.dumps(
            {
                "dataset_schema_version": DATASET_SCHEMA_VERSION,
                "span_labels": LABELS,
                "bio_labels": ["O"]
                              + [f"{prefix}-{label}" for label in LABELS for prefix in ("B", "I")],
                "offset_unit": "Unicode code points",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    report = f"""# HK address dataset audit

This report was generated with the dataset. It describes observed coverage; it
does **not** certify that the data is unbiased or complete.

## Build result

- Source features seen: {audit.counts["features_seen"]:,}
- Structured features written: {audit.counts["structured_records"]:,}
- Training examples written: {total_examples:,}
- Rejected features: {audit.counts["rejected_features"]:,}
- Duplicate address groups observed: {audit.counts["duplicate_group_ids"]:,}
- Overall noisy/messy share: {messy_rate:.2%}
- Structural-messy share: {structural_rate:.2%} (requested: {args.messy_rate:.2%})
- Minor-typo share: {typo_rate:.2%} (requested: {args.typo_rate:.2%})
- Incomplete-input share: {omission_rate:.2%} (requested: {args.component_drop_rate:.2%})
- English-region abbreviation share: {region_abbreviation_rate_eligible:.2%} of rows with an observed English region (requested: {args.region_abbreviation_rate:.2%}; {region_abbreviation_rate:.2%} of all rows)
- Chinese-English code-switched share: {mixed_rate:.2%}
- Address groups appearing in more than one split: {len(overlap):,}

## Split counts

{markdown_table(counter_rows(audit.by_split), "Split")}

## Language counts

{markdown_table(counter_rows(audit.by_language), "Language")}

## 3D provenance

`als` means the floor/unit fields came from the government record. `synthetic`
means they are grammar-generated parsing examples, not verified premises.

{markdown_table(counter_rows(audit.by_three_d_source), "3D source")}

## Floor/unit cases

{markdown_table(counter_rows(audit.by_three_d_case), "Case")}

## Floor shapes

{markdown_table(counter_rows(audit.by_floor_shape), "Floor shape")}

## Unit/flat shapes

{markdown_table(counter_rows(audit.by_unit_shape), "Unit shape")}

## Chinese-English mixing modes

Mixed rows keep parallel canonical components and record the source language of
every rendered chunk.

{markdown_table(counter_rows(audit.by_mix_mode), "Mix mode")}

## Minor typo operations

Typos are applied to names only, never to floor or unit identifiers. Every
corruption stores its original and corrupted value.

{markdown_table(counter_rows(audit.by_typo_operation), "Typo operation")}

## Missing input components

Region has the highest omission weight, followed by district and location.
Street/building/estate/block components are removed less often and only when a
different usable location anchor remains. Missing chunks produce no entity in
the training target.

{markdown_table(counter_rows(audit.by_omitted_component), "Omitted component")}

### Omission combinations

{markdown_table(counter_rows(audit.by_omission_profile), "Omission profile")}

## Observed region forms

English government codes are canonically expanded to `Hong Kong`, `Kowloon`,
and `New Territories`. `HK`, `KLN`, and `NT` remain minority variants. The
`<omitted>` row measures examples with no region entity.

{markdown_table(counter_rows(audit.by_observed_region), "Observed region")}

## Address shapes

One example may have several shape tags.

{markdown_table(counter_rows(audit.by_shape), "Shape")}

## Augmentation scenarios

One example may have several scenario tags.

{markdown_table(counter_rows(audit.by_scenario), "Scenario")}

## District distribution

District equality is not automatically desirable: it may differ from the real
deployment distribution. Compare this table with a sample of actual user input
before choosing sampling weights.

{markdown_table(counter_rows(audit.by_district), "District")}

## Remaining bias and coverage risks

- ALS is not exhaustive, and its registered building-name coverage can lag new
  construction or omit informal names.
- Official 3D floor/unit arrays cover HKHA public housing only. Private housing,
  commercial premises, most shops, and rural addresses lack verified 3D truth.
- Simplified Chinese is converted from Traditional Chinese. It is not a corpus
  of naturally typed Simplified Chinese Hong Kong addresses.
- Synthetic variants include sparse name typos and component-level bilingual
  switching plus importance-weighted omissions, but they do not reproduce the
  true frequency of missing fields, typos, speech-to-text errors, OCR errors,
  nicknames, or mixed-language user behaviour.
- The government-data distribution is not the same as the future user-query
  distribution. A held-out, manually reviewed production-like test set is still
  required.
"""
    (output_dir / "audit_report.md").write_text(report, encoding="utf-8")


def write_dataset_card(output_dir: Path) -> None:
    card = """# Hong Kong multilingual address-component dataset

    ## Files

    - `train.jsonl`, `validation.jsonl`, `test.jsonl`: augmented span-labelled data.
    - `structured_records.jsonl`: loss-minimized extraction of each source feature,
      including all real ALS 3D arrays before training-time sampling.
    - `rejects.jsonl`: records that could not be used, with reasons.
    - `labels.json`: span labels and their BIO equivalents.
    - `audit_summary.json`, `audit_report.md`: distributions and leakage checks.
    - `schema_paths.json`: observed source paths for schema-drift review.

    ## Training fields

    Use `text` as model input and `entities` as character-span supervision.
    `observed_components` contains only component text actually present in the
    input; an omitted component has no entity and no observed output. Each entity's
    `component_kind` distinguishes shared labels such as street versus village
    `BUILDING_NUMBER`. `components` clears source subfields whose owning
    chunk/label was omitted; it is audit metadata, not the training target.
    `parallel_components` retains canonical bilingual source metadata for mixed
    rows. Keep examples with the same `group_id` in one split; the generator
    already enforces this.

    `has_omission` and `omitted_component_kinds` record incomplete inputs. English
    regions are canonicalized to `Hong Kong`, `Kowloon`, and `New Territories`,
    while abbreviations are generated only as a minority scenario.

    Rows whose `language` starts with `mixed-` contain component-level Chinese and
    English switching. `parallel_components` keeps both canonical language forms,
    `component_languages` identifies the source language of each rendered chunk,
    and `mix_mode` describes whether one component, the 3D portion, or roughly half
    the address was switched.

    `has_typo` rows contain exactly one low-rate typo in an address name. The
    `corruptions` array preserves the original and corrupted text. Floor, unit, and
    building-number identifiers are never typo-corrupted.

    `three_d_source` has three possible values:

    - `als`: floor/unit data is present in the government record.
    - `synthetic`: floor/unit syntax was generated to teach parsing only.
    - `none`: a valid 2D/no-floor/no-unit example.

    `floor_shape` and `unit_shape` expose coarse audit categories such as ground,
    basement, duplex/range, alphabetic unit, leading-zero unit, and combined units.

    Never use `synthetic` rows as evidence that a physical flat, floor, or shop
    exists. For exact address resolution, combine a token classifier with retrieval
    against the address gazetteer; a classifier alone cannot reliably invent or
    validate arbitrary 3D premises.
    """
    (output_dir / "DATASET_CARD.md").write_text(card, encoding="utf-8")


def build_dataset(args: argparse.Namespace) -> Path:
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        raise DatasetBuildError(f"Input directory does not exist: {input_dir}")
    probability_args = {
        "--messy-rate": args.messy_rate,
        "--typo-rate": args.typo_rate,
        "--component-drop-rate": args.component_drop_rate,
        "--region-abbreviation-rate": args.region_abbreviation_rate,
        "--synthetic-3d-rate": args.synthetic_3d_rate,
        "--fullwidth-punctuation-rate": args.fullwidth_punctuation_rate,
        "--chinese-separator-noise-rate": args.chinese_separator_noise_rate,
        "--synthetic-subdistrict-rate": args.synthetic_subdistrict_rate,
        "--district-suffix-noise-rate": args.district_suffix_noise_rate,
        "--chinese-region-variation-rate": args.chinese_region_variation_rate,
        "--floor-unit-fusion-rate": args.floor_unit_fusion_rate,
        "--canonical-floor-unit-fusion-rate": args.canonical_floor_unit_fusion_rate,
        "--mixed-variants-per-address": args.mixed_variants_per_address,
        "--filter-missing-building": args.filter_missing_building,
        "--filter-missing-street": args.filter_missing_street,
        "--filter-missing-region": args.filter_missing_region,
        "--filter-missing-district": args.filter_missing_district,
        "--filter-missing-estate": args.filter_missing_estate,
    }
    for option, value in probability_args.items():
        if not (0.0 <= value <= 1.0):
            raise DatasetBuildError(f"{option} must be between 0 and 1")
    if args.variants_per_address < 1:
        raise DatasetBuildError("--variants-per-address must be at least 1")
    if args.max_real_3d_per_address < 0:
        raise DatasetBuildError("--max-real-3d-per-address cannot be negative")
    english_case_weights = (
        args.english_upper_rate,
        args.english_title_rate,
        args.english_lower_rate,
        args.english_mixed_rate,
    )
    if any(weight < 0 for weight in english_case_weights):
        raise DatasetBuildError("English case weights cannot be negative")
    if sum(english_case_weights) <= 0:
        raise DatasetBuildError("At least one English case weight must be positive")
    if not (0.0 < args.train_ratio < 1.0):
        raise DatasetBuildError("--train-ratio must be between 0 and 1")
    if not (0.0 <= args.validation_ratio < 1.0):
        raise DatasetBuildError("--validation-ratio must be between 0 and 1")
    if args.train_ratio + args.validation_ratio >= 1.0:
        raise DatasetBuildError("train + validation ratios must be less than 1")

    input_paths = discover_inputs(input_dir, args.recursive, output_dir)
    if not input_paths:
        raise DatasetBuildError(
            f"No .geojson, .json, or .zip inputs found in {input_dir}"
        )
    converter = load_opencc() if args.simplified else None
    tqdm_factory = load_tqdm() if args.progress else None
    total_features: int | None = None
    if tqdm_factory is not None:
        total_features = count_all_features(
            input_paths,
            input_dir=input_dir,
            max_features=args.max_features,
            tqdm_factory=tqdm_factory,
        )
        LOGGER.info("Found %s source address features", total_features)
    prepare_output_dir(output_dir, args.overwrite)

    audit = Audit()
    seen_groups: Counter[str] = Counter()
    split_handles = {
        split: (output_dir / f"{split}.jsonl").open("w", encoding="utf-8")
        for split in ("train", "validation", "test")
    }
    structured_handle = (output_dir / "structured_records.jsonl").open(
        "w", encoding="utf-8"
    )
    rejects_handle = (output_dir / "rejects.jsonl").open("w", encoding="utf-8")
    generation_progress = (
        tqdm_factory(
            total=total_features,
            desc="Generating address examples",
            unit=" features",
            dynamic_ncols=True,
        )
        if tqdm_factory is not None
        else None
    )

    try:
        for feature, source_ref, feature_index in iter_all_features(
                input_paths, input_dir=input_dir, max_features=args.max_features
        ):
            if generation_progress is not None:
                generation_progress.update(1)
            audit.counts["features_seen"] += 1
            audit.by_source[source_ref.as_string()] += 1
            premises = find_premises_address(feature)
            if not premises:
                audit.counts["rejected_features"] += 1
                write_json_line(
                    rejects_handle,
                    {
                        "source_file": source_ref.as_string(),
                        "feature_index": feature_index,
                        "reason": "PremisesAddress not found",
                    },
                )
                continue

            for path in iter_leaf_paths(premises):
                audit.schema_paths[path] += 1
            for key in premises:
                if key_normal_form(key) not in KNOWN_PREMISES_KEYS:
                    audit.unknown_premises_keys[str(key)] += 1

            components_by_language: dict[str, dict[str, Any]] = {
                "en": normalize_address_components(premises, "en"),
                "zh-Hant": normalize_address_components(premises, "zh-Hant"),
            }
            if converter is not None:
                components_by_language["zh-Hans"] = convert_nested_strings(
                    components_by_language["zh-Hant"], converter
                )

            usable_languages = [
                language
                for language in LANGUAGE_ORDER
                if language in components_by_language
                   and address_has_content(components_by_language[language])
            ]
            if not usable_languages:
                audit.counts["rejected_features"] += 1
                write_json_line(
                    rejects_handle,
                    {
                        "source_file": source_ref.as_string(),
                        "feature_index": feature_index,
                        "reason": "No usable English or Chinese address components",
                    },
                )
                continue
            filter_rng = stable_rng(
                args.seed,
                source_ref.as_string(),
                feature_index,
                "missing_info_filter",
            )

            filter_rates = {
                "building_name": args.filter_missing_building,
                "street_name": args.filter_missing_street,
                "region": args.filter_missing_region,
                "district": args.filter_missing_district,
                "estate_name": args.filter_missing_estate,
            }

            should_skip_row = False
            for comp_key, skip_prob in filter_rates.items():
                missing_in_every_language = all(
                    not clean_text(components_by_language[lang].get(comp_key))
                    for lang in usable_languages
                )
                if skip_prob > 0.0 and missing_in_every_language:
                    if filter_rng.random() < skip_prob:
                        should_skip_row = True
                        audit.counts[f"filtered_missing_{comp_key}"] += 1
                        break

            if should_skip_row:
                audit.counts["filtered_due_to_missing_source_info"] += 1
                continue
            coordinates = extract_coordinates(feature)
            group_id, geo_address, csu_id = derive_group_id(
                premises,
                components_by_language,
                coordinates,
                source_ref,
                feature_index,
            )
            seen_groups[group_id] += 1
            if seen_groups[group_id] > 1:
                audit.counts["duplicate_group_ids"] += 1
            split = choose_split(
                group_id,
                args.seed,
                args.train_ratio,
                args.validation_ratio,
            )

            structured_record = {
                "dataset_schema_version": DATASET_SCHEMA_VERSION,
                "group_id": group_id,
                "geo_address": geo_address,
                "csu_id": csu_id,
                "coordinates": coordinates,
                "split": split,
                "languages": {
                    language: components_by_language[language]
                    for language in usable_languages
                },
                "provenance": {
                    "source_file": source_ref.as_string(),
                    "feature_index": feature_index,
                },
            }
            write_json_line(structured_handle, structured_record)
            audit.counts["structured_records"] += 1

            for language in usable_languages:
                components = components_by_language[language]
                real_profiles = select_real_three_d(
                    ensure_list(components.get("three_d_addresses")),
                    args.max_real_3d_per_address,
                    seed=args.seed,
                    group_id=group_id,
                )
                profile_options: list[tuple[Mapping[str, Any] | None, str]]
                if real_profiles:
                    profile_options = [(profile, "als") for profile in real_profiles]
                else:
                    profile_options = [(None, "none")]

                for profile_index, (real_profile, source_kind) in enumerate(profile_options):
                    for variant_index in range(args.variants_per_address):
                        profile = real_profile
                        three_d_source = source_kind
                        is_village = bool(clean_text(components.get("village_name")))

                        if (
                                profile is None
                                and stable_unit_interval(
                            args.seed,
                            group_id,
                            profile_index,
                            variant_index,
                            "synthetic_gate",
                        )
                                < args.synthetic_3d_rate
                        ):
                            profile = synthetic_three_d(
                                group_id, language, args.seed, (profile_index, variant_index), is_village
                            )
                            three_d_source = "synthetic"

                        example = build_training_example(
                            components=components,
                            language=language,
                            three_d=profile,
                            three_d_source=three_d_source,
                            split=split,
                            group_id=group_id,
                            geo_address=geo_address,
                            csu_id=csu_id,
                            coordinates=coordinates,
                            source_ref=source_ref,
                            feature_index=feature_index,
                            variant_index=variant_index,
                            profile_index=profile_index,
                            variants_per_address=args.variants_per_address,
                            messy_rate=args.messy_rate,
                            typo_rate=args.typo_rate,
                            component_drop_rate=args.component_drop_rate,
                            region_abbreviation_rate=args.region_abbreviation_rate,
                            fullwidth_punctuation_rate=args.fullwidth_punctuation_rate,
                            chinese_separator_noise_rate=args.chinese_separator_noise_rate,
                            synthetic_subdistrict_rate=args.synthetic_subdistrict_rate,
                            district_suffix_noise_rate=args.district_suffix_noise_rate,
                            chinese_region_variation_rate=args.chinese_region_variation_rate,
                            floor_unit_fusion_rate=args.floor_unit_fusion_rate,
                            canonical_floor_unit_fusion_rate=args.canonical_floor_unit_fusion_rate,
                            english_upper_rate=args.english_upper_rate,
                            english_title_rate=args.english_title_rate,
                            english_lower_rate=args.english_lower_rate,
                            english_mixed_rate=args.english_mixed_rate,
                            seed=args.seed,
                        )
                        if not example["text"]:
                            audit.counts["empty_examples_skipped"] += 1
                            continue
                        write_json_line(split_handles[split], example)
                        audit.observe_example(example)

            if (
                    args.mixed_variants_per_address > 0
                    and "en" in usable_languages
                    and any(language.startswith("zh-") for language in usable_languages)
            ):
                english_components = components_by_language["en"]
                english_profiles = ensure_list(
                    english_components.get("three_d_addresses")
                )
                for chinese_language in (
                        language
                        for language in ("zh-Hant", "zh-Hans")
                        if language in usable_languages
                ):
                    chinese_components = components_by_language[chinese_language]
                    chinese_profiles = ensure_list(
                        chinese_components.get("three_d_addresses")
                    )
                    paired_count = min(len(english_profiles), len(chinese_profiles))
                    if len(english_profiles) != len(chinese_profiles):
                        audit.counts["unpaired_bilingual_3d_arrays"] += 1
                    paired_indices = select_real_three_d_indices(
                        paired_count,
                        args.max_real_3d_per_address,
                        seed=args.seed,
                        group_id=group_id,
                    )
                    mixed_profile_options: list[
                        tuple[
                            int, Mapping[str, Any] | None, Mapping[str, Any] | None, str
                        ]
                    ]
                    if paired_indices:
                        mixed_profile_options = [
                            (
                                index,
                                english_profiles[index],
                                chinese_profiles[index],
                                "als",
                            )
                            for index in paired_indices
                        ]
                    else:
                        mixed_profile_options = [(0, None, None, "none")]

                    for (
                            profile_index,
                            real_english_profile,
                            real_chinese_profile,
                            source_kind,
                    ) in mixed_profile_options:
                        # Use deterministic probability instead of a loop
                        if stable_unit_interval(args.seed, group_id, profile_index,
                                                "mixed_prob") < args.mixed_variants_per_address:
                            variant_index = 1  # Set to 1 so the script allows noise augmentations
                            pseudo_variant_count = 2  # Trick the math denominator later

                            english_profile = real_english_profile
                            chinese_profile = real_chinese_profile
                            three_d_source = source_kind
                            is_village = bool(
                                clean_text(english_components.get("village_name"))
                            )

                            if (
                                    english_profile is None
                                    and chinese_profile is None
                                    and stable_unit_interval(
                                args.seed,
                                group_id,
                                profile_index,
                                variant_index,
                                "mixed_synthetic_gate",
                            )
                                    < args.synthetic_3d_rate
                            ):
                                synthetic_key = (
                                    "mixed",
                                    profile_index,
                                    variant_index,
                                )
                                english_profile = synthetic_three_d(
                                    group_id,
                                    "en",
                                    args.seed,
                                    synthetic_key,
                                    is_village,
                                )
                                chinese_profile = synthetic_three_d(
                                    group_id,
                                    chinese_language,
                                    args.seed,
                                    synthetic_key,
                                    is_village,
                                )
                                three_d_source = "synthetic"

                            mode_pool = (
                                MIX_MODES
                                if english_profile is not None
                                   and chinese_profile is not None
                                else MIX_MODES[2:]
                            )
                            mode_index = int(
                                stable_digest(
                                    args.seed,
                                    group_id,
                                    chinese_language,
                                    profile_index,
                                    variant_index,
                                    "mix_mode",
                                )[:8],
                                16,
                            ) % len(mode_pool)
                            mix_mode = mode_pool[mode_index]

                            if mix_mode in {
                                "english_3d_chinese_body",
                                "single_english_component",
                            }:
                                base_components = chinese_components
                                base_language = chinese_language
                                base_profile = chinese_profile
                                alternate_components = english_components
                                alternate_language = "en"
                                alternate_profile = english_profile
                            elif mix_mode in {
                                "chinese_3d_english_body",
                                "single_chinese_component",
                            }:
                                base_components = english_components
                                base_language = "en"
                                base_profile = english_profile
                                alternate_components = chinese_components
                                alternate_language = chinese_language
                                alternate_profile = chinese_profile
                            elif mode_index % 2:
                                base_components = chinese_components
                                base_language = chinese_language
                                base_profile = chinese_profile
                                alternate_components = english_components
                                alternate_language = "en"
                                alternate_profile = english_profile
                            else:
                                base_components = english_components
                                base_language = "en"
                                base_profile = english_profile
                                alternate_components = chinese_components
                                alternate_language = chinese_language
                                alternate_profile = chinese_profile

                            example = build_training_example(
                                components=base_components,
                                language=base_language,
                                three_d=base_profile,
                                three_d_source=three_d_source,
                                split=split,
                                group_id=group_id,
                                geo_address=geo_address,
                                csu_id=csu_id,
                                coordinates=coordinates,
                                source_ref=source_ref,
                                feature_index=feature_index,
                                variant_index=variant_index,
                                profile_index=profile_index,
                                variants_per_address=pseudo_variant_count,
                                messy_rate=args.messy_rate,
                                typo_rate=args.typo_rate,
                                component_drop_rate=args.component_drop_rate,
                                region_abbreviation_rate=args.region_abbreviation_rate,
                                fullwidth_punctuation_rate=args.fullwidth_punctuation_rate,
                                chinese_separator_noise_rate=args.chinese_separator_noise_rate,
                                synthetic_subdistrict_rate=args.synthetic_subdistrict_rate,
                                district_suffix_noise_rate=args.district_suffix_noise_rate,
                                chinese_region_variation_rate=args.chinese_region_variation_rate,
                                floor_unit_fusion_rate=args.floor_unit_fusion_rate,
                                canonical_floor_unit_fusion_rate=args.canonical_floor_unit_fusion_rate,
                                english_upper_rate=args.english_upper_rate,
                                english_title_rate=args.english_title_rate,
                                english_lower_rate=args.english_lower_rate,
                                english_mixed_rate=args.english_mixed_rate,
                                seed=args.seed,
                                alternate_components=alternate_components,
                                alternate_language=alternate_language,
                                alternate_three_d=alternate_profile,
                                output_language=f"mixed-{chinese_language}-en",
                                mix_mode=mix_mode,
                            )
                            if not example["text"]:
                                audit.counts["empty_examples_skipped"] += 1
                                continue
                            if not ensure_mapping(example.get("provenance")).get(
                                    "code_switched"
                            ):
                                audit.counts["code_switch_not_applicable_skipped"] += 1
                                continue
                            write_json_line(split_handles[split], example)
                            audit.observe_example(example)
    finally:
        for handle in split_handles.values():
            handle.close()
        structured_handle.close()
        rejects_handle.close()
        if generation_progress is not None:
            generation_progress.close()

    if audit.counts["examples"] == 0:
        raise DatasetBuildError(
            "No training examples were generated; inspect rejects.jsonl"
        )

    write_audit_files(output_dir, audit, input_paths=input_paths, args=args)
    write_dataset_card(output_dir)
    LOGGER.info(
        "Wrote %s examples from %s structured features to %s",
        audit.counts["examples"],
        audit.counts["structured_records"],
        output_dir,
    )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Read Hong Kong ALS GeoJSON files/ZIPs and create multilingual "
            "span-labelled address parsing data."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=script_dir / "geojson",
        help="Directory containing GeoJSON/JSON files or ALS ZIP archives",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "hk_address_dataset",
        help="Directory to create",
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Search input subdirectories"
    )
    parser.add_argument(
        "--variants-per-address",
        type=int,
        default=2,
        help="Number of text variants for each selected language/3D profile",
    )
    parser.add_argument(
        "--mixed-variants-per-address",
        type=float,
        default=0.05,
        help=(
            "Probability (0.0 to 1.0) of generating a Chinese-English "
            "code-switched variant per bilingual address/3D profile"
        ),
    )
    parser.add_argument(
        "--messy-rate",
        type=float,
        default=0.05,
        help="Target share of variants with structural reordering/format noise",
    )
    parser.add_argument(
        "--typo-rate",
        type=float,
        default=0.02,
        help="Target share with one minor name typo; floor/unit identifiers are protected",
    )
    parser.add_argument(
        "--component-drop-rate",
        type=float,
        default=0.02,
        help=(
            "Target share with importance-weighted missing components; region "
            "and district are omitted more often than address anchors"
        ),
    )
    parser.add_argument(
        "--region-abbreviation-rate",
        type=float,
        default=0.2,
        help=(
            "Minority share using HK/KLN/NT after canonical expansion to Hong "
            "Kong/Kowloon/New Territories"
        ),
    )
    parser.add_argument(
        "--synthetic-3d-rate",
        type=float,
        default=0.9,
        help="For 2D records, chance that each variant receives marked synthetic floor/unit syntax",
    )
    parser.add_argument(
        "--fullwidth-punctuation-rate",
        type=float,
        default=0.1,
        help="Probability to use fullwidth (Chinese) punctuation in augmented variants"
    )
    parser.add_argument(
        "--chinese-separator-noise-rate",
        type=float,
        default=0.10,
        help="Per-augmented-variant probability of separator noise inside Chinese 3D syntax",
    )
    parser.add_argument(
        "--synthetic-subdistrict-rate",
        type=float,
        default=0.30,
        help=(
            "Probability of injecting a sub-district drawn from SUB_DISTRICT_MAP "
            "when the address has a recognisable district and no existing "
            "sub-district (ALS LocationName or prior synthetic).  The value is "
            "never treated as verified locality; it only teaches the model the "
            "real vocabulary of Hong Kong sub-districts.  Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--district-suffix-noise-rate",
        type=float,
        default=0.10,
        help="Per-augmented-variant probability of adding/removing District/區/区",
    )
    parser.add_argument(
        "--chinese-region-variation-rate",
        type=float,
        default=0.10,
        help="Per-augmented-variant probability of a Chinese region-name variant",
    )
    parser.add_argument(
        "--floor-unit-fusion-rate",
        type=float,
        default=0.20,
        help="Per-augmented-variant probability of compact/inverted floor-unit syntax",
    )
    parser.add_argument(
        "--canonical-floor-unit-fusion-rate",
        type=float,
        default=0.0,
        help="Probability of floor-unit fusion in variant zero; zero keeps a clean baseline",
    )
    parser.add_argument(
        "--max-real-3d-per-address",
        type=int,
        default=2,
        help="Cap sampled real ALS floor/unit profiles per building; 0 means no cap",
    )
    parser.add_argument(
        "--filter-missing-building", type=float, default=0.8,
        help="Probability to skip the entire row if building name is missing in the source"
    )
    parser.add_argument(
        "--filter-missing-street", type=float, default=0.3,
        help="Probability to skip the entire row if street name is missing in the source"
    )
    parser.add_argument(
        "--filter-missing-region", type=float, default=0.0,
        help="Probability to skip the entire row if region is missing in the source"
    )
    parser.add_argument(
        "--filter-missing-district", type=float, default=0.3,
        help="Probability to skip the entire row if district is missing in the source"
    )
    parser.add_argument(
        "--filter-missing-estate", type=float, default=0.2,
        help="Probability to skip the entire row if estate name is missing in the source"
    )
    parser.add_argument("--seed", type=int, default=20260714, help="Deterministic seed")
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--english-upper-rate", type=float, default=0.15,
                        help="Relative weight for all-caps English text")
    parser.add_argument("--english-title-rate", type=float, default=0.25,
                        help="Relative weight for Title Case English text")
    parser.add_argument("--english-lower-rate", type=float, default=0.55,
                        help="Relative weight for lowercase English text")
    parser.add_argument("--english-mixed-rate", type=float, default=0.05,
                        help="Relative weight for mixed-case English text")
    parser.add_argument(
        "--simplified",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate zh-Hans rows from government zh-Hant records with OpenCC",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Optional development/testing limit across all sources",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "First count source features, then show tqdm generation progress; "
            "--no-progress skips the counting pass"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace a non-empty output directory",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    try:
        output_dir = build_dataset(args)
    except (DatasetBuildError, OSError) as exc:
        LOGGER.error("%s", exc)
        return 2
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())