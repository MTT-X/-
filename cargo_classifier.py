"""
航运保险货品智能分类系统 - 基于规则的关键词匹配方案
FINA2003 金融科技前沿 期末项目
"""

import re
import pandas as pd
from collections import Counter

# ============================================================
# 第一部分：关键词词典构建
# ============================================================

# 保险除外标的 → 关键词映射（10类拒保商品）
EXCLUSION_RULES = {
    "活动物及动物产品/血制品/冷冻品/疫苗": {
        "keywords_cn": ["活动物", "动物", "血制品", "冷冻", "疫苗", "活体", "牲畜", "禽", "肉类",
                        "冻品", "冰鲜", "冷藏", "血清", "免疫", "接种", "活鱼", "活虾", "活蟹",
                        "海鲜", "水产品", "海产品", "畜", "屠宰", "冷冻食品", "冷鲜", "保鲜"],
        "keywords_en": ["live animal", "livestock", "frozen", "vaccine", "blood", "chilled",
                        "fresh fish", "live fish", "seafood", "poultry", "meat", "beef",
                        "pork", "chicken", "mutton", "lamb", "shrimp", "crab", "lobster",
                        "cold storage", "refrigerated", "carcass", "serum", "immune"],
        "risk": "拒保"
    },
    "艺术品/玉制品/收藏品/古物/金银珠宝/钻石/现金/有价证券": {
        "keywords_cn": ["艺术品", "玉", "收藏品", "古物", "金银", "珠宝", "钻石", "现金", "有价证券",
                        "古董", "字画", "油画", "雕塑", "翡翠", "宝石", "钱币", "纪念币", "邮票",
                        "文物", "手稿", "贵金属", "黄金", "白银", "铂金", "首饰"],
        "keywords_en": ["art", "antique", "jade", "jewelry", "diamond", "gold", "silver",
                        "platinum", "cash", "currency", "painting", "sculpture", "gemstone",
                        "precious metal", "collectible", "coin", "stamp", "artifact",
                        "cultural relic", "manuscript"],
        "risk": "拒保"
    },
    "易腐易蛀品/果仁/花生/大豆/豆粕/谷物/鲜活货": {
        "keywords_cn": ["果仁", "花生", "大豆", "豆粕", "谷物", "农产品", "鱼粉", "菜籽饼",
                        "地瓜干", "木薯干", "海货", "鲜活", "易腐", "易蛀", "易变质",
                        "坚果", "核桃", "杏仁", "腰果", "瓜子", "粮食", "小麦", "玉米",
                        "大米", "豆类", "芝麻", "油菜籽", "面粉", "饲料",
                        "干果", "蜜饯", "红枣", "枸杞", "菌菇", "木耳", "海带"],
        "keywords_en": ["perishable", "nut", "peanut", "soybean", "grain", "cereal",
                        "agricultural", "fishmeal", "dried", "kernel", "almond",
                        "cashew", "walnut", "sunflower seed", "wheat", "corn", "rice",
                        "flour", "animal feed", "bean", "lentil", "pea", "millet",
                        "sorghum", "rapeseed", "sesame"],
        "risk": "拒保"
    },
    "易燃易爆品/烟花爆竹": {
        "keywords_cn": ["易燃", "易爆", "烟花", "爆竹", "火药", "炸药", "鞭炮", "烟火",
                        "打火机", "汽油", "柴油", "煤油", "酒精", "油漆", "溶剂",
                        "气体", "液化气", "天然气", "丙烷", "丁烷", "氢气"],
        "keywords_en": ["flammable", "explosive", "firework", "firecracker", "gasoline",
                        "diesel", "kerosene", "alcohol", "ethanol", "paint", "solvent",
                        "propane", "butane", "hydrogen", "lpg", "lng", "aerosol",
                        "lighter", "gunpowder", "dynamite", "combustible"],
        "risk": "拒保"
    },
    "精密仪器设备(无国内维修)/芯片/液晶/单台>200万": {
        "keywords_cn": ["精密仪器", "芯片", "液晶", "印刷设备", "手机制造", "半导体",
                        "晶圆", "光刻", "蚀刻", "封装测试", "精密设备", "高端设备",
                        "检测设备", "测量仪器", "光谱", "色谱", "质谱", "激光设备"],
        "keywords_en": ["precision instrument", "semiconductor", "chip", "wafer", "LCD",
                        "OLED", "lithography", "etching", "clean room", "laser",
                        "spectrometer", "chromatograph", "nano", "photolithography"],
        "risk": "拒保"
    },
    "军火/军品/原木/鱼粉/水泥/原糖/精制糖/原煤/黄麻": {
        "keywords_cn": ["军火", "军品", "原木", "水泥", "原糖", "精制糖", "原煤",
                        "黄麻", "武器", "弹药", "军用", "枪支", "子弹",
                        "红木", "煤炭", "焦炭", "白砂糖", "原蔗糖",
                        "剑麻", "亚麻", "苎麻"],
        "keywords_en": ["arms", "weapon", "ammunition", "military", "log", "timber",
                        "cement", "raw sugar", "coal", "jute", "gun", "rifle", "bullet",
                        "unprocessed wood", "anthracite", "lignite", "cane sugar",
                        "hemp", "sisal", "linen", "ramie"],
        "risk": "拒保"
    },
    "车辆/航空器/船舶/运输设备": {
        "keywords_cn": ["车辆", "航空器", "船舶", "运输设备", "汽车", "飞机", "轮船",
                        "火车", "摩托车", "电动车", "直升机", "游艇",
                        "货车", "卡车", "巴士", "客车", "地铁", "轻轨", "拖车"],
        "keywords_en": ["vehicle", "aircraft", "ship", "vessel", "boat", "car", "truck",
                        "bus", "motorcycle", "bicycle", "helicopter", "yacht", "airplane",
                        "locomotive", "trailer", "tractor", "bulldozer", "excavator"],
        "risk": "拒保"
    },
    "超大超重件/港机设备/滚装货/龙门吊": {
        "keywords_cn": ["超大件", "超重件", "港机", "滚装", "龙门吊", "起重机", "塔吊",
                        "港口机械", "大型设备", "超限", "巨型", "大型机械", "盾构机"],
        "keywords_en": ["oversized", "overweight", "crane", "gantry", "port machinery",
                        "ro-ro", "roll-on", "heavy lift", "mega", "giant", "tower crane",
                        "bridge crane", "container crane", "tunnel boring machine"],
        "risk": "拒保"
    },
    "玻璃/玻璃制品/陶瓷/石材/易碎品": {
        "keywords_cn": ["玻璃", "陶瓷", "石材", "易碎", "镜子", "瓷砖", "大理石", "花岗岩",
                        "花瓶", "琉璃", "水晶", "灯管", "灯泡", "显示屏", "触摸屏"],
        "keywords_en": ["glass", "ceramic", "porcelain", "stone", "fragile", "marble",
                        "granite", "crystal", "mirror", "tile", "lamp", "bulb", "vase",
                        "screen", "touch panel", "display panel"],
        "risk": "拒保"
    },
    "二手货物": {
        "keywords_cn": ["二手", "旧货", "废旧", "翻新", "用过的", "淘汰", "报废", "回收"],
        "keywords_en": ["used", "second hand", "secondhand", "refurbished", "scrap",
                        "waste", "recycled", "pre-owned", "worn", "salvage", "reclaimed"],
        "risk": "拒保"
    }
}

# HS 22大类 → 关键词映射（用于可承保货物的分类）
HS_CATEGORIES = {
    "活动物;动物产品": {
        "keywords_cn": ["动物产品", "肉类", "水产", "鱼", "虾", "蟹", "贝", "乳制品",
                        "奶", "蛋", "蜂蜜", "动物毛", "羽毛", "骨", "角", "肠衣",
                        "肉制品", "香肠", "火腿", "罐头肉", "动物油脂"],
        "keywords_en": ["meat", "fish", "shrimp", "crab", "shell", "dairy", "milk",
                        "egg", "honey", "wool", "feather", "bone", "sausage", "ham",
                        "animal", "poultry", "livestock", "canned meat", "seafood",
                        "fillet", "salmon", "tuna", "cod", "squid", "octopus"]
    },
    "植物产品": {
        "keywords_cn": ["蔬菜", "水果", "谷物", "粮食", "咖啡", "茶", "香料", "植物",
                        "花卉", "种子", "豆类", "坚果", "药材", "人参", "虫草",
                        "可可", "调味料", "辣椒", "花椒", "八角", "桂皮"],
        "keywords_en": ["vegetable", "fruit", "grain", "cereal", "coffee", "tea",
                        "spice", "plant", "flower", "seed", "bean", "nut", "herb",
                        "ginseng", "cocoa", "pepper", "chili", "cinnamon", "ginger",
                        "quinoa", "edamame", "chickpea", "lentil", "pea", "millet",
                        "sorghum", "buckwheat", "oat", "barley", "rye", "bran", "germ",
                        "soy", "soya", "soybean meal", "canola", "sunflower seed"]
    },
    "动、植物油、脂": {
        "keywords_cn": ["植物油", "动物油", "油脂", "食用油", "棕榈油", "豆油", "菜籽油",
                        "花生油", "橄榄油", "椰子油", "葵花籽油", "猪油", "牛油", "黄油",
                        "人造黄油", "起酥油", "可可脂"],
        "keywords_en": ["oil", "fat", "butter", "margarine", "vegetable oil", "palm oil",
                        "soybean oil", "olive oil", "coconut oil", "sunflower oil",
                        "shortening", "cocoa butter", "lard", "tallow", "grease"]
    },
    "食品；饮料、酒及醋；烟草": {
        "keywords_cn": ["食品", "饮料", "酒", "醋", "烟草", "糖果", "巧克力", "饼干",
                        "糕点", "面包", "面条", "调味品", "酱油", "味精", "零食",
                        "矿泉水", "果汁", "啤酒", "葡萄酒", "白酒", "香烟", "雪茄",
                        "罐头", "果酱", "酵母", "食用色素", "食品添加剂"],
        "keywords_en": ["food", "beverage", "wine", "vinegar", "tobacco", "candy",
                        "chocolate", "biscuit", "cookie", "cake", "bread", "noodle",
                        "pasta", "sauce", "soy sauce", "juice", "beer", "cigarette",
                        "cigar", "canned food", "jam", "yeast", "snack", "confectionery"]
    },
    "矿产品": {
        "keywords_cn": ["矿产", "矿石", "盐", "硫磺", "石墨", "砂石", "黏土", "高岭土",
                        "石油", "天然气", "煤炭", "铁矿", "铜矿", "铝矿", "锌矿",
                        "镍矿", "锂矿", "钴矿", "稀土", "石英", "云母"],
        "keywords_en": ["mineral", "ore", "salt", "sulfur", "graphite", "clay", "kaolin",
                        "petroleum", "crude oil", "natural gas", "coal", "iron ore",
                        "copper ore", "bauxite", "zinc", "nickel", "lithium", "cobalt",
                        "rare earth", "quartz", "mica", "gypsum", "spodumene", "bentonite",
                        "asphalt", "perlite", "vermiculite", "dolomite", "limestone",
                        "basalt", "pumice", "diatomite", "barite", "talc", "magnesite",
                        "rutile", "ilmenite", "zircon", "bitumen", "petroleum coke"]
    },
    "化学工业及其相关工业的产品": {
        "keywords_cn": ["化工", "化学品", "有机", "无机", "药品", "医药",
                        "化肥", "染料", "颜料", "涂料", "精油", "肥皂",
                        "洗涤剂", "胶水", "粘合剂", "催化剂", "试剂", "添加剂",
                        "树脂", "橡胶原料", "农药", "消毒剂", "化妆品", "乳液",
                        "粉末", "提取物", "分子筛", "尿素", "甘油", "活性炭"],
        "keywords_en": ["chemical", "pharmaceutical", "medicine", "drug", "fertilizer",
                        "dye", "pigment", "paint", "coating", "essential oil", "soap",
                        "detergent", "glue", "adhesive", "catalyst", "reagent", "resin",
                        "additive", "pesticide", "disinfectant", "cosmetic",
                        "compound", "solution", "emulsion", "powder", "extract",
                        "molecular sieve", "urea", "polyether", "polyol", "glycerin",
                        "glycerine", "silicone", "surfactant", "acrylate", "methacrylate",
                        "monomer", "polymer", "copolymer", "elastomer", "prepolymer",
                        "isocyanate", "polyurethane", "epoxy", "phenolic", "melamine",
                        "formaldehyde", "glutaraldehyde", "chloride", "sulphate",
                        "sulfate", "phosphate", "nitrate", "carbonate", "bicarbonate",
                        "hydroxide", "oxide", "peroxide", "ammonium", "sodium",
                        "potassium", "calcium", "magnesium", "aluminium oxide",
                        "activated carbon", "silica gel", "desiccant", "absorbent"]
    },
    "塑料及其制品；橡胶及其制品": {
        "keywords_cn": ["塑料", "橡胶", "塑胶", "聚乙烯", "聚丙烯", "PVC", "尼龙", "泡沫",
                        "塑料膜", "塑料袋", "塑料管", "橡胶管", "轮胎",
                        "胶带", "密封圈", "硅胶", "亚克力"],
        "keywords_en": ["plastic", "rubber", "polyethylene", "polypropylene", "PVC",
                        "nylon", "foam", "film", "bag", "pipe", "tube", "tire", "tyre",
                        "tape", "seal", "silicone", "acrylic", "plexiglass", "elastomer",
                        "polymer", "polyurethane"]
    },
    "生皮、皮革、毛皮及其制品": {
        "keywords_cn": ["皮革", "毛皮", "生皮", "皮具", "箱包", "手袋", "钱包", "皮带",
                        "皮衣", "皮鞋", "皮手套", "旅行用品", "人造革", "麂皮", "貂皮"],
        "keywords_en": ["leather", "fur", "hide", "skin", "handbag", "wallet",
                        "belt", "luggage", "suitcase", "glove", "suede", "mink"]
    },
    "木及木制品；木炭": {
        "keywords_cn": ["木材", "木制品", "木炭", "软木", "竹", "藤",
                        "家具", "木板", "胶合板", "密度板", "刨花板", "地板",
                        "木质包装", "木箱", "木托盘", "柳编", "草编"],
        "keywords_en": ["wood", "timber", "charcoal", "cork", "bamboo", "rattan",
                        "furniture", "plywood", "MDF", "particle board", "flooring",
                        "pallet", "wicker", "straw", "veneer", "lumber"]
    },
    "木浆；纸及纸板": {
        "keywords_cn": ["纸", "纸板", "纸浆", "木浆", "纤维素", "瓦楞纸", "卡纸",
                        "卫生纸", "纸巾", "纸箱", "纸袋", "书本", "印刷品", "标签",
                        "墙纸", "过滤纸", "绝缘纸", "复写纸", "包装材料", "包装袋"],
        "keywords_en": ["paper", "paperboard", "pulp", "cellulose", "corrugated",
                        "cardboard", "carton", "tissue", "book", "label", "printing",
                        "wallpaper", "kraft", "linerboard", "packing material",
                        "packaging", "wrapping", "carton box", "paper box"]
    },
    "纺织原料及纺织制品": {
        "keywords_cn": ["纺织", "面料", "布料", "纱线", "棉", "麻", "丝绸", "羊毛",
                        "化纤", "涤纶", "腈纶", "氨纶", "服装", "衣服", "裤子",
                        "衬衫", "裙子", "外套", "内衣", "袜子", "毛巾", "床上用品",
                        "毯子", "窗帘", "地毯", "无纺布", "缝纫线", "拉链", "纽扣",
                        "牛仔", "休闲裤", "短裤", "T恤", "卫衣", "毛衣", "羽绒",
                        "夹克", "西服", "领带", "围巾", "手套", "泳衣", "运动服"],
        "keywords_en": ["textile", "fabric", "yarn", "cotton", "silk", "wool", "linen",
                        "polyester", "acrylic", "spandex", "garment", "cloth", "clothing",
                        "apparel", "shirt", "pants", "trouser", "dress", "jacket", "coat",
                        "underwear", "sock", "towel", "bedding", "blanket", "curtain",
                        "carpet", "nonwoven", "thread", "zipper", "button", "knitted",
                        "woven", "robe", "T-shirt", "pullover", "sweater",
                        "jeans", "denim", "shorts", "hoodie", "cardigan", "blouse",
                        "skirt", "legging", "jumpsuit", "pajama", "swimsuit",
                        "scarf", "glove", "tie", "vest", "blazer", "uniform"]
    },
    "鞋、帽、伞、杖、鞭及其零件": {
        "keywords_cn": ["鞋", "靴", "帽", "伞", "手杖", "人造花", "羽毛制品", "假发",
                        "运动鞋", "皮鞋", "凉鞋", "拖鞋", "帽子", "雨伞", "阳伞",
                        "登山杖", "拐杖", "鞭子", "头饰", "发饰"],
        "keywords_en": ["shoe", "boot", "hat", "umbrella", "walking stick", "artificial flower",
                        "wig", "sneaker", "sandal", "slipper", "cap", "parasol",
                        "headwear", "cane", "whip", "feather"]
    },
    "石料、石膏、水泥、石棉、云母等制品；陶瓷；玻璃": {
        "keywords_cn": ["石料", "石膏", "水泥", "石棉", "云母",
                        "砖", "瓦", "混凝土", "耐火材料", "砂轮", "磨料",
                        "石棉制品", "绝缘材料", "碳纤维", "玻纤"],
        "keywords_en": ["stone", "plaster", "asbestos", "mica", "brick",
                        "concrete", "refractory", "abrasive", "insulation",
                        "carbon fiber", "fiberglass", "slate", "granite slab"]
    },
    "珍珠、宝石、贵金属、仿首饰": {
        "keywords_cn": ["珍珠", "宝石", "半宝石", "贵金属", "首饰", "仿首饰",
                        "翡翠", "玛瑙", "水晶饰品", "项链", "戒指", "耳环", "手镯"],
        "keywords_en": ["pearl", "gem", "precious", "jadeite",
                        "agate", "crystal", "necklace", "ring", "earring", "bracelet"]
    },
    "贱金属及其制品": {
        "keywords_cn": ["钢铁", "金属", "不锈钢", "铝", "铜", "锌", "铅", "锡", "镍",
                        "金属丝", "金属网", "金属管", "金属板", "金属结构",
                        "螺丝", "螺帽", "螺栓", "钉子", "工具", "锁", "铰链",
                        "钢管", "钢板", "钢丝", "铝合金", "铜管", "铜线", "铸件",
                        "锻件", "焊接", "法兰", "阀门", "管件", "弹簧"],
        "keywords_en": ["steel", "iron", "metal", "stainless", "aluminum", "copper",
                        "zinc", "lead", "tin", "nickel", "wire", "tube", "pipe",
                        "plate", "sheet", "screw", "nut", "bolt", "nail", "tool",
                        "lock", "hinge", "alloy", "cast", "forged", "weld", "flange",
                        "valve", "fitting", "spring", "rod", "bar", "coil",
                        "hardware", "bracket", "track", "angle", "handle", "knob",
                        "latch", "washer", "rivet", "pin", "clip", "clamp", "hook",
                        "ring", "chain", "mesh", "grate", "grating", "railing",
                        "fence", "gate", "ladder", "scaffold", "stainless steel",
                        "carbon steel", "galvanized", "galvanised", "brass", "bronze"]
    },
    "机器、机械器具、电气设备及其零件": {
        "keywords_cn": ["机器", "机械", "电气", "电子", "设备", "电机", "发电机",
                        "发动机", "泵", "压缩机", "风机", "轴承", "齿轮", "变速箱",
                        "空调", "冰箱", "洗衣机", "计算机", "打印", "手机",
                        "电路", "变压器", "传感器", "电池", "电缆", "开关", "继电器",
                        "机器人", "数控", "机床", "模具", "自动化", "流水线",
                        "加热器", "过滤器", "液压", "气缸", "太阳能", "光伏",
                        "检测器", "进样器", "割草机", "售货机", "马达", "减速器",
                        "罐", "槽", "搅拌", "分散", "混合", "粉碎", "筛", "输送"],
        "keywords_en": ["machine", "machinery", "mechanical", "electric", "electronic",
                        "equipment", "motor", "engine", "generator", "pump", "compressor",
                        "fan", "bearing", "gear", "gearbox", "air condition", "refrigerator",
                        "computer", "printer", "circuit", "transformer", "sensor",
                        "battery", "cable", "switch", "relay", "robot", "CNC", "lathe",
                        "mold", "mould", "automation", "conveyor", "solar", "module",
                        "filter", "heater", "heating", "cooler", "cooling", "hydraulic",
                        "cylinder", "detector", "extruder", "mixer", "grinder", "cutter",
                        "dispenser", "vending", "actuator", "servo", "inverter",
                        "controller", "regulator", "rectifier", "capacitor", "inductor",
                        "resistor", "diode", "transistor", "thermostat", "thermocouple",
                        "manifold", "exhaust", "intake", "turbine", "impeller",
                        "agitator", "centrifuge", "separator", "cyclone", "scrubber",
                        "chiller", "evaporator", "condenser", "heat exchanger",
                        "conveyor belt", "conveyor system", "vibrating", "sieving",
                        "disperser", "dispersion", "tank", "vessel", "hopper",
                        "silo", "feeder", "screen", "sifter", "shredder", "crusher",
                        "granulator", "pelletizer", "baler", "compactor", "wrapper"]
    },
    "车辆、航空器、船舶及有关运输设备": {
        "keywords_cn": ["车辆", "汽车", "航空航天", "铁路", "机车",
                        "摩托车", "自行车", "电动车", "飞机", "直升机", "轮船",
                        "集装箱", "半挂车", "底盘"],
        "keywords_en": ["vehicle", "automobile", "aircraft", "ship", "vessel", "railway",
                        "locomotive", "motorcycle", "bicycle", "helicopter", "airplane",
                        "container", "trailer", "chassis"]
    },
    "光学、医疗、精密仪器设备；钟表；乐器": {
        "keywords_cn": ["光学", "照相", "摄影", "电影", "计量", "检验", "医疗", "外科",
                        "仪器", "仪表", "钟表", "手表", "乐器", "镜头", "相机",
                        "显微镜", "望远镜", "医疗设备", "诊断", "监护", "超声波",
                        "计量器具", "温度计", "钢琴", "吉他"],
        "keywords_en": ["optical", "photographic", "camera", "medical", "surgical",
                        "instrument", "meter", "watch", "clock", "lens", "microscope",
                        "telescope", "medical device", "diagnostic", "monitor", "ultrasound",
                        "X-ray", "MRI", "CT", "scale", "thermometer", "piano", "guitar",
                        "measuring", "testing", "laboratory"]
    },
    "武器、弹药及其零件、附件": {
        "keywords_cn": ["武器", "弹药", "枪支", "子弹", "军用", "防弹"],
        "keywords_en": ["weapon", "ammunition", "gun", "bullet", "military", "armor",
                        "missile", "bomb", "grenade", "tank"]
    },
    "杂项制品": {
        "keywords_cn": ["家具", "玩具", "游戏", "运动器材", "健身器材", "灯具",
                        "坐具", "床垫", "帐篷", "笔", "文具",
                        "卫生用品", "婴童用品", "圣诞", "节日用品", "装饰品",
                        "行李箱", "背包", "梳子", "发卡", "浴缸", "马桶",
                        "洗手盆", "水龙头", "淋浴", "门", "窗", "百叶窗",
                        "靠垫", "坐垫", "地垫", "脚垫", "婴儿", "尿布",
                        "护理垫", "宠物用品", "猫砂", "狗粮", "猫粮"],
        "keywords_en": ["furniture", "toy", "game", "sport", "fitness", "equipment",
                        "lamp", "lighting", "mattress", "tent", "pen", "stationery",
                        "sanitary", "baby", "infant", "christmas", "festival",
                        "decoration", "backpack", "playground", "amusement",
                        "chair", "table", "sofa", "desk", "cabinet", "shelf", "stool",
                        "bench", "wardrobe", "drawer", "nightstand", "bookcase",
                        "headboard", "furnishing", "led light", "led lamp", "led high",
                        "high bay", "bay light", "floodlight", "spotlight",
                        "luminaire", "chandelier", "bathtub", "shower", "toilet",
                        "sink", "faucet", "bidet", "bathroom", "basin", "mirror",
                        "door", "window", "blind", "shutter", "curtain rod", "handrail",
                        "cushion", "pad", "underpad", "mattress pad", "pet", "cat",
                        "dog", "litter", "diaper", "nappy", "hygiene"]
    },
    "艺术品、收藏品及古物": {
        "keywords_cn": ["艺术品", "收藏品", "古物", "油画", "版画", "雕塑",
                        "标本", "古董", "文物"],
        "keywords_en": ["art", "collector", "antique", "painting", "sculpture",
                        "stamp collection", "specimen", "artifact"]
    },
    "特殊交易品及未分类商品": {
        "keywords_cn": ["软件", "定制", "数字产品", "许可", "版权"],
        "keywords_en": ["software", "customized", "digital", "license"]
    }
}


# ============================================================
# 第二部分：文本预处理
# ============================================================

def preprocess(text):
    """清洗货物描述文本"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.strip()
    text = text.replace("\\n", " ").replace("\n", " ").replace("\t", " ")
    # 全角转半角
    result = []
    for ch in text:
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))
        elif code == 0x3000:
            result.append(' ')
        else:
            result.append(ch)
    text = ''.join(result)
    text = re.sub(r'\s+', ' ', text)
    return text


# ============================================================
# 第三部分：分类引擎
# ============================================================

def match_keywords(text, keywords_cn, keywords_en):
    """在文本中匹配中英文关键词，返回命中的关键词列表
    中文使用子串匹配，英文使用整词匹配避免误判（如 art 不匹配 parts）"""
    text_lower = text.lower()
    matched = []
    for kw in keywords_cn:
        if kw in text:
            matched.append(kw)
    for kw in keywords_en:
        kw_lower = kw.lower()
        # 对短词(<=3字符)使用严格整词匹配，避免子串误判
        if len(kw) <= 3:
            pattern = r'(?<![a-z])' + re.escape(kw_lower) + r'(?![a-z])'
            if re.search(pattern, text_lower):
                matched.append(kw)
        else:
            # 长词直接匹配，但要求至少是完整单词的一部分（避免过度匹配）
            if kw_lower in text_lower:
                matched.append(kw)
    return matched


def classify_exclusion(text):
    """检查是否命中除外标的（拒保类）"""
    for category, rules in EXCLUSION_RULES.items():
        matched = match_keywords(text, rules["keywords_cn"], rules["keywords_en"])
        if matched:
            return True, category, matched
    return False, None, []


def classify_hs(text):
    """匹配HS大类"""
    # 特殊处理：数据质量问题的描述
    low_quality_patterns = [
        "SEE ATTACHMENT", "SEE ATTACHED", "SEE THE ATTACHMENT",
        "AS PER INVOICE", "AS PER ATTACHED", "DETAILS AS PER",
        "SEE BELOW", "AS ABOVE", "AS DESCRIBED",
        "CONTAINER", "PALLET", "PACKAGES",
    ]
    text_upper = text.upper().strip()
    # 如果描述仅包含低质量模式词，直接标记
    if text_upper in low_quality_patterns or len(text_upper) <= 3:
        return "信息缺失/无效描述", []

    best_category = None
    best_score = 0
    best_matches = []

    for category, rules in HS_CATEGORIES.items():
        matched = match_keywords(text, rules["keywords_cn"], rules["keywords_en"])
        score = len(matched)
        if score > best_score:
            best_score = score
            best_category = category
            best_matches = matched

    if best_score == 0:
        return "未分类", []
    return best_category, best_matches


def run_classification(descriptions):
    """对全部货品描述进行分类"""
    results = []
    for i, (idx, desc) in enumerate(descriptions):
        text = preprocess(desc)

        is_excluded, excl_category, excl_matches = classify_exclusion(text)

        if is_excluded:
            hs_category = None
            hs_matches = []
            risk_level = "高风险（拒保）"
        else:
            hs_category, hs_matches = classify_hs(text)
            risk_level = "低风险（可承保）"

        results.append({
            "index": idx,
            "original": desc,
            "cleaned": text,
            "is_excluded": is_excluded,
            "exclusion_category": excl_category,
            "exclusion_matches": ", ".join(excl_matches),
            "hs_category": hs_category,
            "hs_matches": ", ".join(hs_matches),
            "risk_level": risk_level
        })

        if (i + 1) % 5000 == 0:
            print(f"  已处理 {i+1} 条...")

    return results


# ============================================================
# 第四部分：统计与输出
# ============================================================

def print_statistics(results):
    """打印分类统计结果"""
    total = len(results)
    excluded = [r for r in results if r["is_excluded"]]
    insured = [r for r in results if not r["is_excluded"]]

    print(f"\n{'='*60}")
    print(f"  航运保险货品分类统计报告")
    print(f"{'='*60}")
    print(f"\n总数据量: {total} 条")
    print(f"拒保件数: {len(excluded)} 条 ({len(excluded)/total*100:.1f}%)")
    print(f"可承保件数: {len(insured)} 条 ({len(insured)/total*100:.1f}%)")

    print(f"\n--- 拒保原因分布 ---")
    excl_counter = Counter(r["exclusion_category"] for r in excluded)
    for cat, count in excl_counter.most_common():
        print(f"  {cat}: {count} 次")

    print(f"\n--- HS大类分布（可承保货物，前15类）---")
    hs_counter = Counter(r["hs_category"] for r in insured)
    for cat, count in hs_counter.most_common(15):
        print(f"  {cat}: {count} 条 ({count/len(insured)*100:.1f}%)")

    unclassified = [r for r in insured if r["hs_category"] == "未分类"]
    print(f"\n--- 未分类货物（可承保但无关键词匹配）---")
    print(f"  数量: {len(unclassified)} 条 ({len(unclassified)/total*100:.1f}%)")
    print(f"  前10条样例:")
    for r in unclassified[:10]:
        desc = r["cleaned"][:120]
        print(f"    [{r['index']}] {desc}")

    return excluded, insured, unclassified


def save_results(results, output_path):
    """保存完整分类结果到Excel"""
    df = pd.DataFrame(results)
    df.to_excel(output_path, index=False, engine='openpyxl')
    print(f"\n分类结果已保存至: {output_path}")


def save_detailed_report(results, output_path):
    """保存详细报告到文本文件"""
    total = len(results)
    excluded = [r for r in results if r["is_excluded"]]
    insured = [r for r in results if not r["is_excluded"]]
    unclassified = [r for r in insured if r["hs_category"] == "未分类"]

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("  航运保险货品智能分类 - 完整报告\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"总数据量: {total}\n")
        f.write(f"拒保: {len(excluded)} ({len(excluded)/total*100:.1f}%)\n")
        f.write(f"可承保: {len(insured)} ({len(insured)/total*100:.1f}%)\n\n")

        # 拒保明细
        f.write("-" * 40 + "\n")
        f.write("拒保货物明细 (前200条)\n")
        f.write("-" * 40 + "\n")
        for r in excluded[:200]:
            f.write(f"[{r['index']}] {r['cleaned'][:120]}\n")
            f.write(f"  排除原因: {r['exclusion_category']}\n")
            f.write(f"  匹配词: {r['exclusion_matches']}\n\n")

        # 未分类样本
        f.write("-" * 40 + "\n")
        f.write(f"未分类货物 (共{len(unclassified)}条, 展示前100条)\n")
        f.write("-" * 40 + "\n")
        for r in unclassified[:100]:
            f.write(f"[{r['index']}] {r['cleaned'][:120]}\n\n")

        # 分类分布
        f.write("-" * 40 + "\n")
        f.write("HS大类分布（可承保）\n")
        f.write("-" * 40 + "\n")
        hs_counter = Counter(r["hs_category"] for r in insured)
        for cat, count in hs_counter.most_common():
            f.write(f"  {cat}: {count}\n")

    print(f"详细报告已保存至: {output_path}")


# ============================================================
# 主程序入口
# ============================================================

def main():
    print("=" * 60)
    print("  航运保险货品智能分类系统 v1.0")
    print("  基于规则的关键词匹配方案")
    print("=" * 60)

    print("\n[1/4] 加载数据...")
    df = pd.read_excel(r"E:\SKD\金融科技前沿\期末项目\货物描述信息2025.xlsx")
    col = df.columns[0]
    descriptions = [(idx, str(val)) for idx, val in df[col].items() if idx > 0 and pd.notna(val)]
    print(f"  成功加载 {len(descriptions)} 条货物描述")

    print("\n[2/4] 运行分类...")
    results = run_classification(descriptions)

    print("\n[3/4] 统计结果...")
    excluded, insured, unclassified = print_statistics(results)

    print("\n[4/4] 保存结果...")
    output_dir = r"E:\SKD\金融科技前沿\期末项目"
    save_results(results, f"{output_dir}/分类结果.xlsx")
    save_detailed_report(results, f"{output_dir}/分类报告.txt")

    coverage = (1 - len(unclassified)/len(results)) * 100
    print("\n" + "=" * 60)
    print(f"  分类完成!  覆盖率: {coverage:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
