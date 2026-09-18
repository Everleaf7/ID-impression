from __future__ import annotations

from src.semantic.id_parser import tokenize_id
from src.semantic.schemas import Concept


KNOWN_CONCEPTS = {
    "示例春花": {
        "tokens": ["示例", "春", "花"],
        "literal_meanings": ["花"],
        "abstract_concepts": ["春天"],
        "associations": ["春雨", "阳光", "花"],
        "expression_type": "symbolic",
        "association_methods": ["word_split", "semantic_expansion"],
        "association_distance": 2,
        "detail_level": 2,
        "main_concept": "从春天联想到春雨、阳光和花",
        "main_subjects": ["乌云", "雨", "花", "太阳"],
        "secondary_elements": [],
        "mood": ["清新", "轻松"],
        "accent_colors": ["yellow", "green", "blue"],
        "composition": "白底，上方乌云和雨，下方小花与太阳光，大面积留白",
        "image_prompt": "a strange little flower under a dark spring rain cloud, yellow sunlight peeking out",
    },
    "样例果汁": {
        "tokens": ["样例", "果汁"],
        "literal_meanings": ["果汁"],
        "abstract_concepts": [],
        "associations": ["饮料瓶"],
        "expression_type": "object",
        "association_methods": ["literal", "objectification"],
        "association_distance": 0,
        "detail_level": 2,
        "main_concept": "一瓶造型略显奇怪的果汁",
        "main_subjects": ["果汁瓶"],
        "secondary_elements": ["吸管"],
        "mood": ["直接", "俏皮"],
        "accent_colors": ["orange"],
        "composition": "白底，小果汁瓶偏下，大面积留白",
        "image_prompt": "a small pale yellow juice bottle with a bent straw and one orange accent",
    },
    "虚构玫瑰": {
        "tokens": ["虚构", "玫瑰"],
        "literal_meanings": ["玫瑰"],
        "abstract_concepts": ["自由"],
        "associations": ["艺术", "独立", "浪漫", "绘画"],
        "expression_type": "scene",
        "association_methods": ["word_split", "semantic_expansion", "concept_merge"],
        "association_distance": 2,
        "detail_level": 3,
        "main_concept": "一个自由的人正在画玫瑰",
        "main_subjects": ["人物", "画架", "玫瑰"],
        "secondary_elements": ["颜料"],
        "mood": ["自由", "浪漫"],
        "accent_colors": ["red", "green", "purple"],
        "composition": "白底，小场景偏下，人物面对画架，大面积留白",
        "image_prompt": "a clumsy stick figure painting one red rose on a tiny easel",
    },
}


def generate_concept(user_id: str) -> Concept:
    value = user_id.strip()
    if not value:
        raise ValueError("ID / 昵称不能为空")
    if value in KNOWN_CONCEPTS:
        return Concept(id=value, **KNOWN_CONCEPTS[value])

    tokens = tokenize_id(value)
    joined = "".join(tokens)
    if any(word in joined for word in ("花", "玫瑰")):
        kind, subjects, colors = "symbolic", ["一朵花"], ["red", "green"]
        idea = f"把“{value}”直接联想成一朵略显古怪的花"
        methods, distance = ["keyword_selection", "semantic_expansion"], 1
        image_prompt = "one strange little flower with mismatched petals and two color accents"
    elif any(word in joined for word in ("猫", "喵")):
        kind, subjects, colors = "character", ["一只古怪的小猫"], ["black", "red"]
        idea = f"从“{value}”联想到一只造型古怪的小猫"
        methods, distance = ["keyword_selection", "characterization"], 1
        image_prompt = "one odd small cat with an exaggerated tail and a playful pose"
    elif any(word in joined for word in ("星", "月", "日", "太阳")):
        kind, subjects, colors = "symbolic", ["星月符号"], ["yellow", "purple"]
        idea = f"把“{value}”拆成随手画出的星月符号"
        methods, distance = ["word_split", "symbolic_merge"], 1
        image_prompt = "a crooked crescent moon touching one oversized yellow star with tiny rays"
    elif any(word in joined for word in ("海", "浪", "水", "河")):
        kind, subjects, colors = "scene", ["夸张的水浪"], ["blue", "green"]
        idea = f"从“{value}”联想到一小片夸张的水浪"
        methods, distance = ["keyword_selection", "semantic_expansion"], 1
        image_prompt = "three oversized blue waves facing one tiny awkward stick figure"
    elif any(word in joined for word in ("蝶", "蝴蝶")):
        kind, subjects, colors = "symbolic", ["蝴蝶"], ["purple", "yellow"]
        idea = f"从“{value}”联想到一只线条简单的蝴蝶"
        methods, distance = ["keyword_selection", "objectification"], 1
        image_prompt = "one tiny purple butterfly above a thin crooked plant"
    elif any(word in joined for word in ("糖", "奶", "茶", "果汁")):
        kind, subjects, colors = "object", ["拟人饮料或糖果"], ["pink", "green"]
        idea = f"把“{value}”随意物体化成饮料或糖果"
        methods, distance = ["keyword_selection", "objectification"], 1
        image_prompt = "a tiny anthropomorphic drink cup made from candy shapes"
    elif any(word in joined for word in ("龙", "魔", "妖", "怪")):
        kind, subjects, colors = "fantasy_hybrid", ["简化的幻想生物"], ["purple", "green"]
        idea = f"把“{value}”脑补成结构不太准确的幻想生物"
        methods = ["characterization", "subjective_random_association"]
        distance = 2
        image_prompt = "one simplified inaccurate fantasy creature with a comically long body"
    elif any(word in joined for word in ("人", "少女", "萝莉")):
        kind, subjects, colors = "character", ["简化人物"], ["red", "blue"]
        idea = f"把“{value}”角色化成一个简单人物"
        methods, distance = ["characterization"], 1
        image_prompt = "one simple awkward stick figure holding an unexplained small object"
    else:
        kind, subjects, colors = "object", [f"由“{tokens[0]}”联想到的奇怪小物件"], ["blue", "yellow"]
        idea = f"抓住“{tokens[0]}”进行直接、略显随意的物体化联想"
        methods, distance = ["keyword_selection", "objectification"], 1
        image_prompt = "one strange small household object with an absurd handle and uneven proportions"

    return Concept(
        id=value,
        tokens=tokens,
        literal_meanings=[],
        abstract_concepts=[],
        associations=subjects.copy(),
        expression_type=kind,
        association_methods=methods,
        association_distance=distance,
        detail_level=2,
        main_concept=idea,
        main_subjects=subjects,
        mood=["俏皮", "随意"],
        accent_colors=colors,
        composition="白色画布，主体较小且略微偏下，大面积留白",
        image_prompt=image_prompt,
    )
