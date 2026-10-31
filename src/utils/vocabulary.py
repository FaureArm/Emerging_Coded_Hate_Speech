"""Shared anchor variants and experimental retrieval seeds for Raiders.

Anchor variants are preserved from Prepare_Raiders_Posts.ipynb, whose source
attribution is Wu et al. (2026), Appendix A. Seeds below are separate pilot
choices, not a verified reproduction of the paper's retrieval vocabulary.
Seed matches indicate topic relevance, not hateful meaning. Country/demonym
terms can be ambiguous; inspect retrieved posts before fixing an experiment.
Matching is case-insensitive and word-bounded: variants must be listed explicitly.
"""

anchor_terms_by_hate_category = {
    "antisemitism": {
        "kike": ["kike"],
        "blood libel": ["blood libel"],
        "the goyim know": ["the goyim know"],
        "great replacement": ["great replacement"],
        "holocaust/gas chambers": ["holocaust", "gas chambers"],
        "holohoax": ["holohoax"],
        "jew coup": ["jew coup"],
        "jew down": ["jew down"],
        "jewish feature": ["jewish feature"],
        "jewish figure": ["jewish figure"],
        "jewish lighting": ["jewish lighting"],
        "jewish power": ["jewish power"],
        "kosher tax": ["kosher tax"],
        "laughing jew": ["laughing jew"],
        "nazi symbols": ["nazi symbols"],
        "protocols of the elders of zion": ["protocols of the elders of zion"],
        "qanon": ["qanon"],
        "smirking merchant": ["smirking merchant"],
        "zionism is racism": ["zionism is racism"],
        "zionist occupied government": ["zionist occupied government"],
    },
    "anti_black": {
        "nigger": [
            "nigger",
            "niger",
            "nig",
            "nigor",
            "nigra",
            "nigar",
            "niggur",
            "nigga",
            "niggah",
            "niggar",
            "nigguh",
            "niggress",
            "nigette",
            "negro",
            "neger",
        ],
        "jiggaboo": [
            "jiggaboo",
            "jiggerboo",
            "niggerboo",
            "jiggabo",
            "jigarooni",
            "jijjiboo",
        ],
        "cotton picker": ["cotton picker"],
        "moon cricket": ["moon cricket"],
        "darky": ["darky", "darkey", "darkie"],
        "pickaninny": ["pickaninny"],
        "porch monkey": ["porch monkey"],
        "groid": ["groid"],
        "jungle bunny": ["jungle bunny"],
        "sambo": ["sambo"],
        "uncle tom": ["uncle tom"],
        "golliwog": ["golliwog"],
        "spear chucker": ["spear chucker"],
        "alligator bait": ["alligator bait", "’gator bait"],
        "nig-nog": ["nig-nog", "nog", "nignog"],
        "coon": ["coon", "cooney"],
        "buckwheat": ["buckwheat"],
        "spook": ["spook"],
        "ape": ["ape"],
        "jim crow": ["jim crow", "crow"],
        "spade": ["spade"],
        "blackie": ["blackie"],
    },
    "islamophobia": {
        "muzat": ["muzat"],
        "muzzies": ["muzzies","muzzie"],
        "pislam": ["pislam"],
        "mohammedan": ["mohammedan"],
        "mudslime": ["mudslime", "mudslimes", "mudslim"],
        "turk": ["turk", "turks", "turkish"],
    },
    "anti_asian": {
        "slant eye": ["slant eye"],
        "chinaman": ["chinaman"],
        "yellow asian": ["yellow asian"],
        "banana": ["banana"],
        "twinkie": ["twinkie"],
        "ching chong": ["ching chong"],
        "chink": ["chink", "chinky"],
        "chonky": ["chonky"],
        "coolie": ["coolie"],
        "curry-muncher": ["curry-muncher"],
        "zipper head": ["zipper head"],
        "pajeet": ["pajeet"],
        "yellow invader": ["yellow invader"],
        "rice nigger": ["rice nigger"],
        "mongoloid": ["mongoloid"],
        "gook": ["gook"],
    },
    "anti_lgbtq": {
        "faggot": ["faggot", "fagot", "faggotry", "faggoty", "fag", "faggy", "fagged"],
        "homo": ["homo"],
        "tranny": ["tranny"],
        "swishy": ["swishy", "swisher"],
        "dyke": ["dyke", "bulldyke", "bulldagger"],
        "twink": ["twink"],
        "chaser": ["chaser"],
        "gay lord": ["gay lord", "gaylord"],
        "fairy": ["fairy"],
        "daisy": ["daisy", "pansy", "tulip"],
        "cornhole": ["cornhole", "cornholer"],
        "flamer": ["flamer", "flaming"],
        "alphabet people": ["alphabet people"],
        "fruit": ["fruit", "fruity"],
        "fag / hag": ["fag", "hag"],
        "friend of dorothy": ["friend of dorothy"],
        "nancy boy": ["nancy boy"],
        "lipstick lesbian": ["lipstick lesbian"],
        "batting for the other team": ["batting for the other team"],
        "punker": ["punker"],
        "capon": ["capon"],
        "bean queen": ["bean queen"],
        "curry queen": ["curry queen"],
        "harry hoofter": ["harry hoofter"],
    },
}

# Independent lists: changing retrieval seeds does not change the anchors.
seed_words_by_hate_category = {
    "antisemitism": [
        "jew",
        "jews",
        "jewish",
        "zionist",
        "zionism",
        "antisemitism",
        "israel",
    ],
    "anti_black": [
        "black people",
        "black person",
        "black men",
        "black women",
        "african american",
        "african americans",
        "african-american",
        "african-americans",
        "anti-black",
        "antiblack",
    ],
    "islamophobia": [
        "islam",
        "muslim",
        "muslims",
        "islamic",
        "mosque",
        "mosques",
        "quran",
        "koran",
        "islamophobia",
    ],
    "anti_asian": [
        "asian",
        "asians",
        "chinese",
        "japanese",
        "korean",
        "koreans",
        "indian",
        "indians",
        "south asian",
        "east asian",
        "anti-asian",
    ],
    "anti_lgbtq": [
        "lgbt",
        "lgbtq",
        "lgbtqia",
        "gay",
        "gays",
        "lesbian",
        "lesbians",
        "bisexual",
        "transgender",
        "trans people",
        "nonbinary",
        "non-binary",
        "homophobia",
        "transphobia",
    ],
}


HATE_CATEGORIES = tuple(sorted(anchor_terms_by_hate_category.keys() & seed_words_by_hate_category.keys()))


def vocabulary_for_category(category):
    """Select anchors and retrieval seeds together, with no category fallback."""
    if category not in HATE_CATEGORIES:
        raise ValueError(f"Unknown hate category {category!r}. Choose one of: {', '.join(HATE_CATEGORIES)}")
    anchors = anchor_terms_by_hate_category[category]
    seeds = list(seed_words_by_hate_category[category])
    if not anchors or not seeds or any(not isinstance(word, str) or not word.strip() for word in seeds):
        raise ValueError(f"Category {category!r} needs nonempty anchors and retrieval seeds in vocabulary.py.")
    return anchors, seeds
