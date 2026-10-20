import re

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
STEM_LEN = 5

STOPWORDS = frozenset("""
и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по
только ее мне было вот от меня еще нет о из ему теперь когда даже ну ли если уже
или ни быть был него до вас опять уж вам ведь там потом себя ей может они тут где
есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под
будет ж тогда кто этот того потому этого какой совсем ним здесь этом один почти
тем чтобы нее сейчас были куда зачем всех можно при два об другой после над
больше тот через эти нас про всего них какая много три эту этой перед том такой
им более между какие какое какая каком сколько можно нужно
""".split())


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for tok in _TOKEN_RE.findall(text.lower().replace("ё", "е")):
        if len(tok) < 2 or tok in STOPWORDS:
            continue
        out.append(tok[:STEM_LEN])
    return out


def acl_allows(allowed_groups: list[str], user_groups: list[str]) -> bool:
    return bool(set(allowed_groups) & set(user_groups))


def rrf_fuse(dense: list[dict], sparse: list[dict], k_const: int = 60) -> list[dict]:
    scores: dict = {}
    meta: dict = {}
    for ranked in (dense, sparse):
        for rank, item in enumerate(ranked):
            scores[item["id"]] = scores.get(item["id"], 0.0) + 1.0 / (k_const + rank + 1)
            meta.setdefault(item["id"], item)
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [{**meta[i], "rrf_score": sc} for i, sc in fused]


def is_relevant(dense: list[dict], sparse: list[dict], min_dense_score: float) -> bool:
    if any(s.get("score", 0.0) > 0 for s in sparse):
        return True
    return bool(dense) and dense[0]["score"] >= min_dense_score