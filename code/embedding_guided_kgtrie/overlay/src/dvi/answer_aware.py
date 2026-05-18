"""Answer-aware candidate refinement for DVI.

The first DVI implementation treated every generated path tail as a final
answer. On CWQ this is often wrong: a valid KG path may end at an intermediate
entity such as a country, artist, or actor, while the question asks for a
currency, child, or film one hop later. This module uses lightweight question
type heuristics plus local KG expansion to turn intermediate path endpoints
into better final-answer candidates.
"""

from __future__ import annotations

from collections import defaultdict, deque
import re
from typing import Iterable


ANSWER_TYPE_SPECS = {
    "currency": {
        "question": ("currency",),
        "relation": ("currency", "currency_used", "money"),
        "entity": ("peso", "dollar", "euro", "yen", "pound", "franc", "rupee", "currency"),
        "type": ("currency",),
    },
    "film": {
        "question": ("movie", "movies", "film", "films"),
        "relation": ("film", "movie", "featured_in", "performance"),
        "entity": (),
        "type": ("film", "movie"),
    },
    "language": {
        "question": ("language", "spoken"),
        "relation": ("language", "official_language", "languages"),
        "entity": ("language", "spanish", "english", "french", "german", "chinese"),
        "type": ("language",),
    },
    "religion": {
        "question": ("religion",),
        "relation": ("religion",),
        "entity": ("religion", "judaism", "christianity", "islam", "buddhism", "hinduism"),
        "type": ("religion",),
    },
    "government_type": {
        "question": ("governmental type", "government type", "form of government"),
        "relation": ("government_form", "form_of_government", "government_type", "government"),
        "entity": ("state", "republic", "monarchy", "democracy", "government"),
        "type": ("government", "form of government", "government type"),
    },
    "country": {
        "question": ("country", "nation"),
        "relation": ("country", "nationality", "containedby"),
        "entity": (),
        "type": ("country", "nation"),
    },
    "city": {
        "question": ("capital", "city"),
        "relation": ("capital", "citytown", "location", "containedby", "contains"),
        "entity": (),
        "type": ("city", "capital"),
    },
    "team": {
        "question": ("team",),
        "relation": ("team", "sports_team", "winner", "championship"),
        "entity": ("f.c.", "fc", "club", "team"),
        "type": ("sports team", "football team", "baseball team"),
    },
    "stadium": {
        "question": ("stadium", "arena", "park", "field"),
        "relation": ("arena_stadium", "stadium", "sports_facility", "venue"),
        "entity": ("stadium", "park", "field", "arena"),
        "type": ("stadium", "sports facility", "venue"),
    },
    "educational_institution": {
        "question": ("educational institution", "school", "college", "university"),
        "relation": ("education", "school", "institution", "university"),
        "entity": ("school", "college", "university", "institute"),
        "type": ("school", "university", "educational institution"),
    },
    "person": {
        "question": (
            "who",
            "person",
            "actor",
            "actress",
            "daughter",
            "son",
            "child",
            "children",
            "sibling",
            "wife",
            "husband",
            "governor",
            "artist",
        ),
        "relation": (
            "people.person",
            "person",
            "children",
            "child",
            "sibling",
            "spouse",
            "parents",
            "office_holder",
            "governing_officials",
            "actor",
            "artist",
        ),
        "entity": (),
        "type": ("person", "actor", "artist", "politician"),
    },
}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def infer_answer_type(question: str, decomposed: dict | None = None) -> str | None:
    """Infer a coarse answer type from the question and decomposer output."""
    q = _norm(question)
    answer_variable = _norm((decomposed or {}).get("answer_variable", ""))

    # Avoid treating constraint words as answer types.
    if re.search(r"\b(when|what year|which year)\b", q):
        return None

    # Prefer the syntactic answer target near the wh phrase over arbitrary
    # entity words that appear later in the question.
    direct_patterns = [
        ("government_type", r"\b(what|which)\s+(?:is|are|was|were\s+)?(?:the\s+)?(?:type|types)\s+of\s+government\b"),
        ("government_type", r"\bgovernmental\s+types?\b"),
        ("currency", r"\b(what|which)\s+(?:is|are|was|were\s+)?(?:the\s+)?currency\b"),
        ("currency", r"\bcurrency\s+of\b"),
        ("country", r"\b(what|which|in which)\s+countries?\b"),
        ("country", r"\b(what|which)\s+nations?\b"),
        ("language", r"\b(what|which)\s+languages?\b"),
        ("religion", r"\b(what|which)\s+(?:is|are|was|were\s+)?(?:the\s+)?(?:predominant\s+)?religion\b"),
        ("person", r"\b(who|whom)\b"),
        ("person", r"\b(what|which)\s+(?:actor|actress|artist|person|governor|official|daughter|son|child|sibling|wife|husband)\b"),
        ("film", r"\b(what|which)\s+(?:movie|movies|film|films)\b"),
        ("educational_institution", r"\b(what|which)\s+(?:educational institution|school|college|university)\b"),
        ("stadium", r"\b(what|which)\s+(?:stadium|arena|park|field|venue)\b"),
        ("city", r"\b(what|which)\s+(?:city|capital)\b"),
        ("team", r"\b(what|which)\s+(?:team|club)\b"),
        ("team", r"\bname\s+of\s+the\s+team\b"),
    ]
    for answer_type, pattern in direct_patterns:
        if re.search(pattern, q):
            return answer_type

    if answer_variable in ANSWER_TYPE_SPECS and answer_variable != "answer":
        return answer_variable
    if q.startswith("who ") or q.startswith("who is ") or q.startswith("who are "):
        return "person"
    return None


def graph_adjacency(graph: Iterable[Iterable[str]], undirected: bool = False) -> dict[str, list[tuple[str, str]]]:
    adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for edge in graph or []:
        if len(edge) < 3:
            continue
        h, rel, t = map(str, edge[:3])
        adjacency[h].append((rel, t))
        if undirected:
            adjacency[t].append((f"inverse::{rel}", h))
    return adjacency


def entity_type_index(graph: Iterable[Iterable[str]]) -> dict[str, set[str]]:
    types: dict[str, set[str]] = defaultdict(set)
    for edge in graph or []:
        if len(edge) < 3:
            continue
        h, rel, t = map(str, edge[:3])
        rel_l = rel.lower()
        if "notable_types" in rel_l or rel_l.endswith(".type") or "type.object.type" in rel_l:
            types[h].add(t)
    return types


def _matches_any(text: str, needles: Iterable[str]) -> bool:
    text_l = _compact(text)
    return any(_compact(n) and _compact(n) in text_l for n in needles)


def _entity_matches_spec(entity: str, answer_type: str, types: dict[str, set[str]]) -> bool:
    spec = ANSWER_TYPE_SPECS.get(answer_type)
    if not spec:
        return False
    if _matches_any(entity, spec["entity"]):
        return True
    entity_types = " ".join(sorted(types.get(entity, ())))
    return _matches_any(entity_types, spec["type"])


def _relation_matches_spec(relation: str, answer_type: str) -> bool:
    spec = ANSWER_TYPE_SPECS.get(answer_type)
    if not spec:
        return False
    return _matches_any(relation, spec["relation"])


def _path_to_string(path: list[tuple[str, str, str]]) -> str:
    if not path:
        return ""
    parts = [path[0][0]]
    for _, rel, tail in path:
        parts.extend([rel, tail])
    return " -> ".join(parts)


def refine_candidates(
    *,
    question: str,
    data: dict,
    decomposed: dict,
    candidates: Iterable[str],
    max_hops: int = 2,
    max_added: int = 12,
    undirected: bool = False,
) -> tuple[set[str], dict]:
    """Return answer-aware candidates and diagnostics.

    The function keeps the original candidates when no reliable answer type is
    inferred or no answer-aware expansion is found.
    """
    original = [str(c) for c in candidates if str(c).strip()]
    answer_type = infer_answer_type(question, decomposed)
    stats = {
        "answer_type": answer_type,
        "original_size": len(set(original)),
        "typed_original_size": 0,
        "expanded_size": 0,
        "final_size": len(set(original)),
        "expansion_paths": [],
        "used_answer_aware": False,
    }
    if not original or not answer_type or answer_type not in ANSWER_TYPE_SPECS:
        return set(original), stats

    graph = data.get("graph") or []
    adjacency = graph_adjacency(graph, undirected=undirected)
    types = entity_type_index(graph)

    typed_original = {
        entity for entity in original
        if _entity_matches_spec(entity, answer_type, types)
    }

    expanded: set[str] = set()
    expansion_paths: list[str] = []
    for seed in original:
        queue = deque([(seed, [], 0)])
        seen = {seed}
        while queue and len(expanded) < max_added:
            node, path, depth = queue.popleft()
            if depth >= max_hops:
                continue
            for rel, nxt in adjacency.get(node, []):
                if nxt in seen:
                    continue
                seen.add(nxt)
                new_path = path + [(node, rel, nxt)]
                rel_match = _relation_matches_spec(rel, answer_type)
                endpoint_match = _entity_matches_spec(nxt, answer_type, types)
                if rel_match or endpoint_match:
                    expanded.add(nxt)
                    expansion_paths.append(_path_to_string(new_path))
                    if len(expanded) >= max_added:
                        break
                # Continue only through relevant relation chains to avoid
                # exploding into the whole local graph.
                if rel_match:
                    queue.append((nxt, new_path, depth + 1))

    original_set = set(original)
    if expanded:
        refined = original_set | set(expanded) | typed_original
        stats["used_answer_aware"] = True
    elif typed_original and len(typed_original) < len(original_set):
        refined = original_set | typed_original
        stats["used_answer_aware"] = True
    else:
        refined = original_set

    stats["typed_original_size"] = len(typed_original)
    stats["expanded_size"] = len(expanded)
    stats["final_size"] = len(refined)
    stats["expanded_candidates"] = sorted(expanded)
    stats["typed_original_candidates"] = sorted(typed_original)
    stats["expansion_paths"] = expansion_paths[:max_added]
    return refined, stats
