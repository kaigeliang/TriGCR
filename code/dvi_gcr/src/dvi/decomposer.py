"""
QueryDecomposer: 将复杂的 KGQA 问题拆解为原子约束列表。

输入: question (str), q_entities (List[str])
输出: List[Dict] — 每个字典是一个原子约束:
  {
    "id":        "c1",           # 约束编号
    "type":      "relation",     # relation | attribute | filter
    "anchor":    "Christopher Nolan",  # KG 中的已知实体 (relation/attribute 类型)
    "predicate": "birth_year < 1970",  # 字面量谓词 (filter 类型)
    "hint":      "..."           # 自然语言提示, 帮助 KG-LLM 搜索路径
  }

当 LLM 调用失败或返回解析错误时, 自动 fallback 到"每个 q_entity 一个约束"
(与原 GCR baseline 等价, 确保系统不崩溃).
"""

import json
import re
import os
import time
from typing import List, Dict, Optional
from openai import OpenAI
import dotenv

dotenv.load_dotenv()

# ─── Few-shot prompt ─────────────────────────────────────────────────────────

DECOMPOSE_SYSTEM = """You are a knowledge graph query analyzer.
Decompose a complex question into atomic constraints, each anchored to ONE known entity.
Output valid JSON only. No extra text."""

DECOMPOSE_FEW_SHOT = """Decompose the question into atomic constraints over the answer variable.

Use these constraint types:
- "relation":   the answer is reachable from anchor via KG relations (multi-hop ok)
- "attribute":  the answer has a property/award linking to anchor
- "filter":     the answer must satisfy a literal predicate (date, type, count, nationality)

Rules:
1. Every "relation" or "attribute" constraint must have an "anchor" from q_entities.
2. Every "filter" constraint must have a "predicate" (Python-evaluable string when possible).
3. Keep constraints minimal — one constraint per sub-condition.
4. All constraints share the same "answer_variable".

### Example 1
Question: Which American actors born before 1970 have won both a Golden Globe and an Oscar, and starred in films directed by Christopher Nolan?
q_entities: ["Christopher Nolan", "Golden Globe", "Oscar"]
Output:
{
  "answer_variable": "actor",
  "constraints": [
    {"id": "c1", "type": "relation",  "anchor": "Christopher Nolan", "hint": "films directed by Christopher Nolan starring the actor"},
    {"id": "c2", "type": "attribute", "anchor": "Golden Globe",       "hint": "actor won a Golden Globe award"},
    {"id": "c3", "type": "attribute", "anchor": "Oscar",              "hint": "actor won an Oscar award"},
    {"id": "c4", "type": "filter",    "predicate": "birth_year < 1970"},
    {"id": "c5", "type": "filter",    "predicate": "nationality == 'American'"}
  ]
}

### Example 2
Question: What is the capital of the country where the Eiffel Tower is located?
q_entities: ["Eiffel Tower"]
Output:
{
  "answer_variable": "city",
  "constraints": [
    {"id": "c1", "type": "relation", "anchor": "Eiffel Tower", "hint": "country where Eiffel Tower is located, then capital of that country"}
  ]
}

### Example 3
Question: Who are the siblings of Barack Obama's wife?
q_entities: ["Barack Obama"]
Output:
{
  "answer_variable": "person",
  "constraints": [
    {"id": "c1", "type": "relation", "anchor": "Barack Obama", "hint": "Barack Obama's wife, then her siblings"}
  ]
}

### Now decompose:
Question: {question}
q_entities: {q_entities}
Output:
"""


def _parse_json_from_response(text: str) -> Optional[Dict]:
    """从 LLM 响应中提取 JSON, 兼容带 markdown 代码块的格式."""
    # 去掉 ```json ... ``` 包裹
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 尝试提取第一个 {...} 块
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
    return None


def _fallback_constraints(q_entities: List[str]) -> Dict:
    """
    Fallback: 每个 q_entity 生成一个 relation 约束.
    与原 GCR baseline 行为等价.
    """
    constraints = [
        {
            "id": f"c{i+1}",
            "type": "relation",
            "anchor": ent,
            "hint": f"reasoning path starting from {ent}",
        }
        for i, ent in enumerate(q_entities)
    ]
    return {"answer_variable": "answer", "constraints": constraints}


class QueryDecomposer:
    """
    将 KGQA 问题拆解为原子约束列表.

    参数:
        model_name: OpenAI 模型名, 默认 gpt-4o-mini (快且便宜)
        retry: API 失败时的重试次数
        cache: 是否缓存分解结果 (避免重复 API 调用)
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        retry: int = 3,
        cache: bool = True,
    ):
        self.model_name = model_name
        self.retry = retry
        self.cache = cache
        self._cache: Dict[str, Dict] = {}
        self.last_call_metadata: Dict = {}

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not found. Please set it in your .env file."
            )
        client_kwargs = {"api_key": api_key}
        base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)

    def decompose(self, question: str, q_entities: List[str]) -> Dict:
        """
        主入口: 返回结构化约束字典.

        返回格式:
        {
          "answer_variable": "actor",
          "constraints": [
            {"id": "c1", "type": "relation", "anchor": "...", "hint": "..."},
            {"id": "c4", "type": "filter",   "predicate": "birth_year < 1970"},
            ...
          ]
        }
        """
        cache_key = f"{question}||{sorted(q_entities)}"
        if self.cache and cache_key in self._cache:
            self.last_call_metadata = {
                "cache_hit": True,
                "api_calls": 0,
                "used_fallback": False,
                "fallback_reason": None,
            }
            return self._cache[cache_key]

        # 只有一个实体且问题较简单时, 直接 fallback (省 API 调用)
        if len(q_entities) == 1:
            result = _fallback_constraints(q_entities)
            if self.cache:
                self._cache[cache_key] = result
            self.last_call_metadata = {
                "cache_hit": False,
                "api_calls": 0,
                "used_fallback": True,
                "fallback_reason": "single_entity",
            }
            return result

        # The few-shot examples contain literal JSON braces, so avoid str.format().
        prompt = (
            DECOMPOSE_FEW_SHOT
            .replace("{question}", question)
            .replace("{q_entities}", json.dumps(q_entities, ensure_ascii=False))
        )

        api_calls = 0
        for attempt in range(self.retry):
            try:
                api_calls += 1
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": DECOMPOSE_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    timeout=30,
                )
                raw = response.choices[0].message.content.strip()
                parsed = _parse_json_from_response(raw)
                if parsed and "constraints" in parsed and len(parsed["constraints"]) > 0:
                    # 校验每个约束有必要字段
                    valid = True
                    for c in parsed["constraints"]:
                        if c.get("type") in ("relation", "attribute") and "anchor" not in c:
                            valid = False
                            break
                        if c.get("type") == "filter" and "predicate" not in c:
                            valid = False
                            break
                    if valid:
                        if self.cache:
                            self._cache[cache_key] = parsed
                        self.last_call_metadata = {
                            "cache_hit": False,
                            "api_calls": api_calls,
                            "used_fallback": False,
                            "fallback_reason": None,
                        }
                        return parsed
            except Exception as e:
                print(f"[Decomposer] Attempt {attempt+1} failed: {e}")
                if attempt < self.retry - 1:
                    time.sleep(5)

        # 所有重试失败 → fallback
        print(f"[Decomposer] All retries failed for: {question!r}. Using fallback.")
        result = _fallback_constraints(q_entities)
        if self.cache:
            self._cache[cache_key] = result
        self.last_call_metadata = {
            "cache_hit": False,
            "api_calls": api_calls,
            "used_fallback": True,
            "fallback_reason": "api_or_parse_failure",
        }
        return result

    def get_relation_constraints(self, decomposed: Dict) -> List[Dict]:
        """只返回 relation / attribute 类约束 (需要构建 Trie 的)."""
        return [
            c for c in decomposed["constraints"]
            if c["type"] in ("relation", "attribute")
        ]

    def get_filter_constraints(self, decomposed: Dict) -> List[Dict]:
        """只返回 filter 类约束 (直接在 KG 属性上求值的)."""
        return [
            c for c in decomposed["constraints"]
            if c["type"] == "filter"
        ]

    def save_cache(self, path: str):
        """持久化分解缓存到 JSON 文件."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)

    def load_cache(self, path: str):
        """从 JSON 文件加载分解缓存."""
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self._cache = json.load(f)
            print(f"[Decomposer] Loaded {len(self._cache)} cached decompositions from {path}")
