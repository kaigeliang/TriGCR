"""
CandidateIntersector: 对各约束解码出的候选实体集合做程序化交集.

这是 DVI 方案的核心步骤 —— 把"找交集"从 LLM 转移给 Python set 运算,
完全消除 conjunction 类问题的幻觉.

数据流:
  paths_per_constraint: Dict[constraint_id → List[str]]  (每个约束的路径列表)
  → extract_tail_entities()                               (从路径字符串提取尾实体)
  → candidate_sets: Dict[constraint_id → Set[str]]
  → intersect()                                           (程序化交集)
  → final_candidates: Set[str]
"""

import re
from typing import Dict, List, Set, Optional


# ─── 路径解析 ─────────────────────────────────────────────────────────────────

def extract_tail_entity(path_str: str) -> Optional[str]:
    """
    从单条路径字符串中提取尾实体.

    路径格式 (GCR 标准):
      "EntityA -> relation1 -> EntityB -> relation2 -> EntityC"
    尾实体 = 最后一个 " -> " 之后的部分.

    同时处理带 <PATH>...</PATH> 标记的格式.
    """
    # 去掉 <PATH> / </PATH> 标记
    path_str = re.sub(r"</?PATH>", "", path_str).strip()
    # GCR path-with-answer generations can append an answer block after the path.
    # Keep only the path portion before extracting the tail entity.
    path_str = re.split(r"\n\s*#\s*Answer\s*:|\n\s*Answer\s*:", path_str, maxsplit=1)[0].strip()
    # 去掉 "Answer: " 等前缀
    path_str = re.sub(r"^(Answer:|Path:)\s*", "", path_str, flags=re.IGNORECASE).strip()

    if " -> " not in path_str:
        return None

    parts = path_str.split(" -> ")
    # 路径格式: entity -> relation -> entity -> relation -> entity
    # 尾实体在偶数索引位置 (0, 2, 4, ...)
    # 最后一个 token 如果在奇数位, 说明路径截断了, 跳过
    if len(parts) % 2 == 0:
        # 偶数个 token → 最后是 relation, 不完整
        # 取倒数第二个 (最后一个实体)
        return parts[-2].strip()
    else:
        return parts[-1].strip()


def extract_tail_entities(paths: List[str]) -> Set[str]:
    """从多条路径中提取所有尾实体, 去重."""
    entities = set()
    for p in paths:
        tail = extract_tail_entity(p)
        if tail and tail.strip():
            entities.add(tail.strip())
    return entities


# ─── 主类 ─────────────────────────────────────────────────────────────────────

class CandidateIntersector:
    """
    对各约束候选集做程序化交集.

    参数:
        min_candidates: 若交集为空, 逐步放宽约束直到候选数 >= min_candidates.
                        设为 0 则不放宽 (保持严格交集).
        verbose: 是否打印中间信息 (调试用).
    """

    def __init__(self, min_candidates: int = 1, verbose: bool = False):
        self.min_candidates = min_candidates
        self.verbose = verbose
        self.last_stats = {}

    # ── 主入口 ────────────────────────────────────────────────────────────────

    def intersect(
        self,
        paths_per_constraint: Dict[str, List[str]],
        filter_constraints: Optional[List[Dict]] = None,
    ) -> Set[str]:
        """
        主入口: 输入各约束的路径列表, 返回最终候选实体集合.

        参数:
            paths_per_constraint: {constraint_id → 路径字符串列表}
            filter_constraints:   filter 类约束列表 (可选, 暂保留接口)

        返回:
            最终候选实体集合 (Set[str])
        """
        if not paths_per_constraint:
            self.last_stats = {
                "n_constraints": 0,
                "candidate_sizes": {},
                "strict_intersection_size": 0,
                "relaxed": False,
                "final_size": 0,
            }
            return set()

        # Step 1: 提取每个约束的候选尾实体集合
        candidate_sets: Dict[str, Set[str]] = {}
        for cid, paths in paths_per_constraint.items():
            entities = extract_tail_entities(paths)
            candidate_sets[cid] = entities
            if self.verbose:
                print(f"[Intersector] {cid}: {len(entities)} candidates → {list(entities)[:5]}")

        # Step 2: 程序化交集
        result = self._smart_intersect(candidate_sets)
        if self.verbose:
            print(f"[Intersector] Final candidates ({len(result)}): {list(result)[:10]}")
        return result

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _smart_intersect(self, candidate_sets: Dict[str, Set[str]]) -> Set[str]:
        """
        先做完整交集; 若结果为空则按集合大小从小到大逐步放宽约束,
        直到候选数 >= min_candidates 或所有约束都放宽完.
        """
        self.last_stats = {
            "n_constraints": len(candidate_sets),
            "candidate_sizes": {k: len(v) for k, v in candidate_sets.items()},
            "non_empty_constraints": [],
            "strict_intersection_size": 0,
            "relaxed": False,
            "dropped_constraints": [],
            "kept_constraints": [],
            "final_size": 0,
        }
        if not candidate_sets:
            return set()

        # 过滤掉空集合 (该约束没有找到任何路径)
        non_empty = {k: v for k, v in candidate_sets.items() if v}
        self.last_stats["non_empty_constraints"] = sorted(non_empty.keys())
        if not non_empty:
            return set()

        # 完整交集
        all_sets = list(non_empty.values())
        result = all_sets[0].copy()
        for s in all_sets[1:]:
            result &= s
        self.last_stats["strict_intersection_size"] = len(result)

        if len(result) >= self.min_candidates or self.min_candidates == 0:
            self.last_stats["kept_constraints"] = sorted(non_empty.keys())
            self.last_stats["final_size"] = len(result)
            return result

        # 交集为空 → 放宽策略: 按集合大小排序, 逐步去掉最大的集合重试
        if self.verbose:
            print("[Intersector] Full intersection empty. Relaxing constraints...")

        # 按集合大小升序排列 (最小的集合 = 最强约束, 优先保留)
        sorted_keys = sorted(non_empty.keys(), key=lambda k: len(non_empty[k]))

        for drop_count in range(1, len(sorted_keys)):
            kept_keys = sorted_keys[:-drop_count]
            kept_sets = [non_empty[k] for k in kept_keys]
            result = kept_sets[0].copy()
            for s in kept_sets[1:]:
                result &= s
            if self.verbose:
                dropped = sorted_keys[-drop_count:]
                print(f"[Intersector] Dropped {dropped}, result size: {len(result)}")
            if len(result) >= self.min_candidates:
                self.last_stats["relaxed"] = True
                self.last_stats["dropped_constraints"] = sorted(sorted_keys[-drop_count:])
                self.last_stats["kept_constraints"] = sorted(kept_keys)
                self.last_stats["final_size"] = len(result)
                return result

        # 实在没有交集 → 返回最小候选集 (最强约束的结果)
        result = non_empty[sorted_keys[0]]
        self.last_stats["relaxed"] = True
        self.last_stats["dropped_constraints"] = sorted(sorted_keys[1:])
        self.last_stats["kept_constraints"] = [sorted_keys[0]]
        self.last_stats["final_size"] = len(result)
        return result

    # ── 工具方法 (供上层 pipeline 调用) ──────────────────────────────────────

    def collect_evidence_paths(
        self,
        paths_per_constraint: Dict[str, List[str]],
        final_candidates: Set[str],
        max_paths: int = 5,
    ) -> List[str]:
        """
        收集与最终候选实体相关的路径, 供最终 LLM 推理时参考.

        只保留尾实体在 final_candidates 中的路径.
        每个约束最多保留 max_paths 条.
        """
        evidence = []
        for cid, paths in paths_per_constraint.items():
            count = 0
            for p in paths:
                tail = extract_tail_entity(p)
                if tail and tail.strip() in final_candidates:
                    evidence.append(p)
                    count += 1
                    if count >= max_paths:
                        break
        return evidence

    def stats(self, paths_per_constraint: Dict[str, List[str]]) -> Dict:
        """返回各约束的候选集统计, 用于 debug / 论文 case study."""
        result = {}
        for cid, paths in paths_per_constraint.items():
            entities = extract_tail_entities(paths)
            result[cid] = {
                "n_paths": len(paths),
                "n_candidates": len(entities),
                "candidates_sample": list(entities)[:10],
            }
        return result
