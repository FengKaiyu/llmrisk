#!/usr/bin/env python3
"""KB 检索器：31 张类别卡片 + 40 张边界卡片，纯字典检索，无向量库。

索引：
  category_by_id    category_id -> 类别卡片
  boundary_by_pair  (a, b) 有序对 -> [边界卡片]   多类卡按其全部内部两两组合入索引
  boundary_by_cat   category_id -> [单类细化案例]

视图：
  core_view()       B3 注入文本 = 全局原则 + 31 小类判定条件/判断关键（紧凑）
  retrieve()        B4 注入文本 = 候选类别卡片 + 组合命中边界卡 + 单类细化案例（预算内）

已知缺口（有意不补）：A.4a↔A.1f 无边界卡片，检索不到即不注入。
"""
from __future__ import annotations
import json
import re
from itertools import combinations
from pathlib import Path

KB_DIR = Path(__file__).resolve().parent / "kb"

# 边界卡中的溯源行（"题干主题取自…"）是建库来源说明，不是判定规则，注入时剔除
_PROVENANCE = re.compile(r"^题干主题取自.*$", re.M)


def render_category(c: dict) -> str:
    return f"【{c['category_id']} {c['category_name']}】\n{c['rule_text'].strip()}"


def render_boundary(b: dict) -> str:
    body = _PROVENANCE.sub("", b["distinctions"]).strip()
    pair = " vs ".join(b["categories"])
    return f"◆ 边界规则 {b['rule_id']}（{pair}）：{b['title']}\n{body}"


class KBRetriever:
    def __init__(self, kb_dir: Path = KB_DIR, measure=None):
        """measure: 文本长度函数，默认字符数；运行时可传 tokenizer 版做 token 预算。"""
        self._measure = measure or (lambda s: len(s))
        self.categories = json.load(open(kb_dir / "category_rules.json", encoding="utf-8"))
        self.boundaries = json.load(open(kb_dir / "boundary_rules.json", encoding="utf-8"))
        self.global_principles = json.load(open(kb_dir / "global_principles.json", encoding="utf-8"))
        self.category_by_id = {c["category_id"]: c for c in self.categories}
        self.boundary_by_pair: dict[tuple, list] = {}
        self.boundary_by_cat: dict[str, list] = {}
        for b in self.boundaries:
            cats = b["categories"]
            if len(cats) >= 2:
                for pair in combinations(sorted(cats), 2):
                    self.boundary_by_pair.setdefault(pair, []).append(b)
            else:
                self.boundary_by_cat.setdefault(cats[0], []).append(b)
        # 渲染与长度缓存：卡片固定，只算一次
        self._cat_text = {cid: render_category(c) for cid, c in self.category_by_id.items()}
        self._bnd_text = {b["rule_id"]: render_boundary(b) for b in self.boundaries}

    # ---------------- B3 视图 ----------------

    def global_text(self) -> str:
        return self.global_principles["principles_text"].strip()

    def core_view(self) -> str:
        lines = [self.global_text(), "", "三十一小类判定条件与判断关键："]
        for c in self.categories:
            seg = [f"- {c['category_id']} {c['category_name']}："]
            if c["positive_conditions"]:
                seg.append(f"判定条件：{c['positive_conditions'].strip()}")
            if c["required_evidence"]:
                seg.append(f"判断关键：{c['required_evidence'].strip()}")
            lines.append("".join(seg))
        return "\n".join(lines)

    # ---------------- B4 检索 ----------------

    def retrieve(self, cand_ids: list[str], budget: int | None = None, k: int = 3):
        """输入候选小类（取前 k 个有效 ID），返回 (注入文本, rule_id 列表, 统计)。

        优先级：类别卡片 > 候选两两组合命中的边界卡 > 单类细化案例。
        逐张累加长度，放不下则跳过该张（类别卡片始终保留，预算只裁边界部分）。
        """
        cands = [c for c in cand_ids if c in self.category_by_id][:k]
        chosen, rule_ids, used = [], [], 0

        def push(text: str, rid: str) -> bool:
            nonlocal used
            t = self._measure(text)
            if budget is not None and used + t > budget:
                return False
            used += t
            chosen.append(text)
            rule_ids.append(rid)
            return True

        for cid in cands:
            push(self._cat_text[cid], cid)

        seen: set[str] = set(rule_ids)
        pair_cards = []
        for pair in combinations(sorted(cands), 2):
            for card in self.boundary_by_pair.get(pair, []):
                if card["rule_id"] not in seen:
                    seen.add(card["rule_id"])
                    pair_cards.append(card)
        for card in pair_cards:
            push(self._bnd_text[card["rule_id"]], card["rule_id"])

        for cid in cands:
            for card in self.boundary_by_cat.get(cid, []):
                if card["rule_id"] in seen:
                    continue
                seen.add(card["rule_id"])
                if not push(self._bnd_text[card["rule_id"]], card["rule_id"]):
                    break

        return "\n\n".join(chosen), rule_ids, {"cards": len(rule_ids), "len": used}


if __name__ == "__main__":  # 简单自测：python3 kb_retriever.py
    r = KBRetriever()
    core = r.core_view()
    print(f"[core] {len(core)} chars")
    for cands in (["A.3b", "A.1g", "A.4c"], ["A.4a", "A.1f", "SAFE"], ["A.5b"]):
        text, rids, st = r.retrieve(cands)
        print(f"[retrieve] {cands} -> {rids}  {st['len']} chars")
