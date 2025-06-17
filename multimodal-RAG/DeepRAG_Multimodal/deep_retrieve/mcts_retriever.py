"""
Monte‑Carlo Tree Search wrapper for multi‑intent retrieval
Author: <your‑name>
Date  : 2025‑06‑17
License: MIT  (保持与仓库其余代码一致即可)

核心思想：
    先用底层检索器为每个 intent 拿 top‑k 文档，随后用 MCTS 组合这些
    (intent, doc) 对，最大化 “覆盖度 + 文档质量 + 多样性” 奖励。
--------------------------------------------------------------------
与仓库其他模块的耦合点：
    • 需要一个 `base_retriever`，它至少暴露
          retrieve(text_query: str, top_k: int) -> List[Document]
      或者 get_topk(...)；若方法名不同，请在 _prefetch 里改一行即可。
    • 假设每个 Document 至少包含
          .doc_id  (hashable)
          .score   (float, 越大越相关)
          .embedding  (1‑D numpy array)  —— 若无，可把 diversity 权重设 0
--------------------------------------------------------------------
使用示例：
    from deep_retrieve.retriever_multimodal_bge import MultimodalMatcher
    from deep_retrieve.mcts_retriever          import MCTSWrapper

    matcher = MultimodalMatcher(cfg)
    retriever = MCTSWrapper(matcher, rollout_budget=300)
    docs = retriever.search(query, intents)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional
import math
import random
import numpy as np


# ---------- 数据结构 -------------------------------------------------
@dataclass
class State:
    """已选文档集合 & 已覆盖意图集合"""
    docs: Set             # {doc_id, ...}
    covered: Set          # {intent_idx, ...}

    def clone(self) -> "State":
        return State(set(self.docs), set(self.covered))


@dataclass
class MCTSNode:
    state: State
    parent: Optional["MCTSNode"]
    action: Optional[Tuple[int, int]]          # (intent_idx, cand_idx)  — 根节点为 None
    children: List["MCTSNode"] = field(default_factory=list)
    visit_count: int = 0
    total_value: float = 0.0
    untried_actions: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def q(self) -> float:
        return self.total_value / self.visit_count if self.visit_count else 0.0


# ---------- MCTS 包装器 ----------------------------------------------
class MCTSWrapper:
    """
    外部 API:
        search(query: str, intents: List[str]) -> List[Document]
    """
    def __init__(
        self,
        base_retriever,
        rollout_budget: int = 300,
        k_per_intent: int = 5,
        max_depth: int = 10,
        c_puct: float = 1.4,
        reward_weights: Dict[str, float] | None = None,
        diversity_metric: str = "cos",          # ["cos", "l2"]
        random_seed: int | None = None,
    ):
        self.base = base_retriever
        self.rollout_budget = rollout_budget
        self.k = k_per_intent
        self.max_depth = max_depth
        self.c_puct = c_puct
        self.rw = reward_weights or {"coverage": 1.0, "quality": 0.5, "diversity": 0.3}
        self.diversity_metric = diversity_metric
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        # 运行时变量
        self.intent_pool: List[List] = []   # shape: [num_intent][k] = Document
        self.doc_embeddings: Dict = {}      # doc_id -> np.ndarray


    # ---- Public -----------------------------------------------------
    def search(self, query: str, intents: List[str]):
        """
        返回一组 Document；顺序可不作保证，调用方可自行 rerank。
        """
        self._prefetch(intents)

        root = MCTSNode(state=State(set(), set()), parent=None, action=None)
        root.untried_actions = [(i, j) for i in range(len(intents))
                                      for j in range(len(self.intent_pool[i]))]

        for _ in range(self.rollout_budget):
            path = self._select(root)
            leaf  = self._expand(path[-1])
            reward = self._simulate(leaf)
            self._backprop(path, reward)

        # 找访问次数最多的子节点作为 best
        best_child = max(root.children, key=lambda n: n.visit_count, default=None)
        if not best_child:
            return []   # fall back
        final_doc_ids = best_child.state.docs
        return [self._doc_by_id(did) for did in final_doc_ids]


    # ---- MCTS 内部步骤 ----------------------------------------------
    def _select(self, node: MCTSNode) -> List[MCTSNode]:
        """沿着 UCT 最大的 child 下行，直到叶节点"""
        path = [node]
        while node.children and not node.untried_actions:
            log_N = math.log(node.visit_count)
            def uct(n: MCTSNode) -> float:
                return n.q + self.c_puct * math.sqrt(log_N / n.visit_count)
            node = max(node.children, key=uct)
            path.append(node)
        return path

    def _expand(self, node: MCTSNode) -> MCTSNode:
        if not node.untried_actions or len(node.state.docs) >= self.max_depth:
            return node

        action = node.untried_actions.pop(random.randrange(len(node.untried_actions)))
        intent_idx, cand_idx = action
        doc = self.intent_pool[intent_idx][cand_idx]

        new_state = node.state.clone()
        new_state.docs.add(doc.doc_id)
        new_state.covered.add(intent_idx)

        child = MCTSNode(state=new_state, parent=node, action=action)
        # 初始化 child 的未尝试动作（延迟到 select 时再填充更高效）
        child.untried_actions = node.untried_actions.copy()
        node.children.append(child)
        return child

    def _simulate(self, node: MCTSNode) -> float:
        """
        Roll‑out: 从当前 state 随机追加文档直到 max_depth 或覆盖全部意图；
        返回累积 reward（越大越好）。
        """
        state = node.state.clone()
        remaining_actions = node.untried_actions.copy()
        random.shuffle(remaining_actions)

        while remaining_actions \
              and len(state.docs) < self.max_depth \
              and len(state.covered) < len(self.intent_pool):
            intent_idx, cand_idx = remaining_actions.pop()
            doc = self.intent_pool[intent_idx][cand_idx]
            # 若已选过该文档则跳过
            if doc.doc_id in state.docs:
                continue
            state.docs.add(doc.doc_id)
            state.covered.add(intent_idx)

        return self._reward(state)

    def _backprop(self, path: List[MCTSNode], reward: float):
        for node in path:
            node.visit_count += 1
            node.total_value += reward

    # ---- 奖励函数 ----------------------------------------------------
    def _reward(self, state: State) -> float:
        # 1. intent 覆盖奖励
        coverage_r = len(state.covered) / len(self.intent_pool)

        # 2. 文档质量奖励：用检索分数平均
        if state.docs:
            quality_r = np.mean([self._doc_by_id(did).score for did in state.docs])
        else:
            quality_r = 0.0

        # 3. 多样性：平均两两余弦距离
        diversity_r = 0.0
        if len(state.docs) > 1 and self.rw.get("diversity", 0) > 0:
            embs = [self.doc_embeddings[d] for d in state.docs if d in self.doc_embeddings]
            if len(embs) >= 2:
                embs = np.stack(embs)
                norm = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
                embs_norm = embs / norm
                sim = embs_norm @ embs_norm.T          # [n, n]
                upper = sim[np.triu_indices(len(embs), k=1)]
                diversity_r = 1.0 - np.mean(upper)     # 越大越 diverse

        return (self.rw["coverage"]  * coverage_r +
                self.rw["quality"]   * quality_r  +
                self.rw["diversity"] * diversity_r)

    # ---- 工具函数 ----------------------------------------------------
    def _prefetch(self, intents: List[str]):
        """一次性取出每个 intent 的 top‑k 文档，缓存 embeddings 以便多样性计算。"""
        self.intent_pool.clear()
        self.doc_embeddings.clear()

        for intent in intents:
            # --- 如方法名不相符，在此替换即可 --------------------------
            docs = self.base.retrieve(intent, top_k=self.k)
            # docs = self.base.get_topk(intent, self.k)
            # ----------------------------------------------------------
            self.intent_pool.append(docs)
            for doc in docs:
                if hasattr(doc, "embedding") and doc.embedding is not None:
                    self.doc_embeddings[doc.doc_id] = np.asarray(doc.embedding)

    def _doc_by_id(self, doc_id):
        """根据 doc_id 在 intent_pool 中反查对象（最常用的方式）。"""
        for bucket in self.intent_pool:
            for doc in bucket:
                if doc.doc_id == doc_id:
                    return doc
        raise KeyError(f"Unknown doc_id {doc_id}")
