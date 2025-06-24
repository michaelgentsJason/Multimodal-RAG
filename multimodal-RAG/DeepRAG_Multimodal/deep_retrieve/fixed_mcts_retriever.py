"""
修复后的MCTS检索器 - 解决接口不匹配和数据结构问题
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional, Any
import math
import random
import numpy as np
from copy import deepcopy


# ---------- 适配器类 -------------------------------------------------
@dataclass
class Document:
    """标准化的文档格式"""
    doc_id: str
    page_content: str
    score: float = 0.0
    embedding: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, doc_dict: dict, doc_id: str = None) -> "Document":
        """从字典创建Document对象"""
        return cls(
            doc_id=doc_id or str(hash(doc_dict.get("text", ""))),
            page_content=doc_dict.get("text", ""),
            score=doc_dict.get("score", 0.0),
            embedding=doc_dict.get("embedding", None),
            metadata=doc_dict.get("metadata", {})
        )


class MultimodalMatcherAdapter:
    """MultimodalMatcher的适配器，提供MCTS需要的接口"""

    def __init__(self, base_matcher, documents: List[dict]):
        self.base_matcher = base_matcher
        self.documents = documents  # 缓存所有文档

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """
        MCTS期望的接口：只传入query和top_k
        """
        try:
            # 调用原始的MultimodalMatcher.retrieve方法
            results = self.base_matcher.retrieve(query, self.documents)

            # 转换为Document格式并限制数量
            documents = []
            for i, result in enumerate(results[:top_k]):
                doc = Document.from_dict(result, doc_id=f"doc_{i}")
                documents.append(doc)

            return documents
        except Exception as e:
            print(f"检索出错: {str(e)}")
            return []


# ---------- 数据结构 -------------------------------------------------
@dataclass
class State:
    """已选文档集合 & 已覆盖意图集合"""
    docs: Set  # {doc_id, ...}
    covered: Set  # {intent_idx, ...}

    def clone(self) -> "State":
        return State(set(self.docs), set(self.covered))


@dataclass
class MCTSNode:
    state: State
    parent: Optional["MCTSNode"]
    action: Optional[Tuple[int, int]]  # (intent_idx, cand_idx)
    children: List["MCTSNode"] = field(default_factory=list)
    visit_count: int = 0
    total_value: float = 0.0
    untried_actions: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def q(self) -> float:
        return self.total_value / self.visit_count if self.visit_count else 0.0


# ---------- 修复后的MCTS包装器 ----------------------------------------------
class MCTSWrapper:
    """
    修复后的MCTS包装器，解决接口不匹配问题
    """

    def __init__(
            self,
            base_retriever,
            rollout_budget: int = 100,  # 降低默认值
            k_per_intent: int = 3,  # 降低默认值
            max_depth: int = 5,  # 降低默认值
            c_puct: float = 1.0,
            reward_weights: Dict[str, float] = None,
            diversity_metric: str = "cos",
            random_seed: int = None,
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
        self.intent_pool: List[List] = []
        self.doc_embeddings: Dict = {}
        self.adapter = None  # 适配器将在retrieve时创建

    def retrieve(self, query: str, documents: List[dict]) -> List[dict]:
        """
        🔥 新的主要接口：接收query和documents，返回检索结果
        这个方法替代了原来的search方法，提供更符合现有代码的接口
        """
        print(f"🎯 MCTS检索开始，文档数量: {len(documents)}")

        # 创建适配器
        self.adapter = MultimodalMatcherAdapter(self.base, documents)

        # 1. 首先进行意图拆解（简化版本，你可以集成DeepSearch_Beta的拆解逻辑）
        intents = self._split_query_intents(query)
        print(f"🔍 拆解意图: {intents}")

        # 2. 使用MCTS进行检索
        final_docs = self._mcts_search(query, intents)

        # 3. 转换回原始格式
        results = []
        for doc in final_docs:
            results.append({
                "text": doc.page_content,
                "score": doc.score,
                "metadata": doc.metadata
            })

        return results

    def _split_query_intents(self, query: str) -> List[str]:
        """
        简化的意图拆解（你可以替换为DeepSearch_Beta的拆解逻辑）
        """
        # 这里是一个简化的实现，你可以替换为你的多意图拆解逻辑
        intents = [
            query,  # 原始查询
            f"详细解释 {query}",  # 理解意图
            f"分析 {query} 的关键信息",  # 推理意图
            f"找到关于 {query} 的具体内容"  # 定位意图
        ]
        return intents

    def _mcts_search(self, original_query: str, intents: List[str]) -> List[Document]:
        """
        MCTS搜索主逻辑
        """
        try:
            self._prefetch(intents)

            root = MCTSNode(state=State(set(), set()), parent=None, action=None)
            root.untried_actions = [(i, j) for i in range(len(intents))
                                    for j in range(len(self.intent_pool[i]))]

            # 🔥 添加检查避免无限循环
            if not root.untried_actions:
                print("⚠️ 没有可用的动作，返回空结果")
                return []

            for iteration in range(min(self.rollout_budget, 50)):  # 限制最大迭代次数
                try:
                    path = self._select(root)
                    leaf = self._expand(path[-1])
                    reward = self._simulate(leaf)
                    self._backprop(path, reward)
                except Exception as e:
                    print(f"⚠️ MCTS迭代 {iteration} 出错: {str(e)}")
                    continue

            # 找访问次数最多的子节点作为best
            if not root.children:
                print("⚠️ 没有子节点，返回前k个文档")
                # 回退策略：返回第一个意图的前几个文档
                if self.intent_pool:
                    return self.intent_pool[0][:self.max_depth]
                return []

            best_child = max(root.children, key=lambda n: n.visit_count)
            final_doc_ids = best_child.state.docs

            # 转换doc_id回Document对象
            final_docs = []
            for doc_id in final_doc_ids:
                doc = self._doc_by_id(doc_id)
                if doc:
                    final_docs.append(doc)

            return final_docs

        except Exception as e:
            print(f"❌ MCTS搜索出错: {str(e)}")
            # 回退策略：使用基础检索器
            if self.adapter:
                return self.adapter.retrieve(original_query, self.max_depth)
            return []

    def _prefetch(self, intents: List[str]):
        """预取每个意图的文档"""
        self.intent_pool.clear()
        self.doc_embeddings.clear()

        for intent in intents:
            try:
                # 🔥 使用适配器调用
                docs = self.adapter.retrieve(intent, top_k=self.k)
                self.intent_pool.append(docs)

                # 缓存embeddings
                for doc in docs:
                    if doc.embedding is not None:
                        self.doc_embeddings[doc.doc_id] = np.asarray(doc.embedding)

            except Exception as e:
                print(f"⚠️ 预取意图 '{intent}' 失败: {str(e)}")
                self.intent_pool.append([])  # 添加空列表避免索引错误

    def _select(self, node: MCTSNode) -> List[MCTSNode]:
        """沿着UCT最大的child下行，直到叶节点"""
        path = [node]
        max_iterations = 20  # 防止无限循环
        iteration = 0

        while node.children and not node.untried_actions and iteration < max_iterations:
            if not node.children:
                break

            log_N = math.log(max(node.visit_count, 1))  # 避免log(0)

            def uct(n: MCTSNode) -> float:
                if n.visit_count == 0:
                    return float('inf')  # 优先访问未访问的节点
                return n.q + self.c_puct * math.sqrt(log_N / n.visit_count)

            node = max(node.children, key=uct)
            path.append(node)
            iteration += 1

        return path

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """扩展节点"""
        if not node.untried_actions or len(node.state.docs) >= self.max_depth:
            return node

        action = node.untried_actions.pop(random.randrange(len(node.untried_actions)))
        intent_idx, cand_idx = action

        # 检查索引有效性
        if (intent_idx >= len(self.intent_pool) or
                cand_idx >= len(self.intent_pool[intent_idx])):
            return node

        doc = self.intent_pool[intent_idx][cand_idx]

        new_state = node.state.clone()
        new_state.docs.add(doc.doc_id)
        new_state.covered.add(intent_idx)

        child = MCTSNode(state=new_state, parent=node, action=action)
        child.untried_actions = [a for a in node.untried_actions
                                 if a[0] != intent_idx or self.intent_pool[a[0]][a[1]].doc_id not in new_state.docs]
        node.children.append(child)
        return child

    def _simulate(self, node: MCTSNode) -> float:
        """模拟rollout"""
        state = node.state.clone()
        remaining_actions = node.untried_actions.copy()
        random.shuffle(remaining_actions)

        simulation_steps = 0
        max_simulation_steps = min(10, self.max_depth)  # 限制模拟步数

        while (remaining_actions and
               len(state.docs) < self.max_depth and
               len(state.covered) < len(self.intent_pool) and
               simulation_steps < max_simulation_steps):

            action = remaining_actions.pop()
            intent_idx, cand_idx = action

            # 检查索引有效性
            if (intent_idx >= len(self.intent_pool) or
                    cand_idx >= len(self.intent_pool[intent_idx])):
                continue

            doc = self.intent_pool[intent_idx][cand_idx]
            if doc.doc_id in state.docs:
                continue

            state.docs.add(doc.doc_id)
            state.covered.add(intent_idx)
            simulation_steps += 1

        return self._reward(state)

    def _backprop(self, path: List[MCTSNode], reward: float):
        """反向传播奖励"""
        for node in path:
            node.visit_count += 1
            node.total_value += reward

    def _reward(self, state: State) -> float:
        """计算奖励"""
        try:
            # 1. 意图覆盖奖励
            coverage_r = len(state.covered) / max(len(self.intent_pool), 1)

            # 2. 文档质量奖励
            if state.docs:
                scores = []
                for doc_id in state.docs:
                    doc = self._doc_by_id(doc_id)
                    if doc:
                        scores.append(doc.score)
                quality_r = np.mean(scores) if scores else 0.0
            else:
                quality_r = 0.0

            # 3. 多样性奖励
            diversity_r = 0.0
            if len(state.docs) > 1 and self.rw.get("diversity", 0) > 0:
                valid_embeddings = []
                for doc_id in state.docs:
                    if doc_id in self.doc_embeddings:
                        valid_embeddings.append(self.doc_embeddings[doc_id])

                if len(valid_embeddings) >= 2:
                    embs = np.stack(valid_embeddings)
                    norm = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
                    embs_norm = embs / norm
                    sim = embs_norm @ embs_norm.T
                    upper = sim[np.triu_indices(len(embs), k=1)]
                    diversity_r = 1.0 - np.mean(upper) if len(upper) > 0 else 0.0

            total_reward = (self.rw["coverage"] * coverage_r +
                            self.rw["quality"] * quality_r +
                            self.rw["diversity"] * diversity_r)

            return max(0.0, min(1.0, total_reward))  # 限制在[0,1]范围内

        except Exception as e:
            print(f"⚠️ 奖励计算出错: {str(e)}")
            return 0.0

    def _doc_by_id(self, doc_id):
        """根据doc_id查找文档"""
        for bucket in self.intent_pool:
            for doc in bucket:
                if doc.doc_id == doc_id:
                    return doc
        return None