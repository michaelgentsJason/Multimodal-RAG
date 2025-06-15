#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Beam Search即插即用模块
可以轻松集成到现有的多意图检索系统中

使用方法:
from beam_search_module import BeamSearchWrapper

# 包装你的现有检索器
wrapped_retriever = BeamSearchWrapper(
    base_retriever=your_retriever,
    matcher=your_matcher,
    reranker=your_reranker,
    enable_beam_search=True,  # 开关控制
    beam_width=3
)

# 像使用原检索器一样使用
results = wrapped_retriever.search_retrieval(data, retriever=matcher)
"""

import os
import sys
import json
import time
import heapq
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class BeamSearchConfig:
    """Beam Search配置类"""
    # 基础配置
    enable: bool = True  # 是否启用Beam Search
    beam_width: int = 3  # beam宽度

    # 意图拆解配置
    max_decomposition_methods: int = 4  # 最大拆解方法数
    enable_detailed_decomposition: bool = True  # 是否启用细粒度拆解
    enable_coarse_decomposition: bool = True  # 是否启用粗粒度拆解
    enable_hierarchical_decomposition: bool = True  # 是否启用层次化拆解

    # 评分权重
    diversity_weight: float = 0.25  # 多样性权重
    coverage_weight: float = 0.35  # 覆盖度权重
    relevance_weight: float = 0.30  # 相关性权重
    intent_quality_weight: float = 0.10  # 意图质量权重

    # 阈值配置
    min_path_score: float = 0.05  # 最小路径分数阈值
    similarity_threshold: float = 0.7  # 路径相似度阈值

    # 结果配置
    max_final_results: int = 15  # 最大最终结果数
    enable_result_reranking: bool = True  # 是否启用结果重排序

    # 调试配置
    debug_mode: bool = False  # 调试模式
    log_paths: bool = False  # 是否记录路径信息


@dataclass
class SearchPath:
    """搜索路径类"""
    path_id: str
    intent_decomposition: List[str]
    retrieval_results: List[Dict[str, Any]]
    scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_score(self) -> float:
        return self.scores.get('total', 0.0)

    def __lt__(self, other):
        return self.total_score > other.total_score  # 用于优先队列，分数高的优先


class IntentDecomposer:
    """意图拆解器"""

    def __init__(self, base_retriever, config: BeamSearchConfig):
        self.base_retriever = base_retriever
        self.config = config

    def decompose_with_beam_search(self, query: str) -> List[Tuple[List[str], float, str]]:
        """使用Beam Search进行意图拆解"""
        decomposition_candidates = []

        # 方法1: 标准拆解（始终启用）
        try:
            standard_intents = self.base_retriever._split_query_intent(query)
            score1 = self._score_decomposition(query, standard_intents)
            decomposition_candidates.append((standard_intents, score1, "standard"))

            if self.config.debug_mode:
                logger.info(f"标准拆解: {len(standard_intents)} 个意图, 分数: {score1:.3f}")
        except Exception as e:
            logger.warning(f"标准拆解失败: {e}")
            # 备用方案
            decomposition_candidates.append(([query], 0.5, "fallback"))

        # 方法2: 细粒度拆解
        if self.config.enable_detailed_decomposition:
            try:
                detailed_intents = self._generate_detailed_decomposition(query)
                score2 = self._score_decomposition(query, detailed_intents)
                decomposition_candidates.append((detailed_intents, score2, "detailed"))

                if self.config.debug_mode:
                    logger.info(f"细粒度拆解: {len(detailed_intents)} 个意图, 分数: {score2:.3f}")
            except Exception as e:
                logger.warning(f"细粒度拆解失败: {e}")

        # 方法3: 粗粒度拆解
        if self.config.enable_coarse_decomposition:
            try:
                if len(decomposition_candidates) > 0:
                    base_intents = decomposition_candidates[0][0]  # 使用标准拆解作为基础
                    coarse_intents = self._generate_coarse_decomposition(base_intents)
                    score3 = self._score_decomposition(query, coarse_intents)
                    decomposition_candidates.append((coarse_intents, score3, "coarse"))

                    if self.config.debug_mode:
                        logger.info(f"粗粒度拆解: {len(coarse_intents)} 个意图, 分数: {score3:.3f}")
            except Exception as e:
                logger.warning(f"粗粒度拆解失败: {e}")

        # 方法4: 层次化拆解
        if self.config.enable_hierarchical_decomposition:
            try:
                hierarchical_intents = self._generate_hierarchical_decomposition(query)
                score4 = self._score_decomposition(query, hierarchical_intents)
                decomposition_candidates.append((hierarchical_intents, score4, "hierarchical"))

                if self.config.debug_mode:
                    logger.info(f"层次化拆解: {len(hierarchical_intents)} 个意图, 分数: {score4:.3f}")
            except Exception as e:
                logger.warning(f"层次化拆解失败: {e}")

        # 排序并返回top-k
        decomposition_candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = decomposition_candidates[:self.config.max_decomposition_methods]

        if self.config.debug_mode:
            logger.info(f"意图拆解完成，共生成 {len(top_candidates)} 个方案")

        return top_candidates

    def _generate_detailed_decomposition(self, query: str) -> List[str]:
        """生成细粒度拆解"""
        # 获取基础意图
        try:
            base_intents = self.base_retriever._split_query_intent(query)
        except:
            base_intents = [query]

        detailed_intents = []

        for intent in base_intents:
            detailed_intents.append(intent)

            # 为复杂意图添加细节子查询
            words = intent.split()
            if len(words) > 4:
                # 添加具体化的子查询
                if 'features' in intent.lower():
                    detailed_intents.extend([
                        intent.replace('features', 'core features'),
                        intent.replace('features', 'technical specifications'),
                        intent.replace('features', 'user benefits')
                    ])
                elif 'how' in intent.lower():
                    detailed_intents.extend([
                        intent + ' - step by step process',
                        intent + ' - requirements and prerequisites'
                    ])
                else:
                    detailed_intents.append(intent + ' - detailed explanation')

        return list(set(detailed_intents))  # 去重

    def _generate_coarse_decomposition(self, base_intents: List[str]) -> List[str]:
        """生成粗粒度拆解"""
        if len(base_intents) <= 2:
            return base_intents

        # 简单的合并策略：将相似的意图合并
        coarse_intents = []
        used_indices = set()

        for i, intent in enumerate(base_intents):
            if i in used_indices:
                continue

            # 查找相似意图
            similar_indices = []
            intent_words = set(intent.lower().split())

            for j in range(i + 1, len(base_intents)):
                if j in used_indices:
                    continue

                other_words = set(base_intents[j].lower().split())
                overlap = len(intent_words.intersection(other_words))
                union = len(intent_words.union(other_words))

                if union > 0 and overlap / union > 0.5:  # 相似度阈值
                    similar_indices.append(j)

            if similar_indices:
                # 合并相似意图
                all_words = intent_words.copy()
                for idx in similar_indices:
                    all_words.update(base_intents[idx].lower().split())
                combined_intent = f"Combined aspects: {' '.join(sorted(list(all_words))[:8])}"
                coarse_intents.append(combined_intent)
                used_indices.update([i] + similar_indices)
            else:
                coarse_intents.append(intent)
                used_indices.add(i)

        return coarse_intents

    def _generate_hierarchical_decomposition(self, query: str) -> List[str]:
        """生成层次化拆解"""
        # 主要意图
        main_intent = f"Primary focus: {query}"

        # 支撑意图
        try:
            base_intents = self.base_retriever._split_query_intent(query)
            supporting_intents = [f"Supporting aspect: {intent}" for intent in base_intents[:2]]
        except:
            supporting_intents = [f"Supporting details for: {query}"]

        return [main_intent] + supporting_intents

    def _score_decomposition(self, original_query: str, intents: List[str]) -> float:
        """评估意图拆解质量"""
        if not intents:
            return 0.0

        try:
            # 覆盖度分数
            coverage = self._calculate_coverage(original_query, intents)

            # 多样性分数
            diversity = self._calculate_diversity(intents)

            # 复杂度惩罚
            complexity_penalty = max(0, 1 - (len(intents) - 3) * 0.1)

            # 综合分数
            total_score = coverage * 0.4 + diversity * 0.4 + complexity_penalty * 0.2

            return max(0.0, min(1.0, total_score))
        except Exception as e:
            logger.warning(f"意图拆解评分失败: {e}")
            return 0.5  # 默认分数

    def _calculate_coverage(self, query: str, intents: List[str]) -> float:
        """计算覆盖度"""
        query_words = set(query.lower().split())
        if not query_words:
            return 0.0

        intent_words = set()
        for intent in intents:
            intent_words.update(intent.lower().split())

        overlap = query_words.intersection(intent_words)
        return len(overlap) / len(query_words)

    def _calculate_diversity(self, intents: List[str]) -> float:
        """计算多样性"""
        if len(intents) <= 1:
            return 1.0

        similarities = []
        for i in range(len(intents)):
            for j in range(i + 1, len(intents)):
                words_i = set(intents[i].lower().split())
                words_j = set(intents[j].lower().split())

                if not words_i or not words_j:
                    continue

                overlap = len(words_i.intersection(words_j))
                union = len(words_i.union(words_j))
                similarity = overlap / union if union > 0 else 0
                similarities.append(similarity)

        if not similarities:
            return 1.0

        avg_similarity = np.mean(similarities)
        return 1 - avg_similarity


class PathEvaluator:
    """路径评估器"""

    def __init__(self, config: BeamSearchConfig):
        self.config = config

    def evaluate_path(self, path: SearchPath, original_query: str) -> SearchPath:
        """评估搜索路径"""
        try:
            scores = {}

            # 相关性分数
            scores['relevance'] = self._calculate_relevance_score(path.retrieval_results)

            # 多样性分数
            scores['diversity'] = self._calculate_diversity_score(path.retrieval_results)

            # 覆盖度分数
            scores['coverage'] = self._calculate_coverage_score(
                path.intent_decomposition, path.retrieval_results
            )

            # 意图质量分数（从metadata中获取，如果有的话）
            scores['intent_quality'] = path.metadata.get('intent_score', 0.5)

            # 计算总分
            scores['total'] = (
                    self.config.relevance_weight * scores['relevance'] +
                    self.config.diversity_weight * scores['diversity'] +
                    self.config.coverage_weight * scores['coverage'] +
                    self.config.intent_quality_weight * scores['intent_quality']
            )

            path.scores = scores

        except Exception as e:
            logger.warning(f"路径评估失败: {e}")
            path.scores = {'total': 0.0, 'relevance': 0.0, 'diversity': 0.0, 'coverage': 0.0}

        return path

    def _calculate_relevance_score(self, results: List[Dict]) -> float:
        """计算相关性分数"""
        if not results:
            return 0.0

        scores = [r.get('score', 0) for r in results]
        return np.mean(scores) if scores else 0.0

    def _calculate_diversity_score(self, results: List[Dict]) -> float:
        """计算多样性分数"""
        if len(results) <= 1:
            return 1.0

        # 简化的多样性计算
        texts = [r.get('text', '')[:100] for r in results]
        similarities = []

        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                words_i = set(texts[i].lower().split())
                words_j = set(texts[j].lower().split())

                if not words_i or not words_j:
                    continue

                overlap = len(words_i.intersection(words_j))
                union = len(words_i.union(words_j))
                similarity = overlap / union if union > 0 else 0
                similarities.append(similarity)

        if not similarities:
            return 1.0

        avg_similarity = np.mean(similarities)
        return 1 - avg_similarity

    def _calculate_coverage_score(self, intents: List[str], results: List[Dict]) -> float:
        """计算覆盖度分数"""
        if not intents or not results:
            return 0.0

        intent_coverage = []

        for intent in intents:
            intent_words = set(intent.lower().split())
            max_coverage = 0

            for result in results:
                result_text = result.get('text', '').lower()
                result_words = set(result_text.split())

                if intent_words and result_words:
                    overlap = len(intent_words.intersection(result_words))
                    coverage = overlap / len(intent_words)
                    max_coverage = max(max_coverage, coverage)

            intent_coverage.append(max_coverage)

        return np.mean(intent_coverage) if intent_coverage else 0.0


class BeamSearchWrapper:
    """Beam Search包装器 - 即插即用模块"""

    def _split_query_intent(self, query: str):
        """委托给基础检索器的意图拆分方法"""
        return self.base_retriever._split_query_intent(query)

    def __getattr__(self, name):
        """当访问不存在的属性时，尝试从基础检索器获取"""
        if hasattr(self.base_retriever, name):
            return getattr(self.base_retriever, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __init__(self,
                 base_retriever,
                 matcher,
                 reranker,
                 enable_beam_search: bool = True,
                 beam_config: Optional[BeamSearchConfig] = None,
                 **kwargs):
        """
        初始化Beam Search包装器

        Args:
            base_retriever: 基础检索器 (DeepSearch_Beta)
            matcher: 多模态匹配器 (MultimodalMatcher)
            reranker: 重排序器 (FlagReranker)
            enable_beam_search: 是否启用Beam Search
            beam_config: Beam Search配置
            **kwargs: 其他配置参数，会被应用到beam_config
        """
        self.base_retriever = base_retriever
        self.matcher = matcher
        self.reranker = reranker
        self.enable_beam_search = enable_beam_search

        # 初始化配置
        if beam_config is None:
            beam_config = BeamSearchConfig()

        # 应用额外的配置参数
        for key, value in kwargs.items():
            if hasattr(beam_config, key):
                setattr(beam_config, key, value)

        self.config = beam_config
        self.config.enable = enable_beam_search  # 确保配置与参数一致

        # 初始化组件
        if self.enable_beam_search:
            self.decomposer = IntentDecomposer(base_retriever, self.config)
            self.evaluator = PathEvaluator(self.config)
            self.path_counter = 0

        if self.config.debug_mode:
            logger.info(f"BeamSearchWrapper初始化完成，Beam Search: {'启用' if enable_beam_search else '禁用'}")

    def search_retrieval(self, data: Dict[str, Any], retriever=None, **kwargs) -> List[Dict[str, Any]]:
        """
        主要的检索接口 - 与原有接口完全兼容

        Args:
            data: 包含query和documents的数据字典
            retriever: 检索器（兼容性参数）
            **kwargs: 其他参数

        Returns:
            检索结果列表
        """
        if not self.enable_beam_search or not self.config.enable:
            # 如果未启用Beam Search，直接使用原有方法
            if self.config.debug_mode:
                logger.info("🔄 使用传统检索方法")
            return self.base_retriever.search_retrieval(data, retriever=retriever or self.matcher)

        # 使用Beam Search
        if self.config.debug_mode:
            logger.info("🚀 使用Beam Search检索方法")

        return self._beam_search_retrieval(data)

    def _beam_search_retrieval(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """执行Beam Search检索"""
        query = data['query']
        documents = data['documents']

        try:
            # 第一阶段：意图拆解
            intent_candidates = self.decomposer.decompose_with_beam_search(query)

            if self.config.log_paths:
                logger.info(f"生成了 {len(intent_candidates)} 个意图拆解方案")

            # 第二阶段：路径构建和评估
            all_paths = []

            for intents, intent_score, method in intent_candidates:
                # 执行检索
                path_results = self._execute_retrieval_for_intents(intents, documents)

                # 创建路径
                path = SearchPath(
                    path_id=f"path_{self.path_counter}_{method}",
                    intent_decomposition=intents,
                    retrieval_results=path_results,
                    metadata={'intent_score': intent_score, 'method': method}
                )
                self.path_counter += 1

                # 评估路径
                path = self.evaluator.evaluate_path(path, query)

                if path.total_score >= self.config.min_path_score:
                    all_paths.append(path)

                if self.config.log_paths:
                    logger.info(f"路径 {path.path_id}: 分数={path.total_score:.3f}, 方法={method}")

            # 第三阶段：路径选择
            selected_paths = self._select_diverse_paths(all_paths)

            # 第四阶段：结果合成
            final_results = self._synthesize_final_results(selected_paths, query)

            if self.config.debug_mode:
                logger.info(f"Beam Search检索完成，返回 {len(final_results)} 个结果")

            return final_results

        except Exception as e:
            logger.error(f"Beam Search检索失败: {e}")
            # 回退到传统方法
            logger.info("回退到传统检索方法")
            return self.base_retriever.search_retrieval(data, retriever=self.matcher)

    def _execute_retrieval_for_intents(self, intents: List[str], documents: List[Dict]) -> List[Dict]:
        """为意图列表执行检索"""
        all_results = []

        for intent in intents:
            try:
                # 执行单个意图的检索
                intent_results = self.matcher.retrieve(intent, documents)

                # 标记来源意图
                for result in intent_results:
                    result['source_intent'] = intent

                all_results.extend(intent_results)

            except Exception as e:
                logger.warning(f"意图检索失败 '{intent}': {e}")
                continue

        # 去重
        unique_results = self._deduplicate_results(all_results)

        # 重排序
        if self.config.enable_result_reranking and unique_results:
            unique_results = self._rerank_results(intents[0] if intents else "", unique_results)

        return unique_results

    def _select_diverse_paths(self, paths: List[SearchPath]) -> List[SearchPath]:
        """选择多样化的路径"""
        if not paths:
            return []

        # 按分数排序
        paths.sort(reverse=True)

        selected = []

        for path in paths:
            if len(selected) >= self.config.beam_width:
                break

            # 检查多样性
            if self._is_diverse_enough(path, selected):
                selected.append(path)

        # 如果选择的路径不够，补充高分路径
        while len(selected) < self.config.beam_width and len(selected) < len(paths):
            for path in paths:
                if path not in selected:
                    selected.append(path)
                    break

        return selected

    def _is_diverse_enough(self, new_path: SearchPath, selected_paths: List[SearchPath]) -> bool:
        """检查路径是否足够多样化"""
        if not selected_paths:
            return True

        for selected_path in selected_paths:
            similarity = self._calculate_path_similarity(new_path, selected_path)
            if similarity > self.config.similarity_threshold:
                return False

        return True

    def _calculate_path_similarity(self, path1: SearchPath, path2: SearchPath) -> float:
        """计算路径相似度"""
        # 意图相似度
        intents1_text = ' '.join(path1.intent_decomposition).lower()
        intents2_text = ' '.join(path2.intent_decomposition).lower()

        words1 = set(intents1_text.split())
        words2 = set(intents2_text.split())

        if not words1 or not words2:
            return 0.0

        overlap = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return overlap / union if union > 0 else 0.0

    def _synthesize_final_results(self, paths: List[SearchPath], original_query: str) -> List[Dict]:
        """合成最终结果"""
        all_results = []

        # 收集所有路径的结果
        for path in paths:
            for result in path.retrieval_results:
                result['path_id'] = path.path_id
                result['path_score'] = path.total_score
                all_results.append(result)

        # 去重
        unique_results = self._deduplicate_results(all_results)

        # 重新计算分数：结合路径分数和检索分数
        for result in unique_results:
            original_score = result.get('score', 0)
            path_score = result.get('path_score', 0)
            # 加权组合
            result['final_score'] = 0.7 * original_score + 0.3 * path_score

        # 按最终分数排序
        unique_results.sort(key=lambda x: x.get('final_score', 0), reverse=True)

        # 限制最终结果数量
        final_count = min(len(unique_results), self.config.max_final_results)
        return unique_results[:final_count]

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """去重结果"""
        seen_texts = set()
        unique_results = []

        for result in results:
            text = result.get('text', '')
            text_key = text[:100] if text else ''  # 使用前100字符作为标识

            if text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_results.append(result)

        return unique_results

    def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """重排序结果"""
        if not results or not query or not self.reranker:
            return results

        try:
            pairs = [[query, result.get('text', '')] for result in results]
            rerank_scores = self.reranker.compute_score(pairs, normalize=True)

            for i, result in enumerate(results):
                if i < len(rerank_scores):
                    result['rerank_score'] = rerank_scores[i]
                    # 更新分数
                    original_score = result.get('score', 0)
                    result['score'] = 0.6 * rerank_scores[i] + 0.4 * original_score

            results.sort(key=lambda x: x.get('score', 0), reverse=True)

        except Exception as e:
            logger.warning(f"重排序失败: {e}")

        return results

    # 提供便捷的配置方法
    def enable_beam_search_mode(self, enable: bool = True):
        """启用/禁用Beam Search模式"""
        self.enable_beam_search = enable
        self.config.enable = enable

        if self.config.debug_mode:
            logger.info(f"Beam Search模式已{'启用' if enable else '禁用'}")

    def set_beam_width(self, width: int):
        """设置beam宽度"""
        self.config.beam_width = width

        if self.config.debug_mode:
            logger.info(f"Beam宽度设置为: {width}")

    def set_debug_mode(self, debug: bool = True):
        """设置调试模式"""
        self.config.debug_mode = debug

        if debug:
            logger.info("调试模式已启用")

    def get_config(self) -> BeamSearchConfig:
        """获取当前配置"""
        return self.config

    def update_config(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                if self.config.debug_mode:
                    logger.info(f"配置更新: {key} = {value}")


# 便捷的工厂函数
def create_beam_search_retriever(base_retriever, matcher, reranker,
                                 enable_beam_search: bool = True,
                                 beam_width: int = 3,
                                 debug_mode: bool = False,
                                 **config_kwargs) -> BeamSearchWrapper:
    """
    便捷的工厂函数，用于创建Beam Search检索器

    Args:
        base_retriever: 基础检索器
        matcher: 匹配器
        reranker: 重排序器
        enable_beam_search: 是否启用Beam Search
        beam_width: beam宽度
        debug_mode: 是否启用调试模式
        **config_kwargs: 其他配置参数

    Returns:
        BeamSearchWrapper实例
    """
    config = BeamSearchConfig(
        enable=enable_beam_search,
        beam_width=beam_width,
        debug_mode=debug_mode,
        **config_kwargs
    )

    return BeamSearchWrapper(
        base_retriever=base_retriever,
        matcher=matcher,
        reranker=reranker,
        enable_beam_search=enable_beam_search,
        beam_config=config
    )


# 使用示例和测试代码
if __name__ == "__main__":
    # 这里可以放一些简单的测试代码
    print("🚀 Beam Search模块加载成功！")
    print("使用方法:")
    print("from beam_search_module import BeamSearchWrapper, create_beam_search_retriever")
    print()
    print("# 方式1: 直接包装")
    print("wrapped_retriever = BeamSearchWrapper(base_retriever, matcher, reranker, enable_beam_search=True)")
    print()
    print("# 方式2: 使用工厂函数")
    print("wrapped_retriever = create_beam_search_retriever(base_retriever, matcher, reranker, beam_width=3)")
    print()
    print("# 使用（与原接口完全兼容）")
    print("results = wrapped_retriever.search_retrieval(data, retriever=matcher)")