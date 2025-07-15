import json
from copy import deepcopy
from typing import List, Dict, Annotated, Optional, Tuple, Set
from collections import defaultdict

from .agent_gpt4 import AzureGPT4Chat, create_response_format
from datetime import datetime
import sys
from ..retriever_multimodal_bge import DocumentRetriever, RetrieverConfig, MultimodalMatcher
import asyncio
import concurrent.futures
from textwrap import dedent
from langchain_core.documents import Document
from FlagEmbedding import FlagReranker, FlagModel
from ..deepsearch import DeepSearch_Alpha
import numpy as np
import pytesseract
from pdf2image import convert_from_path
import os
import logging
import spacy
import re, string, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# # Ensure the directory for the log file exists
# log_file_path = "/Users/chloe/Documents/Academic/AI/Project/基于Colpali的多模态检索标准框架/multimodal-RAG/DeepRAG_Multimodal/deep_retrieve/ming/deepsearch.log"
# os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
#
# # Configure logger
# logger.basicConfig(
#     filename=log_file_path,
#     level=logger.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )
#
# logger.info("logger setup complete. This is a test log message.")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("DeepSearch_Beta模块初始化")
print(f"当前模块的日志器名称: {logger.name}")


class IntentScoringAgent:
    """意图评分代理，简化版本避免HTTP 500错误"""

    def __init__(self):
        self.llm = AzureGPT4Chat()
        self.max_retries = 2

    def score_and_rank_intents(self, original_query: str, intent_queries: List[str], context_docs: List[Dict] = None) -> \
            List[Dict]:
        """对意图进行评分和排序 - 简化版本"""

        logger.info(f"🎯 开始评分 {len(intent_queries)} 个意图")

        # 🔥 首先尝试简化的LLM评分
        try:
            return self._simplified_llm_scoring(original_query, intent_queries, context_docs)
        except Exception as e:
            logger.warning(f"⚠️ LLM评分失败，使用规则评分: {e}")
            return self._rule_based_scoring(original_query, intent_queries)

    def _simplified_llm_scoring(self, original_query: str, intent_queries: List[str],
                                context_docs: List[Dict] = None) -> List[Dict]:
        """简化的LLM评分"""

        # 🔥 极简prompt，避免复杂性
        SIMPLE_SCORING_PROMPT = dedent("""
        Rate each query's relevance to the original query (0-1 scale).
        Return JSON: {"scores": [{"query": "text", "score": 0.8}, ...]}
        """)

        # 🔥 限制输入长度
        intents_str = json.dumps(intent_queries[:3], ensure_ascii=False)  # 只处理前3个
        if len(intents_str) > 500:
            intents_str = intents_str[:500] + "...]"

        messages = [
            {"role": "system", "content": SIMPLE_SCORING_PROMPT},
            {"role": "user", "content": f"""
            Original: {original_query[:100]}
            Queries: {intents_str}
            Rate relevance (0-1).
            """}
        ]

        try:
            response = self.llm.chat_with_message_format(message_list=messages)
            result = parse_llm_response(response)
            scores = result.get("scores", [])

            # 转换为标准格式
            evaluations = []
            for i, intent in enumerate(intent_queries):
                # 找到对应的分数
                score = 0.6  # 默认分数
                for score_item in scores:
                    if intent[:20] in score_item.get("query", "")[:20]:
                        score = float(score_item.get("score", 0.6))
                        break

                evaluations.append({
                    "intent": intent,
                    "relevance_score": score,
                    "specificity_score": score,
                    "uniqueness_score": max(0.5, 0.9 - i * 0.1),
                    "retrievability_score": score,
                    "overall_score": score,
                    "reasoning": f"LLM score: {score:.2f}",
                    "suggested_action": "prioritize" if score > 0.6 else "skip"
                })

            logger.info(f"✅ LLM评分成功")
            return evaluations

        except Exception as e:
            logger.error(f"❌ LLM评分失败: {e}")
            raise

    def _rule_based_scoring(self, original_query: str, intent_queries: List[str]) -> List[Dict]:
        """基于规则的评分，完全不依赖LLM"""
        logger.info("🔧 使用规则评分")

        evaluations = []
        original_words = set(original_query.lower().split())

        for i, intent in enumerate(intent_queries):
            # 计算词汇重叠度
            intent_words = set(intent.lower().split())
            word_overlap = len(original_words.intersection(intent_words)) / max(len(original_words), 1)

            # 长度相似性
            length_ratio = min(len(intent) / max(len(original_query), 1), 1.0)

            # 综合分数，后面的意图分数递减
            base_score = (word_overlap * 0.7 + length_ratio * 0.3)
            final_score = max(0.4, min(0.9, base_score * (0.95 - i * 0.05)))

            evaluations.append({
                "intent": intent,
                "relevance_score": word_overlap,
                "specificity_score": length_ratio,
                "uniqueness_score": max(0.5, 0.9 - i * 0.1),
                "retrievability_score": final_score,
                "overall_score": final_score,
                "reasoning": f"Rule-based: overlap={word_overlap:.2f}, final={final_score:.2f}",
                "suggested_action": "prioritize" if final_score > 0.6 else "skip"
            })

        logger.info(f"✅ 规则评分完成，平均分数: {np.mean([e['overall_score'] for e in evaluations]):.3f}")
        return evaluations

    def should_refine_intents(self, original_query: str, current_results: List[Dict],
                              intent_scores: List[Dict]) -> Dict:
        """🔥 关键方法：决定是否需要细化意图以及细化哪些意图"""

        logger.info("🤔 开始细化决策分析...")

        # 计算当前结果质量指标
        avg_score = np.mean([item['overall_score'] for item in intent_scores]) if intent_scores else 0.5
        result_count = len(current_results)

        # 分析结果质量
        if current_results:
            result_scores = [r.get('score', 0) for r in current_results]
            avg_result_score = np.mean(result_scores) if result_scores else 0.0
            max_result_score = max(result_scores) if result_scores else 0.0
        else:
            avg_result_score = 0.0
            max_result_score = 0.0

        # 🔥 细化决策逻辑
        should_refine = False
        reasoning = ""
        intents_to_refine = []

        # 条件1：结果数量不足
        if result_count < 3:
            should_refine = True
            reasoning = f"结果数量不足 ({result_count} < 3)"

        # 条件2：意图平均分数较低且结果质量不高
        elif avg_score < 0.65 and avg_result_score < 0.5:
            should_refine = True
            reasoning = f"意图分数低 ({avg_score:.2f}) 且结果质量差 ({avg_result_score:.2f})"

        # 条件3：有高分意图但结果数量中等
        elif avg_score > 0.7 and result_count < 8:
            should_refine = True
            reasoning = f"高质量意图 ({avg_score:.2f}) 但可获取更多结果 ({result_count} < 8)"

        else:
            should_refine = False
            reasoning = f"当前结果充足: {result_count}个结果, 意图分数: {avg_score:.2f}, 结果质量: {avg_result_score:.2f}"

        # 🔥 选择需要细化的意图
        if should_refine and intent_scores:
            # 策略：选择中等分数的意图进行细化（太高分数的已经很好，太低分数的没价值）
            medium_score_intents = [
                item for item in intent_scores
                if 0.5 <= item['overall_score'] <= 0.75
            ]

            if medium_score_intents:
                # 选择分数最高的中等意图
                best_medium_intent = max(medium_score_intents, key=lambda x: x['overall_score'])
                intents_to_refine = [best_medium_intent['intent']]
            else:
                # 如果没有中等分数意图，选择最高分的意图
                best_intent = max(intent_scores, key=lambda x: x['overall_score'])
                intents_to_refine = [best_intent['intent']]

        decision = {
            "should_refine": should_refine,
            "confidence": 0.8,
            "reasoning": reasoning,
            "intents_to_refine": intents_to_refine,
            "refinement_strategy": "expand_scope",
            # 添加调试信息
            "debug_info": {
                "avg_intent_score": avg_score,
                "result_count": result_count,
                "avg_result_score": avg_result_score,
                "max_result_score": max_result_score
            }
        }

        logger.info(f"🤔 细化决策: {should_refine}")
        logger.info(f"   原因: {reasoning}")
        if intents_to_refine:
            logger.info(f"   需要细化的意图: {[intent[:30] + '...' for intent in intents_to_refine]}")

        return decision


class DeepSearch_Beta(DeepSearch_Alpha):
    def __init__(self, max_iterations: int = 2, reranker: FlagReranker = None, params: dict = None):
        super().__init__(max_iterations, reranker, params)

    def result_processor(self, results):
        matched_docs = []
        for doc, score in results:
            matched_docs.append({
                'text': doc.page_content,
                'score': 1 - score,
                'image_score': doc.metadata.get('image_score', 0),
                'text_score': doc.metadata.get('text_score', 0)
            })
        return matched_docs

    def llm_rerank(self, query, retrieval_list, reranker, topk=None):
        pairs = [[query, doc['text']] for doc in retrieval_list]
        rerank_scores = reranker.compute_score(pairs, normalize=True)
        alpha = self.get_visual_weight_llm(query)

        fused_list = []
        for t_score, doc in zip(rerank_scores, retrieval_list):
            img_score = doc.get('image_score', 0.0)  # 若没算图像得分则当 0
            combo = (1 - alpha) * t_score + alpha * img_score

            fused_list.append({
                'text': doc['text'],
                'page': doc['metadata']['page_index'],
                'score': combo,
                'image_score': img_score,
                'text_score': t_score
            })

        # ④ 排序 + 截取 top-k ----------------------------
        fused_list.sort(key=lambda x: x['score'], reverse=True)
        if topk is not None:
            fused_list = fused_list[:topk]

        return fused_list

    def rerank_index_processor(self, results):
        """默认的相似性搜索结果处理器"""
        matched_docs = []
        for doc, score in results:
            # 创建新字典而不是修改Document对象
            matched_doc = {
                'text': doc.page_content,
                'score': 1 - score,
                # 复制元数据（如果有需要）
                **doc.metadata
            }
            matched_docs.append(matched_doc)
        return matched_docs

    def detect_page_span(self, q: str, allow_bare_range: bool = False):
        # ① 连续页码：pages 12-15, pp. 4–5
        PAGES_RANGE_PAT = r"(?:p(?:ages?|p\.?)\s*)(\d{1,4})\s*[\-–—~]\s*(\d{1,4})"

        # ② 多个独立页码：page 3, 4, and 5 / page 6 and 7
        PAGES_LIST_PAT = r"(?:p(?:ages?|p\.?)\s*)([\d,\sand]+)"

        # ③ 裸范围（谨慎开启）
        RANGE_PAT = r"(\d{1,4})\s*[\-–—~]\s*(\d{1,4})"

        pages = set()

        # Step 1: 匹配形如 pages 12–15
        for m in re.finditer(PAGES_RANGE_PAT, q, re.I):
            start, end = int(m.group(1)), int(m.group(2))
            pages.update(range(start, end + 1))

        # Step 2: 匹配形如 page 3, 4 and 5
        for m in re.finditer(PAGES_LIST_PAT, q, re.I):
            candidates = re.findall(r"\d{1,4}", m.group(1))
            pages.update(int(p) for p in candidates)

        # Step 3: 裸范围 15-17（可选）
        if allow_bare_range:
            for m in re.finditer(RANGE_PAT, q):
                start, end = int(m.group(1)), int(m.group(2))
                pages.update(range(start, end + 1))

        if not pages:
            return None

        return min(pages), max(pages)

    def get_visual_weight_llm(self, query: str):
        '''动态调整模态权重'''
        SYSTEM_MESSAGE = dedent("""
                You are a classifier that decides whether answering a question about a PDF mainly requires VISUAL inspection (figures, images, charts, counting objects) or TEXTUAL reading.

                Return **exactly** one word:
                VISUAL   -> if an image, photo, diagram, chart, color, or counting objects is necessary
                TEXTUAL  -> if the answer clearly lies in plain text or tables.

                Examples:
                Q: What color are the birds in the golden sunbird disc design?
                A: VISUAL
                Q: What year is the report for?
                A: TEXTUAL
                Q: According to the chart on page 8, how many students chose Engineering?
                A: VISUAL
                Q: Where was Gestalt psychology conceived?
                A: TEXTUAL
                """)

        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": f"""
                    Please analyze the following query and decide whether it is vision-based or text-based question：

                    Query: {query}
                    """}
        ]

        response = AzureGPT4Chat().chat_with_message_format(
            message_list=messages
        )

        cleaned = response.strip()
        label = cleaned.split()[0].upper() if cleaned else "TEXTUAL"
        logger.info(f"This query is {label}-based")
        if label == "VISUAL":
            return 0.7
        else:
            return 0.3

    def search_retrieval(self, data: dict, multi_intent: False, retriever: MultimodalMatcher):
        original_query = deepcopy(data['query'])
        data_ori = deepcopy(data)
        embedding_topk = self.params['embedding_topk']
        rerank_topk = self.params['rerank_topk']

        all_search_results = {}
        final_search_results = []
        seen_texts = set()

        if multi_intent:
            # 初步探索检索
            initial_retrieval_list = retriever.retrieve(original_query, data['documents'])
            initial_retrieval_list = initial_retrieval_list[:embedding_topk]
            for r in initial_retrieval_list:
                if r['text'] not in seen_texts:
                    seen_texts.add(r['text'])
                    all_search_results[original_query] = [r['text']]
                    final_search_results.append(r)

        # 第一步：使用LLM拆分查询意图
        if multi_intent:
            intent_queries = self._split_query_intent(original_query,
                                                      json.dumps(all_search_results, ensure_ascii=False, indent=2))
            # intent_queries = self._split_query_intent(original_query)
            logger.info(f"🔍 意图拆分结果: {intent_queries}")
        else:
            intent_queries = [original_query]

        # 第二步：对每个意图进行第一轮检索
        for intent_idx, intent_query in enumerate(intent_queries):
            logger.info(f"🔍 检索意图 {intent_idx + 1}/{len(intent_queries)}: {intent_query}")

            retrieval_list = retriever.retrieve(intent_query, data['documents'])
            retrieval_list = retrieval_list[:embedding_topk // len(intent_queries)]
            for r in retrieval_list:
                if r['text'] not in seen_texts:
                    seen_texts.add(r['text'])
                    all_search_results[intent_query] = [r['text']]
                    final_search_results.append(r)

        # 第三步：基于第一轮检索结果进行意图细化
        if multi_intent:
            # refined_intent_queries = self._refine_query_intent(original_query, intent_queries,
            #                                                    json.dumps(all_search_results, ensure_ascii=False, indent=2))
            # logger.info(f"意图细化结果: {refined_intent_queries}")

            refined_intent_queries = self._refine_query_intent_with_knowledge_graph(
                original_query,
                intent_queries,
                json.dumps(all_search_results, ensure_ascii=False, indent=2)
            )
            logger.info(f"知识图谱精准化结果: {refined_intent_queries}")
        else:
            refined_intent_queries = [original_query]

        # 第四步：对细化后的意图进行第二轮检索
        if set(refined_intent_queries) != set(intent_queries):
            for intent_idx, intent_query in enumerate(refined_intent_queries):
                logger.info(f"🔍 检索细化意图 {intent_idx + 1}/{len(refined_intent_queries)}: {intent_query}")

                retrieval_list = retriever.retrieve(intent_query, data_ori['documents'])

                # 合并结果并去重
                for result in retrieval_list:
                    if result['text'] not in seen_texts:
                        seen_texts.add(result['text'])
                        final_search_results.append(result)

        # 第五步：对所有结果进行最终排序
        final_search_results = self.llm_rerank(original_query, final_search_results, self.reranker, rerank_topk)

        logger.info(f"📊 最终结果: {len(final_search_results)} 条")
        logger.info([doc['score'] for doc in final_search_results])

        # 提取最终结果的页码
        final_results_with_pages = [
            {
                "text": doc['text'],
                "score": doc['score'],
                "page": doc['page']  # 获取页码
            }
            for doc in final_search_results
        ]
        # if len(page_evidence) != 0:
        #     if len(page_evidence) <= rerank_topk:
        #         final_results_with_pages[-len(page_evidence):] = page_evidence
        #     else:
        #         final_results_with_pages = page_evidence[:rerank_topk]

        return final_results_with_pages

    # 6.28修改
    def _split_query_intent(self, query: str, context=None) -> List[str]:
        """将查询拆分为多个不同维度的意图查询"""
        # SYSTEM_MESSAGE = dedent("""
        # You are a professional expert in analyzing query intentions. Your task is to analyze the user's query and break it down into multiple sub-queries of different dimensions.
        #
        # Please follow the following rules:
        # 1. If the query contains multiple different information requirements or concerns, split it into multiple sub-queries.
        # 2. Ensure that each sub-query focuses on a different dimension or aspect to maintain diversity.
        # 3. Do not merely change the form of the question; instead, focus on different information dimensions.
        # 4. If the original query is already very clear and only focuses on a single dimension, there is no need to split it.
        # 5. Sub-queries should be more specific and clear, which helps to retrieve more accurate information.
        # 6. The split sub-queries must be relevant to the context of the document.
        #
        # Please return in JSON format, including the following fields:
        # {
        #     "intent_queries": ["subquery1", "subquery2", ...]
        # }
        # """)
        # messages = [
        #     {"role": "system", "content": SYSTEM_MESSAGE},
        #     {"role": "user", "content": f"""Please analyze the following query and break it down into multiple sub-queries:
        #
        #     Original query:
        #     {query}
        #
        #     """}
        # ]
        '''改进后的prompt'''
        SYSTEM_MESSAGE = dedent("""
                You are a professional expert in analyzing query intentions. Your task is to analyze the user's query based on the retrieved context information of the document and break it down into multiple sub-queries of different dimensions.

                Please follow the following rules:
                1. If the query contains multiple different information requirements or concerns, split it into multiple sub-queries.
                2. Ensure that each sub-query focuses on a different dimension or aspect to maintain diversity.
                3. Do not merely change the form of the question; instead, focus on different information dimensions.
                4. If the original query is already very clear and only focuses on a single dimension, there is no need to split it.
                5. Sub-queries should be more specific and clear, which helps to retrieve more accurate information.
                6. The split sub-queries must be relevant to the context of the document.

                Please return in JSON format, including the following fields:
                {
                    "intent_queries": ["subquery1", "subquery2", ...]
                }
                """)

        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": f"""
                    Please analyze the following query and break it down into multiple sub-queries based on different dimensions based on retrieved context.：

                    Original Query: {query}

                    Retrieved Context:
                    {context}
                    """}
        ]
        # SYSTEM_MESSAGE = dedent("""
        #                 You are a professional expert in analyzing query intentions. Your task is to analyze the user's query based on the retrieved context information of the document and break it down into multiple sub-queries of different dimensions.
        #                 Your task has *two stages*:
        #                 **Stage 1 · Clean the query**
        #                 • Remove any words that do NOT help locate information inside the document:
        #                   – answer-format instructions (e.g. "write in float format", "return as integer",
        #                     "round to two decimals", "answer Yes/No");
        #                   – general politeness / meta phrases ("please", "thanks", "根据文档…");
        #                   – output-scene hints ("for a presentation", "for my homework");
        #                   – citations of page numbers UNLESS the page itself is the target of the question.
        #                 • Preserve domain keywords, entities, units, and page numbers **when** they are
        #                   essential for retrieval.
        #
        #                 **Stage 2 · Split the query**
        #                 Please follow the following rules:
        #                 1. If the query contains multiple different information requirements or concerns, split it into multiple sub-queries.
        #                 2. Ensure that each sub-query focuses on a different dimension or aspect to maintain diversity.
        #                 3. Do not merely change the form of the question; instead, focus on different information dimensions.
        #                 4. If the original query is already very clear and only focuses on a single dimension, there is no need to split it.
        #                 5. Sub-queries should be more specific and clear, which helps to retrieve more accurate information.
        #                 6. The split sub-queries must be relevant to the context of the document.
        #
        #                 Please return in JSON format, including the following fields:
        #                 {
        #                     "intent_queries": ["subquery1", "subquery2", ...]
        #                 }
        #                 """)
        #
        # messages = [
        #     {"role": "system", "content": SYSTEM_MESSAGE},
        #     {"role": "user", "content": f"""
        #                     Please analyze the following query. First clean it, then break it down into multiple sub-queries based on different dimensions based on retrieved context.：
        #
        #                     Original Query: {query}
        #
        #                     Retrieved Context:
        #                     {context}
        #                     """}
        # ]

        response_format = create_response_format({
            "intent_queries": {
                "type": "array",
                "description": "拆分后的子查询列表",
                "items": {"type": "string"}
            }
        })

        response = AzureGPT4Chat().chat_with_message_format(
            message_list=messages,
            # response_format=response_format
        )

        try:
            result = parse_llm_response(response)
            print("response:", response)
            intent_queries = result.get("intent_queries", [query])
            print("intent_queries:", intent_queries)
            return intent_queries if intent_queries else [query]
        except Exception as e:
            logger.error(f"意图拆分出错: {e}")
            return [query]

    def _refine_query_intent_with_knowledge_graph(
            self,
            original_query: str,
            intent_queries: List[str],
            context: str
    ) -> List[str]:
        """
        ★ 核心创新：基于多意图协同关系的知识图谱精准化

        重点分析多个子意图之间的关系和互补性：
        1. 构建跨意图的统一知识图谱
        2. 分析意图间的依赖、互补、层次关系
        3. 基于意图协同效应生成精准化查询
        """
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 构建多意图协同分析的系统提示
        SYSTEM_MESSAGE = dedent("""
        You are a knowledge-enhanced query refinement assistant. Your task is to **expand or rephrase the given sub-query** using associated entities and concepts from a knowledge graph to **broaden semantic coverage** and uncover related documents that might not match the original phrasing.

        Objective:
        1. Rewrite the sub-query to include related entities, broader or narrower types, or semantically close concepts.
        2. Suggest variations that shift the focus to attributes, consequences, roles, or relationships of the entity.
        3. Leverage the knowledge graph to:
           - resolve coreference (e.g., "he", "it", "this event"),
           - expand from entity to class (e.g., "Lehman Brothers" → "investment banks"),
           - include related factual aspects (e.g., events, policies, outcomes).

        Optimization Principles:
        - Prioritize the use of specific terms that appear in the search results
        - Avoid generating queries that have no evidence support in the search results
        - Maintain the core intention of the original query unchanged
        - Each optimized query should be independently retrievable 
        - The quantity of refined sub-queries doe not exceed 5.

        Return your output in JSON format with the following structure:
        {
            "refined_intent_queries": ["Refined sub-query 1", "Refined sub-query 2", ...]
        }

        """)

        messages = [
            {
                "role": "system",
                "content": SYSTEM_MESSAGE
            },
            {
                "role": "user",
                "content": dedent(f"""
                【original query】
                {original_query}

                【Decomposed intent queries】
                {json.dumps(intent_queries, ensure_ascii=False, indent=2)}

                【Retrieved context】
                {context}

                【Task】
                Based on the decomposed queries and the related knowledge graph information (entity names, types, timelines, and relationships), generate multiple **semantically diverse** and **professionally phrased** queries to expand retrieval coverage.

                【Guidelines】
                1. Expand vague terms into **multiple specific alternatives**.
                2. Rewrite each query in **different phrasings, scopes, and styles**.
                3. Use **synonyms, paraphrases, temporal variants, and entity-role substitutions**.
                4. Focus on **recall maximization**, not precision.
                5. You may generalize, specify, or reframe the query depending on its focus.

                """)
            }
        ]

        # 这里需要调用LLM API进行多意图协同分析
        response = AzureGPT4Chat().chat_with_message_format(
            message_list=messages,
            # response_format=self._create_multi_intent_kg_response_format()
        )

        try:
            result = parse_llm_response(response)
            refined_queries = result.get("refined_intent_queries", intent_queries)
            print("Refined intent queries:", refined_queries)
            return refined_queries if refined_queries else intent_queries
        except Exception as e:
            logger.error(f"意图细化出错: {e}")
            return intent_queries

    def _refine_query_intent(self, original_query: str, intent_queries: List[str], context: str) -> List[str]:
        """基于检索结果细化查询意图"""

        SYSTEM_MESSAGE = dedent("""
        You are a professional query intent optimization expert. Your task is to refine and enhance the user's search intent based on the retrieved content.

        Please follow these guidelines:
        1. Analyze the retrieved content to identify information gaps and areas that require further exploration.
        2. Based on the original query and the decomposed intent queries, generate more precise and targeted sub-queries.
        3. Ensure that the new sub-queries address the information needs that were not fully satisfied by the original query.
        4. Sub-queries should be more specific, incorporating domain-specific terminology and clearly defined information requirements.
        5. Avoid generating overly similar sub-queries; ensure diversity and coverage of different aspects.
        6. Limit the number of refined sub-queries to a maximum of **three**.

        Return your output in JSON format with the following structure:
        {
            "refined_intent_queries": ["Refined sub-query 1", "Refined sub-query 2", ...]
        }
        """)

        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": f"""
            Original query:
            {original_query}

            Decomposed intent queries:
            {json.dumps(intent_queries, ensure_ascii=False)}

            Retrieved context:
            {context}

            Based on the information above, please refine and optimize the search intent:
            """}
        ]

        response_format = create_response_format({
            "refined_intent_queries": {
                "type": "array",
                "description": "细化后的子查询列表",
                "items": {"type": "string"}
            }
        })

        response = AzureGPT4Chat().chat_with_message_format(
            message_list=messages,
            # response_format=response_format
        )

        try:
            result = parse_llm_response(response)
            refined_queries = result.get("refined_intent_queries", intent_queries)
            print("Refined intent queries:", refined_queries)
            return refined_queries if refined_queries else intent_queries
        except Exception as e:
            logger.error(f"意图细化出错: {e}")
            return intent_queries

    def _prepare_context(self, search_results: List[Dict[str, str]]) -> str:
        """Prepare context from search results."""
        return '\n'.join([f"{index + 1}. {result['text']}" for index, result in enumerate(search_results)])


class EnhancedDeepSearch_Beta(DeepSearch_Beta):
    """增强版DeepSearch_Beta，添加意图评分和动态筛选机制"""

    def __init__(self, max_iterations: int = 2, reranker=None, params: dict = None):
        super().__init__(max_iterations, reranker, params)
        self.intent_scorer = IntentScoringAgent()
        logger.info("✅ 已启用增强版意图评分机制")

    def search_retrieval(self, data: dict, multi_intent: bool = True, retriever=None):
        """优化的检索流程，加入意图评分机制"""
        original_query = deepcopy(data['query'])

        if not multi_intent:
            # 单意图检索保持原有逻辑
            return super().search_retrieval(data, multi_intent, retriever)

        data_ori = deepcopy(data)
        embedding_topk = self.params['embedding_topk']
        rerank_topk = self.params['rerank_topk']

        # 🔥 第一步：意图拆解
        intent_queries = self._split_query_intent(original_query, "")
        logger.info(f"🔍 初始意图拆解: {intent_queries}")

        # 🔥 第二步：意图评分与筛选（新增）
        if self.params.get('intent_scoring_enabled', True):
            scored_intents = self.intent_scorer.score_and_rank_intents(
                original_query=original_query,
                intent_queries=intent_queries,
                context_docs=data.get('documents', [])
            )
            logger.info(f"📊 意图评分结果:")
            for item in scored_intents:
                logger.info(f"   {item['intent']}... -> 评分: {item['overall_score']:.3f}")

            # 根据评分筛选高质量意图
            selected_intents = self._select_intents_by_score(scored_intents)
            logger.info(f"✅ 筛选后意图: {[item['intent'] + '...' for item in selected_intents]}")
        else:
            # 如果未启用评分，使用原有逻辑
            selected_intents = [
                {"intent": intent, "priority": "medium", "resource_allocation": embedding_topk // len(intent_queries)}
                for intent in intent_queries]

        # 🔥 第三步：分层检索策略
        all_search_results = []
        seen_texts = set()

        # 初步探索检索（保持原有逻辑）
        initial_retrieval_list = retriever.retrieve(original_query, data['documents'])
        initial_retrieval_list = initial_retrieval_list[:embedding_topk // 2]
        for r in initial_retrieval_list:
            if r['text'] not in seen_texts:
                seen_texts.add(r['text'])
                all_search_results.append(r)

        # 高优先级意图：更多检索资源
        high_priority_intents = [item for item in selected_intents if item.get('priority') == 'high']
        for intent_item in high_priority_intents:
            intent_query = intent_item['intent']
            allocation = intent_item.get('resource_allocation', 5)

            retrieval_list = retriever.retrieve(intent_query, data['documents'])
            retrieval_list = retrieval_list[:allocation]

            for result in retrieval_list:
                if result['text'] not in seen_texts:
                    seen_texts.add(result['text'])
                    all_search_results.append(result)

        # 中等优先级意图：适中检索资源
        medium_priority_intents = [item for item in selected_intents if item.get('priority') == 'medium']
        for intent_item in medium_priority_intents:
            intent_query = intent_item['intent']
            allocation = intent_item.get('resource_allocation', 3)

            retrieval_list = retriever.retrieve(intent_query, data['documents'])
            retrieval_list = retrieval_list[:allocation]

            for result in retrieval_list:
                if result['text'] not in seen_texts:
                    seen_texts.add(result['text'])
                    all_search_results.append(result)

        # 🔥 第四步：动态细化决策
        if self.params.get('intent_scoring_enabled', True) and len(selected_intents) > 0:
            refinement_decision = self.intent_scorer.should_refine_intents(
                original_query=original_query,
                current_results=all_search_results,
                intent_scores=selected_intents
            )

            logger.info(
                f"🤔 细化决策: {refinement_decision['should_refine']}, 原因: {refinement_decision['reasoning']}...")

            if refinement_decision['should_refine']:
                # 只对指定的意图进行细化
                intents_to_refine = refinement_decision['intents_to_refine']
                logger.info(f"🔄 需要细化的意图: {intents_to_refine}")

                for intent_query in intents_to_refine:
                    refined_queries = self._refine_single_intent(
                        original_query,
                        intent_query,
                        json.dumps([r['text'] for r in all_search_results], ensure_ascii=False)
                    )

                    for refined_query in refined_queries:
                        retrieval_list = retriever.retrieve(refined_query, data_ori['documents'])
                        for result in retrieval_list[:3]:  # 限制细化检索数量
                            if result['text'] not in seen_texts:
                                seen_texts.add(result['text'])
                                all_search_results.append(result)
        else:
            # 原有细化逻辑（如果未启用评分）
            refined_intent_queries = self._refine_query_intent_with_knowledge_graph(
                original_query,
                intent_queries,
                json.dumps([r['text'] for r in all_search_results], ensure_ascii=False, indent=2)
            )

            if set(refined_intent_queries) != set(intent_queries):
                for intent_idx, intent_query in enumerate(refined_intent_queries):
                    retrieval_list = retriever.retrieve(intent_query, data_ori['documents'])
                    for result in retrieval_list[:3]:
                        if result['text'] not in seen_texts:
                            seen_texts.add(result['text'])
                            all_search_results.append(result)

        # 🔥 第五步：最终重排序
        final_search_results = self.llm_rerank(original_query, all_search_results, self.reranker, rerank_topk)

        logger.info(f"📊 最终结果: {len(final_search_results)} 条")
        logger.info([doc['score'] for doc in final_search_results])

        # 提取最终结果的页码
        final_results_with_pages = [
            {
                "text": doc['text'],
                "score": doc['score'],
                "page": doc['page']  # 获取页码
            }
            for doc in final_search_results
        ]

        return final_results_with_pages

    def _select_intents_by_score(self, scored_intents: List[Dict]) -> List[Dict]:
        """根据评分筛选意图"""
        # 按相关性分数排序
        scored_intents.sort(key=lambda x: x['overall_score'], reverse=True)

        selected = []
        total_budget = self.params.get('embedding_topk', 15)
        intent_score_threshold = self.params.get('intent_score_threshold', 0.6)
        max_intents = self.params.get('max_intents_to_process', 3)

        for intent_item in scored_intents:
            if intent_item['overall_score'] >= intent_score_threshold and len(selected) < max_intents:
                # 根据分数分配检索资源
                if intent_item['overall_score'] >= 0.8:
                    priority = 'high'
                    allocation = min(self.params.get('high_priority_allocation', 8), total_budget // 2)
                elif intent_item['overall_score'] >= 0.7:
                    priority = 'medium'
                    allocation = min(self.params.get('medium_priority_allocation', 5), total_budget // 3)
                else:
                    priority = 'low'
                    allocation = min(self.params.get('low_priority_allocation', 3), total_budget // 4)

                intent_item['priority'] = priority
                intent_item['resource_allocation'] = allocation
                selected.append(intent_item)

                total_budget -= allocation
                if total_budget <= 0:
                    break

        # 如果没有任何意图通过筛选，至少保留最高分的一个
        if not selected and scored_intents:
            best_intent = scored_intents[0]
            best_intent['priority'] = 'medium'
            best_intent['resource_allocation'] = self.params.get('embedding_topk', 15) // 2
            selected.append(best_intent)

        return selected

    def _refine_single_intent(self, original_query: str, intent_query: str, context: str) -> List[str]:
        """对单个意图进行细化"""
        SYSTEM_MESSAGE = dedent("""
        You are an expert at refining search intents. Given an original query, a specific intent, and current search context, generate 1-2 refined queries that can better capture the intent.

        Return in JSON format:
        {
            "refined_queries": ["refined query 1", "refined query 2"]
        }
        """)

        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": f"""
            Original query: {original_query}
            Intent to refine: {intent_query}
            Current context: {context[:1000]}

            Please generate 1-2 refined queries for this specific intent.
            """}
        ]

        try:
            response = AzureGPT4Chat().chat_with_message_format(message_list=messages)
            result = parse_llm_response(response)
            return result.get("refined_queries", [intent_query])
        except:
            return [intent_query]


def parse_llm_response(response_text: str) -> dict:
    """
    从LLM响应中提取JSON数据，处理各种可能的格式
    Args:
        response_text: 模型返回的原始文本
    Returns:
        dict: 解析后的JSON对象
    """
    import re
    import json

    # 1. 清理可能的markdown代码块格式
    cleaned_text = re.sub(r'```(?:json|python)?', '', response_text)
    cleaned_text = re.sub(r'`', '', cleaned_text).strip()

    # 2. 尝试直接解析JSON
    try:
        result = json.loads(cleaned_text)
        return result
    except json.JSONDecodeError:
        pass

    # 3. 尝试查找JSON内容
    json_pattern = r'\{[\s\S]*\}'
    match = re.search(json_pattern, cleaned_text)
    if match:
        try:
            result = json.loads(match.group(0))
            return result
        except json.JSONDecodeError:
            pass

    # 4. 回退方案：手动提取关键字段
    output_dict = {}

    # 提取简单的字符串数组字段
    array_fields = ['intent_queries', 'refined_intent_queries']
    for field in array_fields:
        pattern = f'"{field}"\\s*:\\s*\\[(.*?)\\]'
        match = re.search(pattern, cleaned_text, re.DOTALL)
        if match:
            items = re.findall(r'"([^"]+)"', match.group(1))
            output_dict[field] = items

    # 提取 refined_queries（包含对象数组）
    refined_queries_pattern = r'"refined_queries"\s*:\s*\[(.*?)\]'
    refined_match = re.search(refined_queries_pattern, cleaned_text, re.DOTALL)
    if refined_match:
        try:
            # 尝试解析对象数组
            queries_content = refined_match.group(1)
            # 查找对象模式 {"query": "...", "sources": [...]}
            object_pattern = r'\{\s*"query"\s*:\s*"([^"]+)"\s*,\s*"sources"\s*:\s*\[(.*?)\]\s*\}'
            object_matches = re.findall(object_pattern, queries_content, re.DOTALL)

            refined_queries = []
            for query_text, sources_content in object_matches:
                sources = re.findall(r'"([^"]+)"', sources_content)
                refined_queries.append({
                    "query": query_text,
                    "sources": sources
                })

            if refined_queries:
                output_dict["refined_queries"] = refined_queries
        except Exception:
            # 如果解析失败，尝试作为简单字符串数组处理
            items = re.findall(r'"([^"]+)"', refined_match.group(1))
            if items:
                output_dict["refined_queries"] = [{"query": item, "sources": []} for item in items]

    # 提取 graph_triples（三元组对象数组）
    triples_pattern = r'"graph_triples"\s*:\s*\[(.*?)\]'
    triples_match = re.search(triples_pattern, cleaned_text, re.DOTALL)
    if triples_match:
        try:
            triples_content = triples_match.group(1)
            # 查找三元组对象模式 {"s": "...", "p": "...", "o": "...", "from": [...]}
            triple_pattern = r'\{\s*"s"\s*:\s*"([^"]+)"\s*,\s*"p"\s*:\s*"([^"]+)"\s*,\s*"o"\s*:\s*"([^"]+)"\s*,\s*"from"\s*:\s*\[(.*?)\]\s*\}'
            triple_matches = re.findall(triple_pattern, triples_content, re.DOTALL)

            graph_triples = []
            for s, p, o, from_content in triple_matches:
                from_list = re.findall(r'"([^"]+)"', from_content)
                graph_triples.append({
                    "s": s,
                    "p": p,
                    "o": o,
                    "from": from_list
                })

            if graph_triples:
                output_dict["graph_triples"] = graph_triples
        except Exception:
            pass

    # 提取其他可能的数组字段（通用处理）
    other_array_pattern = r'"([^"]+)"\s*:\s*\[(.*?)\]'
    other_matches = re.findall(other_array_pattern, cleaned_text, re.DOTALL)
    for field_name, content in other_matches:
        if field_name not in output_dict and field_name not in ['refined_queries', 'graph_triples']:
            # 尝试解析为字符串数组
            items = re.findall(r'"([^"]+)"', content)
            if items:
                output_dict[field_name] = items

    return output_dict


class PathScoringDeepSearch(DeepSearch_Beta):
    def __init__(self, max_iterations: int = 2, reranker: FlagReranker = None, params: dict = None):
        super().__init__(max_iterations, reranker, params)

        # 路径评分相关参数
        self.path_scoring_params = {
            'alpha': 0.4,  # Rel1权重 (首跳相似度)
            'beta': 0.4,  # Rel2权重 (细化后相似度)
            # 'gamma': 0.2,  # Coherence权重 (两跳证据连贯度)
            'delta': 0.1,  # Novelty权重 (新信息)
            'epsilon': 0.1,  # Answerability权重 (可答性)
            'beam_size': 5,  # 保留的top路径数量
            'lambda_rerank': 0.6,  # rerank score权重
        }
        self.path_scoring_params.update(params.get('path_scoring', {}))

        # 缓存嵌入向量和计算结果
        self.embedding_cache = {}

        # 初始化BGE模型用于精确的embedding计算
        self._init_embedding_models()

        logger.info(f"PathScoringDeepSearch初始化完成，路径评分参数: {self.path_scoring_params}")

    def _init_embedding_models(self):
        """初始化用于路径评分的embedding模型"""
        try:
            # 优先使用现有的reranker模型
            if self.reranker and hasattr(self.reranker, 'model'):
                self.embedding_model = self.reranker.model
                self.embedding_tokenizer = self.reranker.tokenizer
                logger.info("使用reranker的embedding模型进行路径评分")
            else:
                # 备选方案：加载独立的BGE模型
                from transformers import AutoTokenizer, AutoModel
                model_name = "BAAI/bge-large-en-v1.5"
                self.embedding_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.embedding_model = AutoModel.from_pretrained(model_name)
                logger.info(f"加载独立BGE模型用于路径评分: {model_name}")

            # 检查设备
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if hasattr(self.embedding_model, 'to'):
                self.embedding_model = self.embedding_model.to(device)
            self.device = device

        except Exception as e:
            logger.error(f"BGE模型初始化失败: {e}")
            self.embedding_model = None
            self.embedding_tokenizer = None
            self.device = torch.device('cpu')

    def result_processor(self, results):
        matched_docs = []
        for doc, score in results:
            matched_docs.append({
                'text': doc.page_content,
                'score': 1 - score,
                'image_score': doc.metadata.get('image_score', 0),
                'text_score': doc.metadata.get('text_score', 0),
                'metadata': doc.metadata
            })
        return matched_docs

    def llm_rerank(self, query, retrieval_list, reranker, topk=None):
        pairs = [[query, doc['text']] for doc in retrieval_list]
        rerank_scores = reranker.compute_score(pairs, normalize=True)
        output_list = []
        for score, doc in sorted(zip(rerank_scores, retrieval_list), key=lambda x: x[0], reverse=True):
            output_list.append({
                'text': doc['text'],
                'page': doc.get('metadata', {}).get('page_index', doc.get('page')),
                'score': score,
                'metadata': doc.get('metadata', {}),
                'rerank_score': score  # 保存原始rerank分数
            })
        if topk is not None:
            output_list = output_list[:topk]
        return output_list

    def search_retrieval_with_path_scoring(self, data: dict, multi_intent: False, retriever: MultimodalMatcher):
        """增强版检索，集成路径评分"""
        original_query = deepcopy(data['query'])
        data_ori = deepcopy(data)
        embedding_topk = self.params['embedding_topk']
        rerank_topk = self.params['rerank_topk']

        # Step 1: 初步探索检索
        logger.info("🔍 Step 1: 初步探索检索")
        initial_retrieval_list = retriever.retrieve(original_query, data['documents'])
        initial_retrieval_list = initial_retrieval_list[:embedding_topk]

        # Step 2: 第一轮意图拆分
        if multi_intent:
            intent_queries = self._split_query_intent_with_context(
                original_query,
                self._format_documents_for_context(initial_retrieval_list)
            )
            logger.info(f"🔍 Step 2: 意图拆分结果: {intent_queries}")
        else:
            intent_queries = [original_query]

        # Step 3: 第一轮路径构建 (Hop-1)
        logger.info("🔍 Step 3: 第一轮路径构建")
        hop1_paths = []
        seen_texts = set()

        for intent_idx, intent_query in enumerate(intent_queries):
            logger.info(f"检索意图 {intent_idx + 1}/{len(intent_queries)}: {intent_query}")
            retrieval_list = retriever.retrieve(intent_query, data['documents'])
            retrieval_list = retrieval_list[:embedding_topk // len(intent_queries)]

            for doc in retrieval_list:
                if doc['text'] not in seen_texts:
                    seen_texts.add(doc['text'])
                    hop1_paths.append({
                        'intent_id': f"I{intent_idx}",
                        'intent_query': intent_query,
                        'hop1_doc': doc,
                        'hop1_score': doc['score']
                    })

        # Step 4: 带路径追踪的意图细化
        if multi_intent:
            logger.info("🔍 Step 4: 带路径追踪的意图细化")
            refined_intents_with_paths = self._refine_intent_with_path_tracking(
                original_query, intent_queries, hop1_paths
            )
        else:
            refined_intents_with_paths = [{'query': original_query, 'sources': ['I0']}]

        # Step 5: 第二轮检索 (Hop-2)
        logger.info("🔍 Step 5: 第二轮检索构建完整路径")
        complete_paths = []

        for refined_intent in refined_intents_with_paths:
            refined_query = refined_intent['query']
            source_paths = refined_intent.get('sources', [])

            logger.info(f"检索细化意图: {refined_query}")
            hop2_results = retriever.retrieve(refined_query, data_ori['documents'])

            for hop2_doc in hop2_results[:embedding_topk]:
                # 为每个来源路径创建完整路径
                for source in source_paths:
                    # 找到对应的hop1路径
                    matching_hop1 = [p for p in hop1_paths if p['intent_id'] == source]

                    if matching_hop1:
                        hop1_path = matching_hop1[0]
                        complete_path = {
                            'hop1_intent': hop1_path['intent_query'],
                            'hop1_doc': hop1_path['hop1_doc'],
                            'hop2_intent': refined_query,
                            'hop2_doc': hop2_doc,
                            'source_ids': source_paths
                        }
                        complete_paths.append(complete_path)

        # Step 6: 路径评分
        logger.info("🔍 Step 6: 计算路径评分")
        scored_paths = []
        for path in complete_paths:
            score = self._compute_path_score(original_query, path)
            scored_paths.append((score, path))

        # Step 7: Beam剪枝
        beam_size = self.path_scoring_params['beam_size']
        scored_paths = sorted(scored_paths, key=lambda x: x[0], reverse=True)[:beam_size]
        logger.info(f"🔍 Step 7: Beam剪枝保留 {len(scored_paths)} 条路径")

        # Step 8: 证据去重和权重融合
        logger.info("🔍 Step 8: 证据去重和权重融合")
        doc_weights, unique_docs = self._fuse_evidence_with_weights(scored_paths)

        # Step 9: 最终排序融合
        logger.info("🔍 Step 9: 最终排序融合")
        final_results = self._final_scoring_and_ranking(
            original_query, unique_docs, doc_weights, rerank_topk
        )

        logger.info(f"📊 最终结果: {len(final_results)} 条")
        logger.info(f"📊 最终分数: {[doc['final_score'] for doc in final_results]}")

        return final_results

    def _split_query_intent_with_context(self, query: str, context: str) -> List[str]:
        """带上下文的查询意图拆分"""
        SYSTEM_MESSAGE = dedent("""
        You are a professional expert in analyzing query intentions. Your task is to analyze the user's query based on the retrieved context information of the document and break it down into multiple sub-queries of different dimensions.

        Please follow the following rules:
        1. If the query contains multiple different information requirements or concerns, split it into multiple sub-queries.
        2. Ensure that each sub-query focuses on a different dimension or aspect to maintain diversity.
        3. Do not merely change the form of the question; instead, focus on different information dimensions.
        4. If the original query is already very clear and only focuses on a single dimension, there is no need to split it.
        5. Sub-queries should be more specific and clear, which helps to retrieve more accurate information.
        6. The split sub-queries must be relevant to the context of the document.

        Please return in JSON format, including the following fields:
        {
            "intent_queries": ["subquery1", "subquery2", ...]
        }
        """)

        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": f"""
            Please analyze the following query and break it down into multiple sub-queries based on different dimensions based on retrieved context.：

            Original Query: {query}

            Retrieved Context:
            {context}
            """}
        ]

        response = AzureGPT4Chat().chat_with_message_format(message_list=messages)

        try:
            result = parse_llm_response(response)
            intent_queries = result.get("intent_queries", [query])
            logger.info(f"意图拆分: {intent_queries}")
            return intent_queries if intent_queries else [query]
        except Exception as e:
            logger.error(f"意图拆分出错: {e}")
            return [query]

    def _refine_intent_with_path_tracking(self, original_query: str, intent_queries: List[str],
                                          hop1_paths: List[dict]) -> List[dict]:
        """带路径追踪的意图细化"""

        # 构建带标记的上下文
        context_with_ids = self._build_context_with_ids(intent_queries, hop1_paths)

        SYSTEM_MESSAGE = dedent("""
        You are an advanced multi-intent refinement and knowledge-graph expert.

        TASKS
        ─────
        1. **Analyse Inputs**
           • Original query (Q0)  
           • Sub-intents [I1…In]  
           • First-hop evidence docs [D1…Dm] (with IDs)  
           • Pay special attention to entity overlaps, temporal hints, numeric ranges, and explicit
             relations (e.g. "launches", "located in", "during 2015–16").

        2. Generate refined queries:
           • Produce concise, highly-specific queries that are directly retrievable.  
           • Ground them in the exact entities, dates or values found in the evidence.  
           • For every refined query, list the IDs of the sub-intents / documents it relies on
             in a `"sources"` array (e.g., ["I2", "D5"]).

        3. Extract key triples:
           • From the same intents and evidence, extract up to **five** RDF-style triples  
             `(subject, predicate, object)` that directly support answering Q0.  
           • Subjects & objects must be canonical entities or values appearing in the text.  
           • Predicates should be short relation phrases such as "launched_in",
             "located_at", "has_duration", "mass".  
           • Add a `"from"` field with the ID(s) of the document(s) that contain the triple.


        OUTPUT FORMAT  (return **exactly** this JSON schema)
        ────────────────────────────────────────────────────
        {
            "refined_queries": [
                {"query": "refined query 1", "sources": ["I1", "D2"]},
                {"query": "refined query 2", "sources": ["I2", "D3"]}
            ]
            "graph_triples": [
            {"s": "<subject>", "p": "<predicate>", "o": "<object>", "from": ["D3"]},
            ...
            ]
            RULES
            ─────
            * Do **not** invent entities or facts absent from the given evidence.  
            * A refined query is **not** a triple—you must output both sections.  
            * JSON must be valid and parseable: no comments, no trailing commas, no extra keys.
        }
        """)

        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": f"""
            Original Query: {original_query}

            {context_with_ids}

            Please generate refined queries with source tracking.
            """}
        ]

        response = AzureGPT4Chat().chat_with_message_format(message_list=messages)

        try:
            result = parse_llm_response(response)
            refined_queries = result.get("refined_queries", [])

            # 如果解析失败，返回默认结果
            if not refined_queries:
                return [{"query": original_query, "sources": [f"I{i}" for i in range(len(intent_queries))]}]

            logger.info(f"细化意图(带路径): {refined_queries}")
            return refined_queries

        except Exception as e:
            logger.error(f"意图细化出错: {e}")
            return [{"query": original_query, "sources": [f"I{i}" for i in range(len(intent_queries))]}]

    def _build_context_with_ids(self, intent_queries: List[str], hop1_paths: List[dict]) -> str:
        """构建带ID的上下文"""
        context_parts = ["## Sub-Intents"]

        # 添加子意图
        for i, intent in enumerate(intent_queries):
            context_parts.append(f"[I{i}] {intent}")

        context_parts.append("\n## Retrieved Evidence")

        # 添加检索到的证据
        for i, path in enumerate(hop1_paths):
            intent_id = path['intent_id']
            doc_text = path['hop1_doc']['text'][:200] + "..." if len(path['hop1_doc']['text']) > 200 else \
                path['hop1_doc']['text']
            context_parts.append(f"[D{i}] (from {intent_id}) {doc_text}")

        return "\n".join(context_parts)

    def _compute_path_score(self, original_query: str, path: dict) -> float:
        """
        S(P) = α·Rel₁ + β·Rel₂ + δ·Novelty + ε·Answerability
        （如需连贯度可再加 γ·Coherence）
        """
        try:
            hop1_doc, hop2_doc = path['hop1_doc'], path['hop2_doc']
            hop1_q, hop2_q = path['hop1_intent'], path['hop2_intent']

            # --- 1. 相似度 ---
            rel1 = self._compute_semantic_similarity(hop1_q, hop1_doc['text'])
            rel2 = self._compute_semantic_similarity(hop2_q, hop2_doc['text'])

            # --- 2. 信息新颖度 ---
            novelty = self._compute_information_novelty(hop1_doc, hop2_doc)

            # --- 3. 可答性 ---
            answerability = self._compute_comprehensive_answerability(
                original_query, hop1_doc, hop2_doc
            )

            rel1 = rel1 or 0.0
            rel2 = rel2 or 0.0
            nov = novelty or 0.0
            ans = answerability or 0.0

            p = self.path_scoring_params  # 取权重
            score = (
                    p['alpha'] * rel1 +
                    p['beta'] * rel2 +
                    p['delta'] * nov +
                    p['epsilon'] * ans
            )
            return max(0.0, min(1.0, score))

        except Exception as e:
            logger.error(f"路径评分计算出错: {e}", exc_info=True)
            return 0.0

    def _compute_semantic_similarity(self, query: str, document_text: str) -> float:
        from transformers.modeling_outputs import SequenceClassifierOutput
        """
        Compute semantic similarity with proper device handling and
        compatibility for both encoder-only models and
        AutoModelForSequenceClassification cross-encoders.
        """
        try:
            cache_key = f"sim_{hash(query)}_{hash(document_text[:200])}"
            if cache_key in self.embedding_cache:
                return self.embedding_cache[cache_key]

            query_clean = self._preprocess_text(query)
            doc_clean = self._preprocess_text(document_text)
            if not query_clean.strip() or not doc_clean.strip():
                return 0.0

            # ── Case 1: reranker 已带 compute_score (交叉编码器) ───────────
            if hasattr(self.reranker, "compute_score"):
                score = self.reranker.compute_score(
                    [[query_clean, doc_clean]], normalize=True
                )[0]
                result = float(max(0.0, min(1.0, score)))

            # ── Case 2: 手动取 embedding (bi-encoder) ───────────────────
            else:
                model = self.reranker.model
                tokenizer = self.reranker.tokenizer
                device = next(model.parameters()).device

                q_inputs = tokenizer(query_clean, return_tensors="pt",
                                     truncation=True, max_length=512).to(device)
                d_inputs = tokenizer(doc_clean, return_tensors="pt",
                                     truncation=True, max_length=512).to(device)

                with torch.no_grad():
                    q_out = model(**q_inputs)
                    d_out = model(**d_inputs)

                # 如果返回的是普通 encoder 输出
                if hasattr(q_out, "last_hidden_state"):
                    q_emb = q_out.last_hidden_state[:, 0]
                    d_emb = d_out.last_hidden_state[:, 0]
                    sim = torch.nn.functional.cosine_similarity(q_emb, d_emb, dim=1)
                    result = float(sim.item())

                # 如果返回的是 SequenceClassifierOutput → 回落到 logits
                elif isinstance(q_out, SequenceClassifierOutput):
                    # logits 维度 [1,1]，直接做 -logits 距离或 sigmoid
                    logits_sim = torch.sigmoid(q_out.logits).item()
                    result = float(logits_sim)

                else:
                    result = 0.0  # unexpected output type → 防御

            self.embedding_cache[cache_key] = result
            return result

        except Exception as e:
            logger.error(f"语义相似度计算出错: {e}", exc_info=True)
            return 0.0

    def _compute_semantic_coherence(self, doc1: dict, doc2: dict) -> float:
        """
        计算两个文档的语义连贯度

        Args:
            doc1: 第一个文档
            doc2: 第二个文档

        Returns:
            float: 连贯度分数 [0, 1]
        """
        try:
            text1 = doc1['text']
            text2 = doc2['text']

            # 检查缓存
            cache_key = f"coherence_{hash(text1[:200])}_{hash(text2[:200])}"
            if cache_key in self.embedding_cache:
                return self.embedding_cache[cache_key]

            # 多层面连贯度计算

            # 1. 语义相似度 (权重: 0.4)
            semantic_sim = self._compute_semantic_similarity("", text1 + " " + text2) if text1 and text2 else 0.0

            # 2. 实体重叠度 (权重: 0.3)
            entity_overlap = self._compute_entity_overlap(text1, text2)

            # 3. 主题连贯度 (权重: 0.2)
            topic_coherence = self._compute_topic_coherence(text1, text2)

            # 4. 词汇连贯度 (权重: 0.1)
            lexical_coherence = self._compute_lexical_coherence(text1, text2)

            # 加权组合
            coherence = (0.4 * semantic_sim +
                         0.3 * entity_overlap +
                         0.2 * topic_coherence +
                         0.1 * lexical_coherence)

            # 缓存结果
            self.embedding_cache[cache_key] = coherence

            logger.debug(f"连贯度详情: semantic={semantic_sim:.3f}, entity={entity_overlap:.3f}, "
                         f"topic={topic_coherence:.3f}, lexical={lexical_coherence:.3f}, total={coherence:.3f}")

            return coherence

        except Exception as e:
            logger.error(f"连贯度计算出错: {e}")
            return 0.5

    def _compute_information_novelty(self, doc1: dict, doc2: dict) -> float:
        """
        计算信息新颖度/多样性

        Args:
            doc1: 第一个文档
            doc2: 第二个文档

        Returns:
            float: 新颖度分数 [0, 1]，越高表示doc2相对doc1包含更多新信息
        """
        try:
            text1 = doc1['text']
            text2 = doc2['text']

            if not text1.strip() or not text2.strip():
                return 0.5

            # 1. 内容重叠度
            content_overlap = self._compute_content_overlap(text1, text2)

            # 2. 信息增益
            info_gain = self._compute_information_gain(text1, text2)

            # # 3. 结构多样性
            # structural_diversity = self._compute_structural_diversity(text1, text2)

            # 新颖度 = 1 - 重叠度 + 信息增益 + 结构多样性
            # novelty = (1.0 - content_overlap) * 0.5 + info_gain * 0.3 + structural_diversity * 0.2
            novelty = (1.0 - content_overlap) * 0.5 + info_gain * 0.5

            return max(0.0, min(1.0, novelty))

        except Exception as e:
            logger.error(f"新颖度计算出错: {e}")
            return 0.5

    def _compute_comprehensive_answerability(self, query: str, doc1: dict, doc2: dict) -> float:
        """
        Answerability = 0.4·覆盖度 + 0.3·证据互补 + 0.2·答案线索 + 0.1·推理可行
        """
        try:
            text1, text2 = doc1['text'], doc2['text']
            combined = f"{text1} {text2}"

            coverage = self._compute_query_coverage(query, combined)  # 已实现
            completeness = self._compute_evidence_completeness(query, text1, text2)  # 已实现
            # clue_strength = self._compute_answer_clues(query, combined)  # 已实现
            # reasoning = self._compute_reasoning_feasibility(query, combined)  # 已实现

            # ans = 0.4 * coverage + 0.3 * completeness + 0.2 * clue_strength + 0.1 * reasoning
            ans = 0.6 * coverage + 0.4 * completeness
            return max(0.0, min(1.0, ans))

        except Exception as e:
            logger.error(f"可答性计算出错: {e}", exc_info=True)
            return 0.5

    def _preprocess_text(self, text: str) -> str:
        """文本预处理"""
        if not text:
            return ""
        # 去除多余空格，保留基本标点
        text = re.sub(r'\s+', ' ', text.strip())
        return text[:1000]  # 限制长度避免计算过慢

    def _fallback_similarity(self, query: str, document: str) -> float:
        """备选相似度计算方法"""
        try:
            query_words = set(query.lower().split())
            doc_words = set(document.lower().split())

            if not query_words or not doc_words:
                return 0.0

            intersection = len(query_words.intersection(doc_words))
            union = len(query_words.union(doc_words))

            return intersection / union if union > 0 else 0.0

        except Exception:
            return 0.0

    def _compute_entity_overlap(self, text1: str, text2: str) -> float:
        """计算实体重叠度"""
        try:
            # 简化版实体提取：大写开头的词组
            entities1 = set(re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', text1))
            entities2 = set(re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', text2))

            if not entities1 and not entities2:
                return 0.0
            if not entities1 or not entities2:
                return 0.0

            intersection = len(entities1.intersection(entities2))
            union = len(entities1.union(entities2))

            return intersection / union if union > 0 else 0.0

        except Exception:
            return 0.0

    def _compute_topic_coherence(self, text1: str, text2: str) -> float:
        """计算主题连贯度"""
        try:
            # 基于关键词的主题相似度
            from collections import Counter

            # 提取关键词（长度>=4的词）
            words1 = [w.lower() for w in re.findall(r'\b\w{4,}\b', text1)]
            words2 = [w.lower() for w in re.findall(r'\b\w{4,}\b', text2)]

            if not words1 or not words2:
                return 0.0

            # 计算词频
            freq1 = Counter(words1)
            freq2 = Counter(words2)

            # 计算cosine相似度
            common_words = set(freq1.keys()).intersection(set(freq2.keys()))

            if not common_words:
                return 0.0

            dot_product = sum(freq1[word] * freq2[word] for word in common_words)
            norm1 = sum(freq ** 2 for freq in freq1.values()) ** 0.5
            norm2 = sum(freq ** 2 for freq in freq2.values()) ** 0.5

            return dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 0.0

        except Exception:
            return 0.0

    def _compute_lexical_coherence(self, text1: str, text2: str) -> float:
        """计算词汇连贯度"""
        try:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())

            if not words1 or not words2:
                return 0.0

            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))

            return intersection / union if union > 0 else 0.0

        except Exception:
            return 0.0

    def _compute_content_overlap(self, text1: str, text2: str) -> float:
        """计算内容重叠度"""
        try:
            # 使用n-gram重叠
            def get_ngrams(text, n=3):
                words = text.lower().split()
                return set(' '.join(words[i:i + n]) for i in range(len(words) - n + 1))

            ngrams1 = get_ngrams(text1)
            ngrams2 = get_ngrams(text2)

            if not ngrams1 or not ngrams2:
                return 0.0

            intersection = len(ngrams1.intersection(ngrams2))
            union = len(ngrams1.union(ngrams2))

            return intersection / union if union > 0 else 0.0

        except Exception:
            return 0.0

    # 4) 具体的增益函数
    def _compute_information_gain(text1: str, text2: str, vec: TfidfVectorizer) -> float:
        # 1) 预加载 spaCy 英文模型 (只做一次)
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except OSError:  # E050: model not found
            import spacy.cli
            spacy.cli.download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm")

        # 2) 停用词 & 简易 tokenizer
        _stop = _nlp.Defaults.stop_words
        _punct = set(string.punctuation)

        def _tokenize(text: str) -> List[str]:
            doc = _nlp(text.lower())
            return [
                tok.lemma_ for tok in doc
                if tok.pos_ in {"NOUN", "PROPN", "VERB", "ADJ"}
                   and tok.lemma_ not in _stop
                   and tok.lemma_ not in _punct
                   and len(tok) > 2
            ]

        # 3) vectorizer 可在类 __init__ 里持久化 (避免反复 fit)
        def build_vectorizer(corpus: List[str]) -> TfidfVectorizer:
            vec = TfidfVectorizer(tokenizer=_tokenize, lowercase=False)
            vec.fit(corpus)
            return vec

        try:
            v1 = vec.transform([text1])
            v2 = vec.transform([text2])

            if v2.nnz == 0:
                return 0.0

            # indices present in v2 but zero in v1
            v1_idxs = set(v1.indices)
            new_mask = [i for i in range(v2.indices.size) if v2.indices[i] not in v1_idxs]

            if not new_mask:
                return 0.0

            gain_weight = np.sum(v2.data[new_mask])
            total_weight = np.sum(v2.data)
            return float(gain_weight / total_weight)

        except Exception as e:
            # log if you have a logger
            return 0.0

    def _compute_query_coverage(self, query: str, text: str) -> float:
        """计算查询覆盖度"""
        try:
            query_words = set(query.lower().split())
            text_words = set(text.lower().split())

            if not query_words:
                return 0.0

            covered = len(query_words.intersection(text_words))
            return covered / len(query_words)

        except Exception:
            return 0.0

    def _compute_evidence_completeness(self, query: str, text1: str, text2: str) -> float:
        """计算证据完整性"""
        try:
            # 检查两个文档是否提供互补证据
            coverage1 = self._compute_query_coverage(query, text1)
            coverage2 = self._compute_query_coverage(query, text2)
            combined_coverage = self._compute_query_coverage(query, text1 + " " + text2)

            # 互补性奖励
            complementarity = combined_coverage - max(coverage1, coverage2)

            return min(combined_coverage + complementarity * 0.5, 1.0)

        except Exception:
            return 0.0

    def _compute_answer_clues(self, query: str, text: str) -> float:
        """计算答案线索强度"""
        try:
            # 检查是否包含典型的答案指示词
            answer_indicators = ['because', 'therefore', 'result',
                                 'data shows', 'research indicates',
                                 'specifically', 'for example']

            indicator_count = sum(1 for indicator in answer_indicators if indicator in text.lower())

            # 数字和具体事实的存在
            numbers = len(re.findall(r'\d+(?:\.\d+)?', text))

            # 综合评分
            indicator_score = min(indicator_count / 3, 1.0)
            number_score = min(numbers / 5, 1.0)

            return (indicator_score + number_score) / 2

        except Exception:
            return 0.0

    def _compute_reasoning_feasibility(self, query: str, text: str) -> float:
        """计算推理可行性"""
        try:
            # 简化版：基于文本长度和复杂度
            words = text.split()
            sentences = re.split(r'[.!?]+', text)

            if not words:
                return 0.0

            # 文本长度适中性（太短或太长都不利于推理）
            length_score = 1.0 - abs(len(words) - 200) / 200 if len(words) <= 400 else 0.5
            length_score = max(0.0, min(1.0, length_score))

            # 句子复杂度
            avg_sentence_len = len(words) / len(sentences) if sentences else 0
            complexity_score = 1.0 if 10 <= avg_sentence_len <= 25 else 0.5

            return (length_score + complexity_score) / 2

        except Exception:
            return 0.5

    def _fuse_evidence_with_weights(self, scored_paths: List[Tuple[float, dict]]) -> Tuple[
        Dict[str, float], List[dict]]:
        """证据去重和权重融合"""
        doc_weights = defaultdict(float)
        unique_docs = {}

        for score, path in scored_paths:
            # 处理hop1和hop2的文档
            for hop_key in ['hop1_doc', 'hop2_doc']:
                doc = path[hop_key]
                doc_id = self._get_doc_id(doc)

                # 累积权重
                doc_weights[doc_id] += score

                # 保存唯一文档
                if doc_id not in unique_docs:
                    unique_docs[doc_id] = doc

        # 归一化权重到[0,1]
        if doc_weights:
            max_weight = max(doc_weights.values())
            if max_weight > 0:
                for doc_id in doc_weights:
                    doc_weights[doc_id] /= max_weight

        unique_docs_list = list(unique_docs.values())

        logger.info(f"证据融合: {len(unique_docs_list)} 个唯一文档, 权重范围: "
                    f"{min(doc_weights.values()):.3f}-{max(doc_weights.values()):.3f}")

        return dict(doc_weights), unique_docs_list

    def _get_doc_id(self, doc: dict) -> str:
        """获取文档唯一ID"""
        # 优先使用metadata中的信息
        metadata = doc.get('metadata', {})
        if 'page_index' in metadata:
            pdf_path = metadata.get('pdf_path', 'unknown')
            page_index = metadata['page_index']
            return f"{pdf_path}_page_{page_index}"

        # 后备方案：使用文本hash
        return str(hash(doc['text'][:100]))

    def _final_scoring_and_ranking(self, query: str, docs: List[dict], doc_weights: Dict[str, float], topk: int) -> \
            List[dict]:
        """最终评分和排序"""
        # 先进行rerank
        reranked_docs = self.llm_rerank(query, docs, self.reranker, len(docs))

        # 融合rerank分数和路径权重
        lambda_rerank = self.path_scoring_params['lambda_rerank']

        for doc in reranked_docs:
            doc_id = self._get_doc_id(doc)
            path_weight = doc_weights.get(doc_id, 0.0)
            rerank_score = doc.get('rerank_score', doc.get('score', 0.0))

            # 最终分数融合
            final_score = lambda_rerank * rerank_score + (1 - lambda_rerank) * path_weight

            doc['final_score'] = final_score
            doc['path_weight'] = path_weight
            doc['rerank_score'] = rerank_score

        # 按最终分数排序
        final_results = sorted(reranked_docs, key=lambda x: x['final_score'], reverse=True)[:topk]

        return final_results

    def _format_documents_for_context(self, docs: List[dict]) -> str:
        """格式化文档为上下文字符串"""
        context_parts = []
        for i, doc in enumerate(docs[:5]):  # 只取前5个文档
            text_preview = doc['text'][:150] + "..." if len(doc['text']) > 150 else doc['text']
            context_parts.append(f"{i + 1}. {text_preview}")
        return "\n".join(context_parts)


class PageCoverageDeepSearch(DeepSearch_Beta):
    """
    页面覆盖度优化的深度搜索器

    核心思想：通过覆盖度最大化策略，一次性选择能够覆盖查询所有关键信息的K个页面，
    避免冗余信息，提高检索结果的信息密度和完整性。
    """

    def __init__(self, max_iterations: int = 2, reranker=None, params: dict = None):
        super().__init__(max_iterations, reranker, params)

        # 页面覆盖搜索特有参数
        default_coverage_params = {
            "max_pages": 10,  # 最多返回页面数 K
            "lambda_coverage": 0.4,  # 覆盖度权重
            "lambda_relevance": 0.6,  # 相关度权重
            "sim_threshold": 0.55,  # 子查询过滤阈值
            "max_candidate_queries": 6,  # 最大候选查询数
            "min_marginal_gain": 0.02,  # 贪心算法最小边际收益
            "dedup_threshold": 0.9,  # 页面去重阈值
            "stop_words": {  # 覆盖度计算停用词
                "the", "of", "and", "in", "to", "a", "is", "for", "on", "with",
                "an", "by", "at", "as", "from", "that", "this", "these", "those",
                "be", "have", "has", "had", "was", "were", "are", "am", "will", "would"
            }
        }
        self.embedding_cache = {}

        # 合并参数
        self.coverage_params = default_coverage_params.copy()
        if params and "page_coverage" in params:
            self.coverage_params.update(params["page_coverage"])

        logger.info(f"PageCoverageDeepSearch 初始化完成，覆盖度参数: {self.coverage_params}")

    def _preprocess_text(self, text: str) -> str:
        """文本预处理"""
        if not text:
            return ""
        # 去除多余空格，保留基本标点
        text = re.sub(r'\s+', ' ', text.strip())
        return text  # 限制长度避免计算过慢

    def search_retrieval_with_coverage(self, data: dict, multi_intent: bool = True, retriever=None) -> List[dict]:
        """
        主检索接口 - 使用页面覆盖度优化策略

        Args:
            data: 包含查询和文档的数据字典
            multi_intent: 是否启用多意图分析
            retriever: 检索器实例

        Returns:
            List[dict]: 按覆盖度优化排序的检索结果
        """
        original_query = data['query']
        documents = data['documents']

        logger.info(f"🔍 开始页面覆盖度检索: {original_query}")

        # Step 1: 候选查询生成
        candidate_queries = self._generate_candidate_queries(original_query, multi_intent)
        logger.info(f"📝 生成 {len(candidate_queries)} 个候选查询: {candidate_queries}")

        # Step 2: 多通道召回
        recall_pool = self._multi_channel_recall(candidate_queries, retriever, documents)
        logger.info(f"📊 召回池大小: {len(recall_pool)}")

        if not recall_pool:
            logger.warning("召回池为空，返回空结果")
            return []

        # Step 3: 页面去重
        unique_pages = self._deduplicate_pages(recall_pool)
        logger.info(f"🔄 去重后页面数: {len(unique_pages)}")

        # Step 4: 计算语义相关度 (Rel₀)
        pages_with_relevance = self._compute_relevance_scores(original_query, unique_pages)

        # Step 5: 计算查询覆盖度
        pages_with_coverage = self._compute_coverage_scores(original_query, pages_with_relevance)

        # Step 6: 贪心选择最优页面组合
        selected_pages = self._greedy_max_coverage_selection(original_query, pages_with_coverage)
        logger.info(f"🎯 贪心选择 {len(selected_pages)} 个页面")

        # Step 7: 最终重排序
        final_results = self._final_reranking(original_query, selected_pages)

        logger.info(f"✅ 最终返回 {len(final_results)} 个结果")
        logger.info(f"📄 最终页码: {[r.get('page', r.get('metadata', {}).get('page_index')) for r in final_results]}")

        return final_results

    def _generate_candidate_queries(self, query: str, multi_intent: bool) -> List[str]:
        """生成候选查询列表"""
        candidate_queries = []

        # 1. 原始查询
        candidate_queries.append(query)

        if not multi_intent:
            return candidate_queries

        # 2. 尝试规则分解
        rule_based_queries = self._rule_based_query_decomposition(query)
        if rule_based_queries and len(rule_based_queries) > 1:
            candidate_queries.extend(rule_based_queries)
            logger.info(f"规则分解产生 {len(rule_based_queries)} 个查询")
        else:
            # 3. LLM语义分解
            llm_queries = self._llm_based_query_decomposition(query)
            candidate_queries.extend(llm_queries)
            logger.info(f"LLM分解产生 {len(llm_queries)} 个查询")

        # 4. 去重和过滤
        filtered_queries = self._filter_candidate_queries(query, candidate_queries)

        return filtered_queries[:self.coverage_params["max_candidate_queries"]]

    def _rule_based_query_decomposition(self, query: str) -> List[str]:
        """基于规则的查询分解"""
        # 页码指定规则
        page_pattern = re.compile(r'page(?:s)?\s*(\d+(?:\s*-\s*\d+)?)', re.I)
        if page_pattern.search(query):
            page_span = page_pattern.search(query).group(1)
            if '-' in page_span:
                start, end = page_span.split('-')
                return [f"content on page {start.strip()}", f"content on page {end.strip()}"]
            else:
                return [f"content on page {page_span.strip()}"]

        # 计数查询规则
        count_pattern = re.compile(r'\bhow many\b|\bnumber of\b', re.I)
        if count_pattern.search(query):
            return [query, "list all items for counting"]

        # 年份查询规则
        year_pattern = re.compile(r'\b(?:FY\s*)?20\d{2}\b')
        if year_pattern.search(query):
            year = year_pattern.search(query).group(0)
            core_query = re.sub(year_pattern, '', query).strip()
            return [f"{core_query} {year}", f"financial data {year}"]

        # 比较查询规则
        comparison_words = ['compare', 'difference', 'versus', 'vs', 'between']
        if any(word in query.lower() for word in comparison_words):
            # 尝试提取比较对象
            entities = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b', query)
            if len(entities) >= 2:
                return [f"information about {entities[0]}", f"information about {entities[1]}"]

        return [query]

    def _llm_based_query_decomposition(self, query: str) -> List[str]:
        """基于LLM的查询分解"""
        SYSTEM_MESSAGE = dedent("""
        You are an expert at decomposing complex user queries into multiple focused sub-queries
        for comprehensive information retrieval.

        Rules:
        1. Decompose the query ONLY if it contains multiple distinct information needs
        2. Focus on different aspects: entities, time periods, metrics, relationships, etc.
        3. Generate 2-4 sub-queries maximum
        4. Each sub-query should be independently retrievable
        5. Keep sub-queries concise (≤ 15 words each)
        6. If the query is already focused, return it as the single query

        Return JSON format:
        {
            "sub_queries": ["sub-query 1", "sub-query 2", ...]
        }
        """)

        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": f"Decompose this query: {query}"}
        ]

        try:
            response = AzureGPT4Chat().chat_with_message_format(message_list=messages)
            result = parse_llm_response(response)
            sub_queries = result.get("sub_queries", [query])
            return sub_queries if sub_queries else [query]
        except Exception as e:
            logger.error(f"LLM查询分解失败: {e}")
            return [query]

    def _filter_candidate_queries(self, original_query: str, candidates: List[str]) -> List[str]:
        """过滤候选查询"""
        filtered = []
        seen = set()

        for candidate in candidates:
            # 去重
            if candidate.lower() in seen:
                continue
            seen.add(candidate.lower())

            # 相似度过滤
            if candidate != original_query:
                similarity = self._compute_semantic_similarity(original_query, candidate)
                if similarity < self.coverage_params["sim_threshold"]:
                    logger.debug(f"过滤低相似度查询: {candidate} (sim={similarity:.3f})")
                    continue

            filtered.append(candidate)

        return filtered

    def _multi_channel_recall(self, queries: List[str], retriever, documents) -> List[dict]:
        """多通道召回"""
        recall_pool = []
        per_query_limit = max(1, self.params['embedding_topk'] // len(queries))

        for i, sub_query in enumerate(queries):
            logger.debug(f"召回子查询 {i + 1}/{len(queries)}: {sub_query}")

            # 使用检索器召回
            sub_results = retriever.retrieve(sub_query, documents)
            limited_results = sub_results[:per_query_limit]

            # 标记来源查询
            for result in limited_results:
                result['source_query'] = sub_query
                result['source_query_id'] = i

            recall_pool.extend(limited_results)
            logger.debug(f"召回 {len(limited_results)} 个文档")

        return recall_pool

    def _deduplicate_pages(self, pages: List[dict]) -> List[dict]:
        """页面去重 - 同一页面保留最高分数的版本"""
        page_groups = defaultdict(list)

        # 按页面ID分组
        for page in pages:
            page_id = self._get_page_identifier(page)
            page_groups[page_id].append(page)

        # 每组保留最高分数的页面
        unique_pages = []
        for page_id, group in page_groups.items():
            best_page = max(group, key=lambda x: x.get('score', 0))
            unique_pages.append(best_page)

        logger.info(f"去重: {len(pages)} -> {len(unique_pages)} 个唯一页面")
        return unique_pages

    def _get_page_identifier(self, page: dict) -> str:
        """获取页面唯一标识符"""
        metadata = page.get('metadata', {})

        # 优先使用页面索引
        if 'page_index' in metadata:
            pdf_path = metadata.get('pdf_path', 'unknown')
            return f"{pdf_path}#page_{metadata['page_index']}"

    def _compute_relevance_scores(self, query: str, pages: List[dict]) -> List[dict]:
        """计算语义相关度分数 (Rel₀)"""
        if not pages:
            return pages

        # 使用重排序器计算相关度
        pairs = [[query, page['text']] for page in pages]
        relevance_scores = self.reranker.compute_score(pairs, normalize=True)

        # 添加相关度分数
        for page, score in zip(pages, relevance_scores):
            page['relevance_score'] = float(score)

        logger.debug(f"相关度分数范围: {min(relevance_scores):.3f} - {max(relevance_scores):.3f}")
        return pages

    def _compute_coverage_scores(self, query: str, pages: List[dict]) -> List[dict]:
        """计算查询覆盖度分数"""
        query_tokens = self._tokenize_for_coverage(query)

        for page in pages:
            page_tokens = self._tokenize_for_coverage(page['text'])
            coverage = self._calculate_coverage(query_tokens, page_tokens)
            page['coverage_score'] = coverage

        # 计算综合分数
        for page in pages:
            relevance = page.get('relevance_score', 0.5)
            coverage = page.get('coverage_score', 0.0)

            composite_score = (
                    self.coverage_params['lambda_relevance'] * relevance +
                    self.coverage_params['lambda_coverage'] * coverage
            )
            page['composite_score'] = composite_score

        return pages

    def _tokenize_for_coverage(self, text: str) -> Set[str]:
        """为覆盖度计算进行分词"""
        if not text:
            return set()

        # 提取字母数字词汇
        words = re.findall(r'[A-Za-z0-9]+', text.lower())

        # 过滤停用词和短词
        meaningful_words = {
            word for word in words
            if len(word) >= 2 and word not in self.coverage_params['stop_words']
        }

        return meaningful_words

    def _calculate_coverage(self, query_tokens: Set[str], page_tokens: Set[str]) -> float:
        """计算覆盖度"""
        if not query_tokens:
            return 0.0

        covered_tokens = query_tokens.intersection(page_tokens)
        coverage = len(covered_tokens) / len(query_tokens)

        return coverage

    def _greedy_max_coverage_selection(self, query: str, pages: List[dict]) -> List[dict]:
        """贪心最大覆盖选择算法"""
        query_tokens = self._tokenize_for_coverage(query)
        selected_pages = []
        covered_tokens = set()
        remaining_pages = pages.copy()

        max_pages = self.coverage_params['max_pages']
        min_gain = self.coverage_params['min_marginal_gain']

        while len(selected_pages) < max_pages and remaining_pages:
            best_page = None
            best_gain = -1
            best_new_tokens = set()

            # 寻找边际收益最大的页面
            for page in remaining_pages:
                page_tokens = self._tokenize_for_coverage(page['text'])
                new_tokens = page_tokens - covered_tokens
                new_coverage_tokens = new_tokens.intersection(query_tokens)

                # 边际收益 = 新覆盖token数 / 总query token数 + 相关度奖励
                marginal_gain = len(new_coverage_tokens) / len(query_tokens) if query_tokens else 0
                relevance_bonus = page.get('relevance_score', 0.5) * 0.1  # 小的相关度奖励

                total_gain = marginal_gain + relevance_bonus

                if total_gain > best_gain:
                    best_page = page
                    best_gain = total_gain
                    best_new_tokens = new_coverage_tokens

            # 检查是否满足最小边际收益
            if best_page is None or best_gain < min_gain:
                logger.debug(f"贪心选择终止: 最佳边际收益 {best_gain:.4f} < 阈值 {min_gain}")
                break

            # 选择最佳页面
            selected_pages.append(best_page)
            covered_tokens.update(best_new_tokens)
            remaining_pages.remove(best_page)

            coverage_ratio = len(covered_tokens.intersection(query_tokens)) / len(query_tokens) if query_tokens else 0
            logger.debug(f"选择页面 {len(selected_pages)}: 边际收益={best_gain:.4f}, 总覆盖度={coverage_ratio:.4f}")

        # 最终覆盖度统计
        final_coverage = len(covered_tokens.intersection(query_tokens)) / len(query_tokens) if query_tokens else 0
        logger.info(f"贪心选择完成: {len(selected_pages)} 页面, 最终覆盖度: {final_coverage:.4f}")

        return selected_pages

    def _final_reranking(self, query: str, pages: List[dict]) -> List[dict]:
        """最终重排序"""
        if not pages:
            return []

        # 使用父类的重排序方法
        reranked_pages = self.llm_rerank(query, pages, self.reranker, len(pages))

        # 保留覆盖度信息
        for page in reranked_pages:
            # 确保必要字段存在
            if 'page' not in page and 'metadata' in page:
                page['page'] = page['metadata'].get('page_index')

        return reranked_pages

    def _compute_semantic_similarity(self, query: str, document_text: str) -> float:
        """
        计算查询和文档之间的语义相似度

        Args:
            query: 查询文本
            document_text: 文档文本

        Returns:
            float: 相似度分数 [0, 1]
        """
        try:
            # 使用缓存避免重复计算
            cache_key = f"sim_{hash(query)}_{hash(document_text[:200])}"
            if cache_key in self.embedding_cache:
                return self.embedding_cache[cache_key]

            query_clean = self._preprocess_text(query)
            doc_clean = self._preprocess_text(document_text)

            if not query_clean.strip() or not doc_clean.strip():
                return 0.0

            # 使用reranker计算相似度
            if hasattr(self.reranker, "compute_score"):
                score = self.reranker.compute_score([[query_clean, doc_clean]], normalize=True)[0]
                result = float(max(0.0, min(1.0, score)))
            else:
                # 备选方案：简单的词汇重叠
                result = self._simple_text_similarity(query_clean, doc_clean)

            # 缓存结果
            self.embedding_cache[cache_key] = result
            return result

        except Exception as e:
            logger.error(f"语义相似度计算出错: {e}")
            return 0.0

    # 重写父类方法以支持覆盖度搜索
    def search_retrieval(self, data: dict, multi_intent: bool = True, retriever=None) -> List[dict]:
        """重写父类搜索方法，使用覆盖度优化策略"""
        return self.search_retrieval_with_coverage(data, multi_intent, retriever)

# if __name__ == "__main__":
#     # Initialize DeepSearch_Beta instance with parameters
#     retriever = DeepSearch_Beta(params={
#         "embedding_topk": 15,
#         "rerank_topk": 10
#     },
#         reranker=FlagReranker(model_name_or_path="BAAI/bge-reranker-large")
#     )
#
#     # Initialize MultimodalMatcher with external configuration
#     retriever_config = RetrieverConfig(
#         model_name="vidore/colqwen2.5-v0.2",
#         processor_name="vidore/colqwen2.5-v0.1",
#         bge_model_name="BAAI/bge-large-en-v1.5",
#         device="cuda",
#         use_fp16=True,
#         batch_size=32,
#         threshold=0.4,
#         mode="mixed"
#     )
#     matcher = MultimodalMatcher(config=retriever_config)
#
#     # Load test data
#     base_dir = "/Users/chloe/Documents/Academic/AI/Project/基于Colpali的多模态检索标准框架/multimodal-RAG/DeepRAG_Multimodal/picked_LongDoc"
#     test_data_path = "/Users/chloe/Documents/Academic/AI/Project/基于Colpali的多模态检索标准框架/multimodal-RAG/DeepRAG_Multimodal/picked_LongDoc/selected_LongDocURL_public_with_subtask_category.jsonl"
#     with open(test_data_path, 'r', encoding='utf-8') as f:
#         for i, line in enumerate(f):
#             doc_data = json.loads(line)
#             documents = []
#             query = doc_data.get("question", "Provide a query here for testing.")  # Extract query from each record
#
#             if "pdf_path" in doc_data:
#                 # Handle PDF documents by converting them into pages
#                 pdf_pages = matcher._pdf_to_pages(os.path.join(base_dir, doc_data["pdf_path"]))
#                 for page_index, page_content in enumerate(pdf_pages):
#                     documents.append({
#                         "text": page_content.get("text", ""),
#                         "image": page_content.get("image", None),
#                         "metadata": {
#                             **doc_data.get("metadata", {}),
#                             "page_index": page_index + 1  # Ensure page_index is added
#                         }
#                     })
#             else:
#                 # Handle regular documents
#                 documents.append(Document(page_content=doc_data['content'], metadata=doc_data.get('metadata', {})))
#
#             data = {
#                 "query": query,  # Use the extracted query
#                 "documents": documents
#             }
#
#             # Perform search retrieval
#             results = retriever.search_retrieval(data, retriever=matcher)
#
#             # Save results to a file for each doc_data
#             results_output_path = f"retrieval_results_{i}.json"
#             with open(results_output_path, 'w', encoding='utf-8') as f_out:
#                 json.dump(results, f_out, ensure_ascii=False, indent=4)
#             logger.info(f"Results saved to {results_output_path}")
#
#             # Extract retrieved pages from results
#             retrieved_pages = set(result['metadata'].get('page_index') for result in results if 'metadata' in result)
#
#             # Calculate and print accuracy
#             calculate_accuracy(test_data_path, retrieved_pages)