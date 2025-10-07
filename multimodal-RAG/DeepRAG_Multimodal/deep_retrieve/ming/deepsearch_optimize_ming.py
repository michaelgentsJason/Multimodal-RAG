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

        # 初步探索检索
        if multi_intent:
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
            refined_intent_queries = self._refine_query_intent(original_query, intent_queries,
                                                               json.dumps(all_search_results, ensure_ascii=False, indent=2))
            logger.info(f"意图细化结果: {refined_intent_queries}")

            # refined_intent_queries = self._refine_query_intent_with_knowledge_graph(
            #     original_query,
            #     intent_queries,
            #     json.dumps(all_search_results, ensure_ascii=False, indent=2)
            # )

            # 第四步：对细化后的意图进行第二轮检索
            if set(refined_intent_queries) != set(intent_queries):
                for intent_idx, intent_query in enumerate(refined_intent_queries):
                    logger.info(f"🔍 检索细化意图 {intent_idx + 1}/{len(refined_intent_queries)}: {intent_query}")

                    retrieval_list = retriever.retrieve(intent_query, data_ori['documents'])
                    retrieval_list = retrieval_list[:embedding_topk // len(intent_queries)]

                    # 合并结果并去重
                    for result in retrieval_list:
                        if result['text'] not in seen_texts:
                            seen_texts.add(result['text'])
                            final_search_results.append(result)

        # 第五步：对所有结果进行最终排序
        # 这里注释掉的是在0824chunk实验中的保持单模态的问题
        #logger.info(
            # f"召回的页面：{[(doc["metadata"]["page_index"], doc["image_score"], doc["text_score"]) for doc in final_search_results]}，共{len(final_search_results)}条")
        final_search_results = self.llm_rerank(original_query, final_search_results, self.reranker, rerank_topk)

        logger.info(f"📊 最终结果: {len(final_search_results)} 条")
        logger.info(f"page-score: {[(doc['page'], doc['score']) for doc in final_search_results]}")

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

        response = AzureGPT4Chat().chat_with_message_format(
            message_list=messages
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
        """基于检索结果和知识图谱细化查询意图"""

        SYSTEM_MESSAGE = dedent("""
            You are a professional query intent optimization expert with knowledge graph construction capabilities. Your task is to first build a knowledge graph from the retrieved content and decomposed queries, then use this graph to refine and enhance the user's search intent.

        **CRITICAL CONSTRAINT: All refined queries MUST stay strictly within the scope and semantic boundaries of the original query. Do NOT introduce new concepts, domains, or topics not present in the original query.**

        Please follow these steps:

        **Step 1: Knowledge Graph Construction**
        1. Extract key entities from the retrieved context that are directly related to the original query:
           - Named entities (persons, organizations, locations, dates, etc.)
           - Domain-specific concepts and terminologies that appear in both the original query and context
           - Important events, processes, or phenomena that are semantically connected to the original query
           - Technical terms that help answer the original query

        2. Identify relationships between entities, but ONLY those that are relevant to the original query:
           - Hierarchical relationships (is-a, part-of, belongs-to)
           - Functional relationships (causes, affects, enables)
           - Temporal relationships (before, after, during)
           - Spatial relationships (located-in, connected-to)
           - Semantic relationships (related-to, similar-to, opposite-to)

        **Step 2: Intent Refinement with Strict Scope Control**
        Based on the constructed knowledge graph, refine the queries following these STRICT rules:

        **MUST DO:**
        1. Keep all refined queries semantically aligned with the original query's core intent
        2. Only explore aspects, facets, or dimensions of the SAME topic from the original query
        3. Use the knowledge graph to find more specific ways to ask about the SAME information
        4. Maintain the same domain, context, and information type as the original query
        5. Generate queries that are complementary parts of answering the original question

        **MUST NOT DO:**
        1. Introduce completely new topics or domains not in the original query
        2. Shift focus to tangentially related but different questions
        3. Expand beyond the scope of what the original query is asking
        4. Generate queries about general background information unless specifically asked in the original query
        5. Create queries that could be answered independently without contributing to the original question

        **Refinement Guidelines:**
        - Create sub-queries that target different aspects of the SAME answer
        - Use entity relationships to create more precise versions of the SAME question
        - Explore different angles or perspectives on the SAME topic
        - Limit the number of refined sub-queries to a maximum of **three**

        **Validation Check:**
        Before finalizing, ask yourself: "Would answering this refined query directly contribute to answering the original query?" If not, discard it.


        Return your output in JSON format with the following structure:
        {
            "knowledge_graph": {
                "entities": [
                    {"name": "entity_name", "type": "entity_type", "description": "brief_description"},
                    ...
                ],
                "relationships": [
                    {"source": "entity1", "target": "entity2", "relation": "relationship_type", "description": "relationship_description"},
                    ...
                ]
            },
            "refined_intent_queries": ["Refined sub-query 1", "Refined sub-query 2", "Refined sub-query 3"]
        }
        """)

        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": f"""
            Original query:
            {original_query}

            Decomposed intent queries:
            {json.dumps(intent_queries, ensure_ascii=False, indent=2)}

            Retrieved context:
            {context}

            Based on the information above, please:
            1. First construct a knowledge graph from the retrieved context and decomposed queries
            2. Then refine and optimize the search intent using the knowledge graph insights
            """}
        ]

        response = AzureGPT4Chat().chat_with_message_format(
            message_list=messages
        )

        try:
            result = parse_llm_response_with_kg(response)

            # 记录知识图谱信息（用于调试和分析）
            knowledge_graph = result.get("knowledge_graph", {})
            visualize_knowledge_graph(knowledge_graph)
            logger.info(
                f"构建的知识图谱包含 {len(knowledge_graph.get('entities', []))} 个实体和 {len(knowledge_graph.get('relationships', []))} 个关系")

            # 提取细化后的查询
            refined_queries = result.get("refined_intent_queries", intent_queries)
            print("Refined intent queries:", refined_queries)

            logger.info(f"基于知识图谱细化的查询: {refined_queries}")
            return refined_queries

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


def parse_llm_response_with_kg(response_text: str) -> dict:
    """
    从LLM响应中提取包含知识图谱的JSON数据（简化版）
    """
    import re
    import json

    # 1. 清理可能的markdown代码块格式
    cleaned_text = re.sub(r'```(?:json|python)?', '', response_text)
    cleaned_text = re.sub(r'`', '', cleaned_text).strip()

    # 2. 尝试直接解析JSON
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        # 3. 尝试查找JSON内容
        json_pattern = r'\{[\s\S]*\}'
        match = re.search(json_pattern, cleaned_text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

    # 4. 回退方案：手动提取关键字段
    output_dict = {}

    # 提取knowledge_graph
    kg_pattern = r'"knowledge_graph"\s*:\s*\{([\s\S]*?)\}(?=\s*,\s*"refined_intent_queries"|\s*\})'
    kg_match = re.search(kg_pattern, cleaned_text)
    if kg_match:
        try:
            kg_json = "{" + kg_match.group(1) + "}"
            output_dict["knowledge_graph"] = json.loads(kg_json)
        except:
            # 手动提取实体和关系
            entities_pattern = r'"entities"\s*:\s*\[([\s\S]*?)\]'
            relationships_pattern = r'"relationships"\s*:\s*\[([\s\S]*?)\]'

            entities_match = re.search(entities_pattern, kg_match.group(1))
            relationships_match = re.search(relationships_pattern, kg_match.group(1))

            kg_dict = {}
            if entities_match:
                entities = []
                entity_matches = re.findall(r'\{[^}]*"name"\s*:\s*"([^"]+)"[^}]*\}', entities_match.group(1))
                for entity_name in entity_matches:
                    entities.append(
                        {"name": entity_name, "type": "unknown", "description": "", "relevance_to_original": ""})
                kg_dict["entities"] = entities

            if relationships_match:
                relationships = []
                relationship_matches = re.findall(
                    r'\{[^}]*"source"\s*:\s*"([^"]+)"[^}]*"target"\s*:\s*"([^"]+)"[^}]*"relation"\s*:\s*"([^"]+)"[^}]*\}',
                    relationships_match.group(1))
                for src, tgt, rel in relationship_matches:
                    relationships.append(
                        {"source": src, "target": tgt, "relation": rel, "description": "", "relevance_to_original": ""})
                kg_dict["relationships"] = relationships

            output_dict["knowledge_graph"] = kg_dict

        # 提取refined_queries数组
        queries_pattern = r'"refined_queries"\s*:\s*\[(.*?)\]'
        queries_match = re.search(queries_pattern, cleaned_text, re.DOTALL)
        if queries_match:
            # 更复杂的解析
            query_items = re.findall(r'\{[^}]*\}', queries_match.group(1))
            refined_queries = []
            for item in query_items:
                query_match = re.search(r'"query"\s*:\s*"([^"]+)"', item)
                sources_match = re.search(r'"sources"\s*:\s*\[(.*?)\]', item)
                if query_match:
                    query = query_match.group(1)
                    sources = []
                    if sources_match:
                        sources = re.findall(r'"([^"]+)"', sources_match.group(1))
                    refined_queries.append({"query": query, "sources": sources})
            output_dict["refined_queries"] = refined_queries

        # 提取intent_queries数组（如果有）
        intent_pattern = r'"intent_queries"\s*:\s*\[(.*?)\]'
        intent_match = re.search(intent_pattern, cleaned_text, re.DOTALL)
        if intent_match:
            intent_items = re.findall(r'"([^"]+)"', intent_match.group(1))
            output_dict["intent_queries"] = intent_items

    # 如果都没有找到，返回空结构
    if not output_dict:
        output_dict = {
            "knowledge_graph": {"entities": [], "relationships": []},
            "refined_intent_queries": []
        }

    return output_dict


def calculate_accuracy(json_file_path, retrieved_pages):
    with open(json_file_path, 'r') as f:
        logs = [json.loads(line.strip()) for line in f]

    total = 0
    correct = 0

    for log in logs:
        evidence_pages = set(log.get('evidence_pages', []))
        if evidence_pages:
            total += 1
            if evidence_pages.intersection(retrieved_pages):
                correct += 1

    accuracy = (correct / total * 100) if total > 0 else 0.0
    logger.info("\n===== Retrieval Accuracy =====")
    logger.info(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")


def visualize_knowledge_graph(knowledge_graph: dict):
    """可视化知识图谱"""
    entities = knowledge_graph.get('entities', [])
    relationships = knowledge_graph.get('relationships', [])

    print("\n" + "=" * 60)
    print(" 知识图谱可视化")
    print("=" * 60)

    # 统计信息
    entity_types = {}
    for entity in entities:
        entity_type = entity.get('type', 'Unknown')
        entity_types[entity_type] = entity_types.get(entity_type, 0) + 1

    print(f"\n 统计信息:")
    print(f"   总实体数: {len(entities)}")
    print(f"   总关系数: {len(relationships)}")
    print(f"   实体类型分布:")
    for entity_type, count in entity_types.items():
        print(f"     - {entity_type}: {count}")

    # 按类型分组显示实体
    print(f"\n 按类型分组的实体:")
    for entity_type in entity_types.keys():
        print(f"\n    {entity_type}:")
        type_entities = [e for e in entities if e.get('type') == entity_type]
        for entity in type_entities:
            name = entity.get('name', 'Unknown')
            desc = entity.get('description', '')
            if desc:
                print(f"     • {name} - {desc}")
            else:
                print(f"     • {name}")

    # 关系网络
    print(f"\n 关系网络:")
    relation_types = {}
    for rel in relationships:
        rel_type = rel.get('relation', 'Unknown')
        relation_types[rel_type] = relation_types.get(rel_type, 0) + 1

    print(f"   关系类型分布:")
    for rel_type, count in relation_types.items():
        print(f"     - {rel_type}: {count}")

    print(f"\n   详细关系:")
    for rel in relationships:
        source = rel.get('source', 'Unknown')
        target = rel.get('target', 'Unknown')
        relation = rel.get('relation', 'Unknown')
        desc = rel.get('description', '')

        print(f"      {source} --[{relation}]--> {target}")
        if desc:
            print(f"         {desc}")

    print("=" * 60)
