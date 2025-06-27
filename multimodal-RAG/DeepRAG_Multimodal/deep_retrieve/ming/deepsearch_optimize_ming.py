import json
from copy import deepcopy
from typing import List, Dict, Annotated, Optional, Any
from DeepRAG_Multimodal.deep_retrieve.ming.agent_gpt4 import AzureGPT4Chat, create_response_format
from datetime import datetime
import sys
from DeepRAG_Multimodal.deep_retrieve.retriever_multimodal_bge import DocumentRetriever, RetrieverConfig, MultimodalMatcher
import asyncio
import concurrent.futures
from textwrap import dedent
from langchain_core.documents import Document
from FlagEmbedding import FlagReranker, FlagModel
from DeepRAG_Multimodal.deep_retrieve.deepsearch import DeepSearch_Alpha
import numpy as np
import pytesseract
from pdf2image import convert_from_path
import os
import logging

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
        output_list = []
        for score, doc in sorted(zip(rerank_scores, retrieval_list), key=lambda x: x[0], reverse=True):
            output_list.append({
                'text': doc['text'],
                'page': doc['metadata']['page_index'],
                'score': score,
                # 'image_score': doc['image_score'],
                # 'text_score': doc['text_score']
            })
        if topk is not None:
            output_list = output_list[:topk]
        return output_list

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

    def search_retrieval(self, data: dict, retriever: MultimodalMatcher):
        original_query = deepcopy(data['query'])

        # 🔥 新增：简单查询检测
        if self._is_simple_query(original_query):
            logger.info(f"🔍 检测到简单查询，使用直接检索: {original_query}")
            return self._direct_simple_retrieval(data, retriever)

        data_ori = deepcopy(data)
        embedding_topk = self.params['embedding_topk']
        rerank_topk = self.params['rerank_topk']

        # 第一步：使用LLM拆分查询意图
        intent_queries = self._split_query_intent(original_query)
        # intent_queries = self._split_query_intent_exist(original_query)
        logger.info(f"🔍 意图拆分结果: {intent_queries}")

        all_search_results = {}
        final_search_results = []
        seen_texts = set()

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
        refined_intent_queries = self._refine_query_intent(original_query, intent_queries,
                                                           json.dumps(all_search_results, ensure_ascii=False, indent=2))
        logger.info("意图细化结果：{refined_intent_queries}")
        # refined_intent_queries = self._refine_query_intent_with_knowledge_graph(
        #     original_query,
        #     intent_queries,
        #     json.dumps(all_search_results, ensure_ascii=False, indent=2)
        # )
        # logger.info(f"知识图谱精准化结果: {refined_intent_queries}")

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
        print("final_search_results: ", final_search_results)

        logger.info(f"📊 最终结果: {len(final_search_results)} 条")

        # 提取最终结果的页码
        final_results_with_pages = [
            {
                "text": doc['text'],
                "score": doc['score'],
                "page": doc['page']  # 获取页码
            }
            for doc in final_search_results
        ]
        logger.info([doc['score'] for doc in final_results_with_pages])

        return sorted(final_results_with_pages, key=lambda x: x['score'], reverse=True)[:rerank_topk]


    def _is_simple_query(self, query: str) -> bool:
        """检测是否为简单查询"""
        query_lower = query.lower().strip()

        # 简单查询的特征
        simple_patterns = [
            'what is the title of',
            'what is the name of',
            'how many',
            'which',
            'when',
            'where',
            'who is',
            'what does',
        ]

        # 检查是否匹配简单模式
        for pattern in simple_patterns:
            if pattern in query_lower:
                return True

        # 检查查询长度和复杂度
        word_count = len(query.split())
        question_count = query.count('?')

        # 短查询且只有一个问号，很可能是简单查询
        if word_count <= 12 and question_count == 1:
            return True

        return False

    def _direct_simple_retrieval(self, data: dict, retriever: MultimodalMatcher) -> List[dict]:
        """简单查询的直接检索"""
        original_query = data['query']

        # 直接检索
        retrieval_list = retriever.retrieve(original_query, data['documents'])

        # 重排序
        final_results = self.llm_rerank(original_query, retrieval_list, self.reranker, self.params['rerank_topk'])

        logger.info(f"📊 简单检索结果: {len(final_results)} 条")

        # 格式化返回结果
        return [{
            "text": doc['text'],
            "score": doc['score'],
            "page": doc.get('page', doc.get('metadata', {}).get('page_index'))
        } for doc in final_results]

    def _split_query_intent(self, query: str) -> List[str]:
        """将查询拆分为多个不同维度的意图查询"""
        SYSTEM_MESSAGE = dedent("""
            You are a professional query intent analysis expert. Your task is to analyze the user's query and decompose it into multiple sub-queries covering different dimensions of information needs.

            Please follow these rules:
            1. If the query contains multiple distinct information needs or concerns, split it into separate sub-queries.
            2. Ensure that each sub-query focuses on a different aspect or dimension to maintain diversity.
            3. Do not merely rephrase the original query—each sub-query should target a distinct informational angle.
            4. If the original query is already very specific and focuses on a single dimension, no decomposition is needed.
            5. Each sub-query should be more specific and concrete to support more accurate information retrieval.

            Return your response in JSON format, with the following structure:
            {
                "intent_queries": ["Sub-query 1", "Sub-query 2", ...]
            }
        """)

        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user",
             "content": f"Please analyze the following query and decompose it into sub-queries that cover different dimensions:\n\n{query}"}
        ]

        response_format = create_response_format({
            "intent_queries": {
                "type": "array",
                "description": "List of decomposed sub-queries",
                "items": {"type": "string"}
            }
        })

        response = AzureGPT4Chat().chat_with_message_format(
            message_list=messages,
            # response_format=response_format
        )

        try:
            result = parse_llm_response(response)
            intent_queries = result.get("intent_queries", [query])
            print("intent_queries:", intent_queries)
            return intent_queries if intent_queries else [query]
        except Exception as e:
            logger.error(f"意图拆分出错: {e}")
            return [query]

    def _split_query_intent_exist(self, query: str) -> List[str]:
        """Directly fetch expanded queries from the JSONL file if the question matches the query."""
        jsonl_path = "/Users/chloe/Documents/Academic/AI/Project/基于Colpali的多模态检索标准框架/multimodal-RAG/DeepRAG_Multimodal/deep_retrieve/query_expansion_task.jsonl"
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()  # Remove leading/trailing whitespace
                    if not line:  # Skip empty lines
                        continue
                    try:
                        # Ensure the line is a valid JSON object
                        if line.startswith("[") or line.startswith("]") or line.startswith("}"):
                            continue  # Skip invalid lines like stray brackets
                        task = json.loads(line)  # Parse JSON line
                        if task.get("question") == query:
                            return [
                                task["Understanding"]["expanded_query"],
                                task["Reasoning"]["expanded_query"],
                                task["Locating"]["expanded_query"]
                            ]
                    except json.JSONDecodeError as e:
                        logger.error(f"Skipping invalid JSON line: {line[:50]}... Error: {e}")
        except Exception as e:
            logger.error(f"Error reading expanded queries from {jsonl_path}: {e}")
        return [query]  # Fallback to the original query if no match is found

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

        # Constructing the System Prompt for Multi-Intent Synergistic Analysis
        SYSTEM_MESSAGE = dedent("""
        You are a professional expert in multi-intent synergistic analysis, specializing in identifying knowledge graph relationships and complementarities between multiple user intents.

        【Core Task】
        Do not analyze each intent in isolation. Instead, deeply analyze the relational network among intents to generate refined queries based on their synergistic effects.

        【Multi-Intent Relationship Analysis Framework】

        **Step 1: Construct Intent Relationship Network**
        Analyze the multidimensional relationships between sub-intents:

        1. Complementary Relationships:
           - Which intents offer different perspectives that collectively address the original query?
           - How does Intent A fill the information gap in Intent B?

        2. Dependency Relationships:
           - Which intents rely on the prior understanding of others?
           - Does Intent A require the foundation provided by Intent B?

        3. Hierarchical Relationships:
           - Which intents follow a general-to-specific pattern?
           - Is Intent A a broader concept or a subset of Intent B?

        4. Comparative Relationships:
           - Which intents should be compared to highlight contrasts?
           - What insights emerge from comparing Intent A and Intent B?

        5. Temporal Relationships:
           - Which intents involve a chronological sequence?
           - How does the past in Intent A affect the present in Intent B?

        **Step 2: Build Cross-Intent Knowledge Graph**
        Construct a unified knowledge graph focusing on:

        1. Shared Entity Recognition:
           - Which key entities appear in multiple intents?
           - How do these shared entities connect different intent dimensions?

        2. Cross-Intent Relation Chains:
           - What relational paths connect entities in Intent A to those in Intent B?
           - What deeper links do these paths reveal?

        3. Synergistic Information Gaps:
           - What knowledge gaps require coordination across intents?
           - Which questions cannot be answered by a single intent alone?

        **Step 3: Generate Synergistic Refined Queries**
        Based on the above intent relationship analysis, generate refined and synergistic queries:

        1. Bridging Queries – Connect key information from related intents  
        2. Comparative Queries – Emphasize differences between intents  
        3. Integrative Queries – Synthesize perspectives from multiple intents  
        4. Deepening Queries – Explore layers based on dependency  
        5. Gap-Filling Queries – Target queries to close missing information

        Current Time: {current_time}

        【Output Format】
        {{
            "intent_relationship_analysis": {{
                "complementary_pairs": [
                    {{"intent_a": "Intent 1", "intent_b": "Intent 2", "relationship": "Description of complementarity", "synergy": "Synergistic effect"}},
                    ...
                ],
                "dependency_chains": [
                    {{"prerequisite": "Prior intent", "dependent": "Dependent intent", "reason": "Dependency reasoning"}},
                    ...
                ],
                "hierarchical_structure": [
                    {{"parent": "Higher-level intent", "child": "Lower-level intent", "relationship_type": "Contains / Refines / Instantiates"}},
                    ...
                ],
                "temporal_sequence": [
                    {{"earlier": "Earlier intent", "later": "Later intent", "connection": "Temporal relation"}},
                    ...
                ]
            }},
            "unified_knowledge_graph": {{
                "shared_entities": ["Shared entity 1", "Shared entity 2", ...],
                "cross_intent_relations": [
                    {{"from_intent": "Intent 1", "to_intent": "Intent 2", "via_entity": "Linking entity", "relation_type": "Relation type"}},
                    ...
                ],
                "synergistic_gaps": ["Information gap 1 requiring coordination", "Gap 2", ...]
            }},
            "refined_intent_queries": [
                "Bridging Query – Connects key info between Intent A and Intent B",
                "Comparative Query – In-depth analysis highlighting differences",
                "Integrative Query – Holistic perspective across multiple intents",
                "Deepening Query – Hierarchical exploration based on dependencies",
                "Gap-Filling Query – Targeted search for missing knowledge"
            ]
        }}
        """)

        messages = [
            {
                "role": "system",
                "content": SYSTEM_MESSAGE.format(current_time=current_time)
            },
            {
                "role": "user",
                "content": dedent(f"""
                【Original Query】
                {original_query}

                【Decomposed Intent Queries】
                {json.dumps(intent_queries, ensure_ascii=False, indent=2)}

                【Current Retrieval Context】
                {context}

                【Refinement Requirements】
                Please optimize the decomposed intents by analyzing them through the lens of a knowledge graph.
                Your goal is to generate more specific and professional refined queries, focusing on:
                1. Entity concretization (replace vague terms with precise entity names)
                2. Temporal precision (specify time frames and key events)
                3. Relationship clarification (clearly express the links between entities)
                4. Gap supplementation (formulate targeted queries for missing information)
                """)
            }
        ]

        # Call to LLM API for multi-intent synergistic analysis
        response = AzureGPT4Chat().chat_with_message_format(
            message_list=messages,
            # response_format=self._create_multi_intent_kg_response_format()
        )

        try:
            result = parse_llm_response(response)
            refined_queries = result.get("refined_intent_queries", intent_queries)
            unified_knowledge_graph = result.get("unified_knowledge_graph", {})
            print("Unified knowledge graph:", unified_knowledge_graph)
            print("Refined intent queries:", refined_queries)
            return refined_queries if refined_queries else intent_queries
        except Exception as e:
            logger.error(f"Error in intent refinement: {e}")
            return intent_queries

    def _refine_query_intent(self, original_query: str, intent_queries: List[str], context: str) -> List[str]:
        """基于检索结果细化查询意图"""
        # SYSTEM_MESSAGE = dedent("""
        # 你是一个专业的查询意图优化专家。你的任务是基于已有的检索结果，进一步细化和优化查询意图。

        # 请遵循以下规则：
        # 1. 分析已有检索结果，识别信息缺口和需要进一步探索的方向
        # 2. 基于原始查询和已拆分的意图，生成更加精确的子查询
        # 3. 确保新的子查询能够覆盖原始查询未被满足的信息需求
        # 4. 子查询应该更加具体，包含专业术语和明确的信息需求
        # 5. 避免生成过于相似的子查询，保证多样性

        # 请以JSON格式返回，包含以下字段：
        # {
        #     "refined_intent_queries": ["细化子查询1", "细化子查询2", ...]
        # }
        # """)

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

        # messages = [
        #     {"role": "system", "content": SYSTEM_MESSAGE},
        #     {"role": "user", "content": f"""
        #     原始查询：
        #     {original_query}

        #     已拆分的意图查询：
        #     {json.dumps(intent_queries, ensure_ascii=False)}

        #     已检索到的内容：
        #     {context}

        #     请基于以上信息，细化和优化查询意图：
        #     """}
        # ]

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

    # 提取refined_intent_queries数组
    queries_pattern = r'"refined_intent_queries"\s*:\s*\[(.*?)\]'
    queries_match = re.search(queries_pattern, cleaned_text, re.DOTALL)
    if queries_match:
        query_items = re.findall(r'"([^"]+)"', queries_match.group(1))
        output_dict["refined_intent_queries"] = query_items

    # 提取intent_queries数组（如果有）
    intent_pattern = r'"intent_queries"\s*:\s*\[(.*?)\]'
    intent_match = re.search(intent_pattern, cleaned_text, re.DOTALL)
    if intent_match:
        intent_items = re.findall(r'"([^"]+)"', intent_match.group(1))
        output_dict["intent_queries"] = intent_items

    return output_dict


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