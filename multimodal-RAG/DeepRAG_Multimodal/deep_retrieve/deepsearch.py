import json
from copy import deepcopy
from typing import List, Dict, Annotated, Optional, Any
import sys
sys.path.append("DeepRAG_Multimodal/deep_retrieve")
from DeepRAG_Multimodal.deep_retrieve.ming.agent_gpt4 import AzureGPT4Chat, create_response_format
from datetime import datetime
from DeepRAG_Multimodal.deep_retrieve.tool_retriever_embed import EmbeddingMatcher
import asyncio
import concurrent.futures
from textwrap import dedent
from langchain_core.documents import Document
from FlagEmbedding import FlagReranker


class DeepSearch_Alpha:
    def __init__(self, max_iterations: int = 2, reranker: FlagReranker = None, params: dict = None):
        self.max_iterations = max_iterations
        self.reranker = reranker
        self.params = params

    def result_processor(self, results):
        matched_docs = []
        for doc, score in results:
            matched_docs.append({
                'text': doc.page_content,
                'score': 1 - score,
                'image_score': doc.metadata.get('image_score', 0),  # 添加图像分数
                'text_score': doc.metadata.get('text_score', 0)   # 添加文本分数
            })
        return matched_docs

    def llm_rerank(self, query, retrieval_list, reranker, topk=None):
        pairs = [[query, doc['text']] for doc in retrieval_list]
        rerank_scores = reranker.compute_score(pairs, normalize=True)
        output_list = []
        for score, doc in sorted(zip(rerank_scores, retrieval_list), key=lambda x: x[0], reverse=True):
            output_list.append({
                'text': doc['text'],
                'score': score,
                'image_score': doc['image_score'],
                'text_score': doc['text_score']
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
                'score': 1-score,
                # 复制元数据（如果有需要）
                **doc.metadata
            }
            matched_docs.append(matched_doc)
        return matched_docs

    def dynamic_retrieve_judge(
        self, 
        relevance_scores, 
        outlier_multiplier=1.5, 
        top_mean_threshold=0.7, 
        steepness_threshold=0.15
    ):
        """
        动态阈值算法，用于判断是否有足够的信息。
        
        参数:
        - relevance_scores (List[float]): 文档的相关性得分列表。
        - outlier_multiplier (float): 离群值规则的乘数。增大此值会使得离群值规则更严格。
        - top_mean_threshold (float): 头部均值规则的阈值。增大此值会使得头部均值规则更严格。
        - steepness_threshold (float): 陡度检测的阈值。减小此值会使得陡度检测更严格。
        
        返回:
        - sufficient (bool): 是否有足够的信息。
        """
        # 动态阈值算法
        scores = sorted(relevance_scores, reverse=True)
        n = len(scores)
        
        # 基础统计量
        mean = sum(scores) / n
        std_dev = (sum((x - mean)**2 for x in scores)/n)**0.5
        max_score = scores[0]
        q3 = scores[int(n*0.25)]  # 前25%分位数
        
        # 动态阈值规则（可组合调整）：
        threshold_rules = [
            max_score > (mean + outlier_multiplier * std_dev),  # 离群值规则
            (sum(scores[:int(n*0.25)])/int(n*0.25)) > top_mean_threshold if n >=4 else False,  # 头部均值规则
            (scores[0] - scores[2]) < steepness_threshold if n >=3 else False  # 陡度检测
        ]
        
        # 组合判断逻辑（满足任意两个条件）
        sufficient = sum(threshold_rules) >= 2
        
        # 增强高置信度情况
        if max_score > 0.9 and scores[0] - scores[1] > 0.2:
            sufficient = True
        
        # 增加对图像和文本分数的动态判断
        text_scores = [score['text_score'] for score in relevance_scores]
        image_scores = [score['image_score'] for score in relevance_scores]
        sufficient_text = super().dynamic_retrieve_judge(text_scores, outlier_multiplier, top_mean_threshold, steepness_threshold)
        sufficient_image = super().dynamic_retrieve_judge(image_scores, outlier_multiplier, top_mean_threshold, steepness_threshold)
        return sufficient_text or sufficient_image
    
    def _has_sufficient_info_embed(self, query: str, docs: List[Dict[str, str]]):
        """基于动态阈值的信息充分性判断"""
        def document_converter(docs):
            return [Document(page_content=doc['text']) for doc in docs]
        
        index_dict = {doc['text']: i for i, doc in enumerate(docs)}
        
        retriver = EmbeddingMatcher(document_converter=document_converter)
        relevance_docs = retriver.match_docs(query, docs=docs, result_processor=self.rerank_index_processor)
        
        if not relevance_docs:
            return False, []
        
        relevance_scores = [doc['score'] for doc in relevance_docs]
        sufficient = self.dynamic_retrieve_judge(relevance_scores)
        
        ranked_indices = [index_dict[doc['text']] for doc in relevance_docs]
        
        return sufficient, ranked_indices


    def search_retrieval(self, data: dict, multi_inetent: False, retriever: EmbeddingMatcher):
        all_search_results = []
        original_query = deepcopy(data['query'])
        query = deepcopy(data['query'])
        
        embedding_topk = self.params['embedding_topk']
        rerank_topk = self.params['rerank_topk']
        
        query_list = [query]
        ranked_indices = None

        for iteration in range(self.max_iterations):
            print(f"🔍 Retrieval Iteration {iteration + 1}/N")
            print(f"📝 Query: {data['query']}")
            
            retrieval_list = retriever.match_docs(query, result_processor=self.result_processor)
            retrieval_list = self.llm_rerank(original_query, retrieval_list, self.reranker, rerank_topk)
            
            # 使用字典的文本内容作为去重依据
            seen_texts = set()
            unique_results = []
            for result in all_search_results + retrieval_list:
                if result['text'] not in seen_texts:
                    seen_texts.add(result['text'])
                    unique_results.append(result)
            all_search_results = unique_results
            
            all_search_results = sorted(all_search_results, key=lambda x: x['score'], reverse=True)
            all_search_results = all_search_results[:rerank_topk]
            
            # 增加对图像和文本分数的动态权重组合
            for doc in retrieval_list:
                doc['combined_score'] = doc['text_score'] * self.params.get('text_weight', 0.5) + \
                                        doc['image_score'] * self.params.get('image_weight', 0.5)
            retrieval_list = sorted(retrieval_list, key=lambda x: x['combined_score'], reverse=True)

            context = self._prepare_context(all_search_results)
            print(f"📊 Context: {len(context)} characters | {len(retrieval_list)} results")
            
            # has_sufficient_info, ranked_indices = self._has_sufficient_information(original_query, context)
            # has_sufficient_info, ranked_indices = self._has_sufficient_info_embed(query, all_search_results)
            has_sufficient_info = self.dynamic_retrieve_judge([doc['score'] for doc in all_search_results])
            print([doc['score'] for doc in all_search_results])
            print(f"✅ Sufficient Information: {'Yes' if has_sufficient_info else 'No'}")
            
            if has_sufficient_info:
                print("🎯 Search completed successfully")
                return all_search_results
            
            if iteration >= self.max_iterations - 1:
                print("⚠️ Max iterations reached. Generating answer with available information.")
                return all_search_results
            
            query = self._improve_query(json.dumps(query_list), context)
            query_list.append(query)
            if not query:
                print("🔄 Refined Query: None")
                break
            print(f"🔄 Refined Query: {query}")
            print("---")

        return all_search_results

    def _prepare_context(self, search_results: List[Dict[str, str]]) -> str:
        """Prepare context from search results."""
        return '\n'.join([f"{index+1}. {result['text']}" for index, result in enumerate(search_results)])

    def _has_sufficient_information(self, query: str, context: str):
        """
        检查是否有足够信息回答查询，并返回按相关性排序的文档索引。
        
        Returns:
            tuple: (是否有足够信息, 按相关性排序的文档索引列表)
        """
        SYSTEM_MESSAGE_HAS_SUFFICIENT_INFO1 = dedent("""分析搜索结果并确定是否有足够信息回答用户的查询。
        1. 首先判断是否有足够信息回答问题
        2. 然后按照与查询的相关性对每个搜索结果进行排序，输出排序后的索引号（从0开始）,你需要对每个结果进行排序，输出全部排序结果，不要有遗漏
        
        请以JSON格式返回，包含以下字段：
        {
            "ranked_indices": [index1, index2, ...] # 按相关性从高到低排序的索引列表
            "sufficient": True/False,
        }""")
        
        SYSTEM_MESSAGE_HAS_SUFFICIENT_INFO2 = dedent("""分析搜索结果的相关性和质量。

        请评估搜索结果并完成以下任务：
        1. 评估搜索结果与查询的相关性
        2. 按照相关性对搜索结果进行排序（输出排序后的索引号，从0开始）
        3. 如果发现高度相关的信息（即使不完整），也可以考虑停止搜索
        
        请以JSON格式返回，包含以下字段：
        {
            "sufficient": True/False,  # True: 找到相关度高的结果可以停止搜索; False: 需要继续搜索
            "ranked_indices": [index1, index2, ...]  # 按相关性排序的索引列表
        }""")
        

        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE_HAS_SUFFICIENT_INFO1},
            {"role": "user", "content": f"Query: {query}\n\nSearch Results:\n{context}\n\n请对结果排序并分析信息是否足够。"}
        ]
        
        response_format = create_response_format({
            "ranked_indices": {
                "type": "array", 
                "description": "按相关性排序的文档索引列表",
                "items": {"type": "integer"},
            },
            "sufficient": {
                "type": "boolean", 
                "description": "True/False: True: 找到相关度高的结果可以停止搜索; False: 需要继续搜索",                 
            }
        })
        response = AzureGPT4Chat(model_name="gpt-4o").chat_with_message_format(message_list=messages, response_format=response_format)
        print(f"🔍 Response: {response}")
        if not response:
            return False, []
        try:
            response = json.loads(response)
        except:
            return False, []
        print(f"🔍 Response: {response}")
        return response['sufficient'], response['ranked_indices']

    def _improve_query(self, query_history: str, context: str) -> str:
        """优化搜索查询以增强多维度探索能力"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        SYSTEM_MESSAGE_IMPROVE_QUERY = dedent("""您是搜索优化专家，请按以下步骤优化查询：
        【步骤1】分析现有结果
        - 识别已覆盖的领域和缺失的信息
        - 评估结果的时间相关性和数据完整性

        current_time: {current_time}

        【步骤2】构建知识图谱
        1. 实体提取：
           - 识别关键实体：人物、地点、事件、概念等
        2. 关系识别：
           - 识别实体之间的关系：因果关系、时间顺序、关联性等
        3. 生成中间表示（必须输出JSON）：
           {{
               "entities": ["实体1", "实体2", ...],
               "relations": [
                   {{"from": "实体1", "to": "实体2", "type": "关系类型"}},
                   ...
               ]
           }}
        【步骤3】生成优化后的查询
        - 基于分析结果，生成更精准的搜索查询
        """)

        messages = [
            {
                "role": "system", 
                "content": SYSTEM_MESSAGE_IMPROVE_QUERY.format(current_time=current_time)
            },
            {
                "role": "user",
                "content": dedent(f"""\
                【历史query list】
                {query_history}

                【现有搜索结果】
                {context}

                【优化要求】
                请基于知识图谱分析，生成更精准的搜索查询：""")
            }
        ]
        # 添加响应格式约束
        response_format = create_response_format({
            "entities": {
                "type": "array",
                "description": "实体列表",
                "items": {"type": "string"},
            },
            "relations": {
                "type": "array",
                "description": "关系列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "from": {"type": "string"},
                        "to": {"type": "string"},
                        "type": {"type": "string"}
                    },
                    "required": ["from", "to", "type"],
                    "additionalProperties": False
                }
            },
            "improved_query": {
                "type": "string",
                "description": "优化后的搜索查询语句，包含具体时间范围和专业术语"
            }
        })

        response = AzureGPT4Chat().chat_with_message_format(
            message_list=messages,
            response_format=response_format
        )
        print(f"====🔍 improved_query: {response}")
        # 解析格式化的响应
        try:
            return json.loads(response).get("improved_query", json.loads(query_history)[-1])
        except Exception as e:
            print(f"==== error: {e}")
            return response or json.loads(query_history)[-1]
