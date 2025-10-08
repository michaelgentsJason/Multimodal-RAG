#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import argparse
import glob
import traceback
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any
import numpy as np
import pandas as pd
from PIL import Image
from dotenv import load_dotenv
from tqdm import tqdm
from copy import deepcopy
from textwrap import dedent
import logging
from collections import defaultdict
import networkx as nx
from dataclasses import dataclass
from pdf2image import convert_from_path
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from transformers.utils.import_utils import is_flash_attn_2_available
from transformers import AutoTokenizer, AutoModel

# 添加必要的路径
sys.path.append("/root/autodl-tmp/multimodal-RAG/multimodal-RAG/DeepRAG_Multimodal/deep_retrieve")
# 加载环境变量
load_dotenv("/root/autodl-tmp/multimodal-RAG/multimodal-RAG/DeepRAG_Multimodal/configs/.env")

# 导入必要的库
from DeepRAG_Multimodal.deep_retrieve.ming.agent_gpt4 import AzureGPT4Chat, create_response_format
from DeepRAG_Multimodal.deep_retrieve.retriever_multimodal_bge import MultimodalMatcher, RetrieverConfig
from FlagEmbedding import FlagReranker, FlagModel
from colpali_engine.models import ColPali, ColPaliProcessor
from peft import PeftModel
import torch

# 配置日志
logging.getLogger().handlers.clear()
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("kg_enhanced_longdoc.log", mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class LongDocChunk:
    """LongDoc chunk数据结构"""
    chunk_id: str
    content: str
    pages: List[int] = None  # 添加页面信息
    entities: List[str] = None
    triplets: List[List[str]] = None
    metadata: Dict = None


@dataclass
class KGTriplet:
    """知识图谱三元组"""
    head: str
    relation: str
    tail: str
    source_chunk: str
    confidence: float = 1.0

base_model_path = "/root/autodl-tmp/multimodal-RAG/hf_models/colpaligemma-3b-pt-448-base"
peft_model_dir = "/root/autodl-tmp/multimodal-RAG/hf_models/colpali-v1.3"

base_model = ColPali.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
            local_files_only=True,
)

image_model = PeftModel.from_pretrained(
            base_model,
            peft_model_dir,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            local_files_only=True,
).eval()

processor = ColPaliProcessor.from_pretrained(
            peft_model_dir,  # 或者使用base_model_path，取决于处理器存储位置
            local_files_only=True
)

bge_model_name: str = "/root/autodl-tmp/multimodal-RAG/hf_models/bge-large-en-v1.5"
text_tokenizer = AutoTokenizer.from_pretrained(bge_model_name, use_fast=True)
text_model = AutoModel.from_pretrained(bge_model_name).to("cuda")


class LongDocKnowledgeGraphProcessor:
    """适配LongDoc的知识图谱处理器 - 实现KG2RAG的核心功能"""

    def __init__(self, kg_files_prefix: str):
        """
        初始化KG处理器

        Args:
            kg_files_prefix: KG文件前缀路径 (例如: "data/output/4078345")
        """
        self.kg_files_prefix = kg_files_prefix

        # 数据存储
        self.chunk_subgraphs = {}  # chunk_id -> subgraph data
        self.entity_kg = {}  # entity -> {seq: [triplets]}
        self.chunk_triplets = {}  # chunk_id -> [triplets]
        self.entity_mapping = {}

        # 图结构
        self.entity_graph = nx.Graph()
        self.chunk_graph = nx.Graph()

        # 映射关系
        self.chunk_to_entities = defaultdict(set)
        self.entity_to_chunks = defaultdict(set)
        self.chunk_content_map = {}  # chunk_id -> content

        if self._check_kg_files_exist():
            self._load_kg_data()
            self._build_knowledge_graph()
        else:
            logger.warning(f"知识图谱文件不存在: {kg_files_prefix}")

    def _check_kg_files_exist(self) -> bool:
        """检查KG文件是否存在"""
        required_files = [
            f"{self.kg_files_prefix}_subgraphs.json",
            f"{self.kg_files_prefix}_kg.json",
            f"{self.kg_files_prefix}_triplets.json"
        ]

        for file_path in required_files:
            if not os.path.exists(file_path):
                logger.debug(f"KG文件不存在: {file_path}")
                return False
        return True

    def _load_kg_data(self):
        """加载KG数据文件"""
        logger.info("加载LongDoc知识图谱数据...")

        # 1. 加载chunk级subgraph
        subgraph_path = f"{self.kg_files_prefix}_subgraphs.json"
        if os.path.exists(subgraph_path):
            with open(subgraph_path, 'r', encoding='utf-8') as f:
                self.chunk_subgraphs = json.load(f)
            logger.info(f"✅ 加载了 {len(self.chunk_subgraphs)} 个chunk级subgraph")

        # 2. 加载实体级KG
        kg_path = f"{self.kg_files_prefix}_kg.json"
        if os.path.exists(kg_path):
            with open(kg_path, 'r', encoding='utf-8') as f:
                self.entity_kg = json.load(f)
            logger.info(f"✅ 加载了 {len(self.entity_kg)} 个实体的知识图谱")

        # 3. 加载chunk级三元组
        triplets_path = f"{self.kg_files_prefix}_triplets.json"
        if os.path.exists(triplets_path):
            with open(triplets_path, 'r', encoding='utf-8') as f:
                self.chunk_triplets = json.load(f)
            logger.info(f"✅ 加载了 {len(self.chunk_triplets)} 个chunk的三元组")

    def _build_knowledge_graph(self):
        """构建知识图谱"""
        logger.info("构建知识图谱...")

        # 从chunk subgraphs构建实体图和映射
        for chunk_id, subgraph_data in self.chunk_subgraphs.items():
            entities = subgraph_data.get('entities', [])
            triplets = subgraph_data.get('triplets', [])

            # 建立chunk-entity映射
            for entity in entities:
                self.chunk_to_entities[chunk_id].add(entity)
                self.entity_to_chunks[entity].add(chunk_id)

                # 添加实体到图中
                if not self.entity_graph.has_node(entity):
                    self.entity_graph.add_node(entity, entity_type='unknown')

            # 添加三元组关系
            for triplet in triplets:
                if len(triplet) >= 3:
                    head, relation, tail = triplet[0], triplet[1], triplet[2]
                    if head != tail:  # 避免自环
                        self.entity_graph.add_edge(head, tail,
                                                   relation=relation,
                                                   source_chunk=chunk_id)

            # 存储chunk内容（如果有）
            if 'metadata' in subgraph_data and 'content_preview' in subgraph_data['metadata']:
                self.chunk_content_map[chunk_id] = subgraph_data['metadata']['content_preview']

        # 构建chunk间关系图
        for chunk_id1, entities1 in self.chunk_to_entities.items():
            for chunk_id2, entities2 in self.chunk_to_entities.items():
                if chunk_id1 != chunk_id2:
                    overlap = len(entities1.intersection(entities2))
                    if overlap > 0:
                        self.chunk_graph.add_edge(chunk_id1, chunk_id2,
                                                  overlap=overlap, weight=overlap)

        logger.info(f"知识图谱构建完成:")
        logger.info(f"  - 实体数: {len(self.entity_graph.nodes)}")
        logger.info(f"  - 实体关系数: {len(self.entity_graph.edges)}")
        logger.info(f"  - Chunk数: {len(self.chunk_subgraphs)}")
        logger.info(f"  - Chunk关系数: {len(self.chunk_graph.edges)}")

    def extract_entities_from_text(self, text: str) -> List[str]:
        """从文本中提取实体"""
        found_entities = []
        text_lower = text.lower()

        for entity in self.entity_graph.nodes():
            if entity.lower() in text_lower:
                found_entities.append(entity)

        return found_entities

    def get_entity_subgraph(self, entities: List[str], max_hops: int = 2) -> nx.Graph:
        """获取实体子图 - 实现KG2RAG的图扩展"""
        if not entities:
            return nx.Graph()

        # 找到种子实体
        seed_entities = [e for e in entities if e in self.entity_graph.nodes()]
        if not seed_entities:
            return nx.Graph()

        # 多跳扩展
        expanded_entities = set(seed_entities)
        current_entities = set(seed_entities)

        for hop in range(max_hops):
            next_entities = set()
            for entity in current_entities:
                neighbors = list(self.entity_graph.neighbors(entity))
                next_entities.update(neighbors)

            # 过滤新实体
            new_entities = next_entities - expanded_entities
            if not new_entities:
                break

            expanded_entities.update(new_entities)
            current_entities = new_entities

            logger.debug(f"第{hop + 1}跳扩展: 新增 {len(new_entities)} 个实体")

        # 构建子图
        subgraph = self.entity_graph.subgraph(expanded_entities).copy()
        logger.info(f"实体子图构建完成: {len(subgraph.nodes)} 个实体, {len(subgraph.edges)} 条关系")

        return subgraph

    def get_chunk_expansion(self, seed_chunk_ids: List[str], max_hops: int = 2) -> Set[str]:
        """
        基于chunk级图进行扩展

        Args:
            seed_chunk_ids: 种子chunk ID列表
            max_hops: 最大跳数

        Returns:
            扩展后的chunk ID集合
        """
        if not seed_chunk_ids:
            return set()

        # 转换为字符串格式
        seed_chunk_ids = [str(cid) for cid in seed_chunk_ids]
        expanded_chunks = set(seed_chunk_ids)
        current_chunks = set(seed_chunk_ids)

        logger.info(f"🔍 开始chunk级图扩展，种子chunks: {len(seed_chunk_ids)}")

        for hop in range(max_hops):
            next_chunks = set()

            # 方法1: 基于实体重叠扩展
            seed_entities = set()
            for chunk_id in current_chunks:
                if chunk_id in self.chunk_to_entities:
                    seed_entities.update(self.chunk_to_entities[chunk_id])

            # 找到包含相同实体的其他chunks
            for entity in seed_entities:
                related_chunks = self.entity_to_chunks.get(entity, set())
                next_chunks.update(related_chunks)

            # 方法2: 基于chunk关系图扩展
            for chunk_id in current_chunks:
                if self.chunk_graph.has_node(chunk_id):
                    neighbors = list(self.chunk_graph.neighbors(chunk_id))
                    next_chunks.update(neighbors)

            # 过滤新chunks并按相关性排序
            new_chunks = next_chunks - expanded_chunks
            if not new_chunks:
                break

            # 限制扩展数量，避免过度扩展
            if len(new_chunks) > 20:  # 限制每跳最多扩展20个chunks
                # 按与种子chunks的关联强度排序
                chunk_scores = []
                for chunk_id in new_chunks:
                    score = 0
                    if chunk_id in self.chunk_to_entities:
                        chunk_entities = self.chunk_to_entities[chunk_id]
                        overlap = len(seed_entities.intersection(chunk_entities))
                        score += overlap * 0.5

                    # 基于chunk图的连接强度
                    if self.chunk_graph.has_node(chunk_id):
                        for seed_chunk in seed_chunk_ids:
                            if self.chunk_graph.has_edge(chunk_id, seed_chunk):
                                edge_data = self.chunk_graph.get_edge_data(chunk_id, seed_chunk)
                                score += edge_data.get('weight', 1) * 0.3

                    chunk_scores.append((chunk_id, score))

                # 选择得分最高的chunks
                chunk_scores.sort(key=lambda x: x[1], reverse=True)
                new_chunks = set([chunk_id for chunk_id, _ in chunk_scores[:20]])

            expanded_chunks.update(new_chunks)
            current_chunks = new_chunks

            logger.debug(f"第{hop + 1}跳扩展: 新增 {len(new_chunks)} 个chunks")

        logger.info(f"Chunk扩展完成: 从 {len(seed_chunk_ids)} 个种子扩展到 {len(expanded_chunks)} 个chunks")
        return expanded_chunks

    def organize_subgraph_context(self, subgraph: nx.Graph, query: str) -> List[Dict]:
        """基于知识图谱组织上下文 - 实现KG2RAG的上下文组织"""
        if not subgraph.nodes():
            return []

        # 计算连通分量
        connected_components = list(nx.connected_components(subgraph))
        organized_contexts = []

        for i, component in enumerate(connected_components):
            # 为每个连通分量构建最小生成树
            component_subgraph = subgraph.subgraph(component)

            if len(component_subgraph.edges()) > 0:
                # 使用边权重构建MST
                edge_weights = {}
                for u, v, data in component_subgraph.edges(data=True):
                    weight = len(self.entity_to_chunks.get(u, set())) + \
                             len(self.entity_to_chunks.get(v, set()))
                    edge_weights[(u, v)] = weight

                # 构建MST
                weighted_graph = nx.Graph()
                for (u, v), weight in edge_weights.items():
                    weighted_graph.add_edge(u, v, weight=weight)

                if len(weighted_graph.edges()) > 0:
                    mst = nx.maximum_spanning_tree(weighted_graph)
                else:
                    mst = component_subgraph
            else:
                mst = component_subgraph

            # 组织上下文段落
            context_info = {
                'component_id': i,
                'entities': list(component),
                'relationships': [],
                'related_chunks': set()
            }

            # 收集关系信息
            for u, v, data in mst.edges(data=True):
                relation = data.get('relation', 'related_to')
                context_info['relationships'].append({
                    'head': u,
                    'relation': relation,
                    'tail': v
                })

                # 收集相关文档块
                context_info['related_chunks'].update(self.entity_to_chunks.get(u, set()))
                context_info['related_chunks'].update(self.entity_to_chunks.get(v, set()))

            context_info['related_chunks'] = list(context_info['related_chunks'])
            organized_contexts.append(context_info)

        # 按相关性排序
        organized_contexts = sorted(organized_contexts,
                                    key=lambda x: len(x['related_chunks']),
                                    reverse=True)

        return organized_contexts

    def get_chunk_content(self, chunk_id: str) -> str:
        """获取chunk内容"""
        return self.chunk_content_map.get(str(chunk_id), '')


class LongDocKGRetriever:
    """LongDoc KG增强检索器"""

    def __init__(self, params: Dict):
        self.text_model = text_model
        self.reranker = FlagReranker(model_name_or_path="/root/autodl-tmp/multimodal-RAG/hf_models/bge-reranker-large")
        self.params = params
        self.kg_processor = None
        self.mm_matcher = None

    def set_kg_processor(self, kg_processor: LongDocKnowledgeGraphProcessor):
        """设置知识图谱处理器"""
        self.kg_processor = kg_processor

    def set_multimodal_matcher(self, mm_matcher: MultimodalMatcher):
        """设置多模态匹配器"""
        self.mm_matcher = mm_matcher

    def compute_text_similarity(self, query: str, documents: List[Dict]) -> List[Tuple[Dict, float]]:
        """计算文本相似度"""
        doc_texts = [doc['content'] for doc in documents]

        # 计算嵌入
        query_embedding = self.text_model.encode([query])
        doc_embeddings = self.text_model.encode(doc_texts)

        # 计算相似度
        similarities = query_embedding @ doc_embeddings.T
        similarities = similarities.flatten()

        # 返回文档和相似度的对
        results = []
        for doc, sim in zip(documents, similarities):
            results.append((doc, float(sim)))

        return sorted(results, key=lambda x: x[1], reverse=True)

    def retrieve(self, query: str, documents: List[Dict], topk: int = None) -> List[Dict]:
        """基础检索方法"""
        logger.info(f"🔍 开始单意图检索: {query}")
        if topk is None:
            topk = self.params.get('embedding_topk', 10)
        topk *= 2  # chunk检索2倍数量

        # 格式化返回结果
        results = []
        for doc in documents:
            text = doc.get("content", "")
            text_score = self._compute_text_score(query, text)
            result = {
                'chunk_id': doc['chunk_id'],
                'content': doc['content'],
                'score': text_score,
                'metadata': doc.get('metadata', {})
            }
            results.append(result)

        return results

    def _compute_text_score(self, query: str, text: str) -> float:
        if not text.strip():
            return 0.0
        query_tokens = text_tokenizer(query, padding=True, truncation=True, return_tensors="pt",
                                           max_length=512).to("cuda")
        text_tokens = text_tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512).to("cuda")
        with torch.no_grad():
            query_embedding = text_model(**query_tokens)
            text_embedding = text_model(**text_tokens)
        query_embedding = query_embedding.last_hidden_state[:, 0].cpu().numpy()
        text_embedding = text_embedding.last_hidden_state[:, 0].cpu().numpy()
        # 归一化
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        text_embedding = text_embedding / np.linalg.norm(text_embedding, axis=1, keepdims=True)
        return float(query_embedding @ text_embedding.T)

    def retrieve_chunk(self, query: str, documents: List[dict], topk: int) -> List[dict]:
        results = []
        logger.info(f"🔍 开始检索文档文本chunk")
        for doc in documents:
            try:
                # 获取文本和图像
                text = doc.get("content", "")

                text_score = self._compute_text_score(query, text)

                doc["text_score"] = text_score
                doc["metadata"] = doc.get("metadata", {})  # Ensure metadata exists
            except Exception as e:
                # 处理任何其他错误
                print(f"处理文档时出错: {str(e)}")
                traceback.print_exc()
                continue

        for doc in sorted(documents, key=lambda x: x["text_score"], reverse=True)[:topk]:
            result = {
                'chunk_id': doc['chunk_id'],
                'content': doc['content'],
                'text_score': doc['text_score'],
                'pages': doc['pages']
            }
            results.append(result)
        return results

    def _compute_image_score(self, query: str, image: Optional[Image.Image]) -> float:
        if image is None:
            return 0.0

        width, height = image.size
        if width > 0 and height > 0:
            # 创建一个全新的图像对象
            img_copy = image.copy()
        with torch.no_grad():
            query_embedding = image_model(**processor.process_queries([query]).to(image_model.device))
            image_input = processor.process_images([img_copy]).to(image_model.device)
            image_embedding = image_model(**image_input)
            return float(processor.score_multi_vector(query_embedding, image_embedding).cpu().numpy().flatten()[0])

    def image_retrieve(self, query: str, documents: List[Dict]) -> List[Dict]:
        image_scores = []
        logger.info(f"🔍 开始检索文档图像")
        for doc in documents:
            try:
                image = doc.get("image", None)
                image_score = self._compute_image_score(query, image) / 100

                doc["image_score"] = image_score
                doc["metadata"] = doc.get("metadata", {})  # Ensure metadata exists
                image_scores.append(image_score)
            except Exception as e:
                # 处理任何其他错误
                print(f"处理文档时出错: {str(e)}")
                traceback.print_exc()
                continue

        # Ensure page_index is preserved in the results
        for doc in documents:
            if "metadata" in doc and "page_index" not in doc["metadata"]:
                doc["metadata"]["page_index"] = None  # Default to None if page_index is missing

        return sorted(documents, key=lambda x: x["image_score"], reverse=True)[:self.params['embedding_topk']]

    def _extract_entities_from_retrieved_chunks(self, chunks: List[Dict]) -> List[str]:
        """从检索到的文档块中提取实体"""
        all_entities = set()

        if not self.kg_processor:
            return []

        for chunk in chunks:
            entities = self.kg_processor.extract_entities_from_text(chunk['content'])
            all_entities.update(entities)

        return list(all_entities)

    def _refine_query_with_kg_context(self, original_query: str, kg_context: List[Dict]) -> List[str]:
        """基于知识图谱上下文细化查询"""
        # 构建知识图谱上下文摘要
        kg_summary = []
        for context in kg_context[:3]:  # 只取前3个最相关的上下文
            entities_str = ', '.join(context['entities'][:5])  # 只显示前5个实体
            relations_str = '; '.join([f"{r['head']} -> {r['relation']} -> {r['tail']}"
                                       for r in context['relationships'][:3]])
            kg_summary.append(f"实体: {entities_str}; 关系: {relations_str}")

        kg_context_str = '\n'.join(kg_summary)

        SYSTEM_MESSAGE = dedent("""
        You are a knowledge graph-enhanced query refinement expert for long document retrieval.
        Your task is to refine and expand the user query by incorporating related entities and relationships 
        from the knowledge graph to improve retrieval coverage for long document chunks.

        Based on the provided knowledge graph context, you should:
        1. Expand the query to include related entities, synonyms, and alternative phrasings
        2. Consider different relationship types and entity categories from the KG
        3. Generate queries that can capture different aspects and perspectives of the topic
        4. Include entity-specific and domain-specific variations
        5. Maintain semantic coherence with the original intent
        6. Generate 1-3 refined queries to maximize retrieval recall

        Return in JSON format:
        {
            "refined_queries": ["refined query 1", "refined query 2", ...]
        }
        """)

        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": f"""
            Original Query: {original_query}

            Knowledge Graph Context (entities and relationships):
            {kg_context_str}

            Please generate refined queries that incorporate the knowledge graph information 
            to improve retrieval coverage for this long document.
            """}
        ]

        try:
            response = AzureGPT4Chat().chat_with_message_format(message_list=messages)
            result = self._parse_llm_response(response)
            refined_queries = result.get("refined_queries", [original_query])
            logger.info(f"🔍 KG上下文查询细化结果: {len(refined_queries)} 个查询")
            return refined_queries
        except Exception as e:
            logger.error(f"KG上下文查询细化出错: {e}")
            return [original_query]

    def search_with_kg_enhanced_multi_intent(self, query: str, documents: List[Dict], image_doc: List[Dict]):
        """基于知识图谱增强的多意图检索 - 完整的KG2RAG流程"""
        embedding_topk = self.params.get('embedding_topk', 15)
        rerank_topk = self.params.get('rerank_topk', 10) * 2

        chunk_to_pages = {}
        for doc in documents:
            if doc['pages']:
                chunk_to_pages[doc['chunk_id']] = doc['pages']

        logger.info(f"🔍 开始KG增强多意图检索: {query}")

        # ===== 步骤1: 语义检索获取种子chunks =====
        seed_chunks = self.retrieve_chunk(query, documents, topk=embedding_topk)
        seen_chunk_ids = set()
        all_results = []
        for result in seed_chunks:
            chunk_id = result['chunk_id']
            if chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                # 添加查询来源信息
                result['source_query'] = query
                result['refinement_step'] = 'original_query'
                all_results.append(result)
        seed_images = self.image_retrieve(query, image_doc)
        seen_image_ids = set()
        all_results_image = []
        for image in seed_images:
            page_id = image['metadata']['page_index']
            if page_id not in seen_image_ids:
                seen_image_ids.add(page_id)
                image['source_query'] = query
                image['refinement_step'] = 'original_query'
                all_results_image.append(image)

        logger.info(f"📄 语义检索获得 {len(seed_chunks)} 个种子chunks")

        # ===== 步骤2: 从种子chunks中提取实体 =====
        entities = self._extract_entities_from_retrieved_chunks(seed_chunks)
        logger.info(f"🏷️ 提取实体: {len(entities)} 个")

        # ===== 步骤3: 知识图谱指导的扩展 =====
        if entities and self.kg_processor:
            # 获取种子chunk IDs
            seed_chunk_ids = [chunk['chunk_id'] for chunk in seed_chunks]

            # chunk级扩展
            expanded_chunk_ids = self.kg_processor.get_chunk_expansion(seed_chunk_ids, max_hops=2)
            logger.info(f"🔍 chunk级扩展获得 {len(expanded_chunk_ids)} 个chunks")

            # 构建实体子图
            entity_subgraph = self.kg_processor.get_entity_subgraph(entities, max_hops=2)

            # 组织知识图谱上下文
            kg_contexts = self.kg_processor.organize_subgraph_context(entity_subgraph, query)
            logger.info(f"🕸️ 构建KG上下文: {len(kg_contexts)} 个连通分量")

            # ===== 步骤4: 基于KG上下文的查询细化 =====
            refined_queries = self._refine_query_with_kg_context(query, kg_contexts)
            logger.info(f"✨ KG查询细化: {len(refined_queries)} 个精化查询")

            # ===== 步骤5: 多意图检索与结果聚合 =====
            # 对每个细化后的查询进行检索
            for i, refined_query in enumerate(refined_queries):
                logger.info(f"🔍 检索细化查询 {i + 1}/{len(refined_queries)}: {refined_query}")

                query_results = self.retrieve_chunk(refined_query, documents,
                                              topk=embedding_topk // len(refined_queries))
                refined_images = self.image_retrieve(refined_query, image_doc)
                refined_images = refined_images[:embedding_topk // len(refined_queries)]

                for result in query_results:
                    chunk_id = result['chunk_id']
                    if chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(chunk_id)
                        # 添加查询来源信息
                        result['source_query'] = refined_query
                        result['refinement_step'] = 'kg_enhanced'
                        all_results.append(result)
                for image in refined_images:
                    page_id = image['metadata']['page_index']
                    if page_id not in seen_image_ids:
                        seen_image_ids.add(page_id)
                        image['source_query'] = query
                        image['refinement_step'] = 'original_query'
                        all_results_image.append(image)

            # 添加扩展的chunks（如果还没有被检索到）
            for chunk_id in expanded_chunk_ids:
                if chunk_id not in seen_chunk_ids:
                    content = self.kg_processor.get_chunk_content(chunk_id)
                    if content:
                        all_results.append({
                            'chunk_id': chunk_id,
                            'content': content,
                            'text_score': 0.0,
                            'pages': chunk_to_pages[chunk_id],
                            'source_query': 'kg_expansion',
                            'refinement_step': 'kg_expansion'
                        })
                        seen_chunk_ids.add(chunk_id)

            # ===== 步骤6: 基于KG上下文的重排序 =====
            final_results = self._kg_aware_rerank(query, all_results, kg_contexts, rerank_topk)
            # final_results = self._combined_rerank(query, all_results, all_results_image, rerank_topk)

        else:
            logger.warning("⚠️ 未找到实体或KG处理器不可用，回退到标准检索")
            final_results = seed_chunks[:rerank_topk]

        logger.info(f"✅ KG增强检索完成: {len(final_results)} 个最终结果")
        return final_results, all_results_image

    def _combined_rerank(self, query: str, all_results: List[Dict], all_results_image: List[Dict], topk: int) -> List[Dict]:
        """多模态融合重排序"""
        logger.info("🎨 开始多模态融合重排序...")

        if not all_results:
            logger.warning("无文本检索结果，返回空列表")
            return []

        # 步骤1: 将chunks按page聚合
        logger.info("📄 聚合chunks到pages...")
        page_chunks = defaultdict(list)

        for chunk in all_results:
            chunk_id = chunk['chunk_id']
            pages = chunk.get('pages', [])

            if not pages:
                # 如果chunk没有页面信息，跳过
                logger.warning(f"Chunk {chunk_id} 缺少页面信息，跳过")
                continue

            # 将chunk分配到其所属的所有页面
            for page_id in pages:
                page_chunks[page_id].append(chunk)

        # 步骤2: 处理图像检索结果
        logger.info(f"🖼️ 处理图像检索结果: {len(all_results_image)} 个")
        image_pages = {}

        for image_result in all_results_image:
            page_id = image_result['metadata']['page_index']
            # 保留最高分的图像结果
            if page_id not in image_pages or image_result.get('image_score', 0) > image_pages[page_id].get(
                    'image_score', 0):
                image_pages[page_id] = image_result

        # 步骤3: 取并集，为每个页面计算分数
        logger.info("🔗 合并文本和图像结果...")
        all_page_ids = set(page_chunks.keys()) | set(image_pages.keys())

        page_results = []
        for page_id in all_page_ids:
            page_data = {'page': page_id}

            # 处理文本分数
            if page_id in page_chunks:
                chunks = page_chunks[page_id]
                # 聚合页面文本
                page_text = "\n".join([chunk['content'] for chunk in chunks])
                # 取最高的chunk分数作为页面文本分数
                max_text_score = max([chunk.get('score', 0) for chunk in chunks])

                page_data.update({
                    'text': page_text,
                    'text_score': max_text_score,
                    'chunks': chunks,
                    'chunk_count': len(chunks)
                })
            else:
                # 如果没有文本chunks，设置默认值
                page_data.update({
                    'text': "None",
                    'text_score': 0.0,
                    'chunks': [],
                    'chunk_count': 0
                })

            # 处理图像分数
            if page_id in image_pages:
                page_data['image_score'] = image_pages[page_id].get('image_score', 0.0)
            else:
                # 如果图像检索中没有找到该页面，设为0
                page_data['image_score'] = 0.0

            page_results.append(page_data)

        # 步骤4: 使用reranker重新计算文本分数（如果有reranker）
        if self.reranker:
            logger.info("📝 使用reranker重新计算文本分数...")
            page_texts = [page['text'] for page in page_results]
            pairs = [[query, text] for text in page_texts]

            try:
                rerank_scores = self.reranker.compute_score(pairs, normalize=True)
                for i, page_data in enumerate(page_results):
                    if page_data['text'] is not None:
                        page_data['text_score'] = float(rerank_scores[i])
            except Exception as e:
                logger.error(f"Reranker计算失败: {e}")
                # 使用原始分数

        # 步骤5: 动态权重计算
        alpha = self.get_visual_weight_llm(query)
        logger.info(f"⚖️ 动态视觉权重: {alpha}")

        # 步骤6: 多模态分数融合
        final_results = []
        for page_data in page_results:
            text_score = page_data['text_score']
            image_score = page_data['image_score']

            # 使用与deepsearch_optimize_ming_fusion.py相同的融合公式
            combo_score = (1 - alpha) * text_score + alpha * image_score

            result = {
                'text': page_data['text'],
                'page': page_data['page'],
                'score': combo_score,
                'text_score': text_score,
                'image_score': image_score,
                'chunk_count': page_data['chunk_count'],
                'chunks': page_data['chunks']  # 保留原始chunks信息
            }
            final_results.append(result)

        # 步骤7: 排序并截取top-k
        final_results.sort(key=lambda x: x['score'], reverse=True)
        final_results = final_results[:topk]

        logger.info(f"✅ 多模态融合完成: {len(final_results)} 个结果")

        return final_results

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

    def _kg_aware_rerank(self, query: str, results: List[Dict], kg_contexts: List[Dict], topk: int) -> List[Dict]:
        """基于知识图谱的感知重排序"""
        if not results:
            return results[:topk]

        try:
            # 标准重排序（如果有reranker）
            if self.reranker:
                pairs = [[query, result['content']] for result in results]
                rerank_scores = self.reranker.compute_score(pairs, normalize=True)

                for i, result in enumerate(results):
                    result['base_score'] = float(rerank_scores[i])
            else:
                # 使用原有分数
                for result in results:
                    result['base_score'] = result.get('score', 0.0)

            # 基于KG上下文的分数增强
            for result in results:
                base_score = result['base_score']

                # KG增强分数
                kg_boost = 0.0
                chunk_id = result['chunk_id']

                # 检查chunk是否在KG中
                if self.kg_processor and str(chunk_id) in self.kg_processor.chunk_subgraphs:
                    subgraph_data = self.kg_processor.chunk_subgraphs[str(chunk_id)]

                    # 基于实体数量加分
                    entity_count = subgraph_data.get('entity_count', 0)
                    kg_boost += min(entity_count * 0.02, 0.1)  # 最多加0.1分

                    # 基于三元组数量加分
                    triplet_count = subgraph_data.get('triplet_count', 0)
                    kg_boost += min(triplet_count * 0.01, 0.05)  # 最多加0.05分

                    # 基于与KG上下文的重叠加分
                    chunk_entities = set(subgraph_data.get('entities', []))
                    for context in kg_contexts[:2]:  # 只考虑前2个最相关的上下文
                        context_entities = set(context['entities'])
                        entity_overlap = len(chunk_entities.intersection(context_entities))
                        if entity_overlap > 0:
                            kg_boost += entity_overlap * 0.05  # 每个重叠实体加0.05分

                # 综合分数
                final_score = base_score + kg_boost
                # result['score'] = final_score
                result['score'] = base_score
                result['kg_boost'] = kg_boost
                result['entity_count'] = subgraph_data.get('entity_count', 0) if self.kg_processor and str(
                    chunk_id) in self.kg_processor.chunk_subgraphs else 0
                result['triplet_count'] = subgraph_data.get('triplet_count', 0) if self.kg_processor and str(
                    chunk_id) in self.kg_processor.chunk_subgraphs else 0

            # 按综合分数排序
            return sorted(results, key=lambda x: x['score'], reverse=True)[:topk]

        except Exception as e:
            logger.error(f"KG感知重排序出错: {str(e)}")
            return sorted(results, key=lambda x: x.get('base_score', 0), reverse=True)[:topk]

    def _parse_llm_response(self, response_text: str) -> dict:
        """解析LLM响应"""
        import re

        # 清理响应文本
        cleaned_text = re.sub(r'```(?:json|python)?', '', response_text)
        cleaned_text = re.sub(r'`', '', cleaned_text).strip()

        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            # 查找JSON内容
            json_pattern = r'\{[\s\S]*\}'
            match = re.search(json_pattern, cleaned_text)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass

        # 回退方案：手动提取
        output_dict = {}

        # 提取refined_queries
        refined_pattern = r'"refined_queries"\s*:\s*\[(.*?)\]'
        refined_match = re.search(refined_pattern, cleaned_text, re.DOTALL)
        if refined_match:
            refined_items = re.findall(r'"([^"]+)"', refined_match.group(1))
            output_dict["refined_queries"] = refined_items

        return output_dict


class LongDocDataProcessor:
    """LongDoc数据集处理器"""

    def __init__(self, data_path: str, kg_files_prefix: str, args: argparse.Namespace = None):
        self.data_path = data_path
        self.kg_files_prefix = kg_files_prefix
        self.args = args  # 添加args参数
        self.kg_processor = LongDocKnowledgeGraphProcessor(kg_files_prefix)
        self.chunk_data = self._load_chunk_data()

    def _load_chunk_data(self) -> List[LongDocChunk]:
        """加载LongDoc chunk数据"""
        logger.info(f"加载LongDoc数据: {self.data_path}")

        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 转换为LongDocChunk对象
            chunks = []
            for item in data:
                chunk = LongDocChunk(
                    chunk_id=str(item.get('chunk_id', '')),
                    content=item.get('content', ''),
                    pages=item.get('pages', []),  # 添加页面信息
                    metadata=item.get('metadata', {})
                )
                chunks.append(chunk)

            logger.info(f"数据载入完成，chunk数: {len(chunks)}")
            return chunks

        except Exception as e:
            logger.error(f"加载数据失败: {str(e)}")
            return []

    def prepare_documents_for_retrieval(self) -> List[Dict]:
        """准备用于检索的文档格式"""
        documents = []

        for chunk in self.chunk_data:
            if chunk.content.strip():  # 过滤空内容
                document = {
                    'chunk_id': chunk.chunk_id,
                    'content': chunk.content,
                    'pages': chunk.pages or [],  # 添加页面信息
                    'metadata': chunk.metadata or {}
                }
                documents.append(document)

        logger.info(f"准备了 {len(documents)} 个检索文档")

        # 输出一些样例来验证页面信息
        debug_mode = self.args.debug if self.args else False
        if documents and debug_mode:
            logger.info("📄 文档样例:")
            for i, doc in enumerate(documents[:3]):
                logger.info(
                    f"  文档 {i + 1}: chunk_{doc['chunk_id']} | 页面: {doc['pages']} | 内容长度: {len(doc['content'])}")

        return documents

    def get_chunk_by_id(self, chunk_id: str) -> Optional[LongDocChunk]:
        """根据ID获取chunk"""
        for chunk in self.chunk_data:
            if chunk.chunk_id == str(chunk_id):
                return chunk
        return None


@dataclass
class GroundTruthItem:
    """Ground truth数据结构"""
    query: str
    relevant_chunk_ids: List[str] = None  # 相关的chunk IDs
    relevant_pages: List[int] = None  # 相关的页面
    metadata: Dict = None


class LongDocEvaluator:
    """LongDoc评估器 - 支持ground truth评估"""

    def __init__(self, chunk_data: List[LongDocChunk] = None):
        self.chunk_data = chunk_data or []
        # 建立chunk_id到页面的映射
        self.chunk_to_pages = {}
        self.page_to_chunks = defaultdict(list)
        self.params = {
            "embedding_topk": 15,
            "rerank_topk": 10
        }

        for chunk in self.chunk_data:
            if chunk.pages:
                self.chunk_to_pages[chunk.chunk_id] = chunk.pages
                for page in chunk.pages:
                    self.page_to_chunks[page].append(chunk.chunk_id)

        logger.info(f"📊 页面映射统计:")
        logger.info(f"  - 总chunk数: {len(self.chunk_data)}")
        logger.info(f"  - 有页面信息的chunk数: {len(self.chunk_to_pages)}")
        logger.info(f"  - 总页面数: {len(self.page_to_chunks)}")

        # 显示一些页面映射样例
        if self.page_to_chunks:
            sample_pages = list(self.page_to_chunks.keys())[:5]
            logger.info(f"  - 页面映射样例:")
            for page in sample_pages:
                chunks = self.page_to_chunks[page]
                logger.info(f"    页面 {page}: {len(chunks)} 个chunks {chunks[:3]}{'...' if len(chunks) > 3 else ''}")

    def evaluate_retrieval_with_ground_truth(self, query: str, retrieved_results: List[Dict],
                                             ground_truth: GroundTruthItem) -> Dict:
        """基于ground truth评估检索结果"""
        if not retrieved_results:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'retrieved_count': 0,
                'relevant_count': len(ground_truth.relevant_chunk_ids) if ground_truth.relevant_chunk_ids else 0,
                'true_positives': 0,
                'page_precision': 0.0,
                'page_recall': 0.0,
                'page_f1': 0.0
            }

        retrieved_chunk_ids = set([str(result['chunk_id']) for result in retrieved_results])

        # 基于chunk ID的评估
        chunk_metrics = {}
        if ground_truth.relevant_chunk_ids:
            relevant_chunk_set = set([str(cid) for cid in ground_truth.relevant_chunk_ids])
            true_positives = len(retrieved_chunk_ids.intersection(relevant_chunk_set))

            precision = true_positives / len(retrieved_chunk_ids) if retrieved_chunk_ids else 0.0
            recall = true_positives / len(relevant_chunk_set) if relevant_chunk_set else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            chunk_metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'true_positives': true_positives,
                'retrieved_count': len(retrieved_chunk_ids),
                'relevant_count': len(relevant_chunk_set),
                'retrieved_chunks': list(retrieved_chunk_ids),
                'missed_chunks': list(relevant_chunk_set - retrieved_chunk_ids)
            }

        # 基于页面的评估
        page_metrics = {}
        if ground_truth.relevant_pages:
            # 获取检索到的页面
            retrieved_pages = set()
            for chunk_id in retrieved_chunk_ids:
                    if chunk_id in self.chunk_to_pages:
                        retrieved_pages.update(self.chunk_to_pages[chunk_id])
                        if len(retrieved_pages) >= self.params.get('rerank_topk', 10):
                            break

            relevant_page_set = set(ground_truth.relevant_pages)
            page_true_positives = len(retrieved_pages.intersection(relevant_page_set))

            page_precision = page_true_positives / len(retrieved_pages) if retrieved_pages else 0.0
            page_recall = page_true_positives / len(relevant_page_set) if relevant_page_set else 0.0
            page_f1 = 2 * page_precision * page_recall / (page_precision + page_recall) if (
                                                                                                   page_precision + page_recall) > 0 else 0.0

            page_metrics = {
                'page_precision': page_precision,
                'page_recall': page_recall,
                'page_f1': page_f1,
                'page_true_positives': page_true_positives,
                'retrieved_pages': list(retrieved_pages),
                'relevant_pages': list(relevant_page_set),
                'missed_pages': list(relevant_page_set - retrieved_pages)
            }

        # 合并两种评估结果
        # combined_metrics = {**chunk_metrics, **page_metrics}
        combined_metrics = page_metrics

        # 添加质量指标
        quality_metrics = self.evaluate_retrieval_quality(query, retrieved_results)
        combined_metrics.update(quality_metrics)

        return combined_metrics

    def evaluate_retrieval_quality(self, query: str, retrieved_results: List[Dict]) -> Dict:
        """评估检索质量（无ground truth版本）"""
        if not retrieved_results:
            return {
                'quality_retrieved_count': 0,
                'quality_avg_score': 0.0,
                'quality_kg_boost_avg': 0.0,
                'quality_entity_coverage': 0.0,
                'quality_triplet_coverage': 0.0
            }

        # 基本统计
        retrieved_count = len(retrieved_results)
        avg_score = np.mean([result.get('score', 0) for result in retrieved_results])
        kg_boost_avg = np.mean([result.get('kg_boost', 0) for result in retrieved_results])
        entity_coverage = np.mean([result.get('entity_count', 0) for result in retrieved_results])
        triplet_coverage = np.mean([result.get('triplet_count', 0) for result in retrieved_results])

        # 查询细化效果统计
        refinement_sources = defaultdict(int)
        for result in retrieved_results:
            source = result.get('refinement_step', 'original')
            refinement_sources[source] += 1

        return {
            'quality_retrieved_count': retrieved_count,
            'quality_avg_score': avg_score,
            'quality_kg_boost_avg': kg_boost_avg,
            'quality_entity_coverage': entity_coverage,
            'quality_triplet_coverage': triplet_coverage,
            'quality_refinement_sources': dict(refinement_sources),
            'quality_score_distribution': {
                'min': min([r.get('score', 0) for r in retrieved_results]),
                'max': max([r.get('score', 0) for r in retrieved_results]),
                'std': np.std([r.get('score', 0) for r in retrieved_results])
            }
        }

    def aggregate_chunks_to_pages_union(self, chunk_results: List[Dict]) -> List[int]:
        """将chunk检索结果聚合到页面级别 - Union策略"""
        if not chunk_results:
            return []

        page_scores = {}  # {page_id: max_score}

        for result in chunk_results:
            chunk_score = result.get('score', 0)
            chunk_pages = result.get('pages', [])

            if not chunk_pages:
                logger.warning(f"未找到页面信息: {result.keys()}")
                continue

            for page in chunk_pages:
                if page not in page_scores:
                    page_scores[page] = chunk_score
                else:
                    page_scores[page] = max(page_scores[page], chunk_score)

        sorted_pages = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)
        return [page for page, score in sorted_pages]

    def evaluate_retrieval_with_image(self, query: str, retrieved_results: List[Dict], retrieved_images: List[Dict], ground_truth: GroundTruthItem, image_weight: float) -> Dict:
        """基于ground truth评估检索结果"""
        chunk_pages = self.aggregate_chunks_to_pages_union(retrieved_results)

        # 基于页面的评估
        page_metrics = {}
        if ground_truth.relevant_pages:
            # 获取检索到的页面
            retrieved_pages = set()
            retrieved_pages.update(chunk_pages[:int(self.params.get('rerank_topk', 10)*(1-image_weight))])
            for image in sorted(retrieved_images, key=lambda x: x['image_score'], reverse=True)[:int(self.params.get('rerank_topk', 10)*image_weight)]:
                retrieved_pages.add(image['metadata']['page_index'])

            relevant_page_set = set(ground_truth.relevant_pages)
            page_true_positives = len(retrieved_pages.intersection(relevant_page_set))

            page_precision = page_true_positives / len(retrieved_pages) if retrieved_pages else 0.0
            page_recall = page_true_positives / len(relevant_page_set) if relevant_page_set else 0.0
            page_f1 = 2 * page_precision * page_recall / (page_precision + page_recall) if (
                                                                                                   page_precision + page_recall) > 0 else 0.0

            page_metrics = {
                'page_precision': page_precision,
                'page_recall': page_recall,
                'page_f1': page_f1,
                'page_true_positives': page_true_positives,
                'retrieved_pages': list(retrieved_pages),
                'relevant_pages': list(relevant_page_set),
                'missed_pages': list(relevant_page_set - retrieved_pages)
            }
        return page_metrics

    def evaluate_retrieval(self, query: str, retrieved_results: List[Dict], ground_truth) -> Dict:
        """评估检索结果 - 只基于页面级别"""
        if not retrieved_results:
            logger.warning("检索结果为空")
            return {
                'page_precision': 0.0,
                'page_recall': 0.0,
                'page_f1': 0.0,
                'true_positives': 0,
                'retrieved_count': 0,
                'relevant_count': len(
                    ground_truth.relevant_pages) if ground_truth and ground_truth.relevant_pages else 0,
                'retrieved_pages': [],
                'relevant_pages': [],
                'missed_pages': []
            }

        # 从检索结果中提取页面信息
        retrieved_pages = set()
        for result in retrieved_results:
            page = result.get('page')
            if page is not None:
                retrieved_pages.add(page)

        # 基于页面的评估
        if ground_truth and ground_truth.relevant_pages:
            relevant_page_set = set(ground_truth.relevant_pages)
            true_positives = len(retrieved_pages.intersection(relevant_page_set))

            precision = true_positives / len(retrieved_pages) if retrieved_pages else 0.0
            recall = true_positives / len(relevant_page_set) if relevant_page_set else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'true_positives': true_positives,
                'retrieved_count': len(retrieved_pages),
                'relevant_count': len(relevant_page_set),
                'retrieved_pages': list(retrieved_pages),
                'relevant_pages': list(relevant_page_set),
                'missed_pages': list(relevant_page_set - retrieved_pages)
            }
        else:
            # 没有ground truth的情况
            metrics = {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'true_positives': 0,
                'retrieved_count': len(retrieved_pages),
                'relevant_count': 0,
                'retrieved_pages': list(retrieved_pages),
                'relevant_pages': [],
                'missed_pages': []
            }

        # 添加质量指标
        quality_metrics = self.evaluate_retrieval_quality(query, retrieved_results)
        metrics.update(quality_metrics)

        return metrics

class LongDocKGTester:
    """LongDoc KG增强测试器 - 支持ground truth评估"""

    def __init__(self, data_path: str, kg_files_prefix: str, args: argparse.Namespace = None,
                 ground_truth_path: str = None):
        self.data_path = data_path
        self.kg_files_prefix = kg_files_prefix
        self.args = args
        self.ground_truth_path = ground_truth_path

        # 初始化组件
        self.data_processor = LongDocDataProcessor(data_path, kg_files_prefix, args)  # 传递args
        self.evaluator = LongDocEvaluator(self.data_processor.chunk_data)  # 传递chunk数据

        # 加载ground truth数据
        self.ground_truth_data = self._load_ground_truth() if ground_truth_path else {}

        # 初始化模型
        self._setup_models()

    def _load_ground_truth(self) -> Dict[str, GroundTruthItem]:
        """加载ground truth数据 - 适配LongDoc格式，增强错误处理"""
        if not self.ground_truth_path or not os.path.exists(self.ground_truth_path):
            logger.warning(f"Ground truth文件不存在: {self.ground_truth_path}")
            return {}

        try:
            with open(self.ground_truth_path, 'r', encoding='utf-8') as f:
                # 尝试加载JSONL格式
                if self.ground_truth_path.endswith('.jsonl'):
                    gt_data = []
                    for line_num, line in enumerate(f, 1):
                        if line.strip():
                            try:
                                gt_data.append(json.loads(line.strip()))
                            except json.JSONDecodeError as e:
                                logger.error(f"JSONL第{line_num}行解析失败: {e}")
                                continue
                else:
                    # JSON格式 - 可能是单个对象或数组
                    gt_data = json.load(f)

            logger.info(f"原始加载的数据类型: {type(gt_data)}")

            # 检查数据格式
            if isinstance(gt_data, str):
                logger.error("加载的数据是字符串，可能文件格式错误")
                return {}

            # 处理单个字典对象的情况（LongDoc格式常见）
            if isinstance(gt_data, dict):
                logger.info("检测到单个JSON对象，转换为列表格式")
                gt_data = [gt_data]
            elif isinstance(gt_data, list):
                logger.info(f"检测到JSON数组，包含 {len(gt_data)} 个项目")
            else:
                logger.error(f"不支持的数据格式: {type(gt_data)}")
                return {}

            ground_truth = {}
            processed_count = 0

            for idx, item in enumerate(gt_data):
                try:
                    # 检查item类型
                    if not isinstance(item, dict):
                        logger.warning(f"第{idx}项不是字典格式，跳过: {type(item)}")
                        continue

                    # 适配LongDoc数据格式
                    question = item.get('question', '')
                    evidence_pages = item.get('evidence_pages', [])
                    doc_no = item.get('doc_no', '')

                    if question and evidence_pages:
                        # 根据页面找到对应的chunk IDs
                        relevant_chunk_ids = []
                        for page in evidence_pages:
                            chunks_in_page = self.evaluator.page_to_chunks.get(page, [])
                            relevant_chunk_ids.extend(chunks_in_page)

                        ground_truth[question] = GroundTruthItem(
                            query=question,
                            relevant_chunk_ids=list(set(relevant_chunk_ids)),  # 去重
                            relevant_pages=evidence_pages,
                            metadata={
                                'doc_no': doc_no,
                                'question_id': item.get('question_id', ''),
                                'question_type': item.get('question_type', ''),
                                'task_tag': item.get('task_tag', ''),
                                'answer': item.get('answer', ''),
                                'detailed_evidences': item.get('detailed_evidences', '')
                            }
                        )
                        processed_count += 1

                        logger.info(f"成功解析数据项 {idx + 1}:")
                        logger.info(f"  问题: {question}")
                        logger.info(f"  证据页面: {evidence_pages}")
                        logger.info(f"  对应chunks: {relevant_chunk_ids}")
                        logger.info(f"  答案: {item.get('answer', 'N/A')}")
                    else:
                        logger.warning(f"第{idx}项缺少必要字段:")
                        logger.warning(f"  question: {'✓' if question else '✗'}")
                        logger.warning(f"  evidence_pages: {'✓' if evidence_pages else '✗'}")

                except Exception as e:
                    logger.error(f"处理第{idx}项时出错: {e}")
                    logger.error(f"问题数据: {item}")
                    continue

            logger.info(f"成功加载了 {len(ground_truth)} 个ground truth项")
            return ground_truth

        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            logger.error("请检查文件格式是否正确")
            return {}
        except Exception as e:
            logger.error(f"加载ground truth失败: {str(e)}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            return {}

    def _setup_models(self):
        """初始化模型"""
        logger.info("初始化检索模型...")

        # 参数配置
        params = {
            "embedding_topk": self.args.embedding_topk,
            "rerank_topk": self.args.rerank_topk
        }

        # 原始检索器（作为基线）
        self.baseline_retriever = LongDocKGRetriever(params)

        # KG增强检索器
        self.kg_enhanced_retriever = LongDocKGRetriever(params)
        self.kg_enhanced_retriever.set_kg_processor(self.data_processor.kg_processor)

        logger.info("模型初始化完成")

    def extract_queries_from_ground_truth(self) -> List[str]:
        """从ground truth文件中提取查询"""
        return list(self.ground_truth_data.keys())

    def test_kg_enhancement_with_ground_truth(self):
        """使用ground truth进行KG增强检索测试"""
        if not self.ground_truth_data:
            logger.error("没有ground truth数据，无法进行测试")
            return []

        queries = self.extract_queries_from_ground_truth()
        logger.info(f"从ground truth中提取了 {len(queries)} 个查询")

        return self.test_kg_enhancement_with_queries(queries)

    def test_kg_enhancement_with_queries(self, queries: List[str]):
        """测试KG增强检索效果 - 支持ground truth评估"""
        logger.info("开始LongDoc KG增强检索测试...")

        # 准备文档
        documents = self.data_processor.prepare_documents_for_retrieval()
        logger.info(f"准备了 {len(documents)} 个文档")

        results = []

        for idx, query in enumerate(queries[:self.args.sample_size] if self.args.sample_size > 0 else queries):
            try:
                logger.info(f"\n查询 {idx + 1}/{len(queries)}: {query}")

                # 获取ground truth（如果存在）
                ground_truth = self.ground_truth_data.get(query)
                if ground_truth:
                    logger.info(f"找到ground truth: {len(ground_truth.relevant_chunk_ids or [])} 个相关chunks, "
                                f"{len(ground_truth.relevant_pages or [])} 个相关页面")

                # ===== 基线检索 =====
                baseline_start = time.time()
                baseline_results = self.baseline_retriever.retrieve(
                    query, documents, topk=self.args.rerank_topk
                )
                baseline_time = time.time() - baseline_start

                # ===== KG增强检索 =====
                kg_start = time.time()
                kg_results = self.kg_enhanced_retriever.search_with_kg_enhanced_multi_intent(query, documents)

                kg_time = time.time() - kg_start

                # 评估结果
                if ground_truth:
                    # 使用ground truth评估
                    baseline_eval = self.evaluator.evaluate_retrieval_with_ground_truth(
                        query, baseline_results, ground_truth)
                    kg_eval = self.evaluator.evaluate_retrieval_with_ground_truth(
                        query, kg_results, ground_truth)
                else:
                    # 使用质量评估
                    baseline_eval = self.evaluator.evaluate_retrieval_quality(query, baseline_results)
                    kg_eval = self.evaluator.evaluate_retrieval_quality(query, kg_results)

                # 记录结果
                result = {
                    'query_id': idx,
                    'query': query,
                    'has_ground_truth': ground_truth is not None,
                    'ground_truth': {
                        'relevant_chunk_ids': ground_truth.relevant_chunk_ids if ground_truth else [],
                        'relevant_pages': ground_truth.relevant_pages if ground_truth else []
                    } if ground_truth else None,
                    'baseline': {
                        **baseline_eval,
                        'retrieval_time': baseline_time,
                        'retrieved_results': [{'chunk_id': r['chunk_id'], 'score': r['score']}
                                              for r in baseline_results]
                    },
                    'kg_enhanced': {
                        **kg_eval,
                        'retrieval_time': kg_time,
                        'retrieved_results': [{'chunk_id': r['chunk_id'], 'score': r.get('score', 0.0),
                                               'kg_boost': r.get('kg_boost', 0.0),
                                               'entity_count': r.get('entity_count', 0),
                                               'source_query': r.get('source_query', 'original')}
                                              for r in kg_results]
                    }
                }

                results.append(result)

                # ===== 详细调试信息 =====
                logger.info("\n" + "=" * 50)
                logger.info("🔍 检索结果详细分析")
                logger.info("=" * 50)

                if ground_truth:
                    logger.info(f"📋 Ground Truth信息:")
                    logger.info(f"  证据页面: {ground_truth.relevant_pages}")
                    logger.info(f"  证据chunks: {ground_truth.relevant_chunk_ids}")

                # 基线检索结果
                logger.info(f"\n📄 基线检索结果 ({len(baseline_results)} 个):")
                for i, result in enumerate(baseline_results, 1):
                    chunk_id = result['chunk_id']
                    chunk_pages = self.evaluator.chunk_to_pages.get(chunk_id, [])
                    is_correct = str(chunk_id) in (ground_truth.relevant_chunk_ids if ground_truth else [])
                    logger.info(
                        f"  {i}. chunk_{chunk_id} | 页面: {chunk_pages} | 分数: {result['score']:.3f} | {'✓' if is_correct else '✗'}")

                # KG增强检索结果
                logger.info(f"\n🚀 KG增强检索结果 ({len(kg_results)} 个):")
                for i, result in enumerate(kg_results, 1):
                    chunk_id = result['chunk_id']
                    chunk_pages = self.evaluator.chunk_to_pages.get(chunk_id, [])
                    is_correct = str(chunk_id) in (ground_truth.relevant_chunk_ids if ground_truth else [])
                    kg_boost = result.get('kg_boost', 0)
                    source_query = result.get('source_query', 'original')
                    logger.info(
                        f"  {i}. chunk_{chunk_id} | 页面: {chunk_pages} | 分数: {result.get('score', 0):.3f} | KG增强: {kg_boost:.3f} | 来源: {source_query} | {'✓' if is_correct else '✗'}")

                # 页面到chunk映射检查
                if ground_truth:
                    logger.info(f"\n🗺️ 页面到chunk映射检查:")
                    for page in ground_truth.relevant_pages:
                        chunks_in_page = self.evaluator.page_to_chunks.get(page, [])
                        logger.info(f"  页面 {page}: chunks {chunks_in_page}")

                # 打印评估结果
                if ground_truth:
                    logger.info(f"\n📊 评估指标:")
                    logger.info(f"  基线检索 - P: {baseline_eval.get('precision', 0):.3f}, "
                                f"R: {baseline_eval.get('recall', 0):.3f}, "
                                f"F1: {baseline_eval.get('f1', 0):.3f}")
                    logger.info(f"  KG增强   - P: {kg_eval.get('precision', 0):.3f}, "
                                f"R: {kg_eval.get('recall', 0):.3f}, "
                                f"F1: {kg_eval.get('f1', 0):.3f}")

                    # 页面级评估
                    if 'page_precision' in baseline_eval:
                        logger.info(f"  页面基线 - P: {baseline_eval.get('page_precision', 0):.3f}, "
                                    f"R: {baseline_eval.get('page_recall', 0):.3f}, "
                                    f"F1: {baseline_eval.get('page_f1', 0):.3f}")
                        logger.info(f"  页面KG   - P: {kg_eval.get('page_precision', 0):.3f}, "
                                    f"R: {kg_eval.get('page_recall', 0):.3f}, "
                                    f"F1: {kg_eval.get('page_f1', 0):.3f}")
                else:
                    logger.info(f"\n📊 质量评估:")
                    logger.info(f"  基线检索 - 检索数量: {baseline_eval.get('quality_retrieved_count', 0)}, "
                                f"平均分数: {baseline_eval.get('quality_avg_score', 0):.3f}")
                    logger.info(f"  KG增强   - 检索数量: {kg_eval.get('quality_retrieved_count', 0)}, "
                                f"平均分数: {kg_eval.get('quality_avg_score', 0):.3f}, "
                                f"KG增强: {kg_eval.get('quality_kg_boost_avg', 0):.3f}")

                logger.info("=" * 50)

                if self.args.debug and idx >= 2:
                    break

            except Exception as e:
                logger.error(f"处理查询 {idx} 时出错: {str(e)}")
                continue

        # 保存和分析结果
        self._save_and_analyze_results(results)

        return results

    def _save_and_analyze_results(self, results: List[Dict]):
        """保存并分析结果 - 支持ground truth评估"""
        # 保存详细结果
        result_file = f"longdoc_kg_enhanced_results_{int(time.time())}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"详细结果已保存到: {result_file}")

        if not results:
            logger.warning("没有可用的结果进行分析")
            return

        # 分离有ground truth和无ground truth的结果
        gt_results = [r for r in results if r.get('has_ground_truth', False)]
        quality_results = [r for r in results if not r.get('has_ground_truth', False)]

        # 打印分析结果
        logger.info("\n" + "=" * 80)
        logger.info("LongDoc知识图谱增强检索实验结果分析")
        logger.info("=" * 80)
        logger.info(f"总测试查询数: {len(results)}")
        logger.info(f"有Ground Truth的查询数: {len(gt_results)}")
        logger.info(f"仅质量评估的查询数: {len(quality_results)}")

        # === Ground Truth评估分析 ===
        if gt_results:
            logger.info(f"\n📊 Ground Truth评估结果 ({len(gt_results)} 个查询):")

            # Chunk级别评估
            baseline_precisions = [r['baseline'].get('precision', 0) for r in gt_results]
            baseline_recalls = [r['baseline'].get('recall', 0) for r in gt_results]
            baseline_f1s = [r['baseline'].get('f1', 0) for r in gt_results]

            kg_precisions = [r['kg_enhanced'].get('precision', 0) for r in gt_results]
            kg_recalls = [r['kg_enhanced'].get('recall', 0) for r in gt_results]
            kg_f1s = [r['kg_enhanced'].get('f1', 0) for r in gt_results]

            logger.info(f"  Chunk级别指标:")
            logger.info(f"    基线检索 - P: {np.mean(baseline_precisions):.4f}, "
                        f"R: {np.mean(baseline_recalls):.4f}, F1: {np.mean(baseline_f1s):.4f}")
            logger.info(f"    KG增强   - P: {np.mean(kg_precisions):.4f}, "
                        f"R: {np.mean(kg_recalls):.4f}, F1: {np.mean(kg_f1s):.4f}")

            # 页面级别评估（如果有）
            baseline_page_precisions = [r['baseline'].get('page_precision', 0) for r in gt_results
                                        if 'page_precision' in r['baseline']]
            baseline_page_recalls = [r['baseline'].get('page_recall', 0) for r in gt_results
                                     if 'page_recall' in r['baseline']]
            baseline_page_f1s = [r['baseline'].get('page_f1', 0) for r in gt_results
                                 if 'page_f1' in r['baseline']]

            kg_page_precisions = [r['kg_enhanced'].get('page_precision', 0) for r in gt_results
                                  if 'page_precision' in r['kg_enhanced']]
            kg_page_recalls = [r['kg_enhanced'].get('page_recall', 0) for r in gt_results
                               if 'page_recall' in r['kg_enhanced']]
            kg_page_f1s = [r['kg_enhanced'].get('page_f1', 0) for r in gt_results
                           if 'page_f1' in r['kg_enhanced']]

            if baseline_page_precisions:
                logger.info(f"  页面级别指标:")
                logger.info(f"    基线检索 - P: {np.mean(baseline_page_precisions):.4f}, "
                            f"R: {np.mean(baseline_page_recalls):.4f}, F1: {np.mean(baseline_page_f1s):.4f}")
                logger.info(f"    KG增强   - P: {np.mean(kg_page_precisions):.4f}, "
                            f"R: {np.mean(kg_page_recalls):.4f}, F1: {np.mean(kg_page_f1s):.4f}")

            # 性能提升统计
            precision_improvements = [(kg - baseline) for kg, baseline in zip(kg_precisions, baseline_precisions)]
            recall_improvements = [(kg - baseline) for kg, baseline in zip(kg_recalls, baseline_recalls)]
            f1_improvements = [(kg - baseline) for kg, baseline in zip(kg_f1s, baseline_f1s)]

            logger.info(f"\n📈 Ground Truth性能提升:")
            logger.info(f"  精确率提升: {np.mean(precision_improvements):+.4f}")
            logger.info(f"  召回率提升: {np.mean(recall_improvements):+.4f}")
            logger.info(f"  F1值提升: {np.mean(f1_improvements):+.4f}")

            # 统计显著性检验
            try:
                from scipy import stats
                t_stat, p_value = stats.ttest_rel(kg_f1s, baseline_f1s)
                significance = "显著" if p_value < 0.05 else "不显著"
                logger.info(f"  F1提升显著性: {significance} (p={p_value:.3f})")
            except:
                logger.info("  F1提升显著性: 无法计算")

        # === 质量评估分析 ===
        if quality_results:
            logger.info(f"\n🔍 质量评估结果 ({len(quality_results)} 个查询):")

            baseline_scores = [r['baseline'].get('quality_avg_score', 0) for r in quality_results]
            kg_scores = [r['kg_enhanced'].get('quality_avg_score', 0) for r in quality_results]
            kg_boosts = [r['kg_enhanced'].get('quality_kg_boost_avg', 0) for r in quality_results]

            baseline_times = [r['baseline']['retrieval_time'] for r in quality_results]
            kg_times = [r['kg_enhanced']['retrieval_time'] for r in quality_results]

            logger.info(f"  基线检索 - 平均分数: {np.mean(baseline_scores):.4f}")
            logger.info(f"  KG增强   - 平均分数: {np.mean(kg_scores):.4f}, KG增强: {np.mean(kg_boosts):.4f}")
            logger.info(f"  时间开销 - 基线: {np.mean(baseline_times):.2f}s, KG增强: {np.mean(kg_times):.2f}s")

        logger.info("=" * 80)


# ===== 新增多文档处理器 =====

class LongDocMultiDocumentTester:
    """LongDoc多文档KG增强测试器 - 只负责多文档处理"""

    def __init__(self, chunk_data_dir: str, kg_files_dir: str, test_data_path: str, args: argparse.Namespace = None):
        self.chunk_data_dir = chunk_data_dir
        self.kg_files_dir = kg_files_dir
        self.test_data_path = test_data_path
        self.data_source_path = Path(self.chunk_data_dir).parent
        self.ocr_method = "pytesseract"
        self.args = args

        # 发现可用的文档
        self.available_documents = self._discover_available_documents()
        logger.info(f"发现 {len(self.available_documents)} 个有KG文件的文档")

    def _discover_available_documents(self) -> Dict[str, Dict]:
        """发现有对应KG文件的文档"""
        available_docs = {}

        # 扫描KG文件目录
        kg_pattern = os.path.join(self.kg_files_dir, "*_subgraphs.json")
        kg_files = glob.glob(kg_pattern)

        for kg_file in kg_files:
            basename = os.path.basename(kg_file)
            doc_id = basename.replace("_subgraphs.json", "")

            # 检查对应的chunk数据文件
            chunk_file = os.path.join(self.chunk_data_dir, f"{doc_id}.json")

            if os.path.exists(chunk_file):
                kg_prefix = os.path.join(self.kg_files_dir, doc_id)

                # 验证KG文件完整性
                required_files = [
                    f"{kg_prefix}_subgraphs.json",
                    f"{kg_prefix}_kg.json",
                    f"{kg_prefix}_triplets.json"
                ]

                if all(os.path.exists(f) for f in required_files):
                    available_docs[doc_id] = {
                        'chunk_file': chunk_file,
                        'kg_prefix': kg_prefix,
                        'doc_id': doc_id
                    }

        return available_docs

    def load_test_data(self) -> List[Dict]:
        """加载测试数据，只保留有KG文件的文档"""
        logger.info(f"加载测试数据: {self.test_data_path}")
        all_test_data = []

        try:
            with open(self.test_data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        all_test_data.append(item)

            # 筛选有KG文件的数据
            filtered_data = []
            for item in all_test_data:
                pdf_path = item.get("pdf_path", "")
                doc_id = os.path.splitext(os.path.basename(pdf_path))[0]

                if doc_id in self.available_documents:
                    item['doc_id'] = doc_id
                    filtered_data.append(item)

            # 采样
            if self.args and self.args.sample_size > 0 and len(filtered_data) > self.args.sample_size:
                np.random.seed(42)
                filtered_data = np.random.choice(filtered_data, self.args.sample_size, replace=False).tolist()

            logger.info(f"加载了 {len(filtered_data)} 条有KG文件的测试数据（原始 {len(all_test_data)} 条）")
            return filtered_data

        except Exception as e:
            logger.error(f"加载测试数据失败: {str(e)}")
            return []

    def process_single_document(self, doc_data):
        """处理单个文档，直接使用OCR结果，跳过PDF转换

        根据MultimodalMatcher的接口要求，格式化文档数据
        """
        documents = []

        # 获取PDF文件路径
        pdf_path = os.path.join(self.data_source_path, doc_data["pdf_path"])
        try:
            pages = convert_from_path(pdf_path)
            logger.info(f"成功将PDF转换为 {len(pages)} 页图像")
        except Exception as e:
            logger.error(f"转换PDF时出错：{str(e)}")
            return []

        # 获取预处理的OCR结果
        ocr_file = os.path.join(
            self.data_source_path,
            f"{self.ocr_method}_save",
            f"{os.path.basename(doc_data['pdf_path']).replace('.pdf', '.json')}"
        )

        # # 获取预处理的OCR结果
        # pdf_name = os.path.basename(doc_data['pdf_path'])
        # name_wo_ext = os.path.splitext(pdf_name)[0]
        # json_name = f"{name_wo_ext}_page.json"
        # ocr_file = os.path.join(
        #     self.data_source_path,
        #     f"{self.ocr_method}_save",
        #     json_name
        # )

        # 读取预处理的文本数据
        if os.path.exists(ocr_file):
            with open(ocr_file, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            logger.info(f"成功读取预处理文本文件: {ocr_file}")
        else:
            logger.warning(f"找不到预处理文本文件: {ocr_file}")
            return []

        # 为每一页创建文档对象（不使用PDF图像）
        page_keys = list(loaded_data.keys())
        for idx, (page_key, page_text) in enumerate(loaded_data.items()):
            # 创建文档结构
            page = pages[idx]
            width, height = page.size
            if width <= 0 or height <= 0:
                logger.warning(f"跳过无效页面 {idx + 1}：尺寸 {width}x{height}")
                continue

            documents.append({
                "text": page_text if page_text.strip() else f"第{idx + 1}页内容",
                "image": page,  # 不使用图像，避免PDF转换
                "metadata": {
                    "page_index": idx + 1,
                    "pdf_path": doc_data.get("pdf_path", "")
                }
            })

        logger.info(f"成功创建 {len(documents)} 个文档对象")
        return documents

    def test_multi_documents(self):
        """测试多文档KG增强检索"""
        logger.info("开始多文档KG增强检索测试...")

        test_data = self.load_test_data()
        if not test_data:
            logger.error("没有可用的测试数据")
            return []

        results = []
        successful_tests = 0
        failed_tests = 0

        for idx, doc_data in enumerate(tqdm(test_data, desc="多文档KG增强测试")):
            try:
                doc_id = doc_data['doc_id']
                query = doc_data.get("question", "")
                evidence_pages = doc_data.get("evidence_pages", [])

                logger.info(f"\n测试 {idx + 1}/{len(test_data)}: 文档 {doc_id}")
                logger.info(f"查询: {query}")

                # 获取文档信息
                doc_info = self.available_documents[doc_id]

                # 为当前文档创建测试器（使用原有的LongDocKGTester）
                tester = LongDocKGTester(
                    data_path=doc_info['chunk_file'],
                    kg_files_prefix=doc_info['kg_prefix'],
                    args=self.args
                )

                # 构造ground truth
                # 构造 ground truth 时，根据页面计算相关的chunk IDs
                relevant_chunk_ids = []
                if evidence_pages:
                    for page in evidence_pages:
                        chunks_in_page = tester.evaluator.page_to_chunks.get(page, [])
                        relevant_chunk_ids.extend(chunks_in_page)
                    relevant_chunk_ids = list(set(relevant_chunk_ids))  # 去重

                ground_truth = GroundTruthItem(
                    query=query,
                    relevant_chunk_ids=relevant_chunk_ids,  # 明确设置
                    relevant_pages=evidence_pages,
                    metadata=doc_data
                )

                # 准备文档
                documents = tester.data_processor.prepare_documents_for_retrieval()
                if not documents:
                    logger.warning(f"文档 {doc_id} 没有有效内容，跳过")
                    failed_tests += 1
                    continue
                image_doc = self.process_single_document(doc_data)

                # 基线检索
                baseline_start = time.time()
                baseline_results = tester.baseline_retriever.retrieve(
                    query, documents, topk=self.args.rerank_topk if self.args else 5
                )
                baseline_time = time.time() - baseline_start

                # KG增强检索
                kg_start = time.time()
                # kg_results = tester.kg_enhanced_retriever.search_with_kg_enhanced_multi_intent(query, documents, image_doc)
                kg_results, image_result = tester.kg_enhanced_retriever.search_with_kg_enhanced_multi_intent(query,
                                                                                                             documents,
                                                                                                             image_doc)
                image_weight = tester.kg_enhanced_retriever.get_visual_weight_llm(query)
                kg_time = time.time() - kg_start

                # 评估结果
                baseline_eval = tester.evaluator.evaluate_retrieval_with_ground_truth(
                    query, baseline_results, ground_truth)
                # kg_eval = tester.evaluator.evaluate_retrieval(
                #     query, kg_results, ground_truth)
                kg_eval = tester.evaluator.evaluate_retrieval_with_image(
                    query, kg_results, image_result, ground_truth, image_weight)

                # 记录结果
                result = {
                    "doc_id": doc_id,
                    "query": query,
                    "evidence_pages": evidence_pages,
                    "task_tag": doc_data.get("task_tag", ""),
                    "baseline": {
                        **baseline_eval,
                        "retrieval_time": baseline_time
                    },
                    "kg_enhanced": {
                        **kg_eval,
                        "retrieval_time": kg_time
                    }
                }

                results.append(result)
                successful_tests += 1

                # ===== 详细调试信息 =====
                logger.info("\n" + "=" * 50)
                logger.info("🔍 检索结果详细分析")
                logger.info("=" * 50)

                if ground_truth:
                    logger.info(f"📋 Ground Truth信息:")
                    logger.info(f"  证据页面: {ground_truth.relevant_pages}")
                    logger.info(f"  证据chunks: {ground_truth.relevant_chunk_ids}")

                # # 基线检索结果
                # logger.info(f"\n📄 基线检索结果 ({len(baseline_results)} 个):")
                # for i, result in enumerate(baseline_results, 1):
                #     chunk_id = result['chunk_id']
                #     chunk_pages = tester.evaluator.chunk_to_pages.get(chunk_id, [])
                #     is_correct = str(chunk_id) in (ground_truth.relevant_chunk_ids if ground_truth else [])
                #     logger.info(
                #         f"  {i}. chunk_{chunk_id} | 页面: {chunk_pages} | 分数: {result['score']:.3f} | {'✓' if is_correct else '✗'}")
                #
                # # KG增强检索结果
                # logger.info(f"\n🚀 KG增强检索结果 ({len(kg_results)} 个):")
                # for i, result in enumerate(kg_results, 1):
                #     chunk_id = result['chunk_id']
                #     chunk_pages = tester.evaluator.chunk_to_pages.get(chunk_id, [])
                #     is_correct = str(chunk_id) in (ground_truth.relevant_chunk_ids if ground_truth else [])
                #     kg_boost = result.get('kg_boost', 0)
                #     source_query = result.get('source_query', 'original')
                #     logger.info(
                #         f"  {i}. chunk_{chunk_id} | 页面: {chunk_pages} | 分数: {result.get('score', 0):.3f} | KG增强: {kg_boost:.3f} | 来源: {source_query} | {'✓' if is_correct else '✗'}")

                # # 页面到chunk映射检查
                # if ground_truth:
                #     logger.info(f"\n🗺️ 页面到chunk映射检查:")
                #     for page in ground_truth.relevant_pages:
                #         chunks_in_page = tester.evaluator.page_to_chunks.get(page, [])
                #         logger.info(f"  页面 {page}: chunks {chunks_in_page}")

                # 打印评估结果
                if ground_truth:
                    # logger.info(f"\n📊 评估指标:")
                    # logger.info(f"  基线检索 - P: {baseline_eval.get('precision', 0):.3f}, "
                    #             f"R: {baseline_eval.get('recall', 0):.3f}, "
                    #             f"F1: {baseline_eval.get('f1', 0):.3f}")
                    # logger.info(f"  KG增强   - P: {kg_eval.get('precision', 0):.3f}, "
                    #             f"R: {kg_eval.get('recall', 0):.3f}, "
                    #             f"F1: {kg_eval.get('f1', 0):.3f}")

                    # 页面级评估
                    if 'page_precision' in baseline_eval:
                        logger.info(f"  页面基线 - P: {baseline_eval.get('page_precision', 0):.3f}, "
                                    f"R: {baseline_eval.get('page_recall', 0):.3f}, "
                                    f"F1: {baseline_eval.get('page_f1', 0):.3f}")
                        logger.info(f"  页面KG   - P: {kg_eval.get('page_precision', 0):.3f}, "
                                    f"R: {kg_eval.get('page_recall', 0):.3f}, "
                                    f"F1: {kg_eval.get('page_f1', 0):.3f}")
                else:
                    logger.info(f"\n📊 质量评估:")
                    logger.info(f"  基线检索 - 检索数量: {baseline_eval.get('quality_retrieved_count', 0)}, "
                                f"平均分数: {baseline_eval.get('quality_avg_score', 0):.3f}")
                    logger.info(f"  KG增强   - 检索数量: {kg_eval.get('quality_retrieved_count', 0)}, "
                                f"平均分数: {kg_eval.get('quality_avg_score', 0):.3f}, "
                                f"KG增强: {kg_eval.get('quality_kg_boost_avg', 0):.3f}")

                logger.info("=" * 50)

            except Exception as e:
                logger.error(f"处理文档 {doc_data.get('doc_id', 'unknown')} 时出错: {str(e)}")
                logger.error(f"详细错误信息:\n{traceback.format_exc()}")
                failed_tests += 1
                continue

        # 保存和分析结果
        self._save_results(results)
        self._analyze_results(results)

        logger.info(f"测试完成: 成功 {successful_tests}, 失败 {failed_tests}")
        return results

    def _save_results(self, results: List[Dict]):
        """保存测试结果"""
        try:
            output_dir = self.args.results_dir if self.args and hasattr(self.args, 'results_dir') else './test_results'
            os.makedirs(output_dir, exist_ok=True)

            result_file = os.path.join(output_dir, 'multi_document_kg_enhanced_results.json')

            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            logger.info(f"结果已保存到: {result_file}")

        except Exception as e:
            logger.error(f"保存结果失败: {str(e)}")

    def _analyze_results(self, results: List[Dict]):
        """分析并输出平均性能指标"""
        if not results:
            logger.warning("没有结果可供分析")
            return

        # # 收集指标
        # baseline_recalls_chunk = [r["baseline"].get("recall", 0) for r in results]
        # baseline_precisions_chunk = [r["baseline"].get("precision", 0) for r in results]
        # baseline_f1s_chunk = [r["baseline"].get("f1", 0) for r in results]
        # baseline_times = [r["baseline"]["retrieval_time"] for r in results]
        #
        # kg_recalls_chunk = [r["kg_enhanced"].get("recall", 0) for r in results]
        # kg_precisions_chunk = [r["kg_enhanced"].get("precision", 0) for r in results]
        # kg_f1s_chunk = [r["kg_enhanced"].get("f1", 0) for r in results]
        # kg_times = [r["kg_enhanced"]["retrieval_time"] for r in results]
        #
        # baseline_success = sum(1 for r in results if r["baseline"].get("success", False))
        # kg_success = sum(1 for r in results if r["kg_enhanced"].get("success", False))

        baseline_recalls_page = [r["baseline"].get("page_recall", 0) for r in results]
        baseline_precisions_page = [r["baseline"].get("page_precision", 0) for r in results]
        baseline_f1s_page = [r["baseline"].get("page_f1", 0) for r in results]

        kg_recalls_page = [r["kg_enhanced"].get("page_recall", 0) for r in results]
        kg_precisions_page = [r["kg_enhanced"].get("page_precision", 0) for r in results]
        kg_f1s_page = [r["kg_enhanced"].get("page_f1", 0) for r in results]

        # # 计算平均值(chunk)
        # avg_baseline_recall_chunk = np.mean(baseline_recalls_chunk)
        # avg_baseline_precision_chunk = np.mean(baseline_precisions_chunk)
        # avg_baseline_f1_chunk = np.mean(baseline_f1s_chunk)
        # avg_baseline_time = np.mean(baseline_times)
        #
        # avg_kg_recall_chunk = np.mean(kg_recalls_chunk)
        # avg_kg_precision_chunk = np.mean(kg_precisions_chunk)
        # avg_kg_f1_chunk = np.mean(kg_f1s_chunk)
        # avg_kg_time = np.mean(kg_times)

        # 计算平均值(page)
        avg_baseline_recall_page = np.mean(baseline_recalls_page)
        avg_baseline_precision_page = np.mean(baseline_precisions_page)
        avg_baseline_f1_page = np.mean(baseline_f1s_page)

        avg_kg_recall_page = np.mean(kg_recalls_page)
        avg_kg_precision_page = np.mean(kg_precisions_page)
        avg_kg_f1_page = np.mean(kg_f1s_page)

        # baseline_success_rate = baseline_success / len(results) * 100
        # kg_success_rate = kg_success / len(results) * 100
        #
        # # 计算提升幅度
        # recall_improvement_chunk = (
        #                                  avg_kg_recall_chunk - avg_baseline_recall_chunk) / avg_baseline_recall_chunk * 100 if avg_baseline_recall_chunk > 0 else 0
        # precision_improvement_chunk = (
        #                                     avg_kg_precision_chunk - avg_baseline_precision_chunk) / avg_baseline_precision_chunk * 100 if avg_baseline_precision_chunk > 0 else 0
        # f1_improvement_chunk = (avg_kg_f1_chunk - avg_baseline_f1_chunk) / avg_baseline_f1_chunk * 100 if avg_baseline_f1_chunk > 0 else 0

        recall_improvement_page = (
                                           avg_kg_recall_page - avg_baseline_recall_page) / avg_baseline_recall_page * 100 if avg_baseline_recall_page > 0 else 0
        precision_improvement_page = (
                                              avg_kg_precision_page - avg_baseline_precision_page) / avg_baseline_precision_page * 100 if avg_baseline_precision_page > 0 else 0
        f1_improvement_page = (
                                           avg_kg_f1_page - avg_baseline_f1_page) / avg_baseline_f1_page * 100 if avg_baseline_f1_page > 0 else 0

        # 输出结果
        logger.info("\n" + "=" * 80)
        logger.info("🔍 多文档KG增强检索性能分析")
        logger.info("=" * 80)

        logger.info(f"\n📊 总体统计:")
        logger.info(f"  测试文档数: {len(results)}")
        logger.info(f"  有KG文件的文档: {len(self.available_documents)}")

        # logger.info(f"\n📈 平均性能指标(chunk):")
        # logger.info(f"  基线检索:")
        # logger.info(f"    Recall:    {avg_baseline_recall_chunk:.4f}")
        # logger.info(f"    Precision: {avg_baseline_precision_chunk:.4f}")
        # logger.info(f"    F1:        {avg_baseline_f1_chunk:.4f}")
        # logger.info(f"    成功率:    {baseline_success_rate:.2f}%")
        # logger.info(f"    平均时间:  {avg_baseline_time:.2f}s")
        #
        # logger.info(f"\n  KG增强检索:")
        # logger.info(f"    Recall:    {avg_kg_recall_chunk:.4f}")
        # logger.info(f"    Precision: {avg_kg_precision_chunk:.4f}")
        # logger.info(f"    F1:        {avg_kg_f1_chunk:.4f}")
        # logger.info(f"    成功率:    {kg_success_rate:.2f}%")
        # logger.info(f"    平均时间:  {avg_kg_time:.2f}s")
        #
        # logger.info(f"\n🚀 性能提升:")
        # logger.info(f"  Recall 提升:    {recall_improvement_chunk:+.2f}%")
        # logger.info(f"  Precision 提升: {precision_improvement_chunk:+.2f}%")
        # logger.info(f"  F1 提升:        {f1_improvement_chunk:+.2f}%")
        # logger.info(f"  成功率提升:     {kg_success_rate - baseline_success_rate:+.2f}%")

        logger.info(f"\n📈 平均性能指标(page):")
        logger.info(f"  基线检索:")
        logger.info(f"    Recall:    {avg_baseline_recall_page:.4f}")
        logger.info(f"    Precision: {avg_baseline_precision_page:.4f}")
        logger.info(f"    F1:        {avg_baseline_f1_page:.4f}")

        logger.info(f"\n  KG增强检索:")
        logger.info(f"    Recall:    {avg_kg_recall_page:.4f}")
        logger.info(f"    Precision: {avg_kg_precision_page:.4f}")
        logger.info(f"    F1:        {avg_kg_f1_page:.4f}")

        logger.info(f"\n🚀 性能提升:")
        logger.info(f"  Recall 提升:    {recall_improvement_page:+.2f}%")
        logger.info(f"  Precision 提升: {precision_improvement_page:+.2f}%")
        logger.info(f"  F1 提升:        {f1_improvement_page:+.2f}%")

        # # 按任务类型分析
        # task_types = {}
        # for r in results:
        #     task_tag = r.get("task_tag", "Unknown")
        #     if task_tag not in task_types:
        #         task_types[task_tag] = {"count": 0, "baseline_f1": 0, "kg_f1": 0}
        #     task_types[task_tag]["count"] += 1
        #     task_types[task_tag]["baseline_f1"] += r["baseline"].get("f1", 0)
        #     task_types[task_tag]["kg_f1"] += r["kg_enhanced"].get("f1", 0)
        #
        # if task_types:
        #     logger.info(f"\n📋 按任务类型分析:")
        #     for task_tag, stats in task_types.items():
        #         count = stats["count"]
        #         avg_baseline_f1 = stats["baseline_f1"] / count
        #         avg_kg_f1 = stats["kg_f1"] / count
        #         improvement = (avg_kg_f1 - avg_baseline_f1) / avg_baseline_f1 * 100 if avg_baseline_f1 > 0 else 0
        #
        #         logger.info(f"  {task_tag} (样本数: {count}):")
        #         logger.info(f"    基线 F1:   {avg_baseline_f1:.4f}")
        #         logger.info(f"    KG增强 F1: {avg_kg_f1:.4f}")
        #         logger.info(f"    F1提升:    {improvement:+.2f}%")

        logger.info("=" * 80)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='LongDoc KG增强检索测试 - 支持Ground Truth评估')

    parser.add_argument('--kg_dir', type=str, default=None,
                        help='包含KG文件的目录路径')
    parser.add_argument('--ground_truth_path', type=str, default=None,
                        help='Ground truth文件路径 (LongDoc格式的JSON/JSONL文件)')
    parser.add_argument('--sample_size', type=int, default=50,
                        help='测试查询数量，0表示全部')
    parser.add_argument('--embedding_topk', type=int, default=15,
                        help='嵌入检索Top-K')
    parser.add_argument('--rerank_topk', type=int, default=10,
                        help='重排序Top-K')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式')

    # 新增多文档处理选项
    parser.add_argument('--multi_doc', action='store_true',
                        help='多文档处理模式')
    parser.add_argument('--chunk_data_dir', type=str,
                        help='多文档模式：chunk数据目录路径')
    parser.add_argument('--kg_files_dir', type=str,
                        help='多文档模式：KG文件目录路径')
    parser.add_argument('--test_data_path', type=str,
                        help='多文档模式：测试数据文件路径 (JSONL格式)')
    parser.add_argument('--results_dir', type=str, default='./test_results',
                        help='结果保存目录')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()

    if args.multi_doc:
        # 多文档处理模式
        if not all([args.chunk_data_dir, args.kg_files_dir, args.test_data_path]):
            logger.error("多文档模式需要指定 --chunk_data_dir, --kg_files_dir, --test_data_path")
            return

        # 检查路径
        for path, name in [(args.chunk_data_dir, "chunk数据目录"),
                           (args.kg_files_dir, "KG文件目录"),
                           (args.test_data_path, "测试数据文件")]:
            if not os.path.exists(path):
                logger.error(f"{name}不存在: {path}")
                return

        # 创建多文档测试器
        tester = LongDocMultiDocumentTester(
            chunk_data_dir=args.chunk_data_dir,
            kg_files_dir=args.kg_files_dir,
            test_data_path=args.test_data_path,
            args=args
        )

        # 运行多文档测试
        results = tester.test_multi_documents()
        logger.info("🎉 多文档KG增强检索测试完成！")

    else:
        # 原有的单文档处理模式
        if not os.path.exists(args.data_path):
            logger.error(f"数据文件不存在: {args.data_path}")
            return

        kg_files_prefix = os.path.join(args.kg_dir, os.path.basename(args.data_path).replace('.json', ''))

        # 检查KG文件是否存在
        kg_files_exist = any([
            os.path.exists(f"{kg_files_prefix}_subgraphs.json"),
            os.path.exists(f"{kg_files_prefix}_kg.json"),
            os.path.exists(f"{kg_files_prefix}_triplets.json")
        ])

        if not kg_files_exist:
            logger.error(f"未找到KG文件，请检查前缀路径: {kg_files_prefix}")
            return

        # 创建测试器
        tester = LongDocKGTester(args.data_path, kg_files_prefix, args, args.ground_truth_path)

        # 确定查询列表
        if args.ground_truth_path and os.path.exists(args.ground_truth_path) and tester.ground_truth_data:
            logger.info("使用Ground Truth文件中的查询进行测试")
            results = tester.test_kg_enhancement_with_ground_truth()
        elif args.queries:
            logger.info("使用命令行指定的查询进行测试")
            results = tester.test_kg_enhancement_with_queries(args.queries)
        else:
            logger.error("请提供查询列表或Ground Truth文件")
            return

        logger.info("🎉 LongDoc KG增强检索测试完成！")


if __name__ == "__main__":
    main()
