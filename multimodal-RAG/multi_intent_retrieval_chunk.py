#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import logging
from typing import List, Dict, Set
import torch
from transformers import AutoTokenizer, AutoModel
from dataclasses import dataclass
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# 加载环境变量
load_dotenv("/root/autodl-tmp/multimodal-RAG/multimodal-RAG/DeepRAG_Multimodal/configs/.env")

# 创建日志
log_dir = Path("./log")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "chunk_intent_comparison.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(str(log_file), mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 添加系统路径
sys.path.append("/root/autodl-tmp/multimodal-RAG/multimodal-RAG/DeepRAG_Multimodal/deep_retrieve")

# 导入必要的库
from DeepRAG_Multimodal.deep_retrieve.ming.deepsearch_optimize_ming import DeepSearch_Beta
from FlagEmbedding import FlagReranker

logger.info("=== Chunk级别单意图vs多意图检索对比测试开始 ===")


@dataclass
class ChunkDocument:
    """Chunk文档数据结构"""
    chunk_id: int
    content: str
    pages: List[int]  # 该chunk覆盖的页面列表
    pdf_path: str = ""

    def get_page_set(self) -> Set[int]:
        """获取页面集合"""
        return set(self.pages)

    def __str__(self):
        return f"Chunk-{self.chunk_id}: {self.content[:50]}... (pages: {self.pages})"


class ChunkMatcher:
    """基于Chunk的检索匹配器 - 适配DeepSearch_Beta接口"""

    def __init__(self, bge_model_path: str, device: str = "cuda:0", topk: int = 10):
        self.bge_model_path = bge_model_path
        self.device = device
        self.topk = topk
        self.max_chunk_length = 512
        self.batch_size = 16
        self._setup_models()

    def _setup_models(self):
        """设置BGE模型"""
        logger.info("🔧 初始化BGE模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.bge_model_path,
            local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            self.bge_model_path,
            local_files_only=True
        ).to(self.device)
        logger.info("✅ BGE模型初始化完成")

    def retrieve(self, query: str, documents: List[Dict]) -> List[Dict]:
        """检索相关chunks - 适配DeepSearch接口

        Args:
            query: 查询字符串
            documents: 格式为 [{"text": str, "metadata": {...}}, ...]

        Returns:
            List[Dict]: 检索结果，格式与输入相同
        """
        if not documents:
            return []

        try:
            # 从documents中提取文本和元数据
            texts = []
            for doc in documents:
                if isinstance(doc, dict) and 'text' in doc:
                    texts.append(doc['text'])
                else:
                    logger.warning(f"跳过格式不正确的文档: {type(doc)}")
                    continue

            if not texts:
                logger.warning("没有有效的文本数据")
                return []

            # 编码查询和文档
            query_embedding = self._encode_texts([query])
            doc_embeddings = self._encode_texts(texts)

            if query_embedding.size == 0 or doc_embeddings.size == 0:
                logger.warning("嵌入计算失败")
                return []

            # 计算相似度
            similarities = np.dot(query_embedding, doc_embeddings.T).flatten()

            # 获取top-k候选
            top_indices = np.argsort(similarities)[::-1][:self.topk]

            # 构建结果 - 保持原始格式
            results = []
            for idx in top_indices:
                if idx < len(documents):
                    doc = documents[idx].copy()  # 复制原文档
                    doc["score"] = float(similarities[idx])  # 添加分数
                    results.append(doc)

            return results

        except Exception as e:
            logger.error(f"❌ Chunk检索失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """使用BGE模型编码文本"""
        if not texts:
            return np.array([])

        embeddings = []

        # 分批处理
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_chunk_length
            ).to(self.device)

            # 获取嵌入
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 使用[CLS] token的嵌入
                batch_embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
                embeddings.append(batch_embeddings)

        # 合并所有批次
        if embeddings:
            all_embeddings = np.vstack(embeddings)
            # L2归一化
            norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
            all_embeddings = all_embeddings / (norms + 1e-8)
            return all_embeddings

        return np.array([])


class ChunkIntentTester:
    """Chunk级别单意图vs多意图检索测试类"""

    def __init__(self):
        """初始化测试器"""
        self.config = self.load_config()
        os.makedirs(self.config['results_dir'], exist_ok=True)
        self.chunk_pages_cache = {}
        self.setup_models()

    def load_config(self):
        """加载配置"""
        config = {
            # 路径配置
            'test_data_path': '/root/autodl-tmp/multimodal-RAG/multimodal-RAG/DeepRAG_Multimodal/picked_LongDoc/selected_LongDocURL_public_with_subtask_category.jsonl',
            'pdf_base_dir': '/root/autodl-tmp/multimodal-RAG/multimodal-RAG/DeepRAG_Multimodal/picked_LongDoc',
            'chunk_data_dir': '/root/autodl-tmp/multimodal-RAG/multimodal-RAG/DeepRAG_Multimodal/picked_LongDoc/chunked_ocr',
            'results_dir': './test_results',

            # 模型配置
            'bge_model_path': '/root/autodl-tmp/multimodal-RAG/hf_models/bge-large-en-v1.5',
            'reranker_model_path': '/root/autodl-tmp/multimodal-RAG/hf_models/bge-reranker-large',
            'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',

            # 采样配置
            'sample_size': 0,  # 测试样本数量，0表示全部
            'debug': False,

            # 检索配置
            'max_iterations': 2,
            'embedding_topk': 15,
            'rerank_topk': 5,
            'text_weight': 1.0,  # 纯文本模式
            'image_weight': 0.0,
        }
        return config

    def setup_models(self):
        """初始化检索模型"""
        logger.info("🚀 初始化Chunk级别检索模型...")

        try:
            # 检查模型路径
            if not os.path.exists(self.config['bge_model_path']):
                raise FileNotFoundError(f"BGE模型路径不存在: {self.config['bge_model_path']}")
            if not os.path.exists(self.config['reranker_model_path']):
                raise FileNotFoundError(f"Reranker模型路径不存在: {self.config['reranker_model_path']}")

            # 初始化Reranker
            self.reranker = FlagReranker(
                model_name_or_path=self.config['reranker_model_path'],
                use_fp16=True,
                device=self.config['device'],
                local_files_only=True
            )

            # 初始化Chunk匹配器
            self.chunk_matcher = ChunkMatcher(
                bge_model_path=self.config['bge_model_path'],
                device=self.config['device'],
                topk=self.config['embedding_topk']
            )

            # 初始化单意图检索器
            self.single_intent_search = DeepSearch_Beta(
                max_iterations=self.config['max_iterations'],
                reranker=self.reranker,
                params={
                    "embedding_topk": self.config['embedding_topk'],
                    "rerank_topk": self.config['rerank_topk'],
                    "text_weight": self.config['text_weight'],
                    "image_weight": self.config['image_weight']
                }
            )

            # 🔥 重写单意图检索器的方法，禁用意图拆解
            def single_intent_split(query: str, context: str = "") -> list:
                """单意图模式：直接返回原查询，不进行拆解"""
                return [query]

            def single_intent_refine(original_query: str, intent_queries: list, context: str = "") -> list:
                """单意图模式：直接返回原查询，不进行细化"""
                return [original_query]

            self.single_intent_search._split_query_intent = single_intent_split
            self.single_intent_search._refine_query_intent = single_intent_refine

            # 初始化多意图检索器
            self.multi_intent_search = DeepSearch_Beta(
                max_iterations=self.config['max_iterations'],
                reranker=self.reranker,
                params={
                    "embedding_topk": self.config['embedding_topk'],
                    "rerank_topk": self.config['rerank_topk'],
                    "text_weight": self.config['text_weight'],
                    "image_weight": self.config['image_weight']
                }
            )

            logger.info(f"✅ 模型初始化成功，设备: {self.config['device']}")

        except Exception as e:
            logger.error(f"❌ 模型初始化失败: {str(e)}")
            raise

    def load_test_data(self):
        """加载测试数据（全量，不使用白名单）"""
        logger.info(f"📚 加载测试数据: {self.config['test_data_path']}")
        test_data = []

        with open(self.config['test_data_path'], 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    test_data.append(item)

        if self.config['sample_size'] > 0 and len(test_data) > self.config['sample_size']:
            np.random.seed(42)  # 保证可重复性
            test_data = np.random.choice(test_data, self.config['sample_size'], replace=False).tolist()

        logger.info(f"✅ 成功加载 {len(test_data)} 条测试数据")
        return test_data

    def load_chunk_data(self, doc_data: dict) -> List[ChunkDocument]:
        """加载文档的chunk数据"""
        pdf_name = doc_data['pdf_path'].replace('.pdf', '')
        chunk_file = os.path.join(
            self.config['chunk_data_dir'],
            f"{pdf_name}.json"
        )

        if not os.path.exists(chunk_file):
            logger.warning(f"⚠️ 找不到chunk文件: {chunk_file}")
            return []

        try:
            with open(chunk_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)

            # 转换为ChunkDocument对象
            chunk_docs = []
            pdf_path = doc_data['pdf_path']

            # 缓存逻辑
            if pdf_path not in self.chunk_pages_cache:
                self.chunk_pages_cache[pdf_path] = {}

            for chunk_info in chunks_data:
                chunk_doc = ChunkDocument(
                    chunk_id=chunk_info['chunk_id'],
                    content=chunk_info['content'],
                    pages=chunk_info['pages'],
                    pdf_path=doc_data['pdf_path']
                )
                chunk_docs.append(chunk_doc)

                # 缓存映射
                self.chunk_pages_cache[pdf_path][chunk_info['chunk_id']] = chunk_info['pages']

            logger.info(f"📑 成功加载 {len(chunk_docs)} 个chunks from {pdf_name}")
            return chunk_docs

        except Exception as e:
            logger.error(f"❌ 加载chunk数据失败: {str(e)}")
            return []

    def chunks_to_documents_format(self, chunk_docs: List[ChunkDocument]) -> List[Dict]:
        """将ChunkDocument转换为DeepSearch所需的documents格式"""
        documents = []
        for chunk_doc in chunk_docs:
            documents.append({
                "text": chunk_doc.content,
                "metadata": {
                    "chunk_id": chunk_doc.chunk_id,
                    "pages": chunk_doc.pages,
                    "pdf_path": chunk_doc.pdf_path,
                    "page_index": chunk_doc.pages[0] if chunk_doc.pages else -1
                }
            })
        return documents

    def aggregate_chunks_to_pages_union(self, chunk_results: List[Dict], pdf_path: str = None) -> List[int]:
        """将chunk检索结果聚合到页面级别 - Union策略"""
        if not chunk_results:
            return []

        page_scores = {}  # {page_id: max_score}

        for result in chunk_results:
            chunk_score = result.get('score', 0)
            chunk_pages = []

            # 尝试多种方式获取页面信息
            if 'metadata' in result and isinstance(result['metadata'], dict):
                chunk_pages = result['metadata'].get('pages', [])  # 方式1: 兼容单意图结果
            elif 'pages' in result:
                chunk_pages = result.get('pages', [])  # 方式2: 备用方式

            # 🔥 新增：如果上面都没找到，通过chunk_id从缓存查找
            if not chunk_pages and 'metadata' in result and isinstance(result['metadata'], dict):
                chunk_id = result['metadata'].get('chunk_id')
                if chunk_id is not None and pdf_path and pdf_path in self.chunk_pages_cache:
                    chunk_pages = self.chunk_pages_cache[pdf_path].get(chunk_id, [])

            elif 'page' in result:  # 方式3: 兼容多意图结果（备用方案）
                page_val = result['page']
                if isinstance(page_val, list):
                    chunk_pages = page_val
                else:  # 假设 page_val 是一个单独的页码数字
                    chunk_pages = [page_val]

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

    def evaluate_results(self, chunk_results: List[Dict], evidence_pages: List[int], method_name: str,
                         pdf_path: str = None) -> Dict:
        """评估检索结果 - 提供chunk级别和页面级别的评估"""
        # 1. Chunk级别评估
        chunk_eval = self._evaluate_chunk_level(chunk_results, evidence_pages, pdf_path)

        # 2. 页面级别评估（通过Union聚合）
        predicted_pages = self.aggregate_chunks_to_pages_union(chunk_results, pdf_path)
        page_eval = self._evaluate_page_level(predicted_pages, evidence_pages, len(chunk_results))

        logger.info(f"📊 {method_name}结果:")
        logger.info(
            f"   Chunk级别 - P: {chunk_eval['precision']:.4f}, R: {chunk_eval['recall']:.4f}, F1: {chunk_eval['f1']:.4f}")
        logger.info(
            f"   Page级别 - P: {page_eval['precision']:.4f}, R: {page_eval['recall']:.4f}, F1: {page_eval['f1']:.4f}")
        logger.info(f"   Page成功: {page_eval['success']}")

        return {
            "chunk_level": chunk_eval,
            "page_level": page_eval,
            "aggregation_method": "union"
        }

    def _evaluate_chunk_level(self, chunk_results: List[Dict], evidence_pages: List[int], pdf_path: str = None) -> Dict:
        """Chunk级别评估 - 基于chunk是否覆盖证据页面"""
        if not chunk_results:
            return {"precision": 0, "recall": 0, "f1": 0, "coverage": 0, "relevant_chunks": 0}

        evidence_set = set(evidence_pages)
        relevant_chunks = 0
        total_covered_pages = set()

        # 检查每个chunk是否与证据页面相关
        for i, result in enumerate(chunk_results):
            # 🔥 调试信息：打印前3个结果的详细结构
            if i < 3:
                logger.info(f"🔍 调试多意图结果 {i}: keys={list(result.keys())}")
                if 'metadata' in result:
                    logger.info(f"🔍 metadata: {result['metadata']}")
                if 'page' in result:
                    logger.info(f"🔍 page字段: {result['page']}")
                logger.info(
                    f"🔍 缓存状态: pdf_path={pdf_path}, 有缓存={pdf_path in self.chunk_pages_cache if pdf_path else False}")
                if pdf_path and pdf_path in self.chunk_pages_cache:
                    logger.info(f"🔍 缓存chunk数量: {len(self.chunk_pages_cache[pdf_path])}")

            # 尝试多种方式获取页面信息
            chunk_pages = []
            extraction_method = "none"

            if 'metadata' in result and isinstance(result['metadata'], dict):
                chunk_pages = result['metadata'].get('pages', [])
                if chunk_pages:
                    extraction_method = "metadata.pages"
            elif 'pages' in result:
                chunk_pages = result.get('pages', [])
                if chunk_pages:
                    extraction_method = "direct_pages"

            # 🔥 新增：如果上面都没找到，通过chunk_id从缓存查找
            if not chunk_pages and 'metadata' in result and isinstance(result['metadata'], dict):
                chunk_id = result['metadata'].get('chunk_id')
                if chunk_id is not None and pdf_path and pdf_path in self.chunk_pages_cache:
                    chunk_pages = self.chunk_pages_cache[pdf_path].get(chunk_id, [])
                    if chunk_pages:
                        extraction_method = f"cache_by_chunk_id_{chunk_id}"

            # 🔥 修复：处理多意图结果的page字段
            if not chunk_pages and 'page' in result:
                page_val = result['page']
                if isinstance(page_val, list):
                    chunk_pages = page_val
                    extraction_method = "page_list"
                elif page_val is not None:
                    chunk_pages = [page_val]
                    extraction_method = "page_single"

            # 🔥 调试信息：前3个结果的页面提取情况
            if i < 3:
                logger.info(f"🔍 结果 {i}: 提取方法={extraction_method}, pages={chunk_pages}")

            chunk_pages_set = set(chunk_pages)

            if chunk_pages_set.intersection(evidence_set):
                relevant_chunks += 1
                total_covered_pages.update(chunk_pages_set.intersection(evidence_set))
                if i < 5:  # 前5个相关的chunk
                    logger.info(
                        f"✅ 找到相关chunk {i}: pages={chunk_pages}, 证据交集={chunk_pages_set.intersection(evidence_set)}")

        # 计算指标
        precision = relevant_chunks / len(chunk_results) if chunk_results else 0
        recall = len(total_covered_pages) / len(evidence_set) if evidence_set else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        coverage = len(total_covered_pages) / len(evidence_set) if evidence_set else 0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "coverage": coverage,
            "relevant_chunks": relevant_chunks,
            "total_chunks": len(chunk_results)
        }

    def _evaluate_page_level(self, predicted_pages: List[int], evidence_pages: List[int], topk: int) -> Dict:
        """页面级别评估 - 传统的页面检索评估"""
        predicted_set = set(predicted_pages[:topk])
        evidence_set = set(evidence_pages)
        correct_pages = evidence_set.intersection(predicted_set)

        recall = len(correct_pages) / len(evidence_set) if evidence_set else 0
        precision = len(correct_pages) / len(predicted_set) if predicted_set else 0
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0
        success = len(correct_pages) == len(evidence_set)

        return {
            "predicted_pages": predicted_pages[:topk],
            "correct_pages": list(correct_pages),
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "success": success
        }

    def test_chunk_intent_comparison(self):
        """测试Chunk级别单意图vs多意图检索对比"""
        logger.info("🎯 开始Chunk级别单意图vs多意图检索对比测试...")
        test_data = self.load_test_data()
        results = []

        for idx, doc_data in enumerate(tqdm(test_data, desc="Chunk级别单意图vs多意图对比测试")):
            try:
                query = doc_data.get("question", "")
                evidence_pages = doc_data.get("evidence_pages", [])
                pdf_path = doc_data.get("pdf_path", "")

                logger.info(f"\n{'=' * 60}")
                logger.info(f"🔍 处理文档 {idx + 1}/{len(test_data)}: {doc_data.get('pdf_path', 'Unknown')}")
                logger.info(f"❓ 查询: {query}")
                logger.info(f"📋 证据页面: {evidence_pages}")

                # 加载chunk数据
                chunk_docs = self.load_chunk_data(doc_data)
                if not chunk_docs:
                    logger.warning(f"⚠️ 跳过文档: 无有效chunk数据")
                    continue

                logger.info(f"📑 加载了 {len(chunk_docs)} 个chunks")

                # 转换为DeepSearch所需的格式
                documents = self.chunks_to_documents_format(chunk_docs)

                data = {
                    "query": query,
                    "documents": documents
                }

                # 测试单意图检索
                logger.info("📄 开始单意图检索...")
                single_start = time.time()

                # 直接使用chunk_matcher检索
                single_results = self.chunk_matcher.retrieve(query, documents)

                # 使用reranker重排序
                if single_results:
                    pairs = [[query, result['text']] for result in single_results]
                    rerank_scores = self.reranker.compute_score(pairs, normalize=True)
                    for result, score in zip(single_results, rerank_scores):
                        result['score'] = float(score)
                    single_results = sorted(single_results, key=lambda x: x['score'], reverse=True)
                    single_results = single_results[:self.config['rerank_topk']]

                single_elapsed = time.time() - single_start

                # 测试多意图检索
                logger.info("📄 开始多意图检索...")
                multi_start = time.time()
                multi_results = self.multi_intent_search.search_retrieval(data, multi_intent=True,
                                                                          retriever=self.chunk_matcher)
                multi_elapsed = time.time() - multi_start

                # 评估单意图结果
                single_eval = self.evaluate_results(single_results, evidence_pages, "单意图", pdf_path)

                # 评估多意图结果
                multi_eval = self.evaluate_results(multi_results, evidence_pages, "多意图", pdf_path)

                logger.info(f"⏱️ 单意图检索耗时: {single_elapsed:.2f}秒")
                logger.info(f"⏱️ 多意图检索耗时: {multi_elapsed:.2f}秒")

                # 记录对比结果
                result = {
                    "doc_id": doc_data.get("doc_no", ""),
                    "pdf_path": doc_data.get("pdf_path", ""),
                    "query": query,
                    "evidence_pages": evidence_pages,
                    "task_tag": doc_data.get("task_tag", ""),
                    "subTask": doc_data.get("subTask", []),
                    "total_chunks": len(chunk_docs),

                    # 单意图结果
                    "single_intent": {
                        **single_eval,
                        "retrieval_time": single_elapsed,
                        "retrieved_chunks": len(single_results)
                    },

                    # 多意图结果
                    "multi_intent": {
                        **multi_eval,
                        "retrieval_time": multi_elapsed,
                        "retrieved_chunks": len(multi_results)
                    },

                    # 对比指标
                    "comparison": {
                        "chunk_f1_improvement": multi_eval["chunk_level"]["f1"] - single_eval["chunk_level"]["f1"],
                        "page_f1_improvement": multi_eval["page_level"]["f1"] - single_eval["page_level"]["f1"],
                        "chunk_recall_improvement": multi_eval["chunk_level"]["recall"] - single_eval["chunk_level"][
                            "recall"],
                        "page_recall_improvement": multi_eval["page_level"]["recall"] - single_eval["page_level"][
                            "recall"],
                        "time_overhead": multi_elapsed - single_elapsed,
                        "multi_intent_better_chunk": multi_eval["chunk_level"]["f1"] > single_eval["chunk_level"]["f1"],
                        "multi_intent_better_page": multi_eval["page_level"]["f1"] > single_eval["page_level"]["f1"]
                    }
                }

                results.append(result)

            except Exception as e:
                logger.error(f"❌ 处理文档时出错: {str(e)}")
                import traceback
                traceback.print_exc()

        # 保存和分析结果
        result_file = os.path.join(self.config['results_dir'], 'chunk_intent_comparison_results.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        self.analyze_comparison_results(results)
        logger.info(f"🎉 Chunk级别意图对比测试结果已保存到: {result_file}")
        return results

    def analyze_comparison_results(self, results):
        """分析对比测试结果"""
        if not results:
            logger.warning("⚠️ 没有可用的结果进行分析")
            return

        # 收集指标
        def collect_metrics(key_path):
            """从嵌套字典中收集指标"""
            values = []
            for r in results:
                current = r
                for key in key_path.split('.'):
                    if key in current:
                        current = current[key]
                    else:
                        current = 0
                        break
                values.append(current)
            return values

        # 单意图指标
        single_chunk_recalls = collect_metrics("single_intent.chunk_level.recall")
        single_chunk_f1s = collect_metrics("single_intent.chunk_level.f1")
        single_page_recalls = collect_metrics("single_intent.page_level.recall")
        single_page_f1s = collect_metrics("single_intent.page_level.f1")
        single_times = collect_metrics("single_intent.retrieval_time")
        single_page_success = sum(1 for r in results if r["single_intent"]["page_level"]["success"])

        # 多意图指标
        multi_chunk_recalls = collect_metrics("multi_intent.chunk_level.recall")
        multi_chunk_f1s = collect_metrics("multi_intent.chunk_level.f1")
        multi_page_recalls = collect_metrics("multi_intent.page_level.recall")
        multi_page_f1s = collect_metrics("multi_intent.page_level.f1")
        multi_times = collect_metrics("multi_intent.retrieval_time")
        multi_page_success = sum(1 for r in results if r["multi_intent"]["page_level"]["success"])

        # 改进指标
        chunk_f1_improvements = collect_metrics("comparison.chunk_f1_improvement")
        page_f1_improvements = collect_metrics("comparison.page_f1_improvement")
        multi_better_chunk_count = sum(1 for r in results if r["comparison"]["multi_intent_better_chunk"])
        multi_better_page_count = sum(1 for r in results if r["comparison"]["multi_intent_better_page"])

        logger.info(f"\n{'=' * 80}")
        logger.info(f"📊 Chunk级别单意图 vs 多意图检索性能对比分析")
        logger.info(f"{'=' * 80}")
        logger.info(f"📋 测试文档数: {len(results)}")

        logger.info(f"\n🔹 单意图检索性能:")
        logger.info(
            f"   Chunk级别 - 平均召回率: {np.mean(single_chunk_recalls):.4f}, F1: {np.mean(single_chunk_f1s):.4f}")
        logger.info(f"   Page级别 - 平均召回率: {np.mean(single_page_recalls):.4f}, F1: {np.mean(single_page_f1s):.4f}")
        logger.info(f"   平均检索时间: {np.mean(single_times):.2f}秒")
        logger.info(
            f"   Page成功率: {(single_page_success / len(results)) * 100:.2f}% ({single_page_success}/{len(results)})")

        logger.info(f"\n🔹 多意图检索性能:")
        logger.info(
            f"   Chunk级别 - 平均召回率: {np.mean(multi_chunk_recalls):.4f}, F1: {np.mean(multi_chunk_f1s):.4f}")
        logger.info(f"   Page级别 - 平均召回率: {np.mean(multi_page_recalls):.4f}, F1: {np.mean(multi_page_f1s):.4f}")
        logger.info(f"   平均检索时间: {np.mean(multi_times):.2f}秒")
        logger.info(
            f"   Page成功率: {(multi_page_success / len(results)) * 100:.2f}% ({multi_page_success}/{len(results)})")

        logger.info(f"\n🔸 性能提升分析:")
        logger.info(f"   Chunk级别F1平均提升: {np.mean(chunk_f1_improvements):+.4f}")
        logger.info(f"   Page级别F1平均提升: {np.mean(page_f1_improvements):+.4f}")
        logger.info(f"   平均时间开销: {np.mean([r['comparison']['time_overhead'] for r in results]):+.2f}秒")
        logger.info(
            f"   多意图优于单意图(Chunk): {(multi_better_chunk_count / len(results)) * 100:.2f}% ({multi_better_chunk_count}/{len(results)})")
        logger.info(
            f"   多意图优于单意图(Page): {(multi_better_page_count / len(results)) * 100:.2f}% ({multi_better_page_count}/{len(results)})")

        # 按任务类型分析
        task_stats = {}
        for r in results:
            task_tag = r.get("task_tag", "Unknown")
            if task_tag not in task_stats:
                task_stats[task_tag] = {
                    "count": 0,
                    "single_chunk_f1": 0,
                    "multi_chunk_f1": 0,
                    "single_page_f1": 0,
                    "multi_page_f1": 0,
                    "single_page_success": 0,
                    "multi_page_success": 0
                }
            task_stats[task_tag]["count"] += 1
            task_stats[task_tag]["single_chunk_f1"] += r["single_intent"]["chunk_level"]["f1"]
            task_stats[task_tag]["multi_chunk_f1"] += r["multi_intent"]["chunk_level"]["f1"]
            task_stats[task_tag]["single_page_f1"] += r["single_intent"]["page_level"]["f1"]
            task_stats[task_tag]["multi_page_f1"] += r["multi_intent"]["page_level"]["f1"]
            if r["single_intent"]["page_level"]["success"]:
                task_stats[task_tag]["single_page_success"] += 1
            if r["multi_intent"]["page_level"]["success"]:
                task_stats[task_tag]["multi_page_success"] += 1

        logger.info(f"\n📊 按任务类型对比:")
        for task_tag, stats in task_stats.items():
            count = stats["count"]
            single_chunk_f1 = stats["single_chunk_f1"] / count
            multi_chunk_f1 = stats["multi_chunk_f1"] / count
            single_page_f1 = stats["single_page_f1"] / count
            multi_page_f1 = stats["multi_page_f1"] / count
            single_page_success_rate = (stats["single_page_success"] / count) * 100
            multi_page_success_rate = (stats["multi_page_success"] / count) * 100

            logger.info(f"   {task_tag} ({count}样本):")
            logger.info(f"     Chunk F1 - 单意图: {single_chunk_f1:.4f}, 多意图: {multi_chunk_f1:.4f}")
            logger.info(f"     Page F1 - 单意图: {single_page_f1:.4f}, 多意图: {multi_page_f1:.4f}")
            logger.info(
                f"     Page成功率 - 单意图: {single_page_success_rate:.1f}%, 多意图: {multi_page_success_rate:.1f}%")

        logger.info(f"{'=' * 80}")

        # 创建可视化
        self.create_comparison_visualization(results)

    def create_comparison_visualization(self, results):
        """创建对比可视化图表"""
        try:
            vis_dir = os.path.join(self.config['results_dir'], 'chunk_intent_visualizations')
            os.makedirs(vis_dir, exist_ok=True)

            # 1. Chunk级别和Page级别性能对比
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

            # Chunk级别F1对比
            single_chunk_f1s = [r["single_intent"]["chunk_level"]["f1"] for r in results]
            multi_chunk_f1s = [r["multi_intent"]["chunk_level"]["f1"] for r in results]

            ax1.bar(['单意图', '多意图'],
                    [np.mean(single_chunk_f1s), np.mean(multi_chunk_f1s)],
                    color=['lightblue', 'lightcoral'])
            ax1.set_ylabel('F1 Score')
            ax1.set_title('Chunk级别F1性能对比')
            ax1.set_ylim(0, 1)
            ax1.grid(True, alpha=0.3)

            # Page级别F1对比
            single_page_f1s = [r["single_intent"]["page_level"]["f1"] for r in results]
            multi_page_f1s = [r["multi_intent"]["page_level"]["f1"] for r in results]

            ax2.bar(['单意图', '多意图'],
                    [np.mean(single_page_f1s), np.mean(multi_page_f1s)],
                    color=['lightblue', 'lightcoral'])
            ax2.set_ylabel('F1 Score')
            ax2.set_title('Page级别F1性能对比')
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3)

            # 检索时间对比
            single_times = [r["single_intent"]["retrieval_time"] for r in results]
            multi_times = [r["multi_intent"]["retrieval_time"] for r in results]

            ax3.bar(['单意图', '多意图'],
                    [np.mean(single_times), np.mean(multi_times)],
                    color=['lightgreen', 'orange'])
            ax3.set_ylabel('检索时间 (秒)')
            ax3.set_title('检索时间对比')
            ax3.grid(True, alpha=0.3)

            # 成功率对比
            single_success_rate = sum(1 for r in results if r["single_intent"]["page_level"]["success"]) / len(
                results) * 100
            multi_success_rate = sum(1 for r in results if r["multi_intent"]["page_level"]["success"]) / len(
                results) * 100

            ax4.bar(['单意图', '多意图'],
                    [single_success_rate, multi_success_rate],
                    color=['gold', 'purple'])
            ax4.set_ylabel('成功率 (%)')
            ax4.set_title('Page级别成功率对比')
            ax4.set_ylim(0, 100)
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, 'chunk_intent_performance_comparison.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()

            # 2. F1提升分布图
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            chunk_f1_improvements = [r["comparison"]["chunk_f1_improvement"] for r in results]
            plt.hist(chunk_f1_improvements, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.axvline(0, color='red', linestyle='--', label='无提升线')
            plt.xlabel('Chunk F1提升值')
            plt.ylabel('文档数量')
            plt.title('Chunk级别F1提升分布')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(1, 2, 2)
            page_f1_improvements = [r["comparison"]["page_f1_improvement"] for r in results]
            plt.hist(page_f1_improvements, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            plt.axvline(0, color='red', linestyle='--', label='无提升线')
            plt.xlabel('Page F1提升值')
            plt.ylabel('文档数量')
            plt.title('Page级别F1提升分布')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, 'chunk_intent_f1_improvement_distribution.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"📈 可视化图表已保存到: {vis_dir}")

        except Exception as e:
            logger.error(f"❌ 创建可视化时出错: {str(e)}")

    def run(self):
        """运行测试"""
        logger.info("🚀 开始Chunk级别单意图vs多意图检索对比测试...")
        start_time = time.time()

        try:
            results = self.test_chunk_intent_comparison()
            total_time = time.time() - start_time
            logger.info(f"\n🎉 对比测试完成！总耗时: {total_time:.2f}秒")
            logger.info(f"📁 结果已保存到: {self.config['results_dir']}")

        except Exception as e:
            logger.error(f"❌ 测试过程中出现错误: {str(e)}", exc_info=True)


def main():
    """主函数"""
    print("🎯 Chunk级别单意图 vs 多意图检索对比测试")
    print("=" * 60)
    print("📄 基于Chunk级别数据进行单意图和多意图拆解效果对比")
    print("🔄 使用Union策略将Chunk结果聚合到Page级别")
    print("📊 提供Chunk级别和Page级别的双重评估")
    print("=" * 60)

    parser = argparse.ArgumentParser(description="Chunk级别单意图vs多意图检索对比测试工具")
    parser.add_argument("--sample_size", type=int, default=100, help="测试样本数量，0表示全部")
    parser.add_argument("--embedding_topk", type=int, default=15, help="初始检索数量")
    parser.add_argument("--rerank_topk", type=int, default=10, help="重排序后返回数量")
    parser.add_argument("--chunk_data_dir", type=str, help="chunk数据目录路径")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")

    try:
        args = parser.parse_args()
    except SystemExit:
        args = argparse.Namespace(
            sample_size=0, embedding_topk=15, rerank_topk=10,
            chunk_data_dir=None, debug=False
        )

    logger.info(f"📊 测试样本数量: {args.sample_size if args.sample_size > 0 else '全部'}")
    logger.info(f"🔍 初始检索: Top-{args.embedding_topk}")
    logger.info(f"🎯 最终返回: Top-{args.rerank_topk}")
    logger.info(f"🐛 调试模式: {args.debug}")

    # 先检查模型文件
    bge_path = "/root/autodl-tmp/multimodal-RAG/hf_models/bge-large-en-v1.5"
    reranker_path = "/root/autodl-tmp/multimodal-RAG/hf_models/bge-reranker-large"

    if not os.path.exists(bge_path):
        print(f"❌ BGE模型路径不存在: {bge_path}")
        return

    if not os.path.exists(reranker_path):
        print(f"❌ 重排序器路径不存在: {reranker_path}")
        return

    tester = ChunkIntentTester()

    # 更新配置
    if args.sample_size:
        tester.config['sample_size'] = args.sample_size
    if args.embedding_topk:
        tester.config['embedding_topk'] = args.embedding_topk
    if args.rerank_topk:
        tester.config['rerank_topk'] = args.rerank_topk
    if args.chunk_data_dir:
        tester.config['chunk_data_dir'] = args.chunk_data_dir

    tester.run()


if __name__ == "__main__":
    main()
