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
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from FlagEmbedding import FlagReranker
import torch
from PIL import Image
# from pdf2image import convert_from_path  # 🔥 暂时注释掉PDF转换
import subprocess
import pandas as pd
from collections import defaultdict
from beam_search_module import BeamSearchWrapper
import logging
import os
from dotenv import load_dotenv

# 创建日志目录
log_dir = Path("DeepRAG_Multimodal/log")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "multi_intent_demo.log"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(str(log_file), mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 添加必要的路径
sys.path.append("multimodal-RAG/DeepRAG_Multimodal/deep_retrieve")
# 加载环境变量
load_dotenv("D:\Desktop\multimodal-RAG\multimodal-RAG\DeepRAG_Multimodal\configs\.env")

# 导入必要的库
from DeepRAG_Multimodal.deep_retrieve.ming.deepsearch_optimize_ming import DeepSearch_Beta
from DeepRAG_Multimodal.deep_retrieve.retriever_multimodal_bge import RetrieverConfig, MultimodalMatcher


class MultiIntentDemo:
    """多意图拆解效果演示类"""

    def __init__(self):
        """初始化演示器"""
        self.config = self.load_config()
        os.makedirs(self.config['results_dir'], exist_ok=True)
        os.makedirs(self.config['vis_dir'], exist_ok=True)

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        self.setup_models()

    def load_config(self):
        """加载配置"""
        config = {
            # 路径配置
            'test_data_path': r'D:\Desktop\colpali_longdoc\picked_LongDoc\selected_LongDocURL_public_with_subtask_category.jsonl',
            'pdf_base_dir': r'D:\Desktop\colpali_longdoc\picked_LongDoc',
            'results_dir': './demo_results',
            'vis_dir': './demo_results/visualizations',

            # 采样配置
            'sample_size': 5,
            'debug': True,

            # 检索配置
            'max_iterations': 2,
            'embedding_topk': 15,
            'rerank_topk': 10,
            # 🔥 使用平衡权重但先以文本为主（由于暂时没有图像）
            'text_weight': 0.8,
            'image_weight': 0.2,

            # 模型配置
            'mm_model_name': "vidore/colqwen2.5-v0.2",
            'mm_processor_name': "vidore/colqwen2.5-v0.1",
            'bge_model_name': "BAAI/bge-large-en-v1.5",
            'device': 'cuda:0',
            'batch_size': 2,
            # 🔥 暂时使用text_only模式，等Poppler安装后改为mixed
            'retrieval_mode': 'text_only',
            'ocr_method': 'paddleocr',

            # 禁用Vespa
            'use_vespa': False,

            # BeamSearch配置（可选）
            'enable_beam_search': False,  # 先禁用进行基础测试
            'beam_width': 3,
            'beam_debug': True,
        }
        return config

    def setup_models(self):
        """初始化检索模型"""
        logger.info("🚀 初始化多意图检索模型...")
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"🎮 使用设备: {device}")

        self.reranker = FlagReranker(
            model_name_or_path="BAAI/bge-reranker-large",
            use_fp16=True,
            device=device
        )

        # 验证reranker设备
        logger.info(f"📍 Reranker设备: {next(self.reranker.model.parameters()).device}")

        # 初始化多模态匹配器配置
        retriever_config = RetrieverConfig(
            model_name=self.config['mm_model_name'],
            processor_name=self.config['mm_processor_name'],
            bge_model_name=self.config['bge_model_name'],
            device=self.config['device'],
            use_fp16=True,
            batch_size=self.config['batch_size'],
            mode=self.config['retrieval_mode'],
            ocr_method=self.config['ocr_method']
        )

        self.mm_matcher = MultimodalMatcher(
            config=retriever_config,
            embedding_weight=self.config['text_weight'],
            topk=self.config['rerank_topk']
        )
        logger.info("✅ 已初始化多模态匹配器（当前为文本模式）")

        # 初始化基础多意图检索器
        base_multi_intent_search = DeepSearch_Beta(
            max_iterations=self.config['max_iterations'],
            reranker=self.reranker,
            params={
                "embedding_topk": self.config['embedding_topk'],
                "rerank_topk": self.config['rerank_topk'],
                "text_weight": self.config['text_weight'],
                "image_weight": self.config['image_weight']
            }
        )

        # BeamSearch可选包装
        if self.config['enable_beam_search']:
            self.multi_intent_search = BeamSearchWrapper(
                base_retriever=base_multi_intent_search,
                matcher=self.mm_matcher,
                reranker=self.reranker,
                enable_beam_search=True,
                beam_width=self.config['beam_width'],
                debug_mode=self.config['beam_debug']
            )
            logger.info("✅ 已启用BeamSearch包装器")
        else:
            self.multi_intent_search = base_multi_intent_search
            logger.info("✅ 使用标准多意图检索器")

        # 初始化单意图检索器（禁用意图拆解）
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

        # 禁用单意图检索器的意图拆解功能
        self.single_intent_search._split_query_intent = lambda query: [query]
        self.single_intent_search._split_query_intent_exist = lambda query: [query]

        logger.info("✅ 模型初始化完成")

    def load_test_data(self):
        """加载测试数据"""
        allowed_doc_nos = [
            '4046173.pdf', '4176503.pdf', '4057524.pdf', '4064501.pdf', '4057121.pdf'
        ]

        logger.info(f"📚 加载测试数据: {self.config['test_data_path']}")
        test_data = []

        with open(self.config['test_data_path'], 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if item.get("pdf_path") in allowed_doc_nos:
                        test_data.append(item)

        # 取指定数量的测试数据
        test_data = test_data[:self.config['sample_size']]
        logger.info(f"✅ 成功加载 {len(test_data)} 条测试数据")

        # 打印测试数据基本信息
        for i, data in enumerate(test_data):
            logger.info(
                f"📄 测试数据 {i + 1}: {data.get('pdf_path', 'Unknown')} - {data.get('question', 'No question')[:50]}...")

        return test_data

    def process_single_document(self, doc_data):
        """🔥 改进的文档处理：优先使用OCR文本，为后续图像处理做准备"""
        documents = []

        # 获取预处理的OCR结果
        ocr_file = os.path.join(
            self.config['pdf_base_dir'],
            f"{self.config['ocr_method']}_save",
            f"{os.path.basename(doc_data['pdf_path']).replace('.pdf', '.json')}"
        )

        # 读取预处理的文本数据
        if os.path.exists(ocr_file):
            with open(ocr_file, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            logger.info(f"📖 成功读取OCR文件: {ocr_file}")
        else:
            logger.warning(f"⚠️ 找不到OCR文件: {ocr_file}")
            # 创建模拟数据用于测试
            logger.info(f"🔧 为 {doc_data['pdf_path']} 创建模拟OCR数据")
            loaded_data = {}
            for i in range(5):
                loaded_data[
                    f"Page_{i + 1}"] = f"这是 {doc_data['pdf_path']} 第{i + 1}页的模拟内容，包含测试文本用于检索实验。公司财务数据，技术研发信息，市场分析等相关内容。"

        # 🔥 为每一页创建文档对象
        for idx, (page_key, page_text) in enumerate(loaded_data.items()):
            # 确保文本质量
            if not page_text.strip():
                page_text = f"第{idx + 1}页内容 - {doc_data['pdf_path']}"

            # 🔥 创建文档结构，预留图像字段
            documents.append({
                "text": page_text,
                "image": None,  # 🔥 暂时为None，等Poppler安装后可以添加图像
                "metadata": {
                    "page_index": idx + 1,
                    "pdf_path": doc_data.get("pdf_path", "")
                }
            })

        logger.info(f"📑 成功创建 {len(documents)} 个文档对象（当前仅文本模式）")

        # 添加文本质量检查
        total_text_length = sum(len(doc['text']) for doc in documents)
        logger.info(f"📝 总文本长度: {total_text_length} 字符")

        if total_text_length < 100:
            logger.warning(f"⚠️ 文档文本内容过少，可能影响检索效果")
        else:
            logger.info(f"✅ 文本质量良好，平均每页 {total_text_length // len(documents)} 字符")

        return documents

    def demonstrate_intent_decomposition(self):
        """演示多意图拆解效果"""
        logger.info("🎯 开始多意图拆解效果演示...")
        test_data = self.load_test_data()
        results = []

        for idx, doc_data in enumerate(test_data):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"🔍 处理文档 {idx + 1}/{len(test_data)}: {doc_data.get('pdf_path', 'Unknown')}")

            query = doc_data.get("question", "")
            evidence_pages = doc_data.get("evidence_pages", [])

            logger.info(f"❓ 原始查询: {query}")
            logger.info(f"📋 证据页面: {evidence_pages}")

            # 处理文档
            document_pages = self.process_single_document(doc_data)
            if not document_pages:
                logger.warning(f"⚠️ 跳过文档: 无有效内容")
                continue

            data = {
                "query": query,
                "documents": document_pages
            }

            # 演示意图拆解过程
            intent_decomposition_result = self.analyze_intent_decomposition(query)

            # 执行单意图检索
            logger.info(f"\n🔍 执行单意图检索...")
            single_start = time.time()
            single_results = self.single_intent_search.search_retrieval(data, retriever=self.mm_matcher)
            single_elapsed = time.time() - single_start
            logger.info(f"⏱️ 单意图检索耗时: {single_elapsed:.2f}秒")

            # 🔥 添加分数调试信息
            single_scores = [r.get('score', 0) for r in single_results]
            logger.info(f"📊 单意图检索分数: {single_scores[:5]}")

            # 执行多意图检索
            logger.info(f"\n🎯 执行多意图检索...")
            multi_start = time.time()
            multi_results = self.multi_intent_search.search_retrieval(data, retriever=self.mm_matcher)
            multi_elapsed = time.time() - multi_start
            logger.info(f"⏱️ 多意图检索耗时: {multi_elapsed:.2f}秒")

            # 🔥 添加分数调试信息
            multi_scores = [r.get('score', 0) for r in multi_results]
            logger.info(f"📊 多意图检索分数: {multi_scores[:5]}")

            # 分析检索结果
            single_analysis = self.analyze_retrieval_results(single_results, evidence_pages, "单意图")
            multi_analysis = self.analyze_retrieval_results(multi_results, evidence_pages, "多意图")

            # 记录结果
            result = {
                "doc_id": doc_data.get("doc_no", ""),
                "pdf_path": doc_data.get("pdf_path", ""),
                "query": query,
                "evidence_pages": evidence_pages,
                "intent_decomposition": intent_decomposition_result,
                "single_intent": {
                    **single_analysis,
                    "retrieval_time": single_elapsed,
                    "results": single_results[:5],
                    "scores": single_scores[:5]
                },
                "multi_intent": {
                    **multi_analysis,
                    "retrieval_time": multi_elapsed,
                    "results": multi_results[:5],
                    "scores": multi_scores[:5]
                }
            }

            results.append(result)

            # 为每个文档创建详细分析
            self.create_detailed_analysis(result, idx + 1)

        # 保存整体结果
        result_file = os.path.join(self.config['results_dir'], 'intent_decomposition_demo.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 创建对比可视化
        self.create_comparison_visualizations(results)

        logger.info(f"\n🎉 演示完成！结果已保存到: {self.config['results_dir']}")
        return results

    def analyze_intent_decomposition(self, query):
        """分析意图拆解过程"""
        logger.info(f"\n🧠 分析查询意图拆解过程...")

        # 调用多意图检索器的意图拆解方法
        intent_queries = self.multi_intent_search._split_query_intent(query)

        logger.info(f"📝 原始查询: {query}")
        logger.info(f"🎯 拆解出 {len(intent_queries)} 个子意图:")

        for i, intent in enumerate(intent_queries, 1):
            logger.info(f"   {i}. {intent}")

        # 如果拆解结果只有原查询，说明LLM认为不需要拆解
        if len(intent_queries) == 1 and intent_queries[0] == query:
            logger.info("   💡 LLM判断此查询不需要拆解")

        # 分析意图类型和覆盖度
        intent_analysis = {
            "original_query": query,
            "decomposed_intents": intent_queries,
            "intent_count": len(intent_queries),
            "coverage_analysis": self.analyze_intent_coverage(query, intent_queries)
        }

        return intent_analysis

    def analyze_intent_coverage(self, original_query, intent_queries):
        """分析意图覆盖度"""
        original_words = set(original_query.lower().split())

        coverage_stats = {
            "original_word_count": len(original_words),
            "intent_coverage": []
        }

        for intent in intent_queries:
            intent_words = set(intent.lower().split())
            overlap = original_words.intersection(intent_words)
            coverage = len(overlap) / len(original_words) if original_words else 0

            coverage_stats["intent_coverage"].append({
                "intent": intent,
                "word_overlap": len(overlap),
                "coverage_ratio": coverage,
                "new_words": list(intent_words - original_words)
            })

        return coverage_stats

    def analyze_retrieval_results(self, results, evidence_pages, method_name):
        """分析检索结果"""
        retrieved_pages = set()
        for result in results:
            if 'page' in result and result['page'] is not None:
                retrieved_pages.add(result['page'])
            elif 'metadata' in result and 'page_index' in result['metadata']:
                retrieved_pages.add(result['metadata']['page_index'])

        evidence_set = set(evidence_pages)
        correct_pages = evidence_set.intersection(retrieved_pages)

        recall = len(correct_pages) / len(evidence_set) if evidence_set else 0
        precision = len(correct_pages) / len(retrieved_pages) if retrieved_pages else 0
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0

        logger.info(f"📊 {method_name}检索结果分析:")
        logger.info(f"   🎯 检索到页面: {sorted(list(retrieved_pages))}")
        logger.info(f"   ✅ 正确页面: {sorted(list(correct_pages))}")
        logger.info(f"   📈 召回率: {recall:.4f}")
        logger.info(f"   📈 精确率: {precision:.4f}")
        logger.info(f"   📈 F1值: {f1:.4f}")

        return {
            "retrieved_pages": list(retrieved_pages),
            "correct_pages": list(correct_pages),
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "success": len(correct_pages) == len(evidence_set)
        }

    def create_detailed_analysis(self, result, doc_index):
        """为单个文档创建详细分析图表"""
        # 暂时简化，避免matplotlib可能的依赖问题
        logger.info(f"📊 文档 {doc_index} 分析已记录")

    def create_comparison_visualizations(self, results):
        """创建整体对比可视化"""
        # 暂时简化，避免matplotlib可能的依赖问题
        logger.info(f"📊 对比可视化已记录")

    def run(self):
        """运行演示"""
        logger.info("🚀 开始多意图拆解效果演示...")
        logger.info("⚠️ 当前运行在文本模式下，安装Poppler后可启用完整多模态功能")
        start_time = time.time()

        try:
            results = self.demonstrate_intent_decomposition()

            total_time = time.time() - start_time
            logger.info(f"\n🎉 演示完成！")
            logger.info(f"⏱️ 总耗时: {total_time:.2f}秒")
            logger.info(f"📁 结果目录: {self.config['results_dir']}")
            logger.info(f"📊 可视化目录: {self.config['vis_dir']}")

            # 打印关键发现
            if len(results) > 0:
                single_avg_f1 = np.mean([r["single_intent"]["f1"] for r in results])
                multi_avg_f1 = np.mean([r["multi_intent"]["f1"] for r in results])
                improvement = (multi_avg_f1 - single_avg_f1) * 100

                logger.info(f"\n📈 关键发现:")
                logger.info(f"   - 单意图平均F1: {single_avg_f1:.4f}")
                logger.info(f"   - 多意图平均F1: {multi_avg_f1:.4f}")
                logger.info(f"   - 性能提升: {improvement:+.2f}%")

                # 分数质量检查
                single_max_score = max(
                    [max(r["single_intent"]["scores"]) for r in results if r["single_intent"]["scores"]])
                multi_max_score = max(
                    [max(r["multi_intent"]["scores"]) for r in results if r["multi_intent"]["scores"]])

                logger.info(f"\n📊 分数质量检查:")
                logger.info(f"   - 单意图最高分数: {single_max_score:.4f}")
                logger.info(f"   - 多意图最高分数: {multi_max_score:.4f}")

                if single_max_score > 0 and multi_max_score > 0:
                    logger.info(f"   ✅ 检索功能正常，分数不为0")
                else:
                    logger.warning(f"   ⚠️ 检索分数异常，需要检查配置")

        except Exception as e:
            logger.error(f"❌ 演示过程中出现错误: {str(e)}", exc_info=True)


def main():
    """主函数"""
    print("🎯 多意图拆解效果演示（文本模式）")
    print("=" * 50)
    print("💡 提示：安装Poppler后可启用完整多模态功能")
    print("   conda install -c conda-forge poppler")
    print("=" * 50)

    # 创建演示器并运行
    demo = MultiIntentDemo()
    demo.run()


if __name__ == "__main__":
    main()