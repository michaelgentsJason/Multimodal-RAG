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
        logging.FileHandler(str(log_file), mode='w', encoding='utf-8'),  # 使用'w'模式清空日志
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

            # 采样配置 - 只测试2条数据
            'sample_size': 1,
            'debug': True,

            # 检索配置
            'max_iterations': 2,
            'embedding_topk': 8,
            'rerank_topk': 5,
            'text_weight': 1.0,
            'image_weight': 0.0,

            # 模型配置
            'mm_model_name': "vidore/colqwen2.5-v0.2",
            'mm_processor_name': "vidore/colqwen2.5-v0.1",
            'bge_model_name': "BAAI/bge-large-en-v1.5",
            'device': 'cuda:0',
            'batch_size': 2,
            'retrieval_mode': 'text_only',  # 专注于文本检索
            'ocr_method': 'paddleocr',

            # 禁用Vespa
            'use_vespa': False,

            # 🎯 添加 Beam Search 配置
            'enable_beam_search': True,  # 主开关
            'beam_width': 3,  # beam宽度
            'beam_debug': True,  # 调试模式
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
            use_fp16=True,  # 启用FP16加速
            device=device  # 🔥 强制使用GPU
        )

        # 验证reranker设备
        logger.info(f"📍 Reranker设备: {next(self.reranker.model.parameters()).device}")

        # 初始化重排序器
        self.reranker = FlagReranker(model_name_or_path="BAAI/bge-reranker-large")

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

        # 使用标准多模态匹配器（不使用Vespa）
        self.mm_matcher = MultimodalMatcher(
            config=retriever_config,
            embedding_weight=self.config['text_weight'],
            topk=self.config['rerank_topk']
        )
        logger.info("✅ 已初始化标准多模态匹配器")

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

        # 🎯 用Beam Search包装器包装（可开关控制）
        self.multi_intent_search = BeamSearchWrapper(
            base_retriever=base_multi_intent_search,
            matcher=self.mm_matcher,
            reranker=self.reranker,
            enable_beam_search=True,  # 🔥 在这里控制开关！
            beam_width=3,
            debug_mode=True
        )

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

        # 只取前2条数据
        test_data = test_data[:self.config['sample_size']]
        logger.info(f"✅ 成功加载 {len(test_data)} 条测试数据")

        # 打印测试数据基本信息
        for i, data in enumerate(test_data):
            logger.info(
                f"📄 测试数据 {i + 1}: {data.get('pdf_path', 'Unknown')} - {data.get('question', 'No question')[:50]}...")

        return test_data

    def process_single_document(self, doc_data):
        """处理单个文档，直接使用OCR结果"""
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
                    f"Page_{i + 1}"] = f"这是 {doc_data['pdf_path']} 第{i + 1}页的模拟内容，包含测试文本用于检索实验。"

        # 为每一页创建文档对象
        for idx, (page_key, page_text) in enumerate(loaded_data.items()):
            documents.append({
                "text": page_text if page_text.strip() else f"第{idx + 1}页内容",
                "image": None,
                "metadata": {
                    "page_index": idx + 1,
                    "pdf_path": doc_data.get("pdf_path", "")
                }
            })

        logger.info(f"📑 成功创建 {len(documents)} 个文档对象")
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

            # 执行多意图检索
            logger.info(f"\n🎯 执行多意图检索...")
            multi_start = time.time()
            multi_results = self.multi_intent_search.search_retrieval(data, retriever=self.mm_matcher)
            multi_elapsed = time.time() - multi_start
            logger.info(f"⏱️ 多意图检索耗时: {multi_elapsed:.2f}秒")

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
                    "results": single_results[:5]  # 只保存前5个结果
                },
                "multi_intent": {
                    **multi_analysis,
                    "retrieval_time": multi_elapsed,
                    "results": multi_results[:5]  # 只保存前5个结果
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
        # 简单的关键词覆盖分析
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
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'文档 {doc_index}: 多意图拆解详细分析\n查询: {result["query"][:50]}...', fontsize=14,
                     fontweight='bold')

        # 1. 意图拆解可视化
        ax1 = axes[0, 0]
        intent_data = result["intent_decomposition"]
        intents = intent_data["decomposed_intents"]

        # 创建意图长度条形图
        intent_lengths = [len(intent.split()) for intent in intents]
        bars = ax1.bar(range(len(intents)), intent_lengths, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax1.set_title('拆解意图长度分布', fontweight='bold')
        ax1.set_xlabel('意图编号')
        ax1.set_ylabel('词数')
        ax1.set_xticks(range(len(intents)))
        ax1.set_xticklabels([f'意图{i + 1}' for i in range(len(intents))])

        # 添加数值标签
        for bar, length in zip(bars, intent_lengths):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     str(length), ha='center', va='bottom')

        # 2. 检索性能对比
        ax2 = axes[0, 1]
        metrics = ['召回率', '精确率', 'F1值']
        single_scores = [result["single_intent"]["recall"], result["single_intent"]["precision"],
                         result["single_intent"]["f1"]]
        multi_scores = [result["multi_intent"]["recall"], result["multi_intent"]["precision"],
                        result["multi_intent"]["f1"]]

        x = np.arange(len(metrics))
        width = 0.35

        bars1 = ax2.bar(x - width / 2, single_scores, width, label='单意图', color='#FF6B6B', alpha=0.8)
        bars2 = ax2.bar(x + width / 2, multi_scores, width, label='多意图', color='#4ECDC4', alpha=0.8)

        ax2.set_title('检索性能对比', fontweight='bold')
        ax2.set_ylabel('分数')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics)
        ax2.legend()
        ax2.set_ylim(0, 1)

        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                         f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        # 3. 检索时间对比
        ax3 = axes[1, 0]
        methods = ['单意图', '多意图']
        times = [result["single_intent"]["retrieval_time"], result["multi_intent"]["retrieval_time"]]

        bars = ax3.bar(methods, times, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
        ax3.set_title('检索时间对比', fontweight='bold')
        ax3.set_ylabel('时间 (秒)')

        # 添加数值标签
        for bar, time_val in zip(bars, times):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{time_val:.2f}s', ha='center', va='bottom')

        # 4. 页面检索准确性
        ax4 = axes[1, 1]
        evidence_pages = set(result["evidence_pages"])
        single_pages = set(result["single_intent"]["retrieved_pages"])
        multi_pages = set(result["multi_intent"]["retrieved_pages"])

        # 创建维恩图式的分析
        categories = ['仅单意图', '仅多意图', '两者共同', '遗漏页面']
        single_only = single_pages - multi_pages - evidence_pages
        multi_only = multi_pages - single_pages - evidence_pages
        both_correct = (single_pages & multi_pages) & evidence_pages
        missed = evidence_pages - (single_pages | multi_pages)

        counts = [len(single_only), len(multi_only), len(both_correct), len(missed)]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

        wedges, texts, autotexts = ax4.pie(counts, labels=categories, colors=colors, autopct='%1.0f',
                                           startangle=90)
        ax4.set_title('页面检索分布', fontweight='bold')

        plt.tight_layout()

        # 保存图表
        chart_file = os.path.join(self.config['vis_dir'], f'doc_{doc_index}_detailed_analysis.png')
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"📊 详细分析图表已保存: {chart_file}")

        # 创建意图拆解文本分析
        self.create_intent_text_analysis(result, doc_index)

    def create_intent_text_analysis(self, result, doc_index):
        """创建意图拆解的文本分析报告"""
        report_file = os.path.join(self.config['vis_dir'], f'doc_{doc_index}_intent_analysis.txt')

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"文档 {doc_index} 多意图拆解分析报告\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"📄 文档: {result['pdf_path']}\n")
            f.write(f"❓ 原始查询: {result['query']}\n")
            f.write(f"📋 证据页面: {result['evidence_pages']}\n\n")

            f.write("🎯 意图拆解结果:\n")
            f.write("-" * 30 + "\n")
            intent_data = result["intent_decomposition"]
            for i, intent in enumerate(intent_data["decomposed_intents"], 1):
                f.write(f"{i}. {intent}\n")

            f.write(f"\n📊 拆解统计:\n")
            f.write(f"   - 原始查询词数: {len(result['query'].split())}\n")
            f.write(f"   - 拆解意图数量: {intent_data['intent_count']}\n")

            f.write(f"\n🔍 检索结果对比:\n")
            f.write("-" * 30 + "\n")
            f.write(f"单意图检索:\n")
            f.write(f"   - 检索页面: {result['single_intent']['retrieved_pages']}\n")
            f.write(f"   - 正确页面: {result['single_intent']['correct_pages']}\n")
            f.write(f"   - F1值: {result['single_intent']['f1']:.4f}\n")
            f.write(f"   - 用时: {result['single_intent']['retrieval_time']:.2f}秒\n\n")

            f.write(f"多意图检索:\n")
            f.write(f"   - 检索页面: {result['multi_intent']['retrieved_pages']}\n")
            f.write(f"   - 正确页面: {result['multi_intent']['correct_pages']}\n")
            f.write(f"   - F1值: {result['multi_intent']['f1']:.4f}\n")
            f.write(f"   - 用时: {result['multi_intent']['retrieval_time']:.2f}秒\n\n")

            f.write(f"📈 性能提升:\n")
            f.write("-" * 30 + "\n")
            recall_diff = result['multi_intent']['recall'] - result['single_intent']['recall']
            precision_diff = result['multi_intent']['precision'] - result['single_intent']['precision']
            f1_diff = result['multi_intent']['f1'] - result['single_intent']['f1']

            f.write(f"   - 召回率变化: {recall_diff:+.4f}\n")
            f.write(f"   - 精确率变化: {precision_diff:+.4f}\n")
            f.write(f"   - F1值变化: {f1_diff:+.4f}\n")

            if f1_diff > 0:
                f.write(f"   ✅ 多意图检索表现更好\n")
            elif f1_diff < 0:
                f.write(f"   ❌ 单意图检索表现更好\n")
            else:
                f.write(f"   ➖ 两种方法表现相当\n")

        logger.info(f"📝 意图分析报告已保存: {report_file}")

    def create_comparison_visualizations(self, results):
        """创建整体对比可视化"""
        logger.info(f"📊 创建整体对比可视化...")

        # 1. 整体性能对比图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('多意图拆解整体效果分析', fontsize=16, fontweight='bold')

        # 性能指标对比
        ax1 = axes[0, 0]
        metrics = ['召回率', '精确率', 'F1值']

        single_avg = [
            np.mean([r["single_intent"]["recall"] for r in results]),
            np.mean([r["single_intent"]["precision"] for r in results]),
            np.mean([r["single_intent"]["f1"] for r in results])
        ]

        multi_avg = [
            np.mean([r["multi_intent"]["recall"] for r in results]),
            np.mean([r["multi_intent"]["precision"] for r in results]),
            np.mean([r["multi_intent"]["f1"] for r in results])
        ]

        x = np.arange(len(metrics))
        width = 0.35

        bars1 = ax1.bar(x - width / 2, single_avg, width, label='单意图', color='#FF6B6B', alpha=0.8)
        bars2 = ax1.bar(x + width / 2, multi_avg, width, label='多意图', color='#4ECDC4', alpha=0.8)

        ax1.set_title('平均性能对比', fontweight='bold')
        ax1.set_ylabel('分数')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.legend()
        ax1.set_ylim(0, 1)

        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                         f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        # 2. 时间对比
        ax2 = axes[0, 1]
        single_times = [r["single_intent"]["retrieval_time"] for r in results]
        multi_times = [r["multi_intent"]["retrieval_time"] for r in results]

        methods = ['单意图', '多意图']
        avg_times = [np.mean(single_times), np.mean(multi_times)]

        bars = ax2.bar(methods, avg_times, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
        ax2.set_title('平均检索时间对比', fontweight='bold')
        ax2.set_ylabel('时间 (秒)')

        for bar, time_val in zip(bars, avg_times):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{time_val:.2f}s', ha='center', va='bottom')

        # 3. 意图拆解统计
        ax3 = axes[1, 0]
        intent_counts = [r["intent_decomposition"]["intent_count"] for r in results]

        ax3.hist(intent_counts, bins=range(1, max(intent_counts) + 2), alpha=0.7, color='#45B7D1', edgecolor='black')
        ax3.set_title('意图拆解数量分布', fontweight='bold')
        ax3.set_xlabel('拆解意图数量')
        ax3.set_ylabel('文档数量')

        # 4. 成功率对比
        ax4 = axes[1, 1]
        single_success = sum(1 for r in results if r["single_intent"]["success"])
        multi_success = sum(1 for r in results if r["multi_intent"]["success"])

        categories = ['单意图成功', '多意图成功']
        success_counts = [single_success, multi_success]

        bars = ax4.bar(categories, success_counts, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
        ax4.set_title('完全成功案例数', fontweight='bold')
        ax4.set_ylabel('成功数量')

        for bar, count in zip(bars, success_counts):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     str(count), ha='center', va='bottom')

        plt.tight_layout()

        # 保存图表
        overview_file = os.path.join(self.config['vis_dir'], 'overall_comparison.png')
        plt.savefig(overview_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"📊 整体对比图表已保存: {overview_file}")

        # 创建总结报告
        self.create_summary_report(results)

    def create_summary_report(self, results):
        """创建总结报告"""
        report_file = os.path.join(self.config['results_dir'], 'summary_report.txt')

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("多意图拆解效果演示总结报告\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"📊 测试概况:\n")
            f.write(f"   - 测试文档数量: {len(results)}\n")
            f.write(f"   - 检索模式: {self.config['retrieval_mode']}\n")
            f.write(f"   - OCR方法: {self.config['ocr_method']}\n\n")

            # 计算平均指标
            single_avg_recall = np.mean([r["single_intent"]["recall"] for r in results])
            multi_avg_recall = np.mean([r["multi_intent"]["recall"] for r in results])
            single_avg_precision = np.mean([r["single_intent"]["precision"] for r in results])
            multi_avg_precision = np.mean([r["multi_intent"]["precision"] for r in results])
            single_avg_f1 = np.mean([r["single_intent"]["f1"] for r in results])
            multi_avg_f1 = np.mean([r["multi_intent"]["f1"] for r in results])

            f.write(f"📈 性能指标对比:\n")
            f.write(f"   单意图检索:\n")
            f.write(f"     - 平均召回率: {single_avg_recall:.4f}\n")
            f.write(f"     - 平均精确率: {single_avg_precision:.4f}\n")
            f.write(f"     - 平均F1值: {single_avg_f1:.4f}\n")
            f.write(f"   多意图检索:\n")
            f.write(f"     - 平均召回率: {multi_avg_recall:.4f}\n")
            f.write(f"     - 平均精确率: {multi_avg_precision:.4f}\n")
            f.write(f"     - 平均F1值: {multi_avg_f1:.4f}\n\n")

            f.write(f"🚀 性能提升:\n")
            f.write(f"   - 召回率提升: {multi_avg_recall - single_avg_recall:+.4f}\n")
            f.write(f"   - 精确率提升: {multi_avg_precision - single_avg_precision:+.4f}\n")
            f.write(f"   - F1值提升: {multi_avg_f1 - single_avg_f1:+.4f}\n\n")

            # 时间分析
            single_avg_time = np.mean([r["single_intent"]["retrieval_time"] for r in results])
            multi_avg_time = np.mean([r["multi_intent"]["retrieval_time"] for r in results])

            f.write(f"⏱️ 时间效率:\n")
            f.write(f"   - 单意图平均时间: {single_avg_time:.2f}秒\n")
            f.write(f"   - 多意图平均时间: {multi_avg_time:.2f}秒\n")
            f.write(f"   - 时间增加: {multi_avg_time - single_avg_time:+.2f}秒\n\n")

            # 成功率分析
            single_success = sum(1 for r in results if r["single_intent"]["success"])
            multi_success = sum(1 for r in results if r["multi_intent"]["success"])

            f.write(f"🎯 成功率分析:\n")
            f.write(
                f"   - 单意图完全成功: {single_success}/{len(results)} ({single_success / len(results) * 100:.1f}%)\n")
            f.write(
                f"   - 多意图完全成功: {multi_success}/{len(results)} ({multi_success / len(results) * 100:.1f}%)\n\n")

            # 意图拆解分析
            intent_counts = [r["intent_decomposition"]["intent_count"] for r in results]
            avg_intent_count = np.mean(intent_counts)

            f.write(f"🧠 意图拆解分析:\n")
            f.write(f"   - 平均拆解意图数: {avg_intent_count:.1f}\n")
            f.write(f"   - 拆解范围: {min(intent_counts)} - {max(intent_counts)}\n\n")

            f.write(f"💡 结论:\n")
            if multi_avg_f1 > single_avg_f1:
                f.write(f"   ✅ 多意图拆解方法在F1值上平均提升了 {(multi_avg_f1 - single_avg_f1) * 100:.2f}%\n")
                f.write(f"   ✅ 建议在复杂查询场景中使用多意图拆解方法\n")
            else:
                f.write(f"   ⚠️ 在此测试集上，多意图拆解未显示明显优势\n")
                f.write(f"   ⚠️ 可能需要更大的测试集或调整拆解策略\n")

            f.write(
                f"   ⏱️ 多意图方法平均增加 {((multi_avg_time - single_avg_time) / single_avg_time) * 100:.1f}% 的检索时间\n")

        logger.info(f"📝 总结报告已保存: {report_file}")

    def run(self):
        """运行演示"""
        logger.info("🚀 开始多意图拆解效果演示...")
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

        except Exception as e:
            logger.error(f"❌ 演示过程中出现错误: {str(e)}", exc_info=True)


def main():
    """主函数"""
    print("🎯 多意图拆解效果演示")
    print("=" * 50)

    # 创建演示器并运行
    demo = MultiIntentDemo()
    demo.run()


if __name__ == "__main__":
    main()