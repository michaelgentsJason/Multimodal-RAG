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
from dotenv import load_dotenv
import torch
from pdf2image import convert_from_path
from PIL import Image
import logging
from DeepRAG_Multimodal.deep_retrieve.retriever_multimodal_bge import MultimodalMatcher, RetrieverConfig

# 创建日志
log_dir = Path("./log")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "text_only_baseline_test.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(str(log_file), mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("=== 纯文本Baseline多意图检索测试开始 ===")

# 添加路径
sys.path.append("multimodal-RAG/DeepRAG_Multimodal/deep_retrieve")

# 加载环境变量
load_dotenv("/root/autodl-tmp/multimodal-RAG/multimodal-RAG/DeepRAG_Multimodal/configs/.env")

# 导入必要的库
from DeepRAG_Multimodal.deep_retrieve.ming.deepsearch_optimize_ming import DeepSearch_Beta


class TextOnlyMultiIntentTester:
    """混合多意图检索测试类"""

    def __init__(self):
        """初始化测试器"""
        self.config = self.load_config()
        os.makedirs(self.config['results_dir'], exist_ok=True)
        self.verify_model_files()
        self.setup_models()

    def load_config(self):
        """加载配置"""
        config = {
            # 路径配置
            'test_data_path': '/root/autodl-tmp/multimodal-RAG/multimodal-RAG/DeepRAG_Multimodal/picked_LongDoc/selected_LongDocURL_public_with_subtask_category.jsonl',
            'pdf_base_dir': '/root/autodl-tmp/multimodal-RAG/multimodal-RAG/DeepRAG_Multimodal/picked_LongDoc',
            'results_dir': './test_results',

            # 采样配置
            'sample_size': 1,  # 默认50条数据
            'debug': True,

            # 检索配置
            'max_iterations': 2,
            'embedding_topk': 12,
            'rerank_topk': 5,
            'text_weight': 0.7,
            'image_weight': 0.3,

            # 模型配置 - 只需要文本模型
            'mm_model_name': "/root/autodl-tmp/multimodal-RAG/hf_models/colqwen2.5-v0.2",
            'mm_processor_name': "/root/autodl-tmp/multimodal-RAG/hf_models/colqwen2.5-v0.1",
            'bge_model_name': "/root/autodl-tmp/multimodal-RAG/hf_models/bge-large-en-v1.5",
            'reranker_model_name': "/root/autodl-tmp/multimodal-RAG/hf_models/bge-reranker-large",

            'retrieval_mode': 'mixed',  # 'mixed', 'text_only', 'image_only'

            'device': 'cuda:0',
            'batch_size': 4,
            'ocr_method': 'pytesseract',
        }
        return config

    def verify_model_files(self):
        """验证模型文件"""
        model_paths = [
            self.config['bge_model_name'],
            self.config['reranker_model_name']
        ]

        for model_path in model_paths:
            if not os.path.exists(model_path):
                logger.error(f"❌ 模型路径不存在: {model_path}")
                raise FileNotFoundError(f"模型路径不存在: {model_path}")
            else:
                logger.info(f"✅ 模型路径验证成功: {model_path}")

    def setup_models(self):
        """初始化检索模型"""
        logger.info("🚀 初始化纯文本多意图检索模型...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"🎮 使用设备: {device}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated() / (1024 ** 3)
            logger.info(f"🧹 初始GPU内存使用: {initial_memory:.2f}GB")

        try:
            # 初始化重排序器
            from FlagEmbedding import FlagReranker

            logger.info("⏳ 初始化重排序器...")
            self.reranker = FlagReranker(
                model_name_or_path=self.config['reranker_model_name'],
                use_fp16=True,
                device=device,
                local_files_only=True
            )
            logger.info("✅ 重排序器初始化成功")

            # 初始化纯文本匹配器
            # logger.info("⏳ 初始化纯文本匹配器...")
            # self.text_matcher = TextOnlyMatcher(
            #     bge_model_path=self.config['bge_model_name'],
            #     device=device,
            #     topk=self.config['rerank_topk']
            # )
            # logger.info("✅ 纯文本匹配器初始化成功")

            logger.info("⏳ 初始化多模态匹配器...")
            retriever_config = RetrieverConfig(
                model_name=self.config['mm_model_name'],
                processor_name=self.config['mm_processor_name'],
                bge_model_name=self.config['bge_model_name'],
                device=self.config['device'],
                use_fp16=True,
                batch_size=self.config['batch_size'],
                mode=self.config['retrieval_mode'],  # 'mixed'
                ocr_method=self.config['ocr_method']
            )

            self.text_matcher = MultimodalMatcher(
                config=retriever_config,
                embedding_weight=self.config['text_weight'],
                topk=self.config['rerank_topk']
            )
            logger.info("✅ 多模态匹配器初始化成功")

            # 初始化单意图检索器
            logger.info("⏳ 初始化单意图检索器...")
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
            # # 重写方法，强制单意图
            # self.single_intent_search._split_query_intent = lambda query: [query]
            # self.single_intent_search._refine_query_intent = lambda original_query, intent_queries, context: [
            #     original_query]

            # 初始化多意图检索器
            logger.info("⏳ 初始化多意图检索器...")
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
            logger.info("✅ 多意图检索器初始化成功")

            if torch.cuda.is_available():
                final_memory = torch.cuda.memory_allocated() / (1024 ** 3)
                logger.info(f"📊 最终GPU内存使用: {final_memory:.2f}GB")

            logger.info("✅ 纯文本模型初始化完成")

        except Exception as e:
            logger.error(f"❌ 模型初始化失败: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def load_test_data(self):
        """加载测试数据"""
        allowed_doc_nos = [
            '4046173.pdf', '4176503.pdf', '4057524.pdf', '4064501.pdf', '4057121.pdf', '4174854.pdf',
            '4148165.pdf', '4129570.pdf', '4010333.pdf', '4147727.pdf', '4066338.pdf', '4031704.pdf',
            '4050613.pdf', '4072260.pdf', '4091919.pdf', '4094684.pdf', '4063393.pdf', '4132494.pdf',
            '4185438.pdf', '4129670.pdf', '4138347.pdf', '4190947.pdf', '4100212.pdf', '4173940.pdf',
            '4069930.pdf', '4174181.pdf', '4027862.pdf', '4012567.pdf', '4145761.pdf', '4078345.pdf',
            '4061601.pdf', '4170122.pdf', '4077673.pdf', '4107960.pdf', '4005877.pdf', '4196005.pdf',
            '4126467.pdf', '4088173.pdf', '4106951.pdf', '4086173.pdf', '4072232.pdf', '4111230.pdf',
            '4057714.pdf'
        ]

        logger.info(f"📚 加载测试数据: {self.config['test_data_path']}")
        test_data = []

        with open(self.config['test_data_path'], 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    test_data.append(item)

        if self.config['sample_size'] > 0 and len(test_data) > self.config['sample_size']:
            np.random.seed(42)
            test_data = np.random.choice(test_data, self.config['sample_size'], replace=False).tolist()

        logger.info(f"✅ 成功加载 {len(test_data)} 条测试数据")
        return test_data

    def process_single_document(self, doc_data):
        """处理单个文档 - 仅提取文本"""
        documents = []
        pdf_path = os.path.join(self.config['pdf_base_dir'], doc_data["pdf_path"])

        # 获取OCR结果
        ocr_file = os.path.join(
            self.config['pdf_base_dir'],
            f"{self.config['ocr_method']}_save",
            f"{os.path.basename(doc_data['pdf_path']).replace('.pdf', '.json')}"
        )

        if os.path.exists(ocr_file):
            with open(ocr_file, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            logger.info(f"📖 成功读取OCR文件: {ocr_file}")
        else:
            logger.warning(f"⚠️ 找不到OCR文件: {ocr_file}")
            return []

        # 图像，文本
        page_keys = list(loaded_data.keys())
        for idx, page_key in enumerate(page_keys):
            page_text = loaded_data[page_key]
            if not page_text.strip():
                page_text = f"第{idx + 1}页内容"

            page_image = None
            if self.config['retrieval_mode'] == 'mixed':
                try:
                    # 转换PDF页面为图像
                    pdf_path = os.path.join(self.config['pdf_base_dir'], doc_data["pdf_path"])
                    pages = convert_from_path(pdf_path)
                    if idx < len(pages):
                        page_image = pages[idx]
                except Exception as e:
                    logger.warning(f"⚠️ 无法加载图像页面 {idx + 1}: {str(e)}")

            documents.append({
                "text": page_text,
                "image": page_image,  # 添加图像
                "metadata": {
                    "page_index": idx + 1,
                    "pdf_path": doc_data.get("pdf_path", "")
                }
            })

        logger.info(f"📑 成功创建 {len(documents)} 个文档对象（混合模式）")

        # 文本质量检查
        total_text_length = sum(len(doc['text']) for doc in documents)
        logger.info(f"📝 总文本长度: {total_text_length} 字符")

        return documents

    def evaluate_results(self, results, evidence_pages, method_name):
        """评估检索结果"""
        # 保持页面和分数的对应关系
        page_score_pairs = []
        for result in results:
            page = None
            if 'metadata' in result and 'page_index' in result['metadata']:
                page = result['metadata']['page_index']
            elif 'page' in result and result['page'] is not None:
                page = result['page']

            if page is not None:
                page_score_pairs.append((page, result.get('score', 0)))

        # 按分数排序（保持对应关系）
        page_score_pairs.sort(key=lambda x: x[1], reverse=True)

        retrieved_pages = [pair[0] for pair in page_score_pairs]
        retrieval_scores = [pair[1] for pair in page_score_pairs]

        evidence_set = set(evidence_pages)
        correct_pages = evidence_set.intersection(set(retrieved_pages))

        recall = len(correct_pages) / len(evidence_set) if evidence_set else 0
        precision = len(correct_pages) / len(retrieved_pages) if retrieved_pages else 0
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0

        logger.info(f"📊 {method_name}结果:")
        logger.info(f"   🎯 检索到页面: {retrieved_pages[:5]}")  # 按分数顺序显示
        logger.info(f"   ✅ 正确页面: {sorted(list(correct_pages))}")
        logger.info(f"   📊 检索分数: {retrieval_scores[:5]}")
        logger.info(f"   📊 页面-分数对应: {list(zip(retrieved_pages[:5], retrieval_scores[:5]))}")

        return {
            "retrieved_pages": retrieved_pages,
            "correct_pages": list(correct_pages),
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "retrieval_scores": retrieval_scores[:10],
            "success": len(correct_pages) == len(evidence_set)
        }

    def test_text_only_retrieval(self):
        """测试纯文本单意图 vs 多意图检索对比"""
        logger.info("🎯 开始纯文本单意图 vs 多意图检索对比测试...")
        test_data = self.load_test_data()
        results = []

        for idx, doc_data in enumerate(tqdm(test_data, desc="单意图vs多意图对比测试")):
            try:
                query = doc_data.get("question", "")
                evidence_pages = doc_data.get("evidence_pages", [])

                logger.info(f"\n{'=' * 60}")
                logger.info(f"🔍 处理文档 {idx + 1}/{len(test_data)}: {doc_data.get('pdf_path', 'Unknown')}")
                logger.info(f"❓ 查询: {query}")
                logger.info(f"📋 证据页面: {evidence_pages}")

                document_pages = self.process_single_document(doc_data)
                if not document_pages:
                    logger.warning(f"⚠️ 跳过文档: 无有效内容")
                    continue

                data = {"query": query, "documents": document_pages}

                # 单意图检索
                logger.info("📄 开始单意图检索...")
                single_start_time = time.time()
                single_results = self.single_intent_search.search_retrieval(data, multi_intent=False, retriever=self.text_matcher)
                single_elapsed = time.time() - single_start_time

                # 多意图检索
                logger.info("📄 开始多意图检索...")
                multi_start_time = time.time()
                multi_results = self.multi_intent_search.search_retrieval(data, multi_intent=True, retriever=self.text_matcher)
                multi_elapsed = time.time() - multi_start_time

                # 评估单意图结果
                single_eval = self.evaluate_results(single_results, evidence_pages, "单意图")

                # 评估多意图结果
                multi_eval = self.evaluate_results(multi_results, evidence_pages, "多意图")

                # logger.info(f"⏱️ 单意图检索耗时: {single_elapsed:.2f}秒")
                logger.info(f"⏱️ 多意图检索耗时: {multi_elapsed:.2f}秒")

                # 记录对比结果
                result = {
                    "doc_id": doc_data.get("doc_no", ""),
                    "pdf_path": doc_data.get("pdf_path", ""),
                    "query": query,
                    "evidence_pages": evidence_pages,
                    "task_tag": doc_data.get("task_tag", ""),
                    "subTask": doc_data.get("subTask", []),

                    # 单意图结果
                    "single_intent": {
                        **single_eval,
                        "retrieval_time": single_elapsed
                    },

                    # 多意图结果
                    "multi_intent": {
                        **multi_eval,
                        "retrieval_time": multi_elapsed
                    },

                    # 对比指标
                    "comparison": {
                        "f1_improvement": multi_eval["f1"] - single_eval["f1"],
                        "recall_improvement": multi_eval["recall"] - single_eval["recall"],
                        "precision_improvement": multi_eval["precision"] - single_eval["precision"],
                        "time_overhead": multi_elapsed - single_elapsed,
                        "multi_intent_better": multi_eval["f1"] > single_eval["f1"]
                    }
                }

                results.append(result)

            except Exception as e:
                logger.error(f"❌ 处理文档时出错: {str(e)}")
                import traceback
                traceback.print_exc()

        #保存和分析结果
        result_file = os.path.join(self.config['results_dir'], 'single_vs_multi_intent_comparison.json')
        result_file = os.path.join(self.config['results_dir'], 'improved_multi_intent_comparison.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        self.analyze_comparison_results(results)
        # self.generate_comparison_summary(results)
        logger.info(f"🎉 对比测试结果已保存到: {result_file}")
        return results

    def analyze_comparison_results(self, results):
        """分析对比测试结果"""
        if not results:
            logger.warning("⚠️ 没有可用的结果进行分析")
            return

        # 单意图指标
        single_recalls = [r["single_intent"]["recall"] for r in results]
        single_precisions = [r["single_intent"]["precision"] for r in results]
        single_f1s = [r["single_intent"]["f1"] for r in results]
        single_times = [r["single_intent"]["retrieval_time"] for r in results]
        single_success_count = sum(1 for r in results if r["single_intent"]["success"])

        # 多意图指标
        multi_recalls = [r["multi_intent"]["recall"] for r in results]
        multi_precisions = [r["multi_intent"]["precision"] for r in results]
        multi_f1s = [r["multi_intent"]["f1"] for r in results]
        multi_times = [r["multi_intent"]["retrieval_time"] for r in results]
        multi_success_count = sum(1 for r in results if r["multi_intent"]["success"])

        # 改进指标
        f1_improvements = [r["comparison"]["f1_improvement"] for r in results]
        multi_better_count = sum(1 for r in results if r["comparison"]["multi_intent_better"])

        logger.info(f"\n{'=' * 80}")
        # logger.info(f"📊 单意图 vs 多意图检索性能对比分析")
        logger.info(f"改进多意图检索性能对比分析")
        logger.info(f"{'=' * 80}")
        logger.info(f"📋 测试文档数: {len(results)}")

        logger.info(f"\n🔹 单意图检索性能:")
        logger.info(f"   平均召回率: {np.mean(single_recalls):.4f}")
        logger.info(f"   平均精确率: {np.mean(single_precisions):.4f}")
        logger.info(f"   平均F1值: {np.mean(single_f1s):.4f}")
        logger.info(f"   平均检索时间: {np.mean(single_times):.2f}秒")
        logger.info(
            f"   成功率: {(single_success_count / len(results)) * 100:.2f}% ({single_success_count}/{len(results)})")

        logger.info(f"\n🔹 多意图检索性能:")
        logger.info(f"   平均召回率: {np.mean(multi_recalls):.4f}")
        logger.info(f"   平均精确率: {np.mean(multi_precisions):.4f}")
        logger.info(f"   平均F1值: {np.mean(multi_f1s):.4f}")
        logger.info(f"   平均检索时间: {np.mean(multi_times):.2f}秒")
        logger.info(
            f"   成功率: {(multi_success_count / len(results)) * 100:.2f}% ({multi_success_count}/{len(results)})")

        logger.info(f"\n🔸 性能提升分析:")
        logger.info(f"   平均F1提升: {np.mean(f1_improvements):+.4f}")
        logger.info(f"   平均召回率提升: {np.mean([r['comparison']['recall_improvement'] for r in results]):+.4f}")
        logger.info(f"   平均精确率提升: {np.mean([r['comparison']['precision_improvement'] for r in results]):+.4f}")
        logger.info(f"   平均时间开销: {np.mean([r['comparison']['time_overhead'] for r in results]):+.2f}秒")
        logger.info(
            f"   多意图优于单意图的比例: {(multi_better_count / len(results)) * 100:.2f}% ({multi_better_count}/{len(results)})")

        logger.info(f"{'=' * 80}")

    def generate_comparison_summary(self, results):
        """生成对比汇总报告"""
        if not results:
            return

        logger.info(f"\n{'=' * 80}")
        logger.info(f"📋 单意图 vs 多意图检索 - 总体对比汇总")
        logger.info(f"{'=' * 80}")

        total_docs = len(results)

        # 基础统计
        single_success = sum(1 for r in results if r["single_intent"]["success"])
        multi_success = sum(1 for r in results if r["multi_intent"]["success"])
        multi_better_count = sum(1 for r in results if r["comparison"]["multi_intent_better"])

        # 性能指标
        single_avg_f1 = np.mean([r["single_intent"]["f1"] for r in results])
        multi_avg_f1 = np.mean([r["multi_intent"]["f1"] for r in results])
        avg_f1_improvement = np.mean([r["comparison"]["f1_improvement"] for r in results])

        single_total_time = sum([r["single_intent"]["retrieval_time"] for r in results])
        multi_total_time = sum([r["multi_intent"]["retrieval_time"] for r in results])

        # 按任务类型分析
        task_stats = {}
        for r in results:
            task_tag = r.get("task_tag", "Unknown")
            if task_tag not in task_stats:
                task_stats[task_tag] = {
                    "count": 0,
                    "single_success": 0,
                    "multi_success": 0,
                    "single_f1_sum": 0,
                    "multi_f1_sum": 0,
                    "multi_better": 0
                }
            task_stats[task_tag]["count"] += 1
            if r["single_intent"]["success"]:
                task_stats[task_tag]["single_success"] += 1
            if r["multi_intent"]["success"]:
                task_stats[task_tag]["multi_success"] += 1
            task_stats[task_tag]["single_f1_sum"] += r["single_intent"]["f1"]
            task_stats[task_tag]["multi_f1_sum"] += r["multi_intent"]["f1"]
            if r["comparison"]["multi_intent_better"]:
                task_stats[task_tag]["multi_better"] += 1

        logger.info(f"🎯 总体对比结果:")
        logger.info(f"   测试文档总数: {total_docs}")
        logger.info(f"   单意图成功数: {single_success} ({(single_success / total_docs) * 100:.2f}%)")
        logger.info(f"   多意图成功数: {multi_success} ({(multi_success / total_docs) * 100:.2f}%)")
        logger.info(f"   多意图优于单意图: {multi_better_count} ({(multi_better_count / total_docs) * 100:.2f}%)")
        logger.info(f"   单意图平均F1: {single_avg_f1:.4f}")
        logger.info(f"   多意图平均F1: {multi_avg_f1:.4f}")
        logger.info(f"   平均F1提升: {avg_f1_improvement:+.4f}")
        logger.info(f"   单意图总耗时: {single_total_time:.2f}秒")
        logger.info(f"   多意图总耗时: {multi_total_time:.2f}秒")
        logger.info(f"   时间开销: {multi_total_time - single_total_time:+.2f}秒")

        logger.info(f"\n📊 按任务类型对比:")
        for task_tag, stats in task_stats.items():
            count = stats["count"]
            single_success_rate = (stats["single_success"] / count) * 100
            multi_success_rate = (stats["multi_success"] / count) * 100
            single_avg_f1 = stats["single_f1_sum"] / count
            multi_avg_f1 = stats["multi_f1_sum"] / count
            multi_better_rate = (stats["multi_better"] / count) * 100

            logger.info(f"   {task_tag} ({count}样本):")
            logger.info(f"     单意图: 成功率{single_success_rate:.1f}%, F1:{single_avg_f1:.4f}")
            logger.info(f"     多意图: 成功率{multi_success_rate:.1f}%, F1:{multi_avg_f1:.4f}")
            logger.info(f"     多意图优势: {multi_better_rate:.1f}%")

        # 保存汇总到文件
        summary = {
            "experiment_name": "单意图vs多意图检索对比",
            "total_documents": total_docs,
            "single_intent_results": {
                "successful_retrievals": single_success,
                "success_rate": (single_success / total_docs) * 100,
                "average_f1": single_avg_f1,
                "total_time": single_total_time
            },
            "multi_intent_results": {
                "successful_retrievals": multi_success,
                "success_rate": (multi_success / total_docs) * 100,
                "average_f1": multi_avg_f1,
                "total_time": multi_total_time
            },
            "comparison_metrics": {
                "multi_intent_better_count": multi_better_count,
                "multi_intent_better_rate": (multi_better_count / total_docs) * 100,
                "average_f1_improvement": avg_f1_improvement,
                "time_overhead": multi_total_time - single_total_time
            },
            "task_breakdown": task_stats,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        summary_file = os.path.join(self.config['results_dir'], 'single_vs_multi_intent_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"\n💾 对比汇总报告已保存到: {summary_file}")
        logger.info(f"{'=' * 80}")

    def run(self):
        """运行测试"""
        logger.info("🚀 开始单意图 vs 多意图检索对比测试...")
        start_time = time.time()

        try:
            results = self.test_text_only_retrieval()
            total_time = time.time() - start_time
            logger.info(f"\n🎉 对比测试完成！总耗时: {total_time:.2f}秒")
            logger.info(f"📁 结果已保存到: {self.config['results_dir']}")

        except Exception as e:
            logger.error(f"❌ 测试过程中出现错误: {str(e)}", exc_info=True)


def main():
    """主函数"""
    print("🎯 单意图 vs 多意图检索对比测试")
    print("=" * 50)
    print("📄 对比单意图和多意图拆解在纯文本检索中的效果差异")
    print("=" * 50)

    parser = argparse.ArgumentParser(description="单意图vs多意图检索对比测试工具")
    parser.add_argument("--sample_size", type=int, default=50, help="测试样本数量")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")

    try:
        args = parser.parse_args()
    except SystemExit:
        args = argparse.Namespace(sample_size=50, debug=False)

    logger.info(f"📊 测试样本数量: {args.sample_size}")
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

    tester = TextOnlyMultiIntentTester()
    if args.sample_size:
        tester.config['sample_size'] = args.sample_size

    tester.run()


if __name__ == "__main__":
    main()