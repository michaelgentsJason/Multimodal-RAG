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
from FlagEmbedding import FlagReranker
import torch
from pdf2image import convert_from_path
from PIL import Image
import logging

# 创建日志目录
log_dir = Path("./log")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "multimodal_intent_test.log"

root = logging.getLogger()
for handler in root.handlers[:]:
    root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(str(log_file), mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 开头加入测试日志
logger.info("=== 多意图检索测试开始 ===")

# 添加必要的路径
sys.path.append("multimodal-RAG/DeepRAG_Multimodal/deep_retrieve")
# 加载环境变量
load_dotenv("D:\Desktop\multimodal-RAG\multimodal-RAG\DeepRAG_Multimodal\configs\.env")

# 导入必要的库
from DeepRAG_Multimodal.deep_retrieve.ming.deepsearch_optimize_ming import DeepSearch_Beta
from DeepRAG_Multimodal.deep_retrieve.retriever_multimodal_bge import RetrieverConfig, MultimodalMatcher


class MultiIntentTester:
    """多意图检索测试类"""

    def __init__(self):
        """初始化测试器"""
        self.config = self.load_config()
        os.makedirs(self.config['results_dir'], exist_ok=True)
        self.setup_models()

    def load_config(self):
        """加载配置"""
        config = {
            # 路径配置 - 使用你的数据路径
            'test_data_path': r'D:\Desktop\colpali_longdoc\picked_LongDoc\selected_LongDocURL_public_with_subtask_category.jsonl',
            'pdf_base_dir': r'D:\Desktop\colpali_longdoc\picked_LongDoc',
            'results_dir': './test_results',

            # 采样配置
            'sample_size': 10,  # 测试10个样本
            'debug': True,

            # 检索配置
            'max_iterations': 2,
            'embedding_topk': 12,
            'rerank_topk': 4,
            'text_weight': 0.8,  # 平衡权重
            'image_weight': 0.2,

            # 模型配置
            'mm_model_name': "vidore/colqwen2.5-v0.2",
            'mm_processor_name': "vidore/colqwen2.5-v0.1",
            'bge_model_name': "BAAI/bge-large-en-v1.5",
            'device': 'cuda:0',
            'batch_size': 2,
            'retrieval_mode': 'mixed',
            'ocr_method': 'pytesseract',
        }

        if config['debug']:
            config['sample_size'] = 5  # 调试模式下只测试3个样本

        return config

    def setup_models(self):
        """初始化检索模型"""
        logger.info("🚀 初始化多意图检索模型...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"🎮 使用设备: {device}")

        # 初始化重排序器
        self.reranker = FlagReranker(
            model_name_or_path="BAAI/bge-reranker-large",
            use_fp16=True,
            device=device
        )

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

        # 使用标准多模态匹配器
        self.mm_matcher = MultimodalMatcher(
            config=retriever_config,
            embedding_weight=self.config['text_weight'],
            topk=self.config['rerank_topk']
        )
        logger.info("✅ 已初始化多模态匹配器")

        # 🔥 只初始化多意图检索器
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

        logger.info("✅ 模型初始化完成")

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
                    if item.get("pdf_path") in allowed_doc_nos:
                        test_data.append(item)

        if self.config['sample_size'] > 0 and len(test_data) > self.config['sample_size']:
            np.random.seed(42)  # 设置随机种子确保可重复性
            test_data = np.random.choice(test_data, self.config['sample_size'], replace=False).tolist()

        logger.info(f"✅ 成功加载 {len(test_data)} 条测试数据")
        return test_data

    def process_single_document(self, doc_data):
        """处理单个文档，使用预处理文本和PDF图像"""
        documents = []

        # 获取PDF文件路径
        pdf_path = os.path.join(self.config['pdf_base_dir'], doc_data["pdf_path"])

        try:
            pages = convert_from_path(pdf_path)
            logger.info(f"📖 成功转换PDF: {doc_data['pdf_path']}, 页数: {len(pages)}")
        except Exception as e:
            logger.error(f"❌ PDF转换失败: {doc_data['pdf_path']}, 错误: {str(e)}")
            return []

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
            # 如果没有OCR文件，创建空文本
            loaded_data = {f"Page_{i + 1}": "" for i in range(len(pages))}

        # 验证页面数量匹配
        if len(loaded_data) != len(pages):
            logger.warning(f"⚠️ OCR数据页数 ({len(loaded_data)}) 与PDF页数 ({len(pages)}) 不匹配")
            page_count = min(len(loaded_data), len(pages))
        else:
            page_count = len(pages)

        # 为每一页创建文档对象
        page_keys = list(loaded_data.keys())
        for idx in range(page_count):
            if idx >= len(pages):
                break

            # 检查页面尺寸是否有效
            page = pages[idx]
            width, height = page.size
            if width <= 0 or height <= 0:
                logger.warning(f"⚠️ 跳过无效页面 {idx + 1}：尺寸 {width}x{height}")
                continue

            # 获取OCR文本
            page_text = loaded_data[page_keys[idx]] if idx < len(page_keys) else ""
            if not page_text.strip():
                page_text = f"第{idx + 1}页内容"  # 备用文本

            # 创建文档结构
            documents.append({
                "text": page_text,
                "image": page,
                "metadata": {
                    "page_index": idx + 1,
                    "pdf_path": doc_data.get("pdf_path", "")
                }
            })

        logger.info(f"📑 成功创建 {len(documents)} 个文档对象")

        # 添加文本质量检查
        total_text_length = sum(len(doc['text']) for doc in documents)
        logger.info(f"📝 总文本长度: {total_text_length} 字符")

        return documents

    def test_multi_intent_retrieval(self):
        """测试多意图检索"""
        logger.info("🎯 开始多意图检索测试...")
        test_data = self.load_test_data()
        results = []

        for idx, doc_data in enumerate(tqdm(test_data, desc="多意图检索测试")):
            try:
                query = doc_data.get("question", "")
                evidence_pages = doc_data.get("evidence_pages", [])

                logger.info(f"\n{'=' * 60}")
                logger.info(f"🔍 处理文档 {idx + 1}/{len(test_data)}: {doc_data.get('pdf_path', 'Unknown')}")
                logger.info(f"❓ 查询: {query}")
                logger.info(f"📋 证据页面: {evidence_pages}")

                # 处理文档
                document_pages = self.process_single_document(doc_data)
                if not document_pages:
                    logger.warning(f"⚠️ 跳过文档 {doc_data.get('pdf_path', '')}: 无有效内容")
                    continue

                # 执行多意图检索
                data = {
                    "query": query,
                    "documents": document_pages
                }

                start_time = time.time()
                retrieval_results = self.multi_intent_search.search_retrieval(data, retriever=self.mm_matcher)
                elapsed_time = time.time() - start_time

                # 提取检索结果中的页码
                retrieved_pages = set()
                for result in retrieval_results:
                    if 'metadata' in result and 'page_index' in result['metadata']:
                        retrieved_pages.add(result['metadata']['page_index'])
                    elif 'page' in result and result['page'] is not None:
                        retrieved_pages.add(result['page'])

                # 评估结果
                evidence_set = set(evidence_pages)
                correct_pages = evidence_set.intersection(retrieved_pages)

                recall = len(correct_pages) / len(evidence_set) if evidence_set else 0
                precision = len(correct_pages) / len(retrieved_pages) if retrieved_pages else 0
                f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0

                # 获取检索分数
                retrieval_scores = [r.get('score', 0) for r in retrieval_results]

                logger.info(f"⏱️ 检索耗时: {elapsed_time:.2f}秒")
                logger.info(f"🎯 检索到页面: {sorted(list(retrieved_pages))}")
                logger.info(f"✅ 正确页面: {sorted(list(correct_pages))}")
                logger.info(f"📊 检索分数: {retrieval_scores[:5]}")
                logger.info(f"📈 召回率: {recall:.4f}")
                logger.info(f"📈 精确率: {precision:.4f}")
                logger.info(f"📈 F1值: {f1:.4f}")

                # 记录结果
                result = {
                    "doc_id": doc_data.get("doc_no", ""),
                    "pdf_path": doc_data.get("pdf_path", ""),
                    "query": query,
                    "evidence_pages": evidence_pages,
                    "task_tag": doc_data.get("task_tag", ""),
                    "subTask": doc_data.get("subTask", []),
                    "retrieved_pages": list(retrieved_pages),
                    "correct_pages": list(correct_pages),
                    "recall": recall,
                    "precision": precision,
                    "f1": f1,
                    "retrieval_time": elapsed_time,
                    "retrieval_scores": retrieval_scores[:10],  # 保存前10个分数
                    "success": len(correct_pages) == len(evidence_set)
                }

                results.append(result)

            except Exception as e:
                logger.error(f"❌ 处理文档 {doc_data.get('pdf_path', '')} 时出错: {str(e)}")

        # 保存结果
        result_file = os.path.join(self.config['results_dir'], 'multi_intent_results.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 分析结果
        self.analyze_results(results)

        logger.info(f"🎉 多意图检索测试结果已保存到: {result_file}")
        return results

    def analyze_results(self, results):
        """分析并打印测试结果"""
        if not results:
            logger.warning("⚠️ 没有可用的结果进行分析")
            return

        # 计算平均指标
        recalls = [r["recall"] for r in results]
        precisions = [r["precision"] for r in results]
        f1s = [r["f1"] for r in results]
        times = [r["retrieval_time"] for r in results]
        success_count = sum(1 for r in results if r["success"])

        # 分数质量检查
        all_scores = []
        for r in results:
            all_scores.extend(r["retrieval_scores"])

        non_zero_scores = [s for s in all_scores if s > 0]

        # 计算平均值
        avg_recall = np.mean(recalls)
        avg_precision = np.mean(precisions)
        avg_f1 = np.mean(f1s)
        avg_time = np.mean(times)
        success_rate = success_count / len(results) * 100

        # 打印结果
        logger.info(f"\n{'=' * 60}")
        logger.info(f"📊 多意图检索性能分析")
        logger.info(f"{'=' * 60}")
        logger.info(f"📋 测试文档数: {len(results)}")
        logger.info(f"📈 平均召回率: {avg_recall:.4f}")
        logger.info(f"📈 平均精确率: {avg_precision:.4f}")
        logger.info(f"📈 平均F1值: {avg_f1:.4f}")
        logger.info(f"⏱️ 平均检索时间: {avg_time:.2f}秒")
        logger.info(f"🎯 成功率: {success_rate:.2f}% ({success_count}/{len(results)})")

        # 分数质量分析
        logger.info(f"\n📊 分数质量分析:")
        logger.info(f"   总分数数量: {len(all_scores)}")
        logger.info(f"   非零分数数量: {len(non_zero_scores)}")
        if non_zero_scores:
            logger.info(f"   非零分数比例: {len(non_zero_scores) / len(all_scores) * 100:.1f}%")
            logger.info(f"   最高分数: {max(non_zero_scores):.4f}")
            logger.info(f"   平均非零分数: {np.mean(non_zero_scores):.4f}")

        if len(non_zero_scores) == 0:
            logger.warning(f"⚠️ 所有检索分数都为0，请检查配置！")
        elif len(non_zero_scores) / len(all_scores) < 0.1:
            logger.warning(f"⚠️ 大部分检索分数为0，检索效果可能有问题")
        else:
            logger.info(f"✅ 检索分数正常")

        # 按任务类型分析（如果有）
        task_types = {}
        for r in results:
            task_tag = r.get("task_tag", "Unknown")
            if task_tag not in task_types:
                task_types[task_tag] = {"count": 0, "f1_sum": 0, "success": 0}

            task_types[task_tag]["count"] += 1
            task_types[task_tag]["f1_sum"] += r["f1"]
            task_types[task_tag]["success"] += 1 if r["success"] else 0

        if len(task_types) > 1:
            logger.info(f"\n📋 按任务类型分析:")
            for task_tag, stats in task_types.items():
                count = stats["count"]
                avg_f1 = stats["f1_sum"] / count
                success_rate = stats["success"] / count * 100
                logger.info(f"   {task_tag}: F1={avg_f1:.4f}, 成功率={success_rate:.1f}% ({count}样本)")

        logger.info(f"{'=' * 60}")

    def run(self):
        """运行测试"""
        logger.info("🚀 开始多意图检索测试...")
        start_time = time.time()

        try:
            results = self.test_multi_intent_retrieval()

            total_time = time.time() - start_time
            logger.info(f"\n🎉 测试完成！")
            logger.info(f"⏱️ 总耗时: {total_time:.2f}秒")
            logger.info(f"📁 结果已保存到: {self.config['results_dir']}")

        except Exception as e:
            logger.error(f"❌ 测试过程中出现错误: {str(e)}", exc_info=True)


def main():
    """主函数"""
    print("🎯 多意图检索测试")
    print("=" * 50)

    # 创建测试器并运行
    tester = MultiIntentTester()
    tester.run()


if __name__ == "__main__":
    main()