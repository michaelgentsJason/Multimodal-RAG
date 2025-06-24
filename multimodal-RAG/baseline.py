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

# 创建日志目录
log_dir = Path("./log")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "baseline_retrieval_test.log"

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

logger.info("=== Baseline多意图检索测试开始 ===")

# 添加必要的路径
sys.path.append("multimodal-RAG/DeepRAG_Multimodal/deep_retrieve")

# 远程环境变量加载
load_dotenv("/root/autodl-tmp/multimodal-RAG/multimodal-RAG/DeepRAG_Multimodal/configs/.env")

# 导入必要的库
from DeepRAG_Multimodal.deep_retrieve.ming.deepsearch_optimize_ming import DeepSearch_Beta
from DeepRAG_Multimodal.deep_retrieve.retriever_multimodal_bge import RetrieverConfig, MultimodalMatcher


class BaselineMultiIntentTester:
    """Baseline多意图检索测试类"""

    def __init__(self):
        """初始化测试器"""
        self.config = self.load_config()
        os.makedirs(self.config['results_dir'], exist_ok=True)
        self.verify_model_files()
        self.setup_models()

    def load_config(self):
        """加载配置"""
        config = {
            # 路径配置 - 远程服务器路径
            'test_data_path': '/root/autodl-tmp/multimodal-RAG/multimodal-RAG/DeepRAG_Multimodal/picked_LongDoc/selected_LongDocURL_public_with_subtask_category.jsonl',
            'pdf_base_dir': '/root/autodl-tmp/multimodal-RAG/multimodal-RAG/DeepRAG_Multimodal/picked_LongDoc',
            'results_dir': './test_results',

            # 采样配置
            'sample_size': 5,
            'debug': True,

            # 检索配置
            'max_iterations': 2,
            'embedding_topk': 12,
            'rerank_topk': 5,
            'text_weight': 0.8,
            'image_weight': 0.2,

            # 🔥 模型配置 - 确保使用正确的本地路径
            'mm_model_name': "/root/autodl-tmp/multimodal-RAG/hf_models/colqwen2.5-v0.2",
            'mm_processor_name': "/root/autodl-tmp/multimodal-RAG/hf_models/colqwen2.5-v0.1",
            'bge_model_name': "/root/autodl-tmp/multimodal-RAG/hf_models/bge-large-en-v1.5",
            'reranker_model_name': "/root/autodl-tmp/multimodal-RAG/hf_models/bge-reranker-large",

            'device': 'cuda:0',
            'batch_size': 2,
            'retrieval_mode': 'mixed',
            'ocr_method': 'pytesseract',
        }

        if config['debug']:
            config['sample_size'] = min(config['sample_size'], 3)

        return config

    def verify_model_files(self):
        """验证模型文件是否存在"""
        model_paths = [
            self.config['mm_model_name'],
            self.config['mm_processor_name'],
            self.config['bge_model_name'],
            self.config['reranker_model_name']
        ]

        for model_path in model_paths:
            if not os.path.exists(model_path):
                logger.error(f"❌ 模型路径不存在: {model_path}")
                raise FileNotFoundError(f"模型路径不存在: {model_path}")
            else:
                logger.info(f"✅ 模型路径验证成功: {model_path}")

            # 检查必要文件
            config_file = os.path.join(model_path, "config.json")
            if not os.path.exists(config_file):
                logger.warning(f"⚠️ 缺少config.json: {model_path}")

    def setup_models(self):
        """初始化检索模型"""
        logger.info("🚀 初始化Baseline多意图检索模型...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"🎮 使用设备: {device}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated() / (1024 ** 3)
            logger.info(f"🧹 初始GPU内存使用: {initial_memory:.2f}GB")

        try:
            # 🔥 先导入FlagReranker，确保离线模式
            from FlagEmbedding import FlagReranker

            # 初始化重排序器
            logger.info("⏳ 初始化重排序器...")
            self.reranker = FlagReranker(
                model_name_or_path=self.config['reranker_model_name'],
                use_fp16=True,
                device=device,
                local_files_only=True  # 🔥 强制使用本地文件
            )
            logger.info("✅ 重排序器初始化成功")

            # 🔥 修改RetrieverConfig以确保离线模式
            logger.info("⏳ 初始化多模态匹配器...")
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

            # 🔥 创建离线版本的MultimodalMatcher
            self.mm_matcher = OfflineMultimodalMatcher(
                config=retriever_config,
                embedding_weight=self.config['text_weight'],
                topk=self.config['rerank_topk']
            )
            logger.info("✅ 多模态匹配器初始化成功")

            # 初始化 DeepSearch_Beta
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

            # 最终内存检查
            if torch.cuda.is_available():
                final_memory = torch.cuda.memory_allocated() / (1024 ** 3)
                logger.info(f"📊 最终GPU内存使用: {final_memory:.2f}GB")

            logger.info("✅ Baseline模型初始化完成")

        except Exception as e:
            logger.error(f"❌ 模型初始化失败: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def load_test_data(self):
        """加载测试数据"""
        allowed_doc_nos = [
            '4064501.pdf', '4129670.pdf', '4012567.pdf', '4057714.pdf', '4196005.pdf'
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
            np.random.seed(42)
            test_data = np.random.choice(test_data, self.config['sample_size'], replace=False).tolist()

        logger.info(f"✅ 成功加载 {len(test_data)} 条测试数据")
        return test_data

    def process_single_document(self, doc_data):
        """处理单个文档"""
        documents = []
        pdf_path = os.path.join(self.config['pdf_base_dir'], doc_data["pdf_path"])

        try:
            pages = convert_from_path(pdf_path)
            logger.info(f"📖 成功转换PDF: {doc_data['pdf_path']}, 页数: {len(pages)}")
        except Exception as e:
            logger.error(f"❌ PDF转换失败: {doc_data['pdf_path']}, 错误: {str(e)}")
            return []

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
            loaded_data = {f"Page_{i + 1}": "" for i in range(len(pages))}

        page_count = min(len(loaded_data), len(pages))
        page_keys = list(loaded_data.keys())

        for idx in range(page_count):
            if idx >= len(pages):
                break

            page = pages[idx]
            width, height = page.size
            if width <= 0 or height <= 0:
                logger.warning(f"⚠️ 跳过无效页面 {idx + 1}")
                continue

            page_text = loaded_data[page_keys[idx]] if idx < len(page_keys) else ""
            if not page_text.strip():
                page_text = f"第{idx + 1}页内容"

            documents.append({
                "text": page_text,
                "image": page,
                "metadata": {
                    "page_index": idx + 1,
                    "pdf_path": doc_data.get("pdf_path", "")
                }
            })

        logger.info(f"📑 成功创建 {len(documents)} 个文档对象")
        return documents

    def test_baseline_retrieval(self):
        """测试Baseline多意图检索"""
        logger.info("🎯 开始Baseline多意图检索测试...")
        test_data = self.load_test_data()
        results = []

        for idx, doc_data in enumerate(tqdm(test_data, desc="Baseline检索测试")):
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

                start_time = time.time()
                logger.info("📄 使用多意图拆解 + ColPali检索")

                data = {"query": query, "documents": document_pages}
                retrieval_results = self.multi_intent_search.search_retrieval(data, retriever=self.mm_matcher)

                elapsed_time = time.time() - start_time

                # 处理结果
                retrieved_pages = set()
                processed_results = []

                for result in retrieval_results:
                    text = result.get("text", "")
                    score = result.get("score", 0)
                    metadata = result.get("metadata", {})
                    page_index = result.get("page", metadata.get("page_index", None))

                    if page_index is not None:
                        retrieved_pages.add(page_index)

                    processed_results.append({
                        "text": text,
                        "score": score,
                        "page": page_index,
                        "metadata": metadata
                    })

                # 评估结果
                evidence_set = set(evidence_pages)
                correct_pages = evidence_set.intersection(retrieved_pages)

                recall = len(correct_pages) / len(evidence_set) if evidence_set else 0
                precision = len(correct_pages) / len(retrieved_pages) if retrieved_pages else 0
                f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0

                retrieval_scores = [r.get('score', 0) for r in processed_results]

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
                    "retrieval_scores": retrieval_scores[:10],
                    "success": len(correct_pages) == len(evidence_set),
                    "strategy": "baseline"
                }

                results.append(result)

            except Exception as e:
                logger.error(f"❌ 处理文档时出错: {str(e)}")
                import traceback
                traceback.print_exc()

        # 保存和分析结果
        result_file = os.path.join(self.config['results_dir'], 'baseline_multi_intent_results.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        self.analyze_results(results)
        logger.info(f"🎉 测试结果已保存到: {result_file}")
        return results

    def analyze_results(self, results):
        """分析测试结果"""
        if not results:
            logger.warning("⚠️ 没有可用的结果进行分析")
            return

        recalls = [r["recall"] for r in results]
        precisions = [r["precision"] for r in results]
        f1s = [r["f1"] for r in results]
        times = [r["retrieval_time"] for r in results]
        success_count = sum(1 for r in results if r["success"])

        all_scores = []
        for r in results:
            all_scores.extend(r["retrieval_scores"])
        non_zero_scores = [s for s in all_scores if s > 0]

        avg_recall = np.mean(recalls)
        avg_precision = np.mean(precisions)
        avg_f1 = np.mean(f1s)
        avg_time = np.mean(times)
        success_rate = success_count / len(results) * 100

        logger.info(f"\n{'=' * 60}")
        logger.info(f"📊 Baseline多意图检索性能分析")
        logger.info(f"{'=' * 60}")
        logger.info(f"📋 测试文档数: {len(results)}")
        logger.info(f"📈 平均召回率: {avg_recall:.4f}")
        logger.info(f"📈 平均精确率: {avg_precision:.4f}")
        logger.info(f"📈 平均F1值: {avg_f1:.4f}")
        logger.info(f"⏱️ 平均检索时间: {avg_time:.2f}秒")
        logger.info(f"🎯 成功率: {success_rate:.2f}% ({success_count}/{len(results)})")

        logger.info(f"\n📊 分数质量分析:")
        logger.info(f"   总分数数量: {len(all_scores)}")
        logger.info(f"   非零分数数量: {len(non_zero_scores)}")
        if non_zero_scores:
            logger.info(f"   非零分数比例: {len(non_zero_scores) / len(all_scores) * 100:.1f}%")
            logger.info(f"   最高分数: {max(non_zero_scores):.4f}")
            logger.info(f"   平均非零分数: {np.mean(non_zero_scores):.4f}")
            logger.info(f"✅ 检索分数正常")
        else:
            logger.warning(f"⚠️ 所有检索分数都为0，请检查配置！")

        logger.info(f"{'=' * 60}")

    def run(self):
        """运行测试"""
        logger.info("🚀 开始Baseline多意图检索测试...")
        start_time = time.time()

        try:
            results = self.test_baseline_retrieval()
            total_time = time.time() - start_time
            logger.info(f"\n🎉 测试完成！总耗时: {total_time:.2f}秒")
            logger.info(f"📁 结果已保存到: {self.config['results_dir']}")

        except Exception as e:
            logger.error(f"❌ 测试过程中出现错误: {str(e)}", exc_info=True)


# 🔥 创建离线版本的MultimodalMatcher
class OfflineMultimodalMatcher(MultimodalMatcher):
    """离线版本的MultimodalMatcher，强制使用本地文件"""

    def _setup_models(self):
        """离线模式设置模型"""
        from transformers import AutoTokenizer, AutoModel
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
        from transformers.utils.import_utils import is_flash_attn_2_available

        logger.info("🔧 使用离线模式初始化模型...")

        # 初始化文本模型
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            self.config.bge_model_name,
            use_fast=True,
            local_files_only=True  # 🔥 强制离线
        )
        self.text_model = AutoModel.from_pretrained(
            self.config.bge_model_name,
            local_files_only=True  # 🔥 强制离线
        ).to(self.device)

        # 初始化图像模型
        self.image_model = ColQwen2_5.from_pretrained(
            "/root/autodl-tmp/multimodal-RAG/hf_models/colqwen2.5-v0.2",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
            local_files_only=True  # 🔥 强制离线
        ).eval()

        self.processor = ColQwen2_5_Processor.from_pretrained(
            "/root/autodl-tmp/multimodal-RAG/hf_models/colqwen2.5-v0.1",
            size={"shortest_edge": 512, "longest_edge": 1024},
            local_files_only=True  # 🔥 强制离线
        )

        logger.info("✅ 离线模型初始化完成")


def main():
    """主函数"""
    print("🎯 Baseline多意图检索测试（离线模式）")
    print("=" * 50)
    print("📄 使用多意图拆解 + ColPali多模态检索")
    print("=" * 50)

    parser = argparse.ArgumentParser(description="Baseline多意图检索测试工具")
    parser.add_argument("--sample_size", type=int, default=3, help="测试样本数量")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")

    try:
        args = parser.parse_args()
    except SystemExit:
        args = argparse.Namespace(sample_size=3, debug=False)

    logger.info(f"📊 测试样本数量: {args.sample_size}")
    logger.info(f"🐛 调试模式: {args.debug}")

    tester = BaselineMultiIntentTester()
    if args.sample_size:
        tester.config['sample_size'] = args.sample_size

    tester.run()


if __name__ == "__main__":
    main()