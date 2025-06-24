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
    """纯文本多意图检索测试类"""

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
            'sample_size': 50,
            'debug': True,

            # 检索配置
            'max_iterations': 2,
            'embedding_topk': 12,
            'rerank_topk': 5,
            'text_weight': 1.0,  # 纯文本模式
            'image_weight': 0.0,

            # 模型配置 - 只需要文本模型
            'bge_model_name': "/root/autodl-tmp/multimodal-RAG/hf_models/bge-large-en-v1.5",
            'reranker_model_name': "/root/autodl-tmp/multimodal-RAG/hf_models/bge-reranker-large",

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
            logger.info("⏳ 初始化纯文本匹配器...")
            self.text_matcher = TextOnlyMatcher(
                bge_model_path=self.config['bge_model_name'],
                device=device,
                topk=self.config['rerank_topk']
            )
            logger.info("✅ 纯文本匹配器初始化成功")

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
                    if item.get("pdf_path") in allowed_doc_nos:
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

        # 只处理文本，不处理图像
        page_keys = list(loaded_data.keys())
        for idx, page_key in enumerate(page_keys):
            page_text = loaded_data[page_key]
            if not page_text.strip():
                page_text = f"第{idx + 1}页内容"

            documents.append({
                "text": page_text,
                "metadata": {
                    "page_index": idx + 1,
                    "pdf_path": doc_data.get("pdf_path", "")
                }
            })

        logger.info(f"📑 成功创建 {len(documents)} 个文档对象（纯文本）")

        # 文本质量检查
        total_text_length = sum(len(doc['text']) for doc in documents)
        logger.info(f"📝 总文本长度: {total_text_length} 字符")

        return documents

    def test_text_only_retrieval(self):
        """测试纯文本多意图检索"""
        logger.info("🎯 开始纯文本多意图检索测试...")
        test_data = self.load_test_data()
        results = []

        for idx, doc_data in enumerate(tqdm(test_data, desc="纯文本检索测试")):
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
                logger.info("📄 使用多意图拆解 + 纯文本检索")

                data = {"query": query, "documents": document_pages}
                retrieval_results = self.multi_intent_search.search_retrieval(data, retriever=self.text_matcher)

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
                    "strategy": "text_only_baseline"
                }

                results.append(result)

            except Exception as e:
                logger.error(f"❌ 处理文档时出错: {str(e)}")
                import traceback
                traceback.print_exc()

        # 保存和分析结果
        result_file = os.path.join(self.config['results_dir'], 'text_only_baseline_results.json')
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
        logger.info(f"📊 纯文本Baseline多意图检索性能分析")
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
        logger.info("🚀 开始纯文本多意图检索测试...")
        start_time = time.time()

        try:
            results = self.test_text_only_retrieval()
            total_time = time.time() - start_time
            logger.info(f"\n🎉 测试完成！总耗时: {total_time:.2f}秒")
            logger.info(f"📁 结果已保存到: {self.config['results_dir']}")

        except Exception as e:
            logger.error(f"❌ 测试过程中出现错误: {str(e)}", exc_info=True)


class TextOnlyMatcher:
    """纯文本匹配器"""

    def __init__(self, bge_model_path: str, device: str = "cuda:0", topk: int = 10):
        self.bge_model_path = bge_model_path
        self.device = device
        self.topk = topk
        self._setup_models()

    def _setup_models(self):
        """设置文本模型"""
        from transformers import AutoTokenizer, AutoModel

        logger.info("🔧 初始化纯文本模型...")
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            self.bge_model_path,
            use_fast=True,
            local_files_only=True
        )
        self.text_model = AutoModel.from_pretrained(
            self.bge_model_path,
            local_files_only=True
        ).to(self.device)
        logger.info("✅ 纯文本模型初始化完成")

    def retrieve(self, query: str, documents: list) -> list:
        """检索相关文档"""
        if not documents:
            return []

        try:
            # 计算查询嵌入
            query_embedding = self._compute_text_embedding(query)

            # 计算文档嵌入和相似度
            scored_documents = []
            for doc in documents:
                text = doc.get("text", "")
                if not text.strip():
                    continue

                doc_embedding = self._compute_text_embedding(text)
                similarity = self._compute_similarity(query_embedding, doc_embedding)

                scored_documents.append({
                    "text": text,
                    "score": float(similarity),
                    "metadata": doc.get("metadata", {})
                })

            # 按相似度排序
            scored_documents.sort(key=lambda x: x["score"], reverse=True)
            return scored_documents[:self.topk]

        except Exception as e:
            logger.error(f"❌ 检索失败: {str(e)}")
            return []

    def _compute_text_embedding(self, text: str):
        """计算文本嵌入"""
        inputs = self.text_tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.text_model(**inputs)
            # 使用[CLS]向量
            embedding = outputs.last_hidden_state[:, 0].cpu().numpy()

        return embedding

    def _compute_similarity(self, query_emb, doc_emb):
        """计算余弦相似度"""
        # 归一化
        query_norm = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-8)
        doc_norm = doc_emb / (np.linalg.norm(doc_emb, axis=1, keepdims=True) + 1e-8)

        # 计算余弦相似度
        similarity = np.dot(query_norm, doc_norm.T)
        return similarity[0][0]


def main():
    """主函数"""
    print("🎯 纯文本Baseline多意图检索测试")
    print("=" * 50)
    print("📄 使用多意图拆解 + 纯文本检索（跳过图像模型）")
    print("=" * 50)

    parser = argparse.ArgumentParser(description="纯文本多意图检索测试工具")
    parser.add_argument("--sample_size", type=int, default=3, help="测试样本数量")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")

    try:
        args = parser.parse_args()
    except SystemExit:
        args = argparse.Namespace(sample_size=3, debug=False)

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