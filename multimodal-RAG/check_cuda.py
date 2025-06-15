#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评分诊断测试脚本 - 快速诊断为什么检索分数全是0

用法:
python score_diagnostic_test.py
"""

import os
import sys
import json
import time
import torch
import numpy as np
from dotenv import load_dotenv
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 添加必要的路径
sys.path.append("multimodal-RAG/DeepRAG_Multimodal/deep_retrieve")
load_dotenv("multimodal-RAG/DeepRAG_Multimodal/configs/.env")

# 设置环境变量（如果需要）
if os.getenv("SILICONFLOW_API_KEY"):
    logger.info("✅ 找到硅基流动API Key")
else:
    logger.warning("⚠️ 未找到硅基流动API Key")


class ScoreDiagnosticTester:
    """评分诊断测试器"""

    def __init__(self):
        """初始化测试器"""
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"🎮 使用设备: {self.device}")

        # 测试用例
        self.test_query = "What are the key features of Huddle section?"
        self.test_documents = [
            {
                "text": "Huddle section key features include real-time collaboration, document sharing, video conferencing, project management tools, team communication capabilities, screen sharing, whiteboard tools, calendar integration, and notification systems.",
                "metadata": {"page_index": 1, "pdf_path": "test.pdf"}
            },
            {
                "text": "The Huddle platform offers advanced collaboration features such as customizable workspace layouts, cloud storage integration, API connectivity, mobile accessibility, security encryption, and user permission controls.",
                "metadata": {"page_index": 2, "pdf_path": "test.pdf"}
            },
            {
                "text": "Huddle section encompasses cross-platform compatibility, automated workflow processes, real-time document editing, version control, team performance analytics, meeting scheduling, and comprehensive reporting tools.",
                "metadata": {"page_index": 3, "pdf_path": "test.pdf"}
            }
        ]

        logger.info(f"🧪 测试查询: {self.test_query}")
        logger.info(f"📚 测试文档数量: {len(self.test_documents)}")

    def test_1_cuda_availability(self):
        """测试1: CUDA可用性"""
        logger.info("\n" + "=" * 60)
        logger.info("🧪 测试1: CUDA可用性检查")
        logger.info("=" * 60)

        try:
            logger.info(f"CUDA可用: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"GPU数量: {torch.cuda.device_count()}")
                logger.info(f"当前GPU: {torch.cuda.current_device()}")
                logger.info(f"GPU名称: {torch.cuda.get_device_name()}")
                logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f}GB")

                # 测试GPU计算
                test_tensor = torch.randn(100, 100).to(self.device)
                result = torch.mm(test_tensor, test_tensor.T)
                logger.info(f"✅ GPU计算测试成功: {result.shape}")
                return True
            else:
                logger.warning("⚠️ CUDA不可用，将使用CPU")
                return False
        except Exception as e:
            logger.error(f"❌ CUDA测试失败: {e}")
            return False

    def test_2_bge_model_direct(self):
        """测试2: BGE模型直接测试"""
        logger.info("\n" + "=" * 60)
        logger.info("🧪 测试2: BGE模型直接测试")
        logger.info("=" * 60)

        try:
            from FlagEmbedding import FlagModel

            logger.info("⏳ 加载BGE模型...")
            start_time = time.time()

            bge_model = FlagModel(
                "BAAI/bge-large-en-v1.5",
                query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                use_fp16=True,
                device=self.device
            )

            load_time = time.time() - start_time
            logger.info(f"✅ BGE模型加载完成，耗时: {load_time:.2f}秒")

            # 测试嵌入计算
            query_emb = bge_model.encode([self.test_query])
            text_emb = bge_model.encode([self.test_documents[0]["text"]])

            # 计算相似度
            similarity = float(query_emb @ text_emb.T)

            logger.info(f"📊 查询嵌入形状: {query_emb.shape}")
            logger.info(f"📊 文本嵌入形状: {text_emb.shape}")
            logger.info(f"🎯 相似度分数: {similarity:.6f}")

            if similarity > 0:
                logger.info("✅ BGE模型工作正常")
                return True, bge_model
            else:
                logger.error("❌ BGE模型返回0分数")
                return False, None

        except Exception as e:
            logger.error(f"❌ BGE模型测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False, None

    def test_3_reranker_direct(self):
        """测试3: FlagReranker直接测试"""
        logger.info("\n" + "=" * 60)
        logger.info("🧪 测试3: FlagReranker直接测试")
        logger.info("=" * 60)

        try:
            from FlagEmbedding import FlagReranker

            logger.info("⏳ 加载FlagReranker...")
            start_time = time.time()

            reranker = FlagReranker(
                model_name_or_path="BAAI/bge-reranker-large",
                use_fp16=True,
                device=self.device
            )

            load_time = time.time() - start_time
            logger.info(f"✅ FlagReranker加载完成，耗时: {load_time:.2f}秒")

            # 测试重排序
            pairs = [[self.test_query, doc["text"]] for doc in self.test_documents]
            rerank_scores = reranker.compute_score(pairs, normalize=True)

            logger.info(f"📊 重排序对数: {len(pairs)}")
            logger.info(f"🎯 重排序分数: {rerank_scores}")

            # 检查分数
            max_score = max(rerank_scores) if rerank_scores else 0
            if max_score > 0:
                logger.info("✅ FlagReranker工作正常")
                return True, reranker
            else:
                logger.error("❌ FlagReranker返回全0分数")
                return False, None

        except Exception as e:
            logger.error(f"❌ FlagReranker测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False, None

    def test_4_multimodal_matcher(self):
        """测试4: MultimodalMatcher测试"""
        logger.info("\n" + "=" * 60)
        logger.info("🧪 测试4: MultimodalMatcher测试")
        logger.info("=" * 60)

        try:
            from DeepRAG_Multimodal.deep_retrieve.retriever_multimodal_bge import RetrieverConfig, MultimodalMatcher

            logger.info("⏳ 初始化MultimodalMatcher...")

            # 配置
            retriever_config = RetrieverConfig(
                model_name="vidore/colqwen2.5-v0.2",
                processor_name="vidore/colqwen2.5-v0.1",
                bge_model_name="BAAI/bge-large-en-v1.5",
                device=self.device,
                use_fp16=True,
                batch_size=2,
                mode='text_only',  # 只用文本模式加快测试
                ocr_method='paddleocr'
            )

            matcher = MultimodalMatcher(
                config=retriever_config,
                embedding_weight=1.0,  # 纯文本
                topk=5
            )

            logger.info("✅ MultimodalMatcher初始化完成")

            # 执行检索
            logger.info("⏳ 执行检索...")
            results = matcher.retrieve(self.test_query, self.test_documents)

            logger.info(f"📊 检索结果数量: {len(results)}")

            # 分析结果
            for i, result in enumerate(results):
                score = result.get('score', 0)
                text_preview = result.get('text', '')[:60] + "..."
                logger.info(f"   结果{i + 1}: 分数={score:.6f}, 文本='{text_preview}'")

            # 检查分数
            scores = [r.get('score', 0) for r in results]
            max_score = max(scores) if scores else 0

            if max_score > 0:
                logger.info("✅ MultimodalMatcher工作正常")
                return True, matcher, results
            else:
                logger.error("❌ MultimodalMatcher返回全0分数")

                # 详细调试
                logger.info("🔍 开始详细调试...")
                for i, doc in enumerate(self.test_documents):
                    text = doc.get('text', '')
                    logger.info(f"📄 文档{i + 1} 长度: {len(text)} 字符")

                    # 手动测试文本分数计算
                    try:
                        if hasattr(matcher, '_compute_text_score'):
                            manual_score = matcher._compute_text_score(self.test_query, text)
                            logger.info(f"📄 文档{i + 1} 手动计算分数: {manual_score:.6f}")
                    except Exception as e:
                        logger.error(f"📄 文档{i + 1} 手动计算失败: {e}")

                return False, None, []

        except Exception as e:
            logger.error(f"❌ MultimodalMatcher测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False, None, []

    def test_5_full_pipeline(self):
        """测试5: 完整检索流水线"""
        logger.info("\n" + "=" * 60)
        logger.info("🧪 测试5: 完整检索流水线测试")
        logger.info("=" * 60)

        try:
            # 导入所需模块
            from DeepRAG_Multimodal.deep_retrieve.ming.deepsearch_optimize_ming import DeepSearch_Beta
            from DeepRAG_Multimodal.deep_retrieve.retriever_multimodal_bge import RetrieverConfig, MultimodalMatcher
            from FlagEmbedding import FlagReranker

            logger.info("⏳ 初始化完整流水线...")

            # 初始化组件
            reranker = FlagReranker(
                model_name_or_path="BAAI/bge-reranker-large",
                use_fp16=True,
                device=self.device
            )

            retriever_config = RetrieverConfig(
                model_name="vidore/colqwen2.5-v0.2",
                processor_name="vidore/colqwen2.5-v0.1",
                bge_model_name="BAAI/bge-large-en-v1.5",
                device=self.device,
                use_fp16=True,
                batch_size=2,
                mode='text_only',
                ocr_method='paddleocr'
            )

            matcher = MultimodalMatcher(
                config=retriever_config,
                embedding_weight=1.0,
                topk=5
            )

            # 创建简化的检索器（不使用意图拆解）
            class SimpleRetriever:
                def __init__(self, matcher, reranker):
                    self.matcher = matcher
                    self.reranker = reranker

                def search_retrieval(self, data, retriever=None):
                    query = data['query']
                    documents = data['documents']

                    logger.info(f"🔍 检索查询: {query}")
                    logger.info(f"📚 文档数量: {len(documents)}")

                    # 直接检索
                    results = self.matcher.retrieve(query, documents)

                    # 记录初始分数
                    initial_scores = [r.get('score', 0) for r in results]
                    logger.info(f"📊 初始分数: {initial_scores}")

                    # 重排序（如果有结果）
                    if results and any(s > 0 for s in initial_scores):
                        try:
                            pairs = [[query, r.get('text', '')] for r in results]
                            rerank_scores = self.reranker.compute_score(pairs, normalize=True)

                            # 更新分数
                            for i, result in enumerate(results):
                                if i < len(rerank_scores):
                                    result['rerank_score'] = rerank_scores[i]
                                    # 组合分数
                                    original_score = result.get('score', 0)
                                    result['final_score'] = 0.6 * rerank_scores[i] + 0.4 * original_score

                            logger.info(f"📊 重排序分数: {rerank_scores}")

                        except Exception as e:
                            logger.warning(f"⚠️ 重排序失败: {e}")

                    return results

            simple_retriever = SimpleRetriever(matcher, reranker)

            # 测试数据
            test_data = {
                'query': self.test_query,
                'documents': self.test_documents
            }

            # 执行完整检索
            logger.info("⏳ 执行完整检索流水线...")
            start_time = time.time()

            final_results = simple_retriever.search_retrieval(test_data)

            elapsed_time = time.time() - start_time
            logger.info(f"✅ 完整检索完成，耗时: {elapsed_time:.2f}秒")

            # 分析最终结果
            logger.info(f"📊 最终结果数量: {len(final_results)}")

            for i, result in enumerate(final_results):
                score = result.get('final_score', result.get('score', 0))
                rerank_score = result.get('rerank_score', 'N/A')
                text_preview = result.get('text', '')[:50] + "..."
                logger.info(f"   结果{i + 1}: 最终分数={score:.6f}, 重排序分数={rerank_score}, 文本='{text_preview}'")

            # 检查是否成功
            final_scores = [r.get('final_score', r.get('score', 0)) for r in final_results]
            max_final_score = max(final_scores) if final_scores else 0

            if max_final_score > 0:
                logger.info("✅ 完整流水线工作正常")
                return True
            else:
                logger.error("❌ 完整流水线返回全0分数")
                return False

        except Exception as e:
            logger.error(f"❌ 完整流水线测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_all_tests(self):
        """运行所有测试"""
        logger.info("🚀 开始评分诊断测试...")
        logger.info(f"📊 测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        test_results = {}

        # 测试1: CUDA
        test_results['cuda'] = self.test_1_cuda_availability()

        # 测试2: BGE模型
        bge_success, bge_model = self.test_2_bge_model_direct()
        test_results['bge_direct'] = bge_success

        # 测试3: FlagReranker
        reranker_success, reranker = self.test_3_reranker_direct()
        test_results['reranker_direct'] = reranker_success

        # 测试4: MultimodalMatcher
        matcher_success, matcher, results = self.test_4_multimodal_matcher()
        test_results['multimodal_matcher'] = matcher_success

        # 测试5: 完整流水线
        pipeline_success = self.test_5_full_pipeline()
        test_results['full_pipeline'] = pipeline_success

        # 总结报告
        self.print_summary_report(test_results)

        return test_results

    def print_summary_report(self, test_results):
        """打印总结报告"""
        logger.info("\n" + "=" * 60)
        logger.info("📋 诊断测试总结报告")
        logger.info("=" * 60)

        all_passed = True
        for test_name, result in test_results.items():
            status = "✅ 通过" if result else "❌ 失败"
            logger.info(f"{test_name:20s}: {status}")
            if not result:
                all_passed = False

        logger.info("\n" + "=" * 60)
        if all_passed:
            logger.info("🎉 所有测试通过！评分系统应该正常工作。")
            logger.info("💡 如果demo中仍然是0分，问题可能在数据处理或配置上。")
        else:
            logger.info("⚠️ 发现问题！请查看上面的详细错误信息。")

            # 提供具体建议
            if not test_results.get('cuda', True):
                logger.info("🔧 建议: 检查CUDA安装和GPU驱动")
            if not test_results.get('bge_direct', True):
                logger.info("🔧 建议: 检查BGE模型下载和设备配置")
            if not test_results.get('reranker_direct', True):
                logger.info("🔧 建议: 检查FlagReranker模型和设备配置")
            if not test_results.get('multimodal_matcher', True):
                logger.info("🔧 建议: 检查MultimodalMatcher配置和文本预处理")
            if not test_results.get('full_pipeline', True):
                logger.info("🔧 建议: 检查完整流水线的组件集成")

        logger.info("=" * 60)


def main():
    """主函数"""
    print("🧪 评分诊断测试脚本")
    print("=" * 60)
    print("这个脚本将快速诊断为什么检索分数全是0")
    print("=" * 60)

    try:
        tester = ScoreDiagnosticTester()
        results = tester.run_all_tests()

        # 保存结果
        result_file = "score_diagnostic_results.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'test_results': results,
                'test_query': tester.test_query,
                'device': tester.device
            }, f, indent=2)

        print(f"\n📁 详细结果已保存到: {result_file}")

    except KeyboardInterrupt:
        print("\n⚠️ 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
