"""
Vespa retriever optimized for default Vespa configuration.
This implementation works with the default Vespa config without requiring custom app deployment.
"""

import os
import json
import time
import asyncio
import logging
import requests
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from PIL import Image
import torch
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class VespaConfig:
    """Configuration for Vespa retriever"""
    def __init__(self, 
                endpoint="http://localhost:19071",
                namespace="multimodal",
                document_type="multimodal_document",
                text_field="text",
                image_field="image_embedding",
                text_embedding_field="text_embedding",
                verify_ssl=False,
                timeout=15):
        self.endpoint = endpoint
        self.namespace = namespace
        self.document_type = document_type
        self.text_field = text_field
        self.image_field = image_field
        self.text_embedding_field = text_embedding_field
        self.verify_ssl = verify_ssl
        self.timeout = timeout


class VespaRetriever:
    """Vespa-based retriever for multimodal document retrieval adapted for default Vespa config"""
    
    def __init__(
        self,
        config: VespaConfig,
        text_model: Any,
        image_model: Optional[Any] = None,
        processor: Optional[Any] = None,
        embedding_weight: float = 0.5,
        topk: int = 10
    ):
        """Initialize VespaRetriever
        
        Args:
            config: Vespa configuration
            text_model: Text embedding model (FlagModel)
            image_model: Image embedding model (ColQwen2_5)
            processor: Image processor (ColQwen2_5_Processor)
            embedding_weight: Weight for text embeddings (1.0 = text only, 0.0 = image only)
            topk: Number of results to return
        """
        self.config = config
        self.text_model = text_model
        self.image_model = image_model
        self.processor = processor
        self.text_embedding_weight = embedding_weight
        self.topk = topk
        
        # Verify Vespa connection
        self.connection_ok = self._check_vespa_connection()
        
        # Initialize in-memory storage for handling documents
        self.documents = []
        self.text_embeddings = []
        self.image_embeddings = []

    def _index_documents_to_vespa(self, documents: List[Dict[str, Any]]):
        """使用cells格式索引文档，解决索引越界问题"""
        for i, doc in enumerate(documents):
            doc_id = f"doc_{i}"
            text = doc.get("text", "")
            text_embedding = self._compute_text_embedding(text)
            print(len(text_embedding.tolist()))
            
            # 准备文档数据
            doc_data = {
                "fields": {
                    self.config.text_field: text,
                    self.config.text_embedding_field: {"values": text_embedding.tolist()}, # Convert to list
                    "page_index": doc.get("metadata", {}).get("page_index", 0),
                    "pdf_path": doc.get("metadata", {}).get("pdf_path", "")
                }
            }

            # 添加图像嵌入（如果有）
            if self.image_model and "image" in doc and doc["image"] is not None:
                image_embedding = self._compute_image_embedding(doc["image"]).tolist()
                print(len(image_embedding))
                doc_data["fields"][self.config.image_field] = {"values": image_embedding}
            
            # 发送到Vespa
            try:
                url = f"{self.config.endpoint}/document/v1/{self.config.namespace}/{self.config.document_type}/docid/{doc_id}"
                response = requests.post(
                    url, json=doc_data, timeout=self.config.timeout, verify=False
                )
                if response.status_code == 200:
                    logger.info(f"成功索引文档 {doc_id}")
                else:
                    logger.warning(f"索引文档失败: {response.status_code} - {response.text[:150]}")
            except Exception as e:
                logger.error(f"索引文档出错: {str(e)}")
        
        # 索引完成后等待一秒，确保Vespa完成索引
        time.sleep(1)
        logger.info("索引完成")
    
    def _check_vespa_connection(self) -> bool:
        """检查Vespa是否运行并响应"""
        try:
            response = requests.get(
                f"{self.config.endpoint}/ApplicationStatus",
                timeout=self.config.timeout,
                verify=self.config.verify_ssl  # 使用配置中的SSL验证选项
            )
            if response.status_code == 200:
                logger.info(f"✅ 成功连接到Vespa服务器: {self.config.endpoint}")
                return True
            else:
                logger.warning(f"⚠️ Vespa服务器返回状态码 {response.status_code}")
                logger.info("将使用内存向量搜索")
                return False
        except Exception as e:
            logger.error(f"❌ 无法连接到Vespa服务器: {str(e)}")
            logger.info("将使用内存向量搜索")
            return False

    def retrieve(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """根据查询相似性检索文档"""
        # 如果文档集改变，重新索引
        if documents != self.documents:
            self.documents = documents
            if self.connection_ok:
                self._index_documents_to_vespa(documents)
            else:
                self._precompute_embeddings(documents)
        
        # 计算查询嵌入
        query_embedding = self._compute_text_embedding(query)
        
        # 使用Vespa查询或后备使用内存检索
        if self.connection_ok:
            results = self._query_vespa(query_embedding, query)
        else:
            results = self._compute_similarity_and_rank(query_embedding)
        
        logger.info(f"为查询 '{query}' 检索到 {len(results)} 个文档")
        return results

    def _precompute_embeddings(self, documents: List[Dict[str, Any]]):
        """Precompute embeddings for all documents"""
        logger.info(f"Precomputing embeddings for {len(documents)} documents...")
        self.text_embeddings = []
        self.image_embeddings = []
        
        # 计算文本嵌入
        for doc in documents:
            text = doc.get("text", "")
            self.text_embeddings.append(self._compute_text_embedding(text))
            
            # 如果有图像和图像模型，计算图像嵌入
            if self.image_model and self.processor:
                image = doc.get("image", None)
                if image is not None:
                    self.image_embeddings.append(self._compute_image_embedding(image))
                else:
                    self.image_embeddings.append(None)
        
        # 转换为 NumPy 数组以加速计算
        self.text_embeddings = np.array(self.text_embeddings)
        logger.info("Embeddings precomputed successfully")

    def _compute_similarity_and_rank(self, query_embedding: np.ndarray) -> List[Dict[str, Any]]:
        """Compute similarity scores and rank documents"""
        scores = []
        
        # 计算每个文档的相似度分数
        for i, doc in enumerate(self.documents):
            # 文本相似度（余弦相似度）
            text_score = np.dot(query_embedding, self.text_embeddings[i])
            
            # 图像相似度（如果有）
            image_score = 0.0
            if self.image_model and len(self.image_embeddings) > i and self.image_embeddings[i] is not None:
                img_emb = self.image_embeddings[i]
                image_score = np.dot(query_embedding, img_emb) / 100  # 归一化
            
            # 组合分数
            combined_score = (self.text_embedding_weight * text_score + 
                             (1 - self.text_embedding_weight) * image_score)
            
            scores.append((i, combined_score))
        
        # 排序并选择前K个
        scores.sort(key=lambda x: x[1], reverse=True)
        top_k_scores = scores[:self.topk]
        
        # 格式化结果
        results = []
        for idx, score in top_k_scores:
            doc = self.documents[idx]
            results.append({
                "text": doc.get("text", ""),
                "score": float(score),
                "metadata": doc.get("metadata", {}),
                "page": doc.get("metadata", {}).get("page_index")
            })
        
        return results

    def _compute_text_embedding(self, text: str) -> np.ndarray: 
        """计算文本嵌入并确保维度正确""" 
        if not text.strip(): return np.zeros(384) 
        # 确保这与您的Vespa配置匹配 
        try: 
            with torch.no_grad(): 
                embedding = self.text_model.encode([text]) 
                # 获取实际嵌入向量 
                embedding_vector = embedding[0] 
                # 确保嵌入维度在Vespa期望的范围内 
                max_dim = 1024 # 调整为与您的Vespa配置匹配 
                if len(embedding_vector) > max_dim: 
                    logger.warning(f"截断嵌入维度，从{len(embedding_vector)}到{max_dim}") 
                    return embedding_vector[:max_dim] 
                elif len(embedding_vector) < max_dim: 
                    logger.warning(f"填充嵌入维度，从{len(embedding_vector)}到{max_dim}") 
                    padded = np.zeros(max_dim) 
                    padded[:len(embedding_vector)] = embedding_vector 
                    return padded 
            return embedding_vector 
        except Exception as e: 
            logger.error(f"计算文本嵌入时出错: {str(e)}") 
            return np.zeros(max_dim) # 确保这与您的Vespa配置匹配

    def _compute_image_embedding(self, image: Image.Image) -> np.ndarray:
        """Compute image embedding"""
        if image is None:
            return np.zeros(1024)

        try:
            with torch.no_grad():
                query_embedding = self.image_model(**self.processor.process_queries([""]).to(self.image_model.device))
                image_input = self.processor.process_images([image]).to(self.image_model.device)
                image_embedding = self.image_model(**image_input)
                return image_embedding.cpu().numpy()[0][:1024]
        except Exception as e:
            logger.error(f"Error computing image embedding: {str(e)}")
            return np.zeros(1024)

            
    def _query_vespa(self, query_embedding: np.ndarray, original_query: str) -> List[Dict[str, Any]]:
        """使用向量嵌入查询Vespa"""
        try:
            # 构建Vespa查询
            query_data = {
                "yql": f"select * from sources * where sddocname contains \"{self.config.document_type}\"",
                "hits": self.topk,
                "ranking": "custom_weight",
                f"ranking.features.query({self.config.text_embedding_field})": {"values": query_embedding.tolist()},
                "ranking.features.query(text_weight)": float(self.text_embedding_weight),
                "ranking.features.query(image_weight)": float(1.0 - self.text_embedding_weight)
            }
            
            url = f"{self.config.endpoint}/search/"
            response = requests.post(
                url, 
                json=query_data, 
                timeout=self.config.timeout,
                verify=self.config.verify_ssl
            )
            
            if response.status_code == 200:
                results = response.json()
                hits = results.get("root", {}).get("children", [])
                
                # 格式化结果
                formatted_results = []
                for hit in hits:
                    fields = hit.get("fields", {})
                    formatted_results.append({
                        "text": fields.get(self.config.text_field, ""),
                        "score": hit.get("relevance", 0.0),
                        "metadata": {
                            "page_index": fields.get("page_index", None),
                            "pdf_path": fields.get("pdf_path", "")
                        }
                    })
                
                return formatted_results
            else:
                logger.warning(f"Vespa查询失败，状态码: {response.status_code}")
                # 如果Vespa查询失败，回退到内存检索
                return self._compute_similarity_and_rank(query_embedding) 
        except Exception as e:
            logger.error(f"查询Vespa时出错: {str(e)}")
            # 出错时回退到内存检索
            return self._compute_similarity_and_rank(query_embedding)