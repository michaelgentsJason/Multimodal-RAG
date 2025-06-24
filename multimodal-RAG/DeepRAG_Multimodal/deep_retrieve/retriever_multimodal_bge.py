import torch
from PIL import Image
import json
from tqdm import tqdm
from pdf2image import convert_from_path
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
import os
import numpy as np
from collections import defaultdict
import logging
from FlagEmbedding import FlagModel
import pytesseract
from dataclasses import dataclass
from typing import List, Tuple, Optional
import argparse
from paddleocr import PaddleOCR
import os
from pathlib import Path
import pickle  # Add this import for saving and loading scores
from transformers import AutoTokenizer, AutoModel


@dataclass
class RetrieverConfig:
    model_name: str = "vidore/colqwen2.5-v0.2"
    processor_name: str = "vidore/colqwen2.5-v0.1"
    bge_model_name: str = "BAAI/bge-large-en-v1.5"
    device: str = "cuda"
    use_fp16: bool = True
    batch_size: int = 32
    threshold: float = 0.5
    chunk_size: int = 0
    chunk_overlap: int = 50
    cache_dir: str = 'retrieval_cache'
    log_file: str = 'retrieval_log.jsonl'
    mode: str = 'mixed'  # 'mixed', 'text_only', 'image_only'
    weight_combinations: List[Tuple[float, float]] = None,
    ocr_method: str = 'paddleocr'  # 'pytesseract' or 'paddleocr'

    def __post_init__(self):
        if self.weight_combinations is None:
            self._set_default_weights()

    def _set_default_weights(self):
        if self.mode == 'text_only':
            self.weight_combinations = [(0.0, 1.0)]
        elif self.mode == 'image_only':
            self.weight_combinations = [(1.0, 0.0)]
        else:
            self.weight_combinations = [
                (0.5, 0.5),  # 混合模式
                (0.0, 1.0),  # 纯文本模式
                (1.0, 0.0),  # 纯图像模式
                (0.6, 0.4),
                (0.4, 0.6),
            ]

    @classmethod
    def from_args(cls, args, parent_dir):
        log_file = args.log_file or os.path.join(parent_dir, f'picked_LongDoc/{args.ocr_method}_retrieval_log.jsonl')
        return cls(
            model_name=args.model_name,
            device=args.device,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            mode=args.mode,
            ocr_method=args.ocr_method,
            log_file=log_file
        )


class DocumentRetriever:
    def __init__(self, config: RetrieverConfig):
        self.config = config
        self._setup_models()
        self.conditions = ["最严格条件", "宽容条件top_1", "宽容条件top_2", "宽容条件top_5"]
        self.top_k = [1, 2, 5]

    def _setup_models(self):
        qwen_model_dir = "/root/autodl-tmp/multimodal-RAG/hf_models/vidore/colqwen2.5-v0.2"
        self.model = ColQwen2_5.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.config.device,
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
        ).eval()

        self.processor = ColQwen2_5_Processor.from_pretrained(
            self.config.processor_name,
            size={"shortest_edge": 512, "longest_edge": 1024}
        )

        self.bge_model = FlagModel(
            self.config.bge_model_name,
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
            use_fp16=self.config.use_fp16,
            device=self.config.device,
            truncate=True  # 启用自动截断，避免过长输入产生 chunking
        )

    def extract_text_from_pages(self, pages, save_path):
        ocr_data = {}
        ocr_method = self.config.ocr_method
        if ocr_method == 'pytesseract':
            for i, page in enumerate(pages):
                text = pytesseract.image_to_string(page, lang="chi_sim+chi_tra+eng")
                ocr_data[f"Page_{i + 1}"] = text.strip()

            with open(save_path, 'w', encoding='utf-8') as json_file:
                json.dump(ocr_data, json_file, ensure_ascii=False, indent=4)
        elif ocr_method == 'paddleocr':
            ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=False)
            for i, page in enumerate(pages):
                result = ocr.ocr(np.array(page))
                text = '\n'.join([line[1][0] for line in result[0]])
                ocr_data[f"Page_{i + 1}"] = text.strip()

            with open(save_path, 'w', encoding='utf-8') as json_file:
                json.dump(ocr_data, json_file, ensure_ascii=False, indent=4)

        return ocr_data

    def process_single_document(self, pdf_path: str, query: str,
                                start_page: int, end_page: int,
                                evidence_pages: List[int], record_idx: int, pdf_path_abs: str) -> dict:
        cache_filename = f"{pdf_path_abs}/{self.config.cache_dir}/{pdf_path.replace('.pdf', '')}_{record_idx}_{self.config.mode}_{self.config.ocr_method}.npz"
        os.makedirs(os.path.dirname(cache_filename), exist_ok=True)

        if os.path.exists(cache_filename):
            cached_data = np.load(cache_filename, allow_pickle=True)
            return {
                'text_scores': cached_data['text_scores'],
                'image_scores': cached_data['image_scores'],
                'page_indices': cached_data['page_indices']
            }

        os.makedirs(f'{pdf_path_abs}/{self.config.ocr_method}_save', exist_ok=True)
        processed_text_path = f"{pdf_path_abs}/{self.config.ocr_method}_save/{pdf_path.replace('.pdf', '')}.json"
        pages = convert_from_path(f"{pdf_path_abs}/{pdf_path}")
        if os.path.exists(processed_text_path):
            with open(processed_text_path, 'r') as json_file:
                text_from_pdf = json.load(json_file)
            text_per_page = []
            for value in text_from_pdf.values():
                text_per_page.append(value)
        else:
            text_per_page = self.extract_text_from_pages(pages, processed_text_path)

        if len(text_per_page) != len(pages):
            print(f"文本页数与图像页数不匹配: {pdf_path}, 文本页数: {len(text_per_page)}, 图像页数: {len(pages)}")
            min_len = min(len(text_per_page), len(pages))
            text_per_page = text_per_page[:min_len]
            pages = pages[:min_len]
            if min_len == 0:
                return {}

        page_indices = np.array(list(range(1, len(pages) + 1)))
        text_scores_all = np.zeros(len(pages))
        image_scores_all = np.zeros(len(pages))
        batch_size = self.config.batch_size

        while batch_size > 0:
            try:
                for batch_idx in range(0, len(pages), batch_size):
                    batch_end = min(batch_idx + batch_size, len(pages))
                    batch_images = pages[batch_idx:batch_end]
                    batch_texts = text_per_page[batch_idx:batch_end]

                    if self.config.mode in ['mixed', 'image_only']:
                        with torch.no_grad():
                            query_embedding = self.model(
                                **self.processor.process_queries([query]).to(self.model.device))
                            image_inputs = self.processor.process_images(batch_images).to(self.model.device)
                            image_embeddings = self.model(**image_inputs)
                            batch_image_scores = self.processor.score_multi_vector(query_embedding,
                                                                                   image_embeddings).cpu().numpy().flatten()
                        image_scores_all[batch_idx:batch_end] = batch_image_scores

                    if self.config.mode in ['mixed', 'text_only']:
                        batch_text_scores = self.process_text_batch(batch_texts, query)
                        text_scores_all[batch_idx:batch_end] = batch_text_scores

                    torch.cuda.empty_cache()
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                batch_size //= 2
                print(f"CUDA out of memory. Reducing batch size to {batch_size}")

        np.savez(cache_filename,
                 text_scores=text_scores_all,
                 image_scores=image_scores_all,
                 page_indices=page_indices)

        return {
            'text_scores': text_scores_all,
            'image_scores': image_scores_all,
            'page_indices': page_indices
        }

    def process_text_batch(self, batch_texts, query):
        """
        Process a batch of texts and compute similarity scores with the query.

        Args:
            batch_texts (List[str]): List of texts to process.
            query (str): Query string.

        Returns:
            np.ndarray: Array of similarity scores for the batch.
        """

        def truncate_text(text, max_length=512):
            """
            Truncate text to a maximum length to avoid issues with encoding.

            Args:
                text (str): Input text.
                max_length (int): Maximum number of characters.

            Returns:
                str: Truncated text.
            """
            return text[:max_length]

        processed_texts = [self.preprocess_text(text) for text in batch_texts]
        # processed_texts = [truncate_text(text) for text in processed_texts]
        valid_texts = [text for text in processed_texts if text.strip()]

        if not valid_texts:
            return np.zeros(len(batch_texts))

        batch_size = self.config.batch_size
        all_scores = []

        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i:i + batch_size]
            try:
                query_embedding = self.bge_model.encode([query])
                text_embeddings = self.bge_model.encode(batch)
                scores = query_embedding @ text_embeddings.T
                all_scores.extend(scores.flatten())
            except Exception as e:
                print(f"批次处理失败 {i}/{len(valid_texts)}: {str(e)}")
                all_scores.extend([0.0] * len(batch))

        batch_text_scores = np.zeros(len(batch_texts))
        valid_idx = 0
        for i in range(len(batch_texts)):
            if processed_texts[i].strip():
                if valid_idx < len(all_scores):
                    batch_text_scores[i] = all_scores[valid_idx]
                valid_idx += 1

        return batch_text_scores

    @staticmethod
    def preprocess_text(text):
        if not isinstance(text, str):
            return ""
        text = text.replace('\x00', '')
        text = text.strip()
        text = ' '.join(text.split())
        return text

    def evaluate_document(self, scores_dict: dict, info_dict: dict, evidence_pages: List[int]) -> dict:
        results = defaultdict(dict)

        for image_weight, text_weight in self.config.weight_combinations:
            weight_key = f"image_{image_weight:.1f}_text_{text_weight:.1f}"
            combined_scores = self.combine_ranks(
                scores_dict['text_scores'],
                scores_dict['image_scores'],
                text_weight,
                image_weight
            )

            mask = combined_scores > self.config.threshold
            valid_scores = combined_scores[mask]
            valid_pages = scores_dict['page_indices'][mask]

            if len(valid_scores) == 0:
                continue

            sorted_indices = np.argsort(-valid_scores)
            sorted_pages = valid_pages[sorted_indices]

            # top_k_save = len(evidence_pages)

            for cond in self.conditions:
                if cond == "最严格条件":
                    eval_top_k = len(evidence_pages)
                elif cond == "宽容条件top_1":
                    eval_top_k = len(evidence_pages) + self.top_k[0]
                elif cond == "宽容条件top_2":
                    eval_top_k = len(evidence_pages) + self.top_k[1]
                elif cond == "宽容条件top_5":
                    eval_top_k = len(evidence_pages) + self.top_k[2]

                eval_top_k_pages = sorted_pages[:eval_top_k]

                results[weight_key][cond] = set(evidence_pages).issubset(set(eval_top_k_pages))

                # Add logging for each condition
                log_entry = {
                    'pdf_path': info_dict.get('pdf_path', ''),
                    'query': info_dict.get('query', ''),
                    'eval_top_k_save_pages': sorted_pages[:eval_top_k].tolist(),
                    'evidence_pages': evidence_pages,
                    "eval_top_k": eval_top_k,
                    "image_weight": image_weight,
                    "text_weight": text_weight,
                    "condition": cond,
                    "eval_top_k_pages": eval_top_k_pages.tolist(),
                    "success": set(evidence_pages).issubset(set(eval_top_k_pages)),
                    # Save scores for recalculating weights
                    "text_scores": scores_dict['text_scores'].tolist(),
                    "image_scores": scores_dict['image_scores'].tolist()
                }
                with open(self.config.log_file, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')

        return results

    @staticmethod
    def combine_ranks(text_scores, image_scores, text_weight=0.5, image_weight=0.5):
        image_scores = image_scores / 100
        total_weight = text_weight + image_weight
        text_weight = text_weight / total_weight
        image_weight = image_weight / total_weight
        combined_scores = text_weight * text_scores + image_weight * image_scores
        # print(f"Text : {text_scores}, Image: {image_scores}, final {combined_scores}")
        return combined_scores

    @staticmethod
    def load_dataset(dataset_path):
        with open(dataset_path, 'r') as f:
            return [json.loads(line.strip()) for line in f]

    # TODO: 可能还有点问题
    @staticmethod
    def print_results(results):
        aggregated_results = defaultdict(lambda: defaultdict(lambda: {'success': 0, 'total': 0}))

        for weight_key, cond_results in results.items():
            for cond, count in cond_results.items():
                aggregated_results[weight_key][cond]['success'] += count
                aggregated_results[weight_key][cond]['total'] += 1

        print("\n===== Retrieval Results =====")
        for weight_key, cond_results in aggregated_results.items():
            print(f"\n{weight_key} 权重配置:")
            for cond, counts in cond_results.items():
                total = counts['total']
                success = counts['success']
                success_rate = (success / total * 100) if total > 0 else 0.0
                print(f"  - {cond}: {success_rate:.2f}% ({success}/{total})")

    @staticmethod
    def print_results_from_file(json_file_path):
        with open(json_file_path, 'r') as f:
            logs = [json.loads(line.strip()) for line in f]

        results = defaultdict(lambda: defaultdict(lambda: {'success': 0, 'total': 0}))

        for log in logs:
            weight_key = f"image_{log['image_weight']:.1f}_text_{log['text_weight']:.1f}"
            condition = log['condition']
            success = log['success']
            results[weight_key][condition]['total'] += 1
            if success:
                results[weight_key][condition]['success'] += 1

        print("\n===== Retrieval Results =====")
        for weight_key, cond_results in results.items():
            print(f"\n{weight_key} 权重配置:")
            for cond, counts in cond_results.items():
                total = counts['total']
                success = counts['success']
                success_rate = (success / total * 100) if total > 0 else 0.0
                print(f"  - {cond}: {success_rate:.2f}% ({success}/{total})")

    @staticmethod
    def evaluate_new_weights(log_file_path: str, new_weight_combinations: List[Tuple[float, float]]):
        with open(log_file_path, 'r') as f:
            logs = [json.loads(line.strip()) for line in f]

        results = defaultdict(lambda: defaultdict(lambda: {'success': 0, 'total': 0}))

        for log in logs:
            text_scores = np.array(log['text_scores'])
            image_scores = np.array(log['image_scores'])
            evidence_pages = set(log['evidence_pages'])
            total_pages = len(text_scores)  # Total number of pages

            for image_weight, text_weight in new_weight_combinations:
                combined_scores = DocumentRetriever.combine_ranks(
                    text_scores, image_scores, text_weight, image_weight
                )
                sorted_indices = np.argsort(-combined_scores)
                sorted_pages = np.arange(1, total_pages + 1)[sorted_indices]  # Sequential page indices

                for cond in ["最严格条件", "宽容条件top_1", "宽容条件top_2", "宽容条件top_5"]:
                    if cond == "最严格条件":
                        eval_top_k = len(evidence_pages)
                    elif cond == "宽容条件top_1":
                        eval_top_k = len(evidence_pages) + 1
                    elif cond == "宽容条件top_2":
                        eval_top_k = len(evidence_pages) + 2
                    elif cond == "宽容条件top_5":
                        eval_top_k = len(evidence_pages) + 5

                    eval_top_k_pages = sorted_pages[:eval_top_k]
                    success = evidence_pages.issubset(set(eval_top_k_pages))

                    weight_key = f"image_{image_weight:.1f}_text_{text_weight:.1f}"
                    results[weight_key][cond]['total'] += 1
                    if success:
                        results[weight_key][cond]['success'] += 1

        print("\n===== New Weight Combinations Results =====")
        for weight_key, cond_results in results.items():
            print(f"\n{weight_key} 权重配置:")
            for cond, counts in cond_results.items():
                total = counts['total'] / 20
                success = counts['success'] / 20
                success_rate = (success / total * 100) if total > 0 else 0.0
                print(f"  - {cond}: {success_rate:.2f}% ({success}/{total})")


class MultimodalMatcher:
    def __init__(
            self,
            config: RetrieverConfig,
            embedding_weight: float = 0.2,
            topk: int = 10
    ):
        self.config = config
        self.text_embedding_weight = embedding_weight
        self.topk = topk
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self._setup_models()

    def _setup_models(self):
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.config.bge_model_name, use_fast=True)
        self.text_model = AutoModel.from_pretrained(self.config.bge_model_name).to(self.device)
        # self.text_model = FlagModel(
        #     self.config.bge_model_name,
        #     query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
        #     use_fp16=self.config.use_fp16,
        #     device=self.config.device
        # )
        self.image_model = ColQwen2_5.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,

        ).eval()
        self.processor = ColQwen2_5_Processor.from_pretrained(
            self.config.processor_name,
            size={"shortest_edge": 512, "longest_edge": 1024},

        )

    def save_scores(self, query, pdf_path, text_scores, image_scores, save_path):
        """
        Save query, pdf_path, text scores, and image scores to a local file.

        Args:
            query (str): Query string.
            pdf_path (str): Path to the PDF document.
            text_scores (list): List of text scores.
            image_scores (list): List of image scores.
            save_path (str): Path to save the scores.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump({
                'query': query,
                'pdf_path': pdf_path,
                'text_scores': text_scores,
                'image_scores': image_scores
            }, f)

    def load_scores(self, save_path):
        """
        Load query, pdf_path, text scores, and image scores from a local file.

        Args:
            save_path (str): Path to load the scores from.

        Returns:
            dict: Dictionary containing 'query', 'pdf_path', 'text_scores', and 'image_scores'.
        """
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Score file not found: {save_path}")
        with open(save_path, 'rb') as f:
            return pickle.load(f)

    def retrieve(self, query: str, documents: List[dict], save_path: Optional[str] = None) -> List[dict]:
        text_scores, image_scores = [], []
        processed_documents = self._process_documents(documents)
        valid_documents = []

        for doc in processed_documents:
            try:
                # 获取文本和图像
                text = doc.get("text", "")
                image = doc.get("image", None)

                # 检查图像是否有效
                if image is not None:
                    try:
                        # 验证图像尺寸
                        width, height = image.size
                        if width <= 0 or height <= 0:
                            # 替换无效图像为空白图像
                            image = Image.new('RGB', (100, 100), color='white')
                            doc["image"] = image
                    except (AttributeError, Exception) as e:
                        # 图像对象无效
                        image = Image.new('RGB', (100, 100), color='white')
                        doc["image"] = image

                pdf_path = doc.get("pdf_path", "")
                text_score = self._compute_text_score(query, text)
                image_score = self._compute_image_score(query, image)
                combined_score = self._combine_scores(text_score, image_score)

                doc["score"] = combined_score
                #doc["score"] = image_score
                doc["metadata"] = doc.get("metadata", {})  # Ensure metadata exists
                text_scores.append(text_score)
                image_scores.append(image_score)
                valid_documents.append(doc)
            except Exception as e:
                # 处理任何其他错误
                print(f"处理文档时出错: {str(e)}")
                continue

        # Save scores locally if a save path is provided
        if save_path:
            self.save_scores(query, pdf_path, text_scores, image_scores, save_path)

        # Ensure page_index is preserved in the results
        for doc in processed_documents:
            if "metadata" in doc and "page_index" not in doc["metadata"]:
                doc["metadata"]["page_index"] = None  # Default to None if page_index is missing

        return sorted(processed_documents, key=lambda x: x["score"], reverse=True)[:self.topk]

    def process_saved_scores(self, save_path):
        """
        Process saved scores by loading them and performing operations.

        Args:
            save_path (str): Path to the saved scores file.

        Returns:
            dict: Processed results based on the loaded scores.
        """
        try:
            saved_data = self.load_scores(save_path)
            query = saved_data['query']
            pdf_path = saved_data['pdf_path']
            text_scores = saved_data['text_scores']
            image_scores = saved_data['image_scores']

            # Example operation: Combine scores and return the result
            combined_scores = [
                self._combine_scores(text_score, image_score)
                for text_score, image_score in zip(text_scores, image_scores)
            ]
            return {
                'query': query,
                'pdf_path': pdf_path,
                'combined_scores': combined_scores,
                'text_scores': text_scores,
                'image_scores': image_scores
            }
        except FileNotFoundError as e:
            print(str(e))
            return {}

    def _process_documents(self, documents: List[dict]) -> List[dict]:
        """
        Process documents to handle PDFs by converting them into pages and extracting text or images.

        Args:
            documents (List[dict]): List of documents, each containing either text, image, or a PDF path.

        Returns:
            List[dict]: Processed documents with extracted text or images.
        """
        # processed_documents = []
        # for doc in documents:
        #     if "pdf_path" in doc:
        #         pdf_pages = self._pdf_to_pages(doc["pdf_path"])
        #         for page_index, page_content in enumerate(pdf_pages):
        #             processed_documents.append({
        #                 "text": page_content.get("text", ""),
        #                 "image": page_content.get("image", None),
        #                 "metadata": {**doc.get("metadata", {}), "page_index": page_index + 1}
        #             })
        #     else:
        #         processed_documents.append(doc)
        return documents

    def _pdf_to_pages(self, pdf_path: str) -> List[dict]:
        """
        Convert a PDF into pages and extract text and images, or load from an existing JSON file if available.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            List[dict]: List of pages, each containing extracted text and image.
        """
        parent_dir = '/content/drive/MyDrive/DeepSearch/multimodal-RAG/DeepRAG_Multimodal/picked_LongDoc/paddleocr_save'
        json_path = os.path.join(parent_dir, f"{os.path.basename(pdf_path).replace('.pdf', '.json')}")

        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as json_file:
                loaded_data = json.load(json_file)
            pages = convert_from_path(pdf_path)
            return [
                {
                    "text": page_data,
                    "image": pages[idx] if idx < len(pages) else None
                }
                for idx, page_data in enumerate(loaded_data.values())
            ]

        pages = convert_from_path(pdf_path)
        extracted_pages = []

        for page in pages:
            page_text = self._extract_text_from_image(page)
            extracted_pages.append({
                "text": page_text,
                "image": page  # Keep the PIL Image format for compatibility with image_encoder
            })

        return extracted_pages
        
    
    def _extract_text_from_image(self, image: Image.Image) -> str:
        """
        Extract text from an image using OCR.

        Args:
            image (Image.Image): Image object.

        Returns:
            str: Extracted text.
        """
        if self.config.ocr_method == "pytesseract":
            return pytesseract.image_to_string(image, lang="chi_sim+chi_tra+eng").strip()
        elif self.config.ocr_method == "paddleocr":
            ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=False)
            result = ocr.ocr(np.array(image))
            return '\n'.join([line[1][0] for line in result[0]]).strip()
        return ""

    def _compute_text_score(self, query: str, text: str) -> float:
        if not text.strip():
            return 0.0
        query_tokens = self.text_tokenizer(query, padding=True, truncation=True, return_tensors="pt",
                                           max_length=512).to(self.config.device)
        text_tokens = self.text_tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512).to(self.config.device)
        with torch.no_grad():
            query_embedding = self.text_model(**query_tokens)
            text_embedding = self.text_model(**text_tokens)
        query_embedding = query_embedding.last_hidden_state[:, 0].cpu().numpy()
        text_embedding = text_embedding.last_hidden_state[:, 0].cpu().numpy()
        # 归一化
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        text_embedding = text_embedding / np.linalg.norm(text_embedding, axis=1, keepdims=True)
        return float(query_embedding @ text_embedding.T)

    def _compute_image_score(self, query: str, image: Optional[Image.Image]) -> float:
        if image is None:
            return 0.0
        with torch.no_grad():
            query_embedding = self.image_model(**self.processor.process_queries([query]).to(self.image_model.device))
            image_input = self.processor.process_images([image]).to(self.image_model.device)
            image_embedding = self.image_model(**image_input)
            return float(self.processor.score_multi_vector(query_embedding, image_embedding).cpu().numpy().flatten()[0])

    def _combine_scores(self, text_score: float, image_score: float) -> float:
        image_score = image_score / 100
        # total_weight = self.embedding_weight + (1 - self.embedding_weight)
        text_weight = self.text_embedding_weight
        image_weight = (1 - self.text_embedding_weight)
        return text_weight * text_score + image_weight * image_score


def main():
    parser = argparse.ArgumentParser(description='文档检索评估')
    parser.add_argument('--model_name', default="vidore/colqwen2.5-v0.2")
    parser.add_argument('--device', default="cuda:0")
    parser.add_argument('--ocr_method', choices=['pytesseract', 'paddleocr'], default='pytesseract')
    parser.add_argument('--chunk_size', type=int, default=0)
    parser.add_argument('--chunk_overlap', type=int, default=50)
    parser.add_argument('--mode', choices=['mixed', 'text_only', 'image_only'], default='mixed')
    parser.add_argument('--dataset_path',
                        default='picked_LongDoc/selected_LongDocURL_public_with_subtask_category.jsonl')
    parser.add_argument('--log_file', help='路径到评估日志文件', default=None)
    parser.add_argument('--just_evaluate', default=False, help='仅评估，不执行检索')
    args = parser.parse_args()

    # 获取当前脚本的上级目录路径
    current_dir = Path(__file__).parent.resolve()  # 当前文件所在目录
    parent_dir = current_dir.parent  # 上级目录

    config = RetrieverConfig.from_args(args, parent_dir)
    retriever = DocumentRetriever(config)

    if os.path.exists(config.log_file) and args.just_evaluate:
        retriever.print_results_from_file(config.log_file)
        return

    dataset_path = parent_dir / args.dataset_path
    dataset = retriever.load_dataset(dataset_path)

    results = defaultdict(lambda: defaultdict(int))
    pdf_path_abs = os.path.join(parent_dir, "picked_LongDoc")

    for record_idx, record in enumerate(tqdm(dataset)):
        scores_dict = retriever.process_single_document(
            record['pdf_path'],
            record['question'],
            record['start_end_idx'][0],
            record['start_end_idx'][1],
            record['evidence_pages'],
            record_idx,
            pdf_path_abs
        )

        info_dict = {
            'pdf_path': record['pdf_path'],
            'query': record['question'],
            'evidence_pages': record['evidence_pages']
        }
        doc_results = retriever.evaluate_document(scores_dict, info_dict, record['evidence_pages'])

        for weight_key, cond_results in doc_results.items():
            for cond, success in cond_results.items():
                results[weight_key][cond] += 1 if success else 0

    retriever.print_results(results)


def evalaute_new_weights():
    parser = argparse.ArgumentParser(description='文档检索评估')
    parser.add_argument('--model_name', default="vidore/colqwen2.5-v0.2")
    parser.add_argument('--device', default="cuda:0")
    parser.add_argument('--ocr_method', choices=['pytesseract', 'paddleocr'], default='paddleocr')
    parser.add_argument('--chunk_size', type=int, default=0)
    parser.add_argument('--chunk_overlap', type=int, default=50)
    parser.add_argument('--mode', choices=['mixed', 'text_only', 'image_only'], default='mixed')
    parser.add_argument('--dataset_path',
                        default='picked_LongDoc/selected_LongDocURL_public_with_subtask_category.jsonl')
    parser.add_argument('--log_file', help='路径到评估日志文件', default=None)
    parser.add_argument('--just_evaluate', default=False, help='仅评估，不执行检索')
    args = parser.parse_args()

    # 获取当前脚本的上级目录路径
    current_dir = Path(__file__).parent.resolve()  # 当前文件所在目录
    parent_dir = current_dir.parent  # 上级目录

    config = RetrieverConfig.from_args(args, parent_dir)
    retriever = DocumentRetriever(config)
    retriever.evaluate_new_weights(config.log_file, [  # 混合模式
        (0.0, 1.0),  # 纯文本模式
        (0.5, 0.5),
        (0.6, 0.4),
        (0.4, 0.6),
        (1.0, 0.0),  # 纯图像模式
        (0.2, 0.8),
        (0.8, 0.2)])


if __name__ == "__main__":
    # main()
    # evalaute_new_weights()
    # test_retri = "/home/liuguanming/multimodal-RAG/DeepRAG_Multimodal/picked_LongDoc/paddleocr_retrieval_log.jsonl"
    # print_results_from_file(test_retri)
    # evaluate_new_weights(test_retri, [(0.2, 0.8), (0.8, 0.2)])

    # text_model = FlagModel(
    #         "BAAI/bge-large-en-v1.5",
    #         query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
    #         use_fp16=True,
    #         device="cuda:0"
    #     )
    # query = ["你能告诉我这篇文章的主要内容吗？", "这篇文章的作者是谁？", "这篇文章的主题是什么？"]
    # text_embedding = text_model.encode(query)
    # print(text_embedding.shape)

    # 初始化模型和分词器
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5").to("cuda:0")
    model.eval()


    # 编码函数（支持批量）
    def encode_text(texts):
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to("cuda:0")
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0].cpu().numpy()  # 取 [CLS] 向量


    # 测试
    query = "你能告诉我这篇文章的主要内容吗？"
    text = ['a', 'b']
    query_embedding = encode_text(query)
    text_embedding = encode_text(text)
    print(query_embedding @ text_embedding.T)


