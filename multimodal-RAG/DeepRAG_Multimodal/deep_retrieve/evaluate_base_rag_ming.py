import json
from datetime import datetime
from tqdm import tqdm
from tool_retriever_embed import EmbeddingMatcher
# from deepsearch.retrieval_evaluate import main_eval
from deepsearch import DeepSearch_Alpha
from DeepRAG_Multimodal.deep_retrieve.ming.deepsearch_optimize_ming import DeepSearch_Beta
from langchain.docstore.document import Document
from multiprocessing import Pool, cpu_count
import concurrent.futures
from FlagEmbedding import FlagReranker, FlagModel
from DeepRAG_Multimodal.deep_retrieve.retriever_multimodal_bge import DocumentRetriever, RetrieverConfig
from pathlib import Path
import sys
sys.path.append("DeepRAG_Multimodal/deep_retrieve")

def _prepare_documents(docs):
    try:
        return [
            Document(
                page_content=doc['body'],
                metadata={
                    'title': doc['title'],
                    'source': doc['source'],
                    'published_at': doc['published_at'],
                }
            ) for doc in docs
        ]
    except:
        import pdb; pdb.set_trace()

def result_processor(results):
    matched_docs = []
    for doc, score in results:
        matched_docs.append({
            'text': doc.page_content,
        })
    return matched_docs

def llm_rerank(query, retrieval_list, reranker, topk=None):
    pairs = [[query, doc['text']] for doc in retrieval_list]
    rerank_scores = reranker.compute_score(pairs, normalize=True)
    retrieval_list = [doc for _, doc in sorted(zip(rerank_scores, retrieval_list), key=lambda x: x[0], reverse=True)]
    if topk is not None:
        retrieval_list = retrieval_list[:topk]
    return retrieval_list


def create_retriever_db(params):
    data_path = 'data/MultiHop-RAG/dataset/corpus.json'
    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    retriever = EmbeddingMatcher(
        topk=params['embedding_topk'],
        chunk_size=params['chunk_size'],
        chunk_overlap=params['chunk_overlap'],
        embedding_weight=params['embedding_weight'],
        use_local_embedding=params['use_local_embedding'],
        embedding_model_name=params['embed_model_name'],
        document_converter=_prepare_documents,
        persistent_db=True,
        persistent_db_path=params['persistent_db_path'],
        persistent_collection_name=params['persistent_collection_name'],
        initial_docs=data
    )
    return retriever

def process_single_data(args: tuple[dict, DocumentRetriever]):
    data, retriever = args
    params = get_configs()

    retrieval_list = retriever.process_single_document(
        query=data['query'],
        documents=data['documents'],  # Assuming `documents` is part of the data
        top_k=params['embedding_topk']
    )

    if params['reranker_mode']:
        retrieval_list = llm_rerank(data['query'], retrieval_list, retriever.bge_model, params['rerank_topk'])

    return {
        'query': data['query'],
        'answer': data['answer'],
        'question_type': data['question_type'],
        'retrieval_list': retrieval_list,
        'gold_list': data['evidence_list']
    }

def create_retriever_for_process():
    """为每个进程创建新的retriever实例"""
    params = get_configs()  # 获取配置
    return create_retriever_db(params)

def process_chunk(chunk_data):
    """处理数据块的函数"""
    retriever = create_retriever_for_process()
    reranker = FlagReranker(model_name_or_path="BAAI/bge-reranker-large")
    results = []
    for data in tqdm(chunk_data, desc="Processing items in chunk", leave=False):
        result = process_single_data((data, retriever))
        results.append(result)
    return results

def evaluate_base_rag_benchmark(params):
    test_data = '/home/liuguanming/multimodal-RAG/LongDocURL/LongDocURL_public_with_subtask_category.jsonl'
    datasets = []
    with open(test_data, 'r', encoding='utf-8') as file:
        for line in file:
            datasets.append(json.loads(line.strip()))

    retrieval_save_list = []
    query_expansion_1_dir = "/home/liuguanming/multimodal-RAG/DeepRAG_Multimodal/deep_retrieve/query_expansion_task.jsonl"
    with open(query_expansion_1_dir, 'r', encoding='utf-8') as file:
        query_expansion_1 = [json.loads(line.strip()) for line in file]
    retriever_config = RetrieverConfig(
        model_name=params['embed_model_name'],
        device="cuda:0",
        chunk_size=params['chunk_size'],
        chunk_overlap=params['chunk_overlap'],
        mode='mixed',
        log_file=params['save_file']
    )
    retriever = DocumentRetriever(retriever_config)

    if params.get('debug_mode', False):
        # 使用普通for循环方式（便于调试）
        
        for data, query_expansion in tqdm(zip(datasets,query_expansion_1)):
            pdf_path = data['pdf_path']
            start_page, end_page = data['start_end_idx']
            evidence_pages = data['evidence_pages']
            query = data['question']
            
            original_query = query_expansion['question']
            expand_q = [
                query_expansion['Understanding']['expanded_query'],
                query_expansion['Reasoning']['expanded_query'],
                query_expansion['Locating']['expanded_query']
            ]
            result = process_single_data((data, retriever))
            retrieval_save_list.append(result)

        # for data in tqdm(datasets):
        #     result = process_single_data((data, retriever))
        #     retrieval_save_list.append(result)
    elif params.get('is_mp', False):
        num_processes = params.get('num_processes', 4)
        chunk_size = len(datasets) // num_processes
        chunks = [datasets[i:i + chunk_size] for i in range(0, len(datasets), chunk_size)]

        print(f"Processing {len(datasets)} items with {num_processes} processes")
        with Pool(processes=num_processes) as pool:
            chunk_results = list(tqdm(
                pool.imap(process_chunk, chunks),
                total=len(chunks),
                desc="Processing chunks",
                position=0,
                leave=True
            ))

            retrieval_save_list = [item for sublist in chunk_results for item in sublist]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=params['num_processes']) as executor:
            futures = [executor.submit(process_single_data, (data, retriever)) for data in datasets]

            retrieval_save_list = list(tqdm(
                (future.result() for future in concurrent.futures.as_completed(futures)),
                total=len(datasets),
                desc="Processing data"
            ))

    with open(params['save_file'], 'w') as json_file:
        json.dump(retrieval_save_list, json_file, indent=2, ensure_ascii=False)

    return retrieval_save_list

def get_configs():
    """
    embed_model: ['text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002']
    num_processes: [1, 2, 4, 8, 16, 32]
    is_mp: [True, False]
    debug_mode: [True, False]
    dataset_split: [1, 4] # 1: 使用全部数据; 4: 使用1/4数据
    deepsearch_mode: ['alpha', 'beta', 'base_retrieval']
    """
    
    # 获取当前脚本的上级目录路径
    current_dir = Path(__file__).parent.resolve()  # 当前文件所在目录
    parent_dir = current_dir.parent  # 上级目录
    grandparent_dir = current_dir.parent.parent  # 上上级目录
    
    params = {
        'debug_mode': False,  # 设置为True时使用for循环模式
        'num_processes': 8,  # 添加进程数配置
        # 新增 DeepSearch 相关配置
        'deepsearch_mode': 'beta',
        'reranker_mode': True,
        # Retrieval 相关配置
        "pdf_path": f'{grandparent_dir}/picked_LongDoc',
        "query_expansion_1_dir": f"{parent_dir}/query_expansion_task.jsonl",
        'embedding_topk': 20,
        'rerank_topk': 10,
        'dataset_split': 8,
        'chunk_size': 1000,
        'chunk_overlap': 100,
        'embedding_weight': 1.0,
        'use_local_embedding': False,
        'save_file': 'res/deepsearch/mix_retrieval_save_list.json',
        # 'is_mp': True,
        'embed_model_name': 'text-embedding-ada-002',
        # 'embed_model_name': 'sentence-transformers/all-MiniLM-L6-v2',
        # 新增 EmbeddingMatcher 相关配置
        'persistent_db': True,
        'persistent_db_path': 'db/deepsearch_chroma_3_small',
        'persistent_collection_name': 'persistent_collection',
    }
    return params

def save_result(metrics, params):
    metrics['params'] = params
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json.dump(metrics, open(f'res/deepsearch/metrics_{timestamp}.json', 'w'), indent=4, ensure_ascii=False)

if __name__ == "__main__":
    # from configs.config import load_envs_func
    # load_envs_func()
    
    params = get_configs()

    retrieval_save_list = evaluate_base_rag_benchmark(params)
    # metrics = main_eval(inference_result=retrieval_save_list)

    # save_result(metrics, params)
