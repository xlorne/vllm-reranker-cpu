from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import random
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# =============== 强制确定性 ===============
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
# =========================================

app = FastAPI()

# 最大的长度限制 8192
max_length = 8192

print("Loading reranker model and tokenizer...")
load_start = time.perf_counter()
model_name = "/models/bge-reranker-v2-m3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()
model.to("cpu")  # Explicitly use CPU
# 确保模型完全处于评估模式
for param in model.parameters():
    param.requires_grad = False
load_end = time.perf_counter()
LOAD_DURATION_NS = int((load_end - load_start) * 1e9)
print(f"Reranker model loaded in {(load_end - load_start):.2f} seconds")


class RerankRequest(BaseModel):
    query: str
    documents: List[str]


@app.post("/v1/rerank")
async def rerank(req: RerankRequest):
    try:
        if not req.documents:
            return {
                "results": [],
                "load_duration": LOAD_DURATION_NS,
                "model": "bge-reranker-v2-m3"
            }

        start_time = time.perf_counter()
        
        # 每一对是 [query, doc]，max_length 限制的是每对拼接后的总长度
        pairs = [[req.query, doc] for doc in req.documents]
        
        with torch.no_grad():
            # 禁用tokenizer的并行处理以确保一致性
            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=max_length,  # 每对 [query, doc] 的最大总 token 数
                add_special_tokens=True,
                padding_side='right'  # 明确指定padding方向
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # 计算token数量
            token_count = sum(len(ids) for ids in inputs['input_ids'])
            
            # 禁用自动混合精度，确保浮点数一致性
            with torch.autocast(enabled=False, device_type='cpu'):
                logits = model(**inputs).logits.squeeze(dim=-1)  # shape: [n_docs]
            
            # 使用 sigmoid 将 logits 映射到 (0, 1) 区间
            scores = torch.sigmoid(logits).tolist()

            # BGE-reranker直接输出logits作为相关性分数
            # scores = model(**inputs).logits.squeeze(dim=-1).tolist()

            # 确保 scores 是 list（处理单文档情况）
            if not isinstance(scores, list):
                scores = [scores]

        # 计算处理时间
        end_time = time.perf_counter()
        total_duration_ns = int((end_time - start_time) * 1e9)
        
        results = [
            {"index": i, "score": float(score), "text": doc}
            for i, (score, doc) in enumerate(zip(scores, req.documents))
        ]
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "results": results,
            "model": "bge-reranker-v2-m3",
            "total_duration": total_duration_ns,
            "load_duration": LOAD_DURATION_NS,
            "token_count": token_count
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reranking failed: {str(e)}")
