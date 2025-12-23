from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()

# 最大的长度限制 8192
max_length = 8192

model_name = "/models/bge-reranker-v2-m3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()
model.to("cpu")  # Explicitly use CPU

class RerankRequest(BaseModel):
    query: str
    documents: List[str]

@app.post("/v1/rerank")
async def rerank(req: RerankRequest):
    try:
        if not req.documents:
            return {"results": []}

        # 每一对是 [query, doc]，max_length 限制的是每对拼接后的总长度
        pairs = [[req.query, doc] for doc in req.documents]
        with torch.no_grad():
            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=max_length,  # 每对 [query, doc] 的最大总 token 数
                add_special_tokens=True
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            logits = model(**inputs).logits.squeeze(dim=-1)  # shape: [n_docs]

            # 使用 sigmoid 将 logits 映射到 (0, 1) 区间
            scores = torch.sigmoid(logits).tolist()

            # 确保 scores 是 list（处理单文档情况）
            if not isinstance(scores, list):
                scores = [scores]

        results = [
            {"index": i, "score": float(score), "text": doc}
            for i, (score, doc) in enumerate(zip(scores, req.documents))
        ]
        results.sort(key=lambda x: x["score"], reverse=True)
        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reranking failed: {str(e)}")