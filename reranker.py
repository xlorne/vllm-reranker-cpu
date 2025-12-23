from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()

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
        pairs = [[req.query, doc] for doc in req.documents]
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)
            scores = model(**inputs).logits.squeeze().tolist()
            if not isinstance(scores, list):
                scores = [scores]
        results = [
            {"index": i, "score": float(score), "text": doc}
            for i, (score, doc) in enumerate(zip(scores, req.documents))
        ]
        results.sort(key=lambda x: x["score"], reverse=True)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
