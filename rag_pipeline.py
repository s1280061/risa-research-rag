"""
RiSA Research RAG Pipeline (FAISS version)
- Vector DB : FAISS (local)
- Embedding : OpenAI text-embedding-3-small
- LLM       : OpenAI gpt-4o-mini

Usage:
    pip install openai faiss-cpu numpy python-dotenv
    python rag_pipeline.py

Optional:
    .env file:
        OPENAI_API_KEY=sk-...
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ── 設定 ──────────────────────────────────────────────────────────
# 変更後
JSONL_PATH   = Path("/data/rag_documents.jsonl")
INDEX_PATH   = Path("/data/risa_research.faiss")
META_PATH    = Path("/data/risa_research_meta.json")
EMBED_MODEL  = "text-embedding-3-small"
LLM_MODEL    = "gpt-5-mini"
TOP_K        = 5
BATCH_SIZE   = 20

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "OPENAI_API_KEY が設定されていません。"

        "PowerShellなら $env:OPENAI_API_KEY='sk-...' を実行するか、"
        ".env ファイルに OPENAI_API_KEY=... を書いてください。"
    )

client_oai = OpenAI(api_key=api_key)


# ══════════════════════════════════════════════════════════════════
# 1. JSONL 読み込み
# ══════════════════════════════════════════════════════════════════
def load_records() -> List[Dict[str, Any]]:
    if not JSONL_PATH.exists():
        raise FileNotFoundError(f"JSONLファイルが見つかりません: {JSONL_PATH}")

    records = []
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARNING] {line_num}行目のJSON読み込みに失敗: {e}")

    if not records:
        raise ValueError("有効なレコードが1件も見つかりませんでした。")

    return records


# ══════════════════════════════════════════════════════════════════
# 2. 埋め込み
# ══════════════════════════════════════════════════════════════════
def embed_texts(texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
    vectors = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        response = client_oai.embeddings.create(
            model=EMBED_MODEL,
            input=batch,
        )

        batch_vectors = [item.embedding for item in response.data]
        vectors.extend(batch_vectors)

        print(f"  embedded {min(i + batch_size, len(texts))}/{len(texts)}")

    arr = np.array(vectors, dtype="float32")
    faiss.normalize_L2(arr)  # cosine類似度相当で使うため正規化
    return arr


def embed_query(query: str) -> np.ndarray:
    response = client_oai.embeddings.create(
        model=EMBED_MODEL,
        input=[query],
    )
    qvec = np.array([response.data[0].embedding], dtype="float32")
    faiss.normalize_L2(qvec)
    return qvec


# ══════════════════════════════════════════════════════════════════
# 3. インデックス構築 / 読み込み
# ══════════════════════════════════════════════════════════════════
def build_or_load_index(force: bool = False):
    if INDEX_PATH.exists() and META_PATH.exists() and not force:
        print("既存のFAISSインデックスを読み込みます...")
        index = faiss.read_index(str(INDEX_PATH))
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        print(f"✓ index loaded: {index.ntotal} docs")
        return index, meta

    print("JSONLを読み込み中...")
    records = load_records()
    print(f"Loading {len(records)} records from JSONL ...")

    texts = []
    meta = []

    for r in records:
        chunk_text = r.get("chunk_text", "") or ""
        texts.append(chunk_text)

        meta.append({
            "id": r.get("id", ""),
            "title": r.get("title", ""),
            "authors": r.get("authors", []),
            "year": r.get("year", ""),
            "venue": r.get("venue", ""),
            "category": r.get("category", ""),
            "keywords": r.get("keywords", []),
            "doi": r.get("doi", ""),
            "arxiv_id": r.get("arxiv_id", ""),
            "summary": r.get("summary", ""),
            "chunk_text": chunk_text,
        })

    print("埋め込みを作成中...")
    vectors = embed_texts(texts)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine相当
    index.add(vectors)

    faiss.write_index(index, str(INDEX_PATH))
    META_PATH.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"✓ Ingest complete. Total: {index.ntotal} docs\n")
    return index, meta


# ══════════════════════════════════════════════════════════════════
# 4. 検索
# ══════════════════════════════════════════════════════════════════
def search(
    index,
    meta: List[Dict[str, Any]],
    query: str,
    top_k: int = TOP_K,
    category: Optional[str] = None,
) -> List[Dict[str, Any]]:
    qvec = embed_query(query)
    scores, indices = index.search(qvec, min(top_k * 3, len(meta)))

    hits = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(meta):
            continue

        item = meta[idx]

        if category and item.get("category") != category:
            continue

        hits.append({
            "score": round(float(score), 4),
            "title": item.get("title", ""),
            "year": item.get("year", ""),
            "category": item.get("category", ""),
            "summary": item.get("summary", ""),
            "doi": item.get("doi", ""),
            "chunk": item.get("chunk_text", ""),
        })

        if len(hits) >= top_k:
            break

    return hits


# ══════════════════════════════════════════════════════════════════
# 5. RAG質問応答
# ══════════════════════════════════════════════════════════════════
def ask(
    index,
    meta: List[Dict[str, Any]],
    question: str,
    top_k: int = TOP_K,
    category: Optional[str] = None,
) -> str:
    hits = search(index, meta, question, top_k=top_k, category=category)

    if not hits:
        return "関連する論文が見つかりませんでした。"

    context_parts = []
    for i, h in enumerate(hits, 1):
        context_parts.append(
            f"[{i}] {h['title']} ({h['year']})\n"
            f"カテゴリ: {h['category']}\n"
            f"内容: {h['summary']}\n"
            f"DOI: {h['doi']}"
        )
    context = "\n\n".join(context_parts)

    system_prompt = (
        "あなたは自動運転・コンピュータビジョン分野の研究アシスタントです。"
        "与えられた参考論文だけを根拠に、日本語で簡潔に答えてください。"
        "回答には根拠として使った論文番号を [1] のように明記してください。"
        "参考論文にない情報は推測で補わず、『資料にありません』と述べてください。"
    )

    user_prompt = f"【参考論文】\n{context}\n\n【質問】\n{question}"

    response = client_oai.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    return response.choices[0].message.content


# ══════════════════════════════════════════════════════════════════
# 6. 表示
# ══════════════════════════════════════════════════════════════════
def print_hits(hits: List[Dict[str, Any]]) -> None:
    if not hits:
        print("  検索結果なし\n")
        return

    for i, h in enumerate(hits, 1):
        print(f"  [{i}] (score={h['score']}) {h['title']} ({h['year']})")
        print(f"       {h['category']}")
        print(f"       {h['summary'][:100]}...")
    print()


# ══════════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    index, meta = build_or_load_index(force=False)

    print("=" * 60)
    print("【検索例】リスク評価 × LLM")
    hits = search(index, meta, "LLMを使ったリスク評価・危険予測")
    print_hits(hits)

    print("=" * 60)
    print("【検索例】レーン検出（カテゴリ絞り込み）")
    hits = search(index, meta, "lane detection BEV", category="06_レーン検出")
    print_hits(hits)

    print("=" * 60)
    question = "GPT-4Vを自動運転に使った研究にはどんなものがあるか？"
    print(f"【質問】{question}")
    answer = ask(index, meta, question)
    print(f"\n【回答】\n{answer}\n")

    print("=" * 60)
    print("対話モード（終了: 'q'）")
    while True:
        q = input("\n質問> ").strip()
        if q.lower() in ("q", "quit", "exit", ""):
            break
        ans = ask(index, meta, q)
        print(f"\n{ans}\n")