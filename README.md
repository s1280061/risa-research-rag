# RiSA Research RAG Pipeline

自動運転・コンピュータビジョン分野の研究論文を対象とした、FAISSベースのRAG（Retrieval-Augmented Generation）検索システムです。

## 概要

論文データ（JSONL形式）をベクトルDBに登録し、自然言語で質問すると関連論文を検索・要約して回答します。

- **ベクトルDB**: FAISS（ローカル）
- **埋め込みモデル**: OpenAI `text-embedding-3-small`
- **LLM**: OpenAI `gpt-4o-mini`

## ファイル構成

```
.
├── rag_pipeline.py       # メインのRAGパイプライン
├── requirements.txt      # 依存パッケージ
├── .env.example          # APIキー設定のテンプレート
└── .gitignore
```

## セットアップ

### 1. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 2. APIキーの設定

`.env.example` をコピーして `.env` を作成し、OpenAI APIキーを設定します。

```bash
cp .env.example .env
```

`.env` を編集：

```
OPENAI_API_KEY=sk-ここにAPIキーを入力
```

### 3. 論文データの準備

以下のパスにJSONL形式の論文データを配置してください：

```
C:\Users\User\Downloads\rag_documents.jsonl
```

各行のJSONには以下のフィールドが必要です：

| フィールド | 説明 |
|---|---|
| `id` | 論文ID |
| `title` | タイトル |
| `authors` | 著者リスト |
| `year` | 発行年 |
| `venue` | 掲載誌・会議名 |
| `category` | カテゴリ |
| `keywords` | キーワードリスト |
| `doi` | DOI |
| `arxiv_id` | arXiv ID |
| `summary` | 要約 |
| `chunk_text` | 埋め込み用テキスト |

## 使い方

```bash
python rag_pipeline.py
```

初回実行時はJSONLを読み込んでFAISSインデックスを構築します（少し時間がかかります）。2回目以降はインデックスを再利用するため高速です。

### 検索例

```python
# キーワード検索
hits = search(index, meta, "LLMを使ったリスク評価・危険予測")

# カテゴリ絞り込み検索
hits = search(index, meta, "lane detection BEV", category="06_レーン検出")

# RAG質問応答
answer = ask(index, meta, "GPT-4Vを自動運転に使った研究にはどんなものがあるか？")
```

### 対話モード

スクリプト末尾の対話モードで、ターミナルから自由に質問できます。

```
質問> 点群を使った3D物体検出の最新手法は？

[回答が表示されます]
```

終了するには `q` を入力してください。

## 注意事項

- `.env` ファイルは絶対にGitにコミットしないでください（APIキー漏洩防止）
- `.faiss` / `*_meta.json` ファイルはサイズが大きいため `.gitignore` で除外しています
- 論文データ（`.jsonl`）も機密の場合は除外してください
