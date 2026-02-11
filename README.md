# Minimal RAG Practice

HuggingFace Datasets + SentenceTransformers + Qdrant で最小RAGを試す構成です。  
投入（`ingest.py`）と問い合わせ（`query.py`）を分離しています。

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install datasets sentence-transformers qdrant-client openai python-dotenv
```

必要なら `.env.example` を `.env` にコピーして、`OPENAI_API_KEY` を設定します。
生成モデルは `OPENAI_MODEL=gpt-5-mini`（最安クラス）をデフォルトにしています。

## Run

1) 初回投入（または再構築）

```bash
source .venv/bin/activate
python ingest.py --dataset ag_news --split "train[:200]" --text-column text
```

コレクションを作り直す場合:

```bash
python ingest.py --reindex
```

2) 問い合わせ

```bash
python query.py --query "この記事は何について書かれていますか？"
```

3) まとめ実行（互換ラッパー）

```bash
python main.py --query "What is this article about?"
```

`main.py` は既存コレクションを再利用し、`--reindex` のときだけ再投入します。

```bash
python main.py --query "この記事は何について書かれていますか？" --reindex
```

## Notes

- デフォルトデータセットは `ag_news`（`train[:200]`）です。
- `OPENAI_API_KEY` が未設定でも検索結果までは動作します。
- 生成モデルは `.env` の `OPENAI_MODEL` か `--llm-model` で変更できます。
- ベクトルDBは無料の `Qdrant` ローカルモードで、`./qdrant_data` に永続化されます。
- 既存コレクションがある場合、`ingest.py` はデフォルトで再投入をスキップします（`--reindex` で再構築）。
- 別データセットを使う場合は `--dataset`, `--split`, `--text-column` を指定してください。
