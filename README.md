# Minimal RAG Practice

HuggingFace Datasets + SentenceTransformers + Qdrant で最小RAGを試す構成です。

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

```bash
source .venv/bin/activate
python main.py --query "What is this article about?"
```

日本語で試す例:

```bash
python main.py --query "この記事は何について書かれていますか？"

# Qdrantの保存先とコレクション名を指定する例
python main.py --query "この記事は何について書かれていますか？" --qdrant-path ./qdrant_data --collection rag_practice
```

## Notes

- デフォルトデータセットは `ag_news`（`train[:200]`）です。
- `OPENAI_API_KEY` が未設定でも検索結果までは動作します。
- 生成モデルは `.env` の `OPENAI_MODEL` か `--llm-model` で変更できます。
- ベクトルDBは無料の `Qdrant` ローカルモードで、`./qdrant_data` に永続化されます。
- 別データセットを使う場合は `--dataset`, `--split`, `--text-column` を指定してください。
