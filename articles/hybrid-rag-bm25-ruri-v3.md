---
title: "介護現場向けRAGにBM25 × ruri-v3ハイブリッド検索を実装する — RRF統合・asyncio並列・EmbeddingWorkerPool設計"
emoji: "🔍"
type: "tech"
topics: ["python", "rag", "qdrant", "nlp", "fastapi"]
published: true
---

# 介護現場向けRAGにBM25 × ruri-v3ハイブリッド検索を実装する — RRF統合・asyncio並列・EmbeddingWorkerPool設計

:::message
**この記事で分かること**
- `cl-nagoya/ruri-v3-310m` の選定理由と「クエリ:」/「文章:」プレフィックスルールの詳細
- BM25 + MeCab による日本語特化スパース検索の実装
- RRF（Reciprocal Rank Fusion）によるDense・Sparseスコア統合の実装
- `asyncio.gather()` によるDense・Sparse並列実行とBM25を同期実行のままにした判断根拠
- `EmbeddingWorkerPool`（ThreadPoolExecutor + asyncioキュー）の設計
- チャンキング設定（CHUNK_SIZE=1024 / OVERLAP=128）のパラメータ調整
- 「褥瘡」「清拭」といった介護専門用語がDenseのみでは検索できなかった問題とハイブリッドによる解決
:::

## はじめに

音声AIインカムの質問応答バックエンドにRAGを組み込んだとき、最初の実装（Dense検索のみ）で「褥瘡」が検索できないという問題が発生した。

`ruri-v3` は日本語テキスト埋め込みとして高い精度を持つが、`褥瘡` と `床ずれ` のような表記揺れのある専門用語では埋め込みベクトル空間の距離が想定より離れることがある。介護現場のマニュアルには `褥瘡` という正式名称で記載されているが、スタッフは `ジョクソウ` や `床ずれ` と発音することがある。

BM25によるキーワード一致を組み合わせることで、「褥瘡という単語が含まれるチャンク」を強制的に引き上げられる。本記事はDense（ruri-v3）+ Sparse（BM25）ハイブリッド検索の設計と実装を記録する。

:::message
**背景・ストーリーはnoteに**
「褥瘡が検索できなかった」経緯、介護マニュアルに含まれる専門用語の難しさ、現場でのフィードバックをストーリー形式で書いています。
→ https://note.com/yamashita_aidev/n/n0183117bcb4c
:::

## 対象読者・前提環境

**対象読者**
- PythonでRAGパイプラインを実装しているエンジニア
- 日本語専門用語の検索精度に悩んでいるエンジニア
- Dense検索とSparse検索のハイブリッド手法を調べているエンジニア

**前提環境**

| 項目 | バージョン / 値 |
|------|---------------|
| Python | 3.11 |
| FastAPI | 0.111.x |
| sentence-transformers | 3.x |
| rank-bm25 | 0.2.x |
| MeCab | 0.996 + ipadic-NEologd |
| Qdrant | クライアント 1.9.x |
| ruri-v3 モデル | cl-nagoya/ruri-v3-310m |

**システム構成の前提**

```
// 概念コード — 本番実装とは異なります

Android（音声入力）
       ↓
FastAPI サーバー（クエリ受信）
       ↓
HybridSearchService
  ├── DenseSearcher（ruri-v3 + Qdrant）
  │       ↓ asyncio.gather() で並列実行
  └── SparseSearcher（BM25 + MeCab）
               ↓
          rrf_fusion()（RRF スコア統合）
               ↓
          上位 TOP_K チャンクを LLM プロンプトへ
               ↓
          Anthropic Claude API（回答生成）
               ↓
Android（音声で回答再生）
```

---

## ハイブリッド検索の全体アーキテクチャ

### なぜDense検索だけでは足りないか

Dense検索（`ruri-v3`）は意味的類似度を捉えるのが得意だ。「痛い」と「疼痛」のような同義語、「褥瘡の予防方法」と「床ずれができないようにするには」のようなパラフレーズは高いスコアで引き当てられる。

一方、苦手な領域がある。

```
// Dense 検索が苦手なケース（介護現場の具体例）

クエリ: 「褥瘡」
  → ベクトル空間で「床ずれ」には近い
  → しかし「褥瘡ケア」「褥瘡予防」「褥瘡処置」などが
    どの程度近いかはモデル次第

クエリ: 「清拭」
  → 「体を拭く」「身体を清める」には意味的に近い
  → 「清拭タオルの温度」「清拭の手順」といった手続き的な文書が
    必ずしも上位に来るわけではない

クエリ: 「経管栄養」
  → 「栄養チューブ」「胃ろう」との関係はモデルが学習していれば拾える
  → 「経管栄養 300ml 速度調整」のような具体的な数値は
    キーワード一致の方が確実
```

BM25はこれらの弱点を補完する。クエリに含まれる単語と完全一致するチャンクを強制的に引き上げられる。

### Sparse（BM25）だけでも足りない

BM25単独の問題は、意味的な類似を捉えられないことだ。

```
// BM25 が失敗するケース

クエリ: 「床ずれ」
  → 「褥瘡」という単語を含むチャンクにスコアが付かない
  → 「褥瘡の予防ケア」という文書は0スコアになる

クエリ: 「ご飯を食べられない利用者への対応」
  → 「嚥下困難」「経口摂取困難」を含む文書にスコアが付かない
```

Dense + Sparse の両者を組み合わせることで、それぞれの弱点を補い合う。

---

## ruri-v3 の選定と設定

### モデル選定の比較

| モデル | 精度（日本語） | コスト | 判断 |
|--------|-------------|-------|------|
| text-embedding-3-small（OpenAI） | 高精度 | API従量課金（月次コスト不透明） | 不採用 |
| multilingual-e5-large | 多言語対応・汎用的 | 無料・ローカル | 日本語特化より精度低い |
| cl-nagoya/ruri-v3-310m | 日本語特化・最高水準 | 無料・ローカル | **採用** |

選定の判断には FP16 二宮氏の「[日本語RAGのEmbeddingモデル、結局どれが最強なのか？6構成で2000問ベンチマークした](https://zenn.dev/fp16/articles/aa48dcae23974e)」を参考にした。このベンチマークで ruri-v3 は日本語特化モデルの中でトップクラスの精度を示している。

また、介護という特定ドメインでAPIコストを継続的に払い続けることへの不安があった。オンプレミス運用でコストが固定できるローカルモデルを優先した。

### プレフィックスルール（最重要設定）

ruri-v3 を使用する際には**プレフィックスの付け方が精度に直結する**。これを知らずに実装すると精度が低下する。

```python
# 概念コード — 本番実装とは異なります
# embedding_service.py

from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingService:
    """ruri-v3 による日本語テキスト埋め込みサービス。

    プレフィックスルールに従い、クエリとドキュメントで
    異なるプレフィックスを付与する。
    """

    # ruri-v3 が期待するプレフィックス定数
    QUERY_PREFIX = "クエリ: "
    DOC_PREFIX = "文章: "

    def __init__(self, model_name: str = "cl-nagoya/ruri-v3-310m") -> None:
        self._model = SentenceTransformer(model_name)

    def encode_query(self, query_text: str) -> np.ndarray:
        """検索クエリを埋め込みベクトルに変換する。

        Args:
            query_text: ユーザーの質問文（例: 「褥瘡の予防方法を教えて」）

        Returns:
            埋め込みベクトル（numpy 配列）
        """
        # クエリには「クエリ: 」プレフィックスを付ける
        prefixed = self.QUERY_PREFIX + query_text
        return self._model.encode(prefixed, normalize_embeddings=True)

    def encode_document(self, chunk_text: str) -> np.ndarray:
        """ドキュメントチャンクを埋め込みベクトルに変換する。

        Args:
            chunk_text: チャンク分割されたドキュメントテキスト

        Returns:
            埋め込みベクトル（numpy 配列）
        """
        # ドキュメントには「文章: 」プレフィックスを付ける
        prefixed = self.DOC_PREFIX + chunk_text
        return self._model.encode(prefixed, normalize_embeddings=True)
```

**なぜプレフィックスが重要か**

ruri-v3 は事前学習時に `クエリ:` と `文章:` のプレフィックスを区別してトレーニングされている。クエリとドキュメントで異なるプレフィックスを使うことで、「ユーザーが何を聞きたいか」と「ドキュメントが何について書かれているか」の対応関係を正確に捉えられる。

プレフィックスなしで実装すると、クエリとドキュメントが同一の埋め込み空間に射影されず、精度が低下する。この設定を見落としてコードを書き、精度が低いまま数時間チューニング設定（チャンクサイズ、TOP_K、フィルタ閾値）を調整し続けた経験がある。

---

## BM25 + MeCab の実装

### MeCab による日本語形態素解析

BM25は単語の一致頻度でスコアを計算するため、日本語では形態素解析が前提になる。英語と違い日本語には単語間のスペースがないため、MeCabで単語分割してからBM25インデックスを構築する。

```python
# 概念コード — 本番実装とは異なります
# sparse_searcher.py

import MeCab
from rank_bm25 import BM25Okapi
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class MeCabTokenizer:
    """MeCab を使った日本語形態素解析トークナイザー。

    名詞・動詞・形容詞のみを抽出し、BM25 インデックスの精度を高める。
    """

    # BM25 インデックスに含める品詞
    INCLUDE_POS = {"名詞", "動詞", "形容詞"}

    def __init__(self, mecab_args: str = "-Owakati") -> None:
        # ipadic-NEologd を使用する場合は引数で辞書を指定する
        # 例: MeCab.Tagger("-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
        self._tagger = MeCab.Tagger(mecab_args)

    def tokenize(self, text: str) -> list[str]:
        """テキストを形態素解析してトークンリストを返す。

        Args:
            text: 日本語テキスト

        Returns:
            形態素のリスト（品詞フィルタ済み）
        """
        node = self._tagger.parseToNode(text)
        tokens: list[str] = []

        while node:
            # 品詞情報は feature の最初のフィールド
            pos = node.feature.split(",")[0]
            if pos in self.INCLUDE_POS and node.surface:
                tokens.append(node.surface)
            node = node.next

        return tokens


class BM25SparseSearcher:
    """BM25Okapi を使ったスパース検索エンジン。

    介護マニュアルチャンクの BM25 インデックスを保持し、
    キーワード一致スコアで候補を返す。
    """

    def __init__(self, tokenizer: Optional[MeCabTokenizer] = None) -> None:
        self._tokenizer = tokenizer or MeCabTokenizer()
        self._bm25: Optional[BM25Okapi] = None
        # インデックス構築時のチャンク順序を保持する
        self._indexed_chunks: list[str] = []

    def build_index(self, chunks: list[str]) -> None:
        """チャンクリストから BM25 インデックスを構築する。

        Args:
            chunks: ドキュメントチャンクのリスト（chunk_id 順）
        """
        tokenized_corpus = [self._tokenizer.tokenize(chunk) for chunk in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._indexed_chunks = chunks
        logger.info("BM25 インデックス構築完了: %d チャンク", len(chunks))

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """クエリに対して BM25 スコアで上位チャンクを返す。

        Args:
            query: 検索クエリ文字列
            top_k: 返却する候補数（RRF 統合前の候補プール）

        Returns:
            (chunk_index, score) のリスト（スコア降順）

        Raises:
            RuntimeError: build_index() が呼ばれる前に search() を呼んだ場合
        """
        if self._bm25 is None:
            raise RuntimeError("build_index() を先に呼んでください")

        query_tokens = self._tokenizer.tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        # スコア降順でインデックスを並べ替える
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        return [(idx, float(scores[idx])) for idx in ranked_indices[:top_k]]
```

### BM25の役割と限界

```python
# BM25 の役割: Dense 検索が苦手な専門用語の完全一致を補完する
# 例: 「褥瘡」「清拭」「経管栄養」「バイタルサイン」「移乗介助」

# Dense 検索（ruri-v3）の問題:
# 「床ずれ」と「褥瘡」は意味的に近いが、ベクトル空間での距離が
# モデルの学習状況によっては想定より離れることがある

# BM25（キーワード一致）の補完:
# 「褥瘡」というキーワードを含む文書のランクを確実に引き上げる
# MeCab で形態素解析してから BM25 インデックスを構築するため、
# 「褥瘡の予防」「褥瘡ケア」なども「褥瘡」トークンで一致する
```

---

## RRF スコア統合の実装

Dense と Sparse の2つのスコアを単純加算すると、スコールの正規化問題が生じる。Dense の類似度スコアは -1〜1（コサイン類似度）だが、BM25 のスコアはコーパスサイズや語彙によって大きく変動する。スケールの異なる2つを足すと BM25 のスケールが支配的になる。

RRF（Reciprocal Rank Fusion）は順位情報だけを使うためスコールのスケール差を無視できる。

```python
# 概念コード — 本番実装とは異なります
# rrf_fusion.py

from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScoredChunk:
    """スコア付きチャンク。"""
    chunk_id: str
    text: str
    score: float


def rrf_fusion(
    dense_results: list[ScoredChunk],
    sparse_results: list[ScoredChunk],
    k: int = 60,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> list[ScoredChunk]:
    """RRF でDense・Sparseスコアを統合する。

    Reciprocal Rank Fusion は順位情報だけを使うため、
    スコールのスケール差（Dense: -1〜1, BM25: 0〜任意）を無視できる。

    Args:
        dense_results: Dense 検索結果（スコア降順）
        sparse_results: Sparse 検索結果（スコア降順）
        k: RRF の平滑化パラメータ（デフォルト 60 が一般的）
        dense_weight: Dense スコアの重み（合計が 1.0 になるよう設定）
        sparse_weight: Sparse スコアの重み

    Returns:
        RRF スコアで統合・ソートされた ScoredChunk リスト
    """
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, ScoredChunk] = {}

    # Dense 結果の RRF スコアを加算する
    for rank, chunk in enumerate(dense_results):
        chunk_id = chunk.chunk_id
        rrf_score = dense_weight * (1.0 / (k + rank + 1))
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + rrf_score
        chunk_map[chunk_id] = chunk

    # Sparse 結果の RRF スコアを加算する
    for rank, chunk in enumerate(sparse_results):
        chunk_id = chunk.chunk_id
        rrf_score = sparse_weight * (1.0 / (k + rank + 1))
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + rrf_score
        # chunk_map に登録されていなければ追加する（Sparse のみに存在するケース）
        if chunk_id not in chunk_map:
            chunk_map[chunk_id] = chunk

    # RRF スコアでソートして返す
    sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

    results = [
        ScoredChunk(
            chunk_id=cid,
            text=chunk_map[cid].text,
            score=rrf_scores[cid],
        )
        for cid in sorted_ids
    ]

    logger.debug(
        "RRF 統合完了: dense=%d件 sparse=%d件 → 統合後=%d件",
        len(dense_results),
        len(sparse_results),
        len(results),
    )

    return results
```

### 重みパラメータの決め方

`dense_weight=0.6 / sparse_weight=0.4` はデフォルト値だ。`k=60` はRRFの原著論文で提案された値で、多くのケースでうまく機能する。

介護マニュアルで調整した結果、Denseをやや重めにした方が「意味的な類似」を生かした回答になった。専門用語の完全一致が必要な場面ではSparseの重みを上げると良いが、現状のバランスで実用的な精度が出ている。

---

## asyncio.gather() による並列実行

### Dense と Sparse を並列実行する

Dense（Qdrant ベクトル検索）と Sparse（BM25）は独立した処理なので並列実行できる。

```python
# 概念コード — 本番実装とは異なります
# hybrid_search_service.py

import asyncio
import time
import logging
from typing import Any

logger = logging.getLogger(__name__)


class HybridSearchService:
    """Dense + Sparse ハイブリッド検索サービス。

    asyncio.gather() で Dense・Sparse を並列実行し、
    RRF でスコアを統合する。
    """

    def __init__(
        self,
        embedding_pool: "EmbeddingWorkerPool",
        qdrant_client: Any,
        sparse_searcher: "BM25SparseSearcher",
        collection_name: str,
        top_k: int = 5,
        rrf_k: int = 60,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        score_threshold: float = 0.3,
    ) -> None:
        self._embedding_pool = embedding_pool
        self._qdrant = qdrant_client
        self._sparse = sparse_searcher
        self._collection_name = collection_name
        self._top_k = top_k
        self._rrf_k = rrf_k
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight
        self._score_threshold = score_threshold

    async def search(
        self, query: str, session_id: str
    ) -> tuple[list[ScoredChunk], dict[str, float]]:
        """Dense + Sparse を並列実行して RRF で統合する。

        Args:
            query: 検索クエリ文字列
            session_id: ログ追跡用のセッション ID

        Returns:
            (スコア閾値フィルタ済みの ScoredChunk リスト, タイミング情報)
        """
        t_start = time.perf_counter()

        # クエリを埋め込みベクトルに変換する（「クエリ: 」プレフィックス付き）
        query_embedding = await self._embedding_pool.embed([
            EmbeddingService.QUERY_PREFIX + query
        ])
        t_embed = time.perf_counter()

        # Dense と Sparse を asyncio.gather() で並列実行する
        dense_results, sparse_results = await asyncio.gather(
            self._dense_search(query_embedding[0]),
            self._sparse_search(query),
        )
        t_search = time.perf_counter()

        # RRF でスコアを統合する
        fused = rrf_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            k=self._rrf_k,
            dense_weight=self._dense_weight,
            sparse_weight=self._sparse_weight,
        )

        # スコア閾値でフィルタリングする
        final_results = [r for r in fused if r.score >= self._score_threshold]

        t_end = time.perf_counter()

        timings = {
            "embed_ms": (t_embed - t_start) * 1000,
            "search_ms": (t_search - t_embed) * 1000,
            "total_ms": (t_end - t_start) * 1000,
        }

        logger.info(
            "ハイブリッド検索完了 session=%s: dense=%d件 sparse=%d件 → fused=%d件 → final=%d件 "
            "embed=%.1fms search=%.1fms total=%.1fms",
            session_id,
            len(dense_results),
            len(sparse_results),
            len(fused),
            len(final_results),
            timings["embed_ms"],
            timings["search_ms"],
            timings["total_ms"],
        )

        return final_results, timings

    async def _dense_search(self, query_embedding: list[float]) -> list[ScoredChunk]:
        """Qdrant を使ったベクトル検索（Dense）。

        Args:
            query_embedding: ruri-v3 で変換したクエリベクトル

        Returns:
            ScoredChunk リスト（スコア降順）
        """
        hits = await self._qdrant.search(
            collection_name=self._collection_name,
            query_vector=query_embedding.tolist(),
            limit=self._top_k * 4,  # RRF 統合前は多めに取得する
        )
        return [
            ScoredChunk(
                chunk_id=hit.id,
                text=hit.payload.get("text", ""),
                score=hit.score,
            )
            for hit in hits
        ]

    async def _sparse_search(self, query: str) -> list[ScoredChunk]:
        """BM25 を使ったキーワード検索（Sparse）。

        BM25 は同期 CPU 処理（約 30ms）のため run_in_executor は使わない。
        オーバーヘッドがコストに見合わないと判断し、MVP では同期実行のまま。

        Args:
            query: 検索クエリ文字列

        Returns:
            ScoredChunk リスト（スコア降順）
        """
        # BM25 は同期処理だが await は不要（asyncio イベントループをブロックしない程度の処理）
        ranked = self._sparse.search(query, top_k=self._top_k * 4)
        return [
            ScoredChunk(
                chunk_id=str(idx),
                text=self._sparse._indexed_chunks[idx],
                score=score,
            )
            for idx, score in ranked
            if score > 0.0  # スコアゼロの候補は除外する
        ]
```

### BM25 を同期実行のままにした理由

BM25は純粋なCPU処理で、介護マニュアルの規模（数百チャンク）では **約30ms** で完了する。`asyncio.run_in_executor()` でスレッドプールに投げると、スレッド切り替えのオーバーヘッドが発生する（数ms〜数十ms）。

30msの処理を別スレッドに投げることで得られる並列化メリットより、オーバーヘッドの方が大きいと判断した。MVPの規模（数百チャンク）では同期実行のまま許容範囲内に収まる。チャンク数が数万単位に増えた場合は再評価が必要だ。

---

## EmbeddingWorkerPool の設計

### SentenceTransformer の非同期実行問題

SentenceTransformerはpickle化できないため、`ProcessPoolExecutor`（マルチプロセス）では使えない。`ThreadPoolExecutor` を使う必要がある。

```python
# 概念コード — 本番実装とは異なります
# embedding_worker_pool.py

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingWorkerPool:
    """asyncio キューベースの埋め込み推論プール。

    SentenceTransformer はピクルス化できないため ProcessPoolExecutor は使用不可。
    ThreadPoolExecutor を使用する。

    Attributes:
        _model: SentenceTransformer モデルインスタンス
        _executor: スレッドプール executor
    """

    def __init__(
        self,
        model_name: str = "cl-nagoya/ruri-v3-310m",
        num_workers: int = 2,
        max_queue_size: int = 100,
    ) -> None:
        """EmbeddingWorkerPool を初期化する。

        Args:
            model_name: SentenceTransformer モデル名（HuggingFace Hub ID）
            num_workers: スレッドプールのワーカー数
            max_queue_size: asyncio キューの最大サイズ
        """
        logger.info("EmbeddingWorkerPool 初期化中: model=%s workers=%d", model_name, num_workers)
        self._model = SentenceTransformer(model_name)
        self._executor = ThreadPoolExecutor(max_workers=num_workers)
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        logger.info("EmbeddingWorkerPool 初期化完了")

    async def embed(self, texts: list[str]) -> np.ndarray:
        """テキストリストを埋め込みベクトルに変換する（非同期）。

        Args:
            texts: 埋め込みを計算するテキストのリスト

        Returns:
            numpy 配列（shape: [len(texts), embedding_dim]）
        """
        loop = asyncio.get_event_loop()
        # ThreadPoolExecutor で SentenceTransformer の推論を実行する
        # これにより asyncio イベントループをブロックせず推論できる
        result: np.ndarray = await loop.run_in_executor(
            self._executor,
            lambda: self._model.encode(
                texts,
                normalize_embeddings=True,  # コサイン類似度のために正規化する
                show_progress_bar=False,
            ),
        )
        return result

    def shutdown(self) -> None:
        """スレッドプールをシャットダウンする。

        FastAPI の lifespan イベントで呼ぶこと。
        """
        self._executor.shutdown(wait=True)
        logger.info("EmbeddingWorkerPool シャットダウン完了")
```

### FastAPI lifespan での初期化

```python
# 概念コード — 本番実装とは異なります
# main.py

from contextlib import asynccontextmanager
from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI の起動・終了時のリソース管理。"""
    # --- 起動時 ---
    app.state.embedding_pool = EmbeddingWorkerPool(
        model_name="cl-nagoya/ruri-v3-310m",
        num_workers=2,
    )
    yield
    # --- 終了時 ---
    app.state.embedding_pool.shutdown()


app = FastAPI(lifespan=lifespan)
```

---

## チャンキング設定とパラメータ調整

### 最終的なパラメータ設定

```python
# 概念コード — 本番実装とは異なります
# chunking_config.py

# チャンキング設定
CHUNK_SIZE = 1024       # 1チャンクの最大文字数
CHUNK_OVERLAP = 128     # チャンク間のオーバーラップ（文字数）

# 検索設定
TOP_K = 5               # LLM プロンプトに渡す上位チャンク数
BM25_WEIGHT = 0.4       # RRF の Sparse 重み（Dense は 0.6）
SCORE_THRESHOLD = 0.3   # 最終フィルタの RRF スコア閾値
```

### パラメータ調整の経緯

**CHUNK_SIZE の選定**

当初 512 文字でチャンク分割していた。介護マニュアルは「褥瘡の予防」「清拭の手順」のように、1つの手順説明が5〜10ステップにわたる長い説明で構成されている。512文字では手順が途中で切れてしまい、LLMが不完全な情報で回答するケースが出た。

1024文字にするとステップが1チャンクに収まる確率が高くなり、回答品質が改善した。ただし1チャンクが長くなるため、LLMに渡す総トークン数が増加する点はトレードオフとして意識している。

**CHUNK_OVERLAP の役割**

```
// チャンク境界の問題

チャンク A（文字 0〜1024）: 「褥瘡の予防については、体位変換が重要です。
                              2時間ごとに...（中略）...また、皮膚の観察も」
チャンク B（文字 1024〜2048）: 「大切です。具体的には...」

→ チャンク B は「大切です。」から始まるため何が大切なのか文脈が欠落している
```

128文字のオーバーラップを設けることで、チャンク境界をまたぐ文脈を両方のチャンクが保持できる。

**SCORE_THRESHOLD の意味**

RRFスコアは Dense と Sparse の順位情報から計算した相対的なスコアだ。0.3という閾値は「両方のリストでそれぞれ上位20位以内に入っているか、どちらかで上位5位以内」に近い水準になる。この閾値により、どちらの検索でもほとんどスコアが付かなかった無関係なチャンクをフィルタできる。

---

## ハマりポイント・注意点

### 1. プレフィックスを忘れていた（数時間のデバッグ）

最初の実装でプレフィックスを付けずに ruri-v3 を使っていた。精度が低いと気づき、チャンクサイズ・TOP_K・SCORE_THRESHOLDを変えながら数時間調整したが改善しなかった。

最終的にモデルのドキュメントを読み直し、プレフィックスが必要だと判明した。`クエリ:` と `文章:` を追加した直後に精度が明確に改善した。**チューニングより先にモデルの仕様を読む**という当たり前の教訓を改めて学んだ。

### 2. BM25スコアとDenseスコアの単純加算は機能しない

最初のハイブリッド実装では2つのスコアを単純加算していた。

```python
# NG: スコアのスケールが全く異なるため加算は意味をなさない
final_score = dense_score + bm25_score

# Dense スコア（コサイン類似度）: 0.0〜1.0
# BM25 スコア: コーパスサイズに依存（0〜任意の正の実数）

# 結果: BM25 スコアが支配的になり Dense の意味的類似度が無視された
```

順位情報だけを使う RRF に切り替えることでスコールのスケール差の問題が解決した。

### 3. SentenceTransformer と ProcessPoolExecutor の非互換性

マルチコアを活用しようとして `ProcessPoolExecutor` でワーカーを立てたが、SentenceTransformerがpickle化できないためエラーになった。

```python
# NG: SentenceTransformer は pickle 化できない
with ProcessPoolExecutor(max_workers=2) as executor:
    future = executor.submit(model.encode, texts)  # PicklingError

# OK: ThreadPoolExecutor を使う
# GIL の制限があるが、推論は C++ レイヤーで実行されるため並列化効果がある
with ThreadPoolExecutor(max_workers=2) as executor:
    result = executor.submit(model.encode, texts)
```

### 4. Qdrant クライアントの非同期対応

Qdrant Python クライアントは同期版と非同期版がある。FastAPI の非同期コンテキストで同期版クライアントを使うと、ベクトル検索中に asyncio イベントループがブロックされる。

```python
# NG: 同期クライアントは FastAPI の非同期コンテキストでイベントループをブロックする
from qdrant_client import QdrantClient
client = QdrantClient(url="http://localhost:6333")

# OK: 非同期クライアントを使う
from qdrant_client import AsyncQdrantClient
client = AsyncQdrantClient(url="http://localhost:6333")
```

---

## パフォーマンス計測結果

### 検索レイテンシの実測値

| ステップ | 計測値 |
|---------|--------|
| ruri-v3 埋め込み（クエリ1件） | 80〜120ms |
| Qdrant ベクトル検索 | 5〜15ms |
| BM25 スパース検索（約300チャンク） | 25〜35ms |
| Dense + Sparse 並列実行合計 | 85〜130ms（埋め込み含む） |
| RRF 統合 | < 1ms |
| **検索全体（埋め込み〜フィルタ）** | **90〜150ms** |

埋め込み計算が全体の大半を占める。ruri-v3-310m はローカル推論でこのレイテンシになる。音声AIインカムでは「質問後2〜3秒以内に回答開始」が体感上の許容範囲で、RAG全体（検索 + LLM生成）がこの範囲に収まっている。

### Dense のみ vs ハイブリッドの精度比較

介護マニュアル（約300チャンク）を対象に、代表的な20クエリで手動評価した。

| 手法 | 上位3件に正解チャンクが含まれる割合 |
|------|--------------------------------|
| Dense のみ（ruri-v3） | 65% |
| Sparse のみ（BM25） | 55% |
| **ハイブリッド（RRF）** | **85%** |

「褥瘡」「清拭」「経管栄養」のような介護専門用語を含むクエリでは、ハイブリッドの改善幅が特に大きかった。

---

## まとめ

介護現場向けRAGにおけるDense + Sparseハイブリッド検索の実装から得た教訓を3点に絞る。

- **ruri-v3 のプレフィックスルールは省略できない。`クエリ:` と `文章:` の付け方を誤ると精度が低下し、他のパラメータ調整では回復しない**
- **BM25 と Dense のスコアを統合するには RRF を使う。単純加算はスコールのスケール差で機能しない**
- **SentenceTransformer は ThreadPoolExecutor で非同期実行する。ProcessPoolExecutor は pickle 非対応のため使えない**

Denseのみでは65%だった正解率がハイブリッドで85%に向上し、「褥瘡が検索できない」問題は解消した。BM25が介護専門用語の完全一致を補完し、ruri-v3が意味的な類似を担う役割分担が機能している。

### 今後の課題

- **動的インデックス更新**: マニュアルが更新されたとき、BM25インデックスとQdrantコレクションを差分更新する仕組みが未実装
- **クエリ拡張**: 「床ずれ」クエリに対して「褥瘡」を同義語として展開する前処理を追加すると、Denseのみでの検索精度も向上できる可能性がある
- **チャンクサイズの動的調整**: セクションの境界（見出しタグなど）を検出して、固定文字数ではなくセマンティックな単位でチャンク分割する

---

## 参考リンク

- [cl-nagoya/ruri-v3-310m（HuggingFace Hub）](https://huggingface.co/cl-nagoya/ruri-v3-310m)
- [日本語RAGのEmbeddingモデル、結局どれが最強なのか？6構成で2000問ベンチマークした（FP16 二宮氏）](https://zenn.dev/fp16/articles/aa48dcae23974e)
- [Qdrant 公式ドキュメント](https://qdrant.tech/documentation/)
- [rank-bm25 GitHub](https://github.com/dorianbrown/rank-bm25)
- [sentence-transformers 公式ドキュメント](https://www.sbert.net/)
- [Reciprocal Rank Fusion at a Thousand Queries（Cormack et al., 2009）](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)

---

> **背景・ストーリーはnoteに**
> 「褥瘡が検索できない」と気づいた現場でのエピソード、Dense検索だけでは足りないと判断した経緯、MeCabの辞書選定で迷った話をストーリー形式で書いています。
> → https://note.com/yamashita_aidev/n/n0183117bcb4c
