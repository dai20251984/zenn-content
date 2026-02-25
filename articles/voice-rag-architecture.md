---
title: "Android × FastAPI × Claude APIで作る介護現場向け音声RAGの全体設計"
emoji: "🎙️"
type: "tech"
topics: ["android", "kotlin", "llm", "rag", "音声認識"]
published: true
---

## はじめに

介護施設での日常業務を支援するハンズフリー音声AIインカムを開発しました。Androidデバイス上の音声をリアルタイム処理し、介護用語の文脈に応じた応答を4〜6秒以内に実現するシステム全体設計を解説します。

## 対象読者・動作確認環境

**対象読者：**
- マルチモーダルLLM応用に興味のある開発者
- Android × バックエンド統合の実装経験を積みたい方
- レイテンシ最適化に取り組む方

**動作確認環境：**

| 項目 | 仕様 |
|------|------|
| **Android** | 13以上（Kotlin） |
| **バックエンド** | Python 3.11 + FastAPI 0.104+ |
| **LLM/Embedding API** | Claude 3.5 Sonnet / OpenAI Whisper / Cohere ruri-v3-310m |
| **Vector DB** | Qdrant (local/cloud) |
| **TTS** | Google Cloud Text-to-Speech |

## システム概要と設計課題

介護現場では、両手を使う作業が頻繁です。スマートフォンを操作できない環境で、音声のみで以下のような業務支援が必要です：

- **シーン1:** 患者の症状を説明→介護技法の提案を音声で回答
- **シーン2:** 薬剤名を聞く→用法・用量の確認
- **シーン3:** 緊急時の判断サポート

設計課題は3つ：

1. **低レイテンシ（4〜6秒）の実現**
   - STT: 1〜2秒
   - RAG + LLM: 2〜3秒
   - TTS: 1秒

2. **医療用語の正確な音声認識**
   - 「褥瘡（じょくそう）」など難読文字
   - 「移乗介助」などドメイン用語

3. **コスト効率**
   - 複数ユーザーの同時利用
   - API呼び出しの最適化

## 全体アーキテクチャ

```
┌─────────────────────────────────────────────────────────┐
│ Android (Kotlin)                                        │
│ ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│ │ AudioRecord  │→ │ Vosk VAD     │→ │ WebSocket    │  │
│ │ VOICE_RECOG  │  │ (Endpoint)   │  │ Client       │  │
│ └──────────────┘  └──────────────┘  └──────────────┘  │
│         ↓                                    ↓          │
│   Noise Suppressor          Authorization: Bearer ... │
│                                                         │
│ ┌──────────────────────────────────────────────────┐  │
│ │ Audio Playback (Speaker)                         │  │
│ │ NeuralTTS response (24kHz PCM)                   │  │
│ └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                            ↕ (WebSocket)
┌─────────────────────────────────────────────────────────┐
│ FastAPI Server (Python)                                │
│ ┌──────────────────────────────────────────────────┐  │
│ │ 1. Whisper STT (カスタムプロンプト)              │  │
│ │    └→ 介護用語プロンプト同梱                    │  │
│ └──────────────────────────────────────────────────┘  │
│                         ↓                              │
│ ┌──────────────────────────────────────────────────┐  │
│ │ 2. RAG Pipeline                                  │  │
│ │    ┌────────────────────────────────────────┐   │  │
│ │    │ Query Embedding (ruri-v3)              │   │  │
│ │    │ 「クエリ: 」プレフィックス必須        │   │  │
│ │    └────────────────────────────────────────┘   │  │
│ │               ↓                                   │  │
│ │    ┌────────────────────────────────────────┐   │  │
│ │    │ Hybrid Search                          │   │  │
│ │    │ Dense (Qdrant) + Sparse (BM25)         │   │  │
│ │    │ RRF Fusion                             │   │  │
│ │    └────────────────────────────────────────┘   │  │
│ │               ↓ (Top-K=5)                       │  │
│ └──────────────────────────────────────────────────┘  │
│                         ↓                              │
│ ┌──────────────────────────────────────────────────┐  │
│ │ 3. Claude API (Prompt Cache)                     │  │
│ │    System: [SYSTEM_PROMPT + RAG Context]         │  │
│ │           {cache_control: ephemeral}             │  │
│ │    User: [User Query]                            │  │
│ │    → キャッシュヒット時: 90%コスト削減         │  │
│ └──────────────────────────────────────────────────┘  │
│                         ↓                              │
│ ┌──────────────────────────────────────────────────┐  │
│ │ 4. Google TTS (Neural2-B)                        │  │
│ │    speaking_rate=1.1, 24kHz PCM                  │  │
│ └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                            ↕ (WebSocket)
                    Android Audio Playback
```

## 各コンポーネントの実装詳細

### 1. STT: Whisper API（カスタムプロンプト）

Whisperは強力ですが、介護用語の固有表現には対応していません。**カスタムプロンプト**で方言や専門用語を補助します。

```python
# whisper_client.py
from openai import OpenAI

WHISPER_PROMPT = (
    "介護施設での会話です。"
    "以下の用語が含まれる場合があります: "
    "移乗介助, 体位変換, 褥瘡, バイタルサイン, 経管栄養, 清拭, "
    "摘便, 吸引, おむつ交換, 入浴介助, 排泄介助, 誤嚥, 拘縮, 認知症"
)

class WhisperSTT:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    async def transcribe(self, audio_file) -> str:
        """音声ファイル（バイナリ）をテキストに変換"""
        response = self.client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="ja",
            prompt=WHISPER_PROMPT,  # 介護用語プロンプト
        )
        return response.text
```

:::message
**重要:** Whisperプロンプトは英語のみで、日本語は無視されます。英語で記述した用語の説明を加えても効果があります。例：`"caregiver tasks: patient transfer, bathing assistance, wound care (called 褥瘡, pressure ulcer)"`
:::

### 2. Embedding: ruri-v3-310m（プレフィックスルール）

Cohere `ruri-v3-310m`は**プレフィックス付きプロンプト**を要求します。これを忘れると、検索精度が大幅に低下します。

```python
# embedding_client.py
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str = "ruri-v3-310m"):
        self.model = SentenceTransformer(model_name)

    def encode_query(self, query_text: str):
        """
        クエリの埋め込み（「クエリ: 」プレフィックス必須）
        """
        prefixed_query = f"クエリ: {query_text}"
        embedding = self.model.encode(prefixed_query)
        return embedding

    def encode_document(self, doc_text: str):
        """
        ドキュメント・チャンクの埋め込み（「文章: 」プレフィックス必須）
        """
        prefixed_doc = f"文章: {doc_text}"
        embedding = self.model.encode(prefixed_doc)
        return embedding
```

:::message
**ハマりどころ:** プレフィックスなしで埋め込むと、クエリとドキュメントの意味空間が異なるため、BM25と似た低精度になります。これはプレフィックスの言語効果によるもので、Cohereの意図的な設計です。
:::

### 3. Vector DB: Qdrant + BM25 ハイブリッド検索

単一の検索手法では、セマンティック検索とキーワード検索の両方のメリットが得られません。**RRF（Reciprocal Rank Fusion）**で統合します。

```python
# rag_client.py
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import asyncio
from rank_bm25 import BM25Okapi

class HybridRAG:
    def __init__(self, qdrant_url: str, collection_name: str):
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.embedder = Embedder()
        self.bm25_index = None

    async def _dense_search(self, query_embedding, top_k=5):
        """Qdrant: ベクトル検索"""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
        )
        return results

    async def _sparse_search(self, query_text, top_k=5):
        """BM25: キーワード検索"""
        if self.bm25_index is None:
            raise ValueError("BM25インデックスが未初期化")

        query_tokens = query_text.split()
        scores = self.bm25_index.get_scores(query_tokens)
        # スコアの高い順に並べ替え
        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )
        return ranked[:top_k]

    def rrf_fusion(
        self,
        dense_results,
        sparse_results,
        k=60,
        dense_weight=0.6,
        sparse_weight=0.4
    ):
        """
        RRF: Dense + Sparse の重み付け統合
        - k: RRF計算用のハイパーパラメータ（推奨値 50-100）
        - dense_weight: ベクトル検索の重み
        - sparse_weight: BM25の重み
        """
        rrf_scores = {}

        # Dense検索結果を統合
        for rank, result in enumerate(dense_results):
            chunk_id = result.id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (
                dense_weight * (1.0 / (k + rank + 1))
            )

        # Sparse検索結果を統合
        for rank, (doc_idx, bm25_score) in enumerate(sparse_results):
            chunk_id = self.doc_id_map[doc_idx]  # ドキュメントインデックス→チャンクID変換
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (
                sparse_weight * (1.0 / (k + rank + 1))
            )

        # スコアの高い順に返却
        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    async def search(self, query_text: str, top_k=5):
        """ハイブリッド検索の実行"""
        # 並列実行で効率化
        query_embedding = self.embedder.encode_query(query_text)

        dense_results, sparse_results = await asyncio.gather(
            self._dense_search(query_embedding, top_k=top_k),
            self._sparse_search(query_text, top_k=top_k)
        )

        # RRF統合
        fused = self.rrf_fusion(dense_results, sparse_results)
        return fused[:top_k]
```

### 4. LLM: Claude API + Prompt Cache

Claude Prompt Cacheを使うと、システムプロンプト＋RAGコンテキストをキャッシュでき、コスト削減と若干のレイテンシ削減が可能です。

```python
# llm_client.py
import anthropic
from typing import Optional

SYSTEM_PROMPT = """あなたは介護現場での実務支援に特化したAIアシスタントです。
以下の原則に従って回答してください：

1. **正確性優先**: 不確実な情報は「確認が必要です」と明示
2. **簡潔性**: 50字以内が目安（音声だから読み上げやすく）
3. **安全性**: リスク要因は必ず触れる
4. **方言対応**: 地域用語に対応する姿勢を示す
"""

class ClaudeRAGClient:
    def __init__(self, api_key: str):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def query(
        self,
        user_query: str,
        rag_context: str,
        model: str = "claude-3-5-sonnet-20241022",
        use_cache: bool = True
    ) -> str:
        """
        RAGコンテキスト付きでClaudeにクエリを送信

        Args:
            user_query: ユーザー質問
            rag_context: RAG検索結果を連結したテキスト
            model: モデルID
            use_cache: Prompt Cacheを使用するか

        Returns:
            Claude の応答テキスト
        """

        # システムプロンプト + RAGコンテキストを組み合わせる
        system_content = f"{SYSTEM_PROMPT}\n\n## 参照情報\n{rag_context}"

        system_block = {
            "type": "text",
            "text": system_content,
        }

        # Prompt Cacheを有効化する場合、cache_controlを指定
        if use_cache:
            system_block["cache_control"] = {"type": "ephemeral"}

        response = await self.client.messages.create(
            model=model,
            max_tokens=1024,
            system=[system_block],
            messages=[
                {
                    "role": "user",
                    "content": user_query,
                }
            ],
        )

        return response.content[0].text
```

:::message
**Prompt Cache の要件:**
- キャッシュできるシステムプロンプト＋コンテキストの合計は**最小1,024トークン**必須
- 介護用語辞典など大規模なRAGコンテキスト（3,000～5,000トークン）があれば、キャッシュヒット時は入力トークン単価が**90%削減**されます
- ephemeral（5分TTL）で十分。複数ユーザーで使い回すなら persistent にして共有可能
:::

### 5. Android音声処理（AudioRecord / Vosk VAD）

Androidから高品質な音声を取得し、バックエンドに送信します。重要なポイント：

```kotlin
// AudioManager.kt
import android.media.AudioRecord
import android.media.MediaRecorder
import android.media.AudioFormat
import android.media.audiofx.NoiseSuppressor
import android.os.PowerManager

class AudioManager(context: Context) {
    private var audioRecord: AudioRecord? = null
    private var wakeLock: PowerManager.WakeLock? = null
    private val SAMPLE_RATE = 16_000  // Whisper標準
    private val BUFFER_SIZE = AudioRecord.getMinBufferSize(
        SAMPLE_RATE,
        AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_16BIT
    )

    fun startRecording() {
        // AudioSource.VOICE_RECOGNITION を使用（ノイズキャンセル効果あり）
        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,  // MICではなくこちら
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            BUFFER_SIZE
        )

        // ノイズサプレッサーを有効化
        if (NoiseSuppressor.isAvailable()) {
            audioRecord?.audioSessionId?.let { sessionId ->
                NoiseSuppressor.create(sessionId)?.apply {
                    enabled = true
                }
            }
        }

        // 8時間のWakeLock確保（バッテリー節約モードでも動作継続）
        val powerManager = context.getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = powerManager.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK,
            "AudioManager:RecordingWakeLock"
        ).apply {
            acquire(8 * 60 * 60 * 1000L)  // 8時間
        }

        audioRecord?.startRecording()
    }

    fun readAudioFrame(): ShortArray? {
        audioRecord?.let {
            val buffer = ShortArray(BUFFER_SIZE)
            val numRead = it.read(buffer, 0, BUFFER_SIZE)
            return if (numRead > 0) {
                buffer.sliceArray(0 until numRead)
            } else null
        }
        return null
    }

    fun stopRecording() {
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null

        wakeLock?.takeIf { it.isHeld }?.release()
        wakeLock = null
    }
}
```

:::message
**AudioSource.VOICE_RECOGNITION の重要性:**
AudioSource.MIC では、周囲のノイズを拾いやすく、特に介護施設の雑音環境では精度低下します。VOICE_RECOGNITION を使うと Android が内部的にノイズ抑制を行い、音声認識に最適化します。
:::

### 6. WebSocket通信・リトライ戦略

```kotlin
// WebSocketManager.kt
import okhttp3.*
import kotlinx.coroutines.*
import java.util.concurrent.TimeUnit

class WebSocketManager(
    private val serverUrl: String,
    private val apiKey: String,
) : WebSocketListener() {

    private var webSocket: WebSocket? = null
    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS)  // WebSocketは長接続
        .connectTimeout(10, TimeUnit.SECONDS)
        .build()

    private val sttRetryConfig = RetryConfig(maxRetries = 2, delayMs = 1000, timeoutMs = 5000)
    private val llmRetryConfig = RetryConfig(maxRetries = 1, delayMs = 2000, timeoutMs = 10000)

    fun connect() {
        val request = Request.Builder()
            .url(serverUrl)
            .addHeader("Authorization", "Bearer $apiKey")
            .build()

        webSocket = client.newWebSocket(request, this)
    }

    override fun onOpen(webSocket: WebSocket, response: Response) {
        // 30秒間隔でPing送信（接続保持）
        scheduleHeartbeat()
    }

    private fun scheduleHeartbeat() {
        GlobalScope.launch {
            while (isActive) {
                delay(30 * 1000)  // 30秒
                webSocket?.send("PING")
            }
        }
    }

    suspend fun sendAudioChunk(audioFrame: ShortArray) {
        val audioData = audioFrame.toByteArray()
        retryWithConfig(sttRetryConfig) {
            webSocket?.send(audioData)
        }
    }

    private suspend fun <T> retryWithConfig(
        config: RetryConfig,
        block: suspend () -> T
    ): T {
        var lastException: Exception? = null

        repeat(config.maxRetries) {
            try {
                return withTimeoutOrNull(config.timeoutMs.toLong()) {
                    block()
                } ?: throw TimeoutException("Operation timed out")
            } catch (e: Exception) {
                lastException = e
                delay(config.delayMs.toLong())
            }
        }

        throw lastException ?: Exception("All retries failed")
    }
}

data class RetryConfig(
    val maxRetries: Int,
    val delayMs: Int,
    val timeoutMs: Int
)
```

## レイテンシ設計（4〜6秒を実現した3つの工夫）

| フェーズ | 目標 | 実現方法 |
|---------|------|--------|
| **STT (Whisper)** | 1〜2秒 | ・ 16kHz PCM Mono で圧縮<br>・ WebSocket ストリーミング（最初のトークン到着まで待機）<br>・ バッチ処理より単発呼び出し |
| **RAG + LLM** | 2〜3秒 | ・ Qdrant + BM25 の並列検索 (asyncio.gather)<br>・ RRF 統合に <100ms<br>・ Claude Prompt Cache で再利用時 -50ms<br>・ TOP_K=5 に限定（トークン数抑制） |
| **TTS** | 1秒 | ・ Google Neural2-B with speaking_rate=1.1<br>・ 24kHz PCM で品質維持<br>・ 短い応答に限定（50字程度） |
| **合計** | **4〜6秒** | 全フェーズの並列化 + キャッシュ活用 |

:::message
**レイテンシ最適化のコツ:**
- Whisper はストリーミング対応（複数チャンクを送信して途中結果を取得）することで、最初の1秒で暫定テキストが得られます
- Claude API は最初のトークン生成までが遅く（プリロード時間）、キャッシュヒット時は短縮されます
- TTS は開始から再生開始まで 300ms 程度なので、LLM完了直後すぐに呼び出すのが重要
:::

## チャンキング・検索パラメータ

```python
# config.py
CHUNK_CONFIG = {
    "chunk_size": 1024,      # 1チャンクの最大文字数
    "chunk_overlap": 128,    # オーバーラップで文脈保持
    "top_k": 5,              # 上位5チャンクをLLMに渡す
}

RRF_CONFIG = {
    "k": 60,                 # RRF計算用ハイパーパラメータ
    "dense_weight": 0.6,     # Qdrant (ベクトル検索) の重み
    "sparse_weight": 0.4,    # BM25 (キーワード検索) の重み
}

TTS_CONFIG = {
    "voice_name": "ja-JP-Neural2-B",
    "speaking_rate": 1.1,    # 若干高速再生で自然感を損なわず時間短縮
    "sample_rate_hz": 24000,
    "encoding": "LINEAR16",
}
```

## ハマりどころ・注意点

### 1. ruri-v3 プレフィックスの必須性

**症状:** 検索精度がBM25と変わらない

**原因:** クエリとドキュメント埋め込み時に「クエリ: 」「文章: 」プレフィックスを付けていない

**解決法:**
```python
# NG
query_embedding = model.encode(query_text)
doc_embedding = model.encode(doc_text)

# OK
query_embedding = model.encode("クエリ: " + query_text)
doc_embedding = model.encode("文章: " + doc_text)
```

### 2. Claude Prompt Cache の 1024 トークン最小要件

**症状:** キャッシュが効かない（cache_creation_input_tokens ≠ 0）

**原因:** システムプロンプト + RAGコンテキストが1024トークン未満

**解決法:**
- 介護用語辞典やガイドラインを大規模に含める
- キャッシュ対象範囲を明確にして計測
- persistent キャッシュを使う（共有キャッシュ）

### 3. Kotlin ShortArray とバイト列変換

**症状:** WebSocketで音声送信時にバイナリが破損

**原因:** ShortArray（16ビット符号付き整数）のバイト変換ロジックが不正

**解決法:**
```kotlin
// NG: 単純な toByteArray() は CharArray 変換
val byteArray = shortArray.toByteArray()  // これはダメ

// OK: リトルエンディアン変換が必要
fun ShortArray.toByteArray(): ByteArray {
    val byteArray = ByteArray(this.size * 2)
    for (i in this.indices) {
        byteArray[i * 2] = (this[i].toInt() and 0xFF).toByte()
        byteArray[i * 2 + 1] = ((this[i].toInt() shr 8) and 0xFF).toByte()
    }
    return byteArray
}
```

### 4. AudioSource.VOICE_RECOGNITION vs. MIC

| 項目 | VOICE_RECOGNITION | MIC |
|------|-------------------|-----|
| ノイズ抑制 | ✓ (内蔵) | ✗ |
| 音声品質 | 優先 | 原音忠実 |
| Whisper 相性 | ◎ | △ |
| 消費電力 | 若干大 | 小 |

介護現場は雑音環境が多く、VOICE_RECOGNITION が推奨です。

### 5. WebSocket Ping/Pong の仕様

**問題:** 接続が勝手に切れる

**原因:** ファイアウォール・ロードバランサーの非アクティブタイムアウト（通常15〜30秒）

**解決法:**
```
クライアント → サーバー: Ping (30秒間隔)
サーバー → クライアント: Pong (自動返信)
```

OkHttp は Pong 自動返信をサポート。カスタム実装が必要な場合は、フレームレベルで処理。

## まとめ

本システムは、以下の技術統合により、介護現場での音声AIインカムを実現しました：

1. **マルチモーダル LLM の活用** - Claude Prompt Cache でコスト・レイテンシ改善
2. **ハイブリッド検索の効果** - Dense + Sparse 検索を RRF で統合、精度向上
3. **低レイテンシ設計** - 並列処理 + ストリーミング + キャッシュで 4〜6 秒を実現
4. **ドメイン最適化** - 介護用語に特化した Whisper プロンプト・ruri-v3 プレフィックス

介護現場の実務では、正確さ・速度・使いやすさのバランスが重要です。今後は、音声品質の更なる改善（ビームフォーミング等）や、複数ユーザーの同時利用対応を検討します。

## 参考リンク

- **元記事 (note):** https://note.com/yamashita_aidev/n/n7a3245355187
- **Cohere Embedding Models:** https://docs.cohere.com/
- **Claude Prompt Caching:** https://docs.anthropic.com/en/docs/build-a-resource/prompt-caching
- **Qdrant Vector DB:** https://qdrant.tech/
- **OpenAI Whisper API:** https://platform.openai.com/docs/guides/speech-to-text
- **Google Cloud TTS:** https://cloud.google.com/text-to-speech
