---
title: "「次」のたびにLLMを呼ぶのをやめたら APIコストが90%減った — chunk_answer()の設計と実装"
emoji: "🎙️"
type: "tech"
topics: ["python", "fastapi", "kotlin", "音声認識", "介護テック"]
published: true
---

## はじめに

介護現場向け音声AIインカムを開発しています。スタッフがハンズフリーで業務マニュアルを確認できるシステムです。

本記事では、「次」コマンドのたびに STT + LLM を呼んでいた設計を、初回で全回答を生成しチャンク分割する方式に変更してAPIコスト90%削減を達成した実装を詳しく解説します。

:::message
背景・ストーリーはnoteに → [「次」のAPI呼び出しをチャンキングで90%削減した話](https://note.com/yamashita_aidev/n/n7654a6de99cf)
:::

## 問題: 「次」コマンドのたびにフルパイプラインが走る

当初の実装では「次」という発話も通常の音声認識フローで処理していました。

```
マイク入力 → Whisper API(STT) → コマンド判定 → LLM再生成 → TTS
```

1セッションで「次」を5回使うと、APIコストは質問1回分の**6倍**に膨らんでいました。

## 解決策: 初回で全回答生成 → チャンク分割 → キャッシュ

### 処理フローの変化

| 処理 | 初回質問 | 「次」Before | 「次」After | 「もう一度」 |
|------|---------|------------|------------|------------|
| STT (Whisper) | 実行 | 実行（無駄） | **スキップ** | スキップ |
| RAG 検索 | 実行 | スキップ | スキップ | スキップ |
| LLM (Claude) | 実行 | 実行（無駄） | **スキップ** | 不要 |
| TTS | 実行 | 実行 | 実行 | **スキップ** |
| レイテンシ | 4-6秒 | 4-6秒 | **1秒以下** | **即時** |

## chunk_answer() の実装

### 基本設計: 「。」で3文ずつ分割

```python
def chunk_answer(text: str, sentences_per_chunk: int = 3) -> list[str]:
    """回答テキストを「。」で分割し、指定文数ごとにチャンク化する.

    Args:
        text: LLMが生成した全回答テキスト
        sentences_per_chunk: 1チャンクあたりの文数（デフォルト3）

    Returns:
        分割されたチャンクのリスト
    """
    # 「。」で文を分割し、空文字を除外
    sentences = [s + "。" for s in text.split("。") if s.strip()]
    if not sentences:
        return [text] if text.strip() else []

    chunks: list[str] = []
    for i in range(0, len(sentences), sentences_per_chunk):
        # 指定文数ごとにひとつのチャンクにまとめる
        chunk = "".join(sentences[i:i + sentences_per_chunk])
        chunks.append(chunk)

    return chunks if chunks else [text]
```

### なぜ「3文」なのか

ユーザビリティテストの結果です:

- **2文**: 短すぎて「次」の頻度が増える → スタッフの手間が増加
- **3文**: スピーカーから聞いて理解できる適切な長さ
- **4文**: デバイスのスピーカーでは聞き逃しが発生

### エッジケースへの対応

```python
# テストケース: さまざまな入力パターン
def test_chunk_answer_edge_cases():
    # 句点なしの短文 → そのまま1チャンクとして返す
    assert chunk_answer("句点のない短文") == ["句点のない短文"]

    # 空文字列 → 空リスト
    assert chunk_answer("") == []
    assert chunk_answer("   ") == []

    # 1文だけ → 1チャンク
    assert chunk_answer("これは1文です。") == ["これは1文です。"]

    # ちょうど3文 → 1チャンク
    text = "1文目。2文目。3文目。"
    result = chunk_answer(text)
    assert len(result) == 1

    # 4文 → 2チャンク（3文 + 1文）
    text = "1文目。2文目。3文目。4文目。"
    result = chunk_answer(text)
    assert len(result) == 2
    assert result[0] == "1文目。2文目。3文目。"
    assert result[1] == "4文目。"
```

## セッション管理とチャンクキャッシュ

### セッションコンテキストの構造

```python
from datetime import datetime, timezone

# セッション単位でチャンクをキャッシュ
session_contexts: dict[str, dict] = {}

def register_session(
    session_id: str,
    original_query: str,
    answer_text: str,
    scored_chunks: list[dict],
) -> list[str]:
    """初回回答生成後にセッションコンテキストを登録する.

    Args:
        session_id: WebSocketセッションID（UUID v4推奨）
        original_query: ユーザーの元質問
        answer_text: LLMが生成した全回答テキスト
        scored_chunks: RAG検索結果（デバッグ・再利用用）

    Returns:
        分割されたチャンクリスト
    """
    answer_chunks = chunk_answer(answer_text)

    session_contexts[session_id] = {
        "original_query": original_query,
        "context_chunks": scored_chunks,
        "answer_chunks": answer_chunks,
        "current_chunk_index": 0,         # 現在何チャンク目まで返したか
        "full_answer": answer_text,
        "created_at": datetime.now(timezone.utc),  # TTL管理用
    }

    return answer_chunks
```

:::message alert
**注意**: `session_id` には UUID v4 等の予測不能な値を使用してください。連番IDでは他ユーザーのキャッシュにアクセスされるリスクがあります。また、途中離脱に備えてTTL（例: 30分）による自動クリーンアップの実装を推奨します。
:::

### 「次」コマンドの処理

```python
async def process_next_command(
    websocket: WebSocket,
    session_id: str,
    tts_service,
) -> None:
    """「次」コマンド処理 — STT/RAG/LLMを一切呼ばない.

    セッションコンテキストからチャンクを取り出してTTSに渡すだけ。
    外部APIへの通信は発生しない。
    """
    ctx = session_contexts.get(session_id)
    if ctx is None:
        # セッション切れ: クライアントへエラー通知
        await websocket.send_json({"error": "session_expired"})
        return

    answer_chunks = ctx["answer_chunks"]
    next_index = ctx["current_chunk_index"] + 1

    if next_index >= len(answer_chunks):
        # 全チャンク読み上げ完了
        complete_text = "以上で全ての内容をお伝えしました。"
        audio_bytes, duration_ms = await tts_service.synthesize(complete_text)
        # セッションを破棄してメモリ解放
        session_contexts.pop(session_id, None)
        return

    # 次のチャンクを取得
    chunk_text = answer_chunks[next_index]
    ctx["current_chunk_index"] = next_index

    # TTSのみ実行 — APIコストが発生するのはここだけ
    audio_bytes, duration_ms = await tts_service.synthesize(chunk_text)
```

## Android側: 「もう一度」のゼロ通信実装

「もう一度」はサーバーに一切触れません。最後に再生したWAVデータをメモリにキャッシュするだけです。

```kotlin
// AudioPlayer.kt — 再生バッファキャッシュ
class AudioPlayer {
    // 最後に再生したWAVデータをキャッシュ
    private var lastPlayedWavData: ByteArray? = null

    fun play(wavData: ByteArray) {
        lastPlayedWavData = wavData  // 再生のたびにキャッシュを更新
        // ... 実際の再生処理 ...
    }

    fun replayLast(): Boolean {
        // キャッシュがなければfalseを返す
        val cached = lastPlayedWavData ?: return false
        play(cached)  // キャッシュから即時再生 — サーバー通信ゼロ
        return true
    }
}
```

## プロンプト設計: 音声読み上げ向けの出力制御

チャンキング方式に変更する際、LLMプロンプトも修正が必要でした。

```
# 旧プロンプト（「次」のたびにLLM呼び出し）
「手順のステップ {step_offset} 番目以降から回答してください。」

# 新プロンプト（初回で全文生成）
「手順がある場合はすべてのステップを含めて完全に回答してください。
回答は音声読み上げ専用のため、マークダウン記号は使用しないでください。
接続詞として『まず』『次に』『それから』『最後に』を使ってください。」
```

マークダウン禁止を入れ忘れると、TTSが「シャープシャープシャープ ステップ1」と読み上げます。テキスト出力の確認だけでは気づけない問題で、実機テストで発覚しました。

## 結果

| メトリクス | Before | After |
|-----------|--------|-------|
| 「次」の応答時間 | 4-6秒 | 1秒以下 |
| APIコスト（5回連続ナビ） | 6倍 | 1倍+TTS |
| 「もう一度」の通信 | サーバー往復 | ゼロ |

## まとめ

- `chunk_answer()` で回答を「。」区切り3文ずつに分割
- 初回で全回答を生成し、セッションにキャッシュ
- 「次」はキャッシュからチャンクを取り出してTTSのみ
- 「もう一度」はAndroidの再生バッファから即時再生
- プロンプトにはマークダウン禁止を必ず含める

「何を毎回やる必要があって、何を1回で済ませられるか」を整理するだけで、APIコスト90%削減とレイテンシ改善を同時に実現できました。
