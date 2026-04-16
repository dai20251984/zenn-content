---
title: "Anthropic → OpenAI へLLMバックエンドを移行した話 — Reasoning Modelの罠とpydantic-settings地雷の全記録"
emoji: "🔄"
type: "tech"
topics: ["python", "openai", "llm", "pydantic", "fastapi"]
published: true
---

:::message
**この記事でわかること**
- OpenAI の Reasoning Model（GPT-5-nano 等）で `answer_length=0` になる原因と回避策
- `AsyncAnthropic` → `AsyncOpenAI` への移行で変わるAPIの差分一覧
- pydantic-settings 2.7.1 で `validation_alias` を使ったテストが一斉に落ちる原因と修正
- Docker コンテナのポートバインドを忘れるとキャッシュが無効化されるパターン
:::

## はじめに

介護AIアプリの音声応答パイプライン（STT → RAG → LLM → TTS）で、LLMボトルネック（7秒）を解消するため、AnthropicからOpenAIへのバックエンド移行を行った。移行自体は正解だったが、道中で3つの地雷を踏んだ。

同じ移行を検討しているエンジニア向けに、ハマりどころを全記録する。

## 環境・バージョン情報

| パッケージ | バージョン |
|---|---|
| Python | 3.12 |
| openai | 1.75.0 |
| anthropic | 0.52.0 |
| pydantic-settings | 2.7.1 |
| FastAPI | 0.115.x |

## 移行の背景

音声応答パイプラインのレイテンシ内訳：

```
# 移行前のパイプライン構成
音声入力 → STT(1.6秒) → RAG検索(1.4秒) → LLM生成(7.0秒) → TTS(0.8秒)
合計: 約10秒
```

Claude Sonnetの7秒はボトルネックとして明確で、かつコストも高かった。
`gpt-4.1-nano` への移行で両方を改善できると仮説を立てた。

## 地雷① Reasoning Modelの罠 — `output_tokens=512` なのに `answer_length=0`

### 何が起きたか

最初に選んだのは「最新モデル＝高性能」という安易な判断でGPT-5-nanoだった。

```python
# 概念コード — 本番実装とは異なります

# NG: Reasoning Modelだと知らずに実装
response = await client.chat.completions.create(
    model="gpt-5-nano",
    max_tokens=512,        # ← Reasoning Modelでは無視される
    temperature=0.3,       # ← Reasoning Modelでは非対応（エラー）
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "入浴介助の手順を教えて"},
    ],
)
```

`temperature` を外して実行すると動くが、回答が空文字列になる：

```
# 実際のログ（抜粋）
output_tokens=512, answer_length=0
```

### 原因

GPT-5-nanoはReasoning Model（推論モデル）。このタイプは回答前に内部で「思考」を実行し、その思考にトークンを消費する。

`max_tokens=512` を指定すると、その512トークンが全部「思考」に使われ、実際の回答を生成するトークンが残らない。

Reasoning Modelで正しく使う場合：

```python
# 概念コード — 本番実装とは異なります

# Reasoning Model の正しい使い方（参考）
response = await client.chat.completions.create(
    model="gpt-5-nano",
    max_completion_tokens=2048,  # 思考+回答の合計。大きめに取る必要あり
    reasoning_effort="low",      # 思考トークンを節約するオプション
    # temperature は指定不可
    messages=[...],
)
# → 動くが、思考分トークンが無駄。短い回答生成には不向き
```

### 解決

介護Q&Aのような「知識を引いて短く答える」タスクにはReasoning Modelは不要。
`gpt-4.1-nano`（通常モデル）に切り替えたら一発で動いた。

```python
# 概念コード — 本番実装とは異なります

# OK: 通常モデルなら max_tokens / temperature がそのまま使える
response = await client.chat.completions.create(
    model="gpt-4.1-nano",
    max_tokens=512,
    temperature=0.3,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "入浴介助の手順を教えて"},
    ],
)
# レイテンシ: 約3.1秒（Claude Sonnetの7秒から半分以下）
```

:::message alert
**Reasoning Modelを使う前に確認すること**
1. モデル名でOpenAIのドキュメントを確認し、Reasoning Modelかどうかを確認する
2. `temperature` が使えない（指定するとエラー）
3. `max_tokens` ではなく `max_completion_tokens` を使う
4. 短い回答生成タスクには向かない（思考コストが無駄になる）
:::

## 地雷② API差分 — AsyncAnthropic vs AsyncOpenAI

移行で最も手間がかかったのはAPI差分。同じ「ストリーミングでLLMから回答を受け取る」処理でも、書き方がかなり違う。

### 差分まとめ

| 項目 | Anthropic | OpenAI |
|------|-----------|--------|
| systemプロンプト | 独立パラメータ `system=` | `messages` 内に `role="system"` |
| ストリーミング方式 | コンテキストマネージャ `messages.stream()` | `create(stream=True)` |
| テキスト取得 | `.text_stream` | `chunk.choices[0].delta.content` |
| usage取得 | `get_final_message().usage` | `stream_options={"include_usage": True}` が必要 |
| トークン名 | `input_tokens` / `output_tokens` | `prompt_tokens` / `completion_tokens` |
| プロンプトキャッシュ | `cache_control` を明示指定 | 自動（1024トークン以上で有効） |

### 実装比較コード

```python
# 概念コード — 本番実装とは異なります

# ========== Anthropic（移行前）==========
from anthropic import AsyncAnthropic

client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

async def stream_response_anthropic(
    system_prompt: str,
    user_message: str,
) -> AsyncIterator[str]:
    # コンテキストマネージャ方式でストリーミング
    async with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        system=system_prompt,               # system は独立パラメータ
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        async for text in stream.text_stream:  # .text_stream でテキストだけ取得
            yield text
        # usage は最後にまとめて取得
        final = await stream.get_final_message()
        usage = final.usage  # .input_tokens / .output_tokens


# ========== OpenAI（移行後）==========
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

async def stream_response_openai(
    system_prompt: str,
    user_message: str,
) -> AsyncIterator[str]:
    # stream=True パラメータ方式
    stream = await client.chat.completions.create(
        model="gpt-4.1-nano",
        max_tokens=512,
        temperature=0.3,
        stream=True,
        stream_options={"include_usage": True},  # ← usage取得に必須。忘れると None になる
        messages=[
            {"role": "system", "content": system_prompt},  # system は messages 内
            {"role": "user", "content": user_message},
        ],
    )
    async for chunk in stream:
        # choices が空のチャンク（usage チャンク）が末尾に来る
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            yield delta.content
        # usage は最後のチャンクに含まれる
        if chunk.usage:
            usage = chunk.usage  # .prompt_tokens / .completion_tokens
```

:::message
**`stream_options={"include_usage": True}` を忘れると**
ストリーミング中の全チャンクで `chunk.usage` が `None` になる。
コスト計算やトークン数ログを取る場合は必須。
:::

### プロンプトキャッシュの差分

```python
# 概念コード — 本番実装とは異なります

# Anthropic: cache_control を明示的に指定する
# 1024トークン以上のプロンプトに有効
messages=[{
    "type": "text",
    "text": system_prompt,
    "cache_control": {"type": "ephemeral"},  # ← これを付けないとキャッシュされない
}]

# OpenAI: 自動プロンプトキャッシュ（設定不要）
# 1024トークン以上のプロンプトは自動的にキャッシュされる
# 確認方法: chunk.usage.prompt_tokens_details.cached_tokens
```

## 地雷③ pydantic-settings 2.7.1 — `validation_alias` でテスト12件崩壊

### 何が起きたか

依存パッケージを更新したらテストが12件まとめて落ちた。

```
# エラーログ（抜粋）
FAILED test_settings.py::test_default_model - ValidationError: Extra inputs are not permitted
FAILED test_settings.py::test_custom_model - ValidationError: Extra inputs are not permitted
... (12件)
```

### 原因

pydantic-settings **2.7.1** から `BaseSettings` のデフォルトが変更された：

| バージョン | `extra` のデフォルト |
|---|---|
| 2.7.0 以前 | `'ignore'`（未定義フィールドは無視） |
| **2.7.1 以降** | **`'forbid'`（未定義フィールドはエラー）** |

`validation_alias` を使っていると、この変更が罠になる：

```python
# 概念コード — 本番実装とは異なります

# NG: pydantic-settings 2.7.1 で壊れるパターン
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    llm_model: str = Field(
        default="gpt-4.1-nano",
        validation_alias="LLM_MODEL",  # 環境変数名をエイリアス指定
    )

# テストコード
settings = Settings(LLM_MODEL="gpt-4.1-nano")
# → 2.7.0 まで: OK
# → 2.7.1 から: ValidationError（"Extra inputs are not permitted"）
#
# 理由: extra='forbid' がフィールド名チェックを先に行う。
#       フィールド名は 'llm_model'、入力キーは 'LLM_MODEL' → 一致しない → エラー
#       validation_alias での解決より先に弾かれる。
```

### 修正

```python
# 概念コード — 本番実装とは異なります

# OK: extra="ignore" を明示するだけ
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",  # ← 1行追加。これで元の挙動に戻る
    )

    llm_model: str = Field(
        default="gpt-4.1-nano",
        validation_alias="LLM_MODEL",
    )
```

たった1行だが、原因特定に2時間かかった。エラーメッセージだけでは pydantic-settings のデフォルト変更が原因とは気づきにくい。

:::message alert
**pydantic-settings を使っているプロジェクトが 2.7.1 以上に上げたとき**
`validation_alias` を使っているフィールドのテストが落ちたら、まず `extra="ignore"` を `SettingsConfigDict` に追加してみる。
:::

## 地雷（番外）Dockerのポートバインド忘れ

LLM移行の検証中にQ-Aキャッシュのヒット率が0%のままだった。

```bash
# NG: ポートバインドなし（ホストのFastAPIからアクセスできない）
docker run -d --name postgres pgvector/pgvector:pg16

# OK: ポートバインドあり
docker run -d --name postgres -p 5432:5432 pgvector/pgvector:pg16
```

FastAPIはホスト側で動いているのに、PostgreSQLはコンテナ内でポートが閉じていた。キャッシュ書き込みがサイレントに失敗し続けていた。LLM移行のデバッグに集中していて見落としていた典型的な凡ミス。

## 移行結果

### レイテンシ計測

各段階を `time.perf_counter()` で計測：

| 処理段階 | Claude Sonnet | gpt-4.1-nano | キャッシュヒット時 |
|---------|--------------|--------------|-----------------|
| STT | 1.6秒 | 1.6秒 | 1.6秒 |
| RAG検索 | 1.4秒 | 1.4秒 | 0秒（スキップ） |
| LLM生成 | 7.0秒 | 3.1秒 | 0秒（スキップ） |
| TTS | 0.8秒 | 0.8秒 | 0秒（キャッシュ） |
| **合計** | **約10秒** | **約6秒** | **約1.6秒** |

LLM単体で7.0秒 → 3.1秒（55%削減）。Q-Aキャッシュ併用時は約1.6秒まで短縮できる。

### 移行工数

| 作業 | 所要時間 |
|------|---------|
| GPT-5-nanoのデバッグ（Reasoning Modelの罠） | 3時間 |
| gpt-4.1-nanoへの切り替え | 30分 |
| ストリーミングAPI差分の対応 | 2時間 |
| pydantic-settingsの修正 | 2時間 |
| Dockerポートバインド問題 | 1時間 |
| テスト全件パス確認 | 1時間 |
| **合計** | **約9.5時間** |

GPT-5-nanoの3時間は、最初にドキュメントを読んでいれば0時間だった。

## まとめ

| 地雷 | 原因 | 解決策 |
|------|------|--------|
| Reasoning Modelで回答空 | `max_tokens` が思考に全消費 | 通常モデルに切り替え or `max_completion_tokens` を大きく取る |
| OpenAI ストリーミングで usage が None | `stream_options={"include_usage": True}` の指定忘れ | オプションを追加 |
| pydantic-settings テスト12件崩壊 | 2.7.1 で `extra='forbid'` がデフォルト化 | `extra="ignore"` を明示 |
| Q-Aキャッシュ無効 | Dockerのポートバインド漏れ | `-p 5432:5432` を追加 |

移行自体は正解だった。同じ移行を検討している方の参考になれば。

## 参考リンク

- [OpenAI Reasoning Models Guide](https://platform.openai.com/docs/guides/reasoning)
- [pydantic-settings 2.7.1 Changelog](https://docs.pydantic.dev/latest/changelog/)
- [OpenAI Streaming with Usage](https://platform.openai.com/docs/api-reference/chat/streaming)

> 背景・ストーリーはnoteに → https://note.com/yamashita_aidev/n/n6fb379a97424
