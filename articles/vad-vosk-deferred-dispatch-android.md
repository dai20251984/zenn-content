---
title: "Android音声AIのVAD×Vosk競合を400msのDeferred Dispatchで解決する"
emoji: "🎙️"
type: "tech"
topics: ["android", "kotlin", "vosk", "vad", "介護テック"]
published: true
---

# Android音声AIのVAD×Vosk競合を400msのDeferred Dispatchで解決する

:::message
**この記事で分かること**
- SpeechDetector（Silero VAD）とVosk音声認識エンジンのタイミング競合が起きるメカニズム
- boolean フラグおよび CountDownLatch アプローチが失敗する根本的な理由
- Deferred Dispatch パターンの完全な実装（handleSpeechComplete + Voskコールバック + 3箇所のクリーンアップ）
- 400msデッドラインの数値根拠（実測データつき）
- タイマー系Runnableを正しく後始末するためのライフサイクル設計
:::

## はじめに

VADとASRエンジンを組み合わせた音声パイプラインでは、両者のタイミング差が原因で「コマンドが別の操作として送信される」競合状態が発生する。本記事はその競合をAndroid/Kotlinで再現し、Deferred Dispatchパターンで解決した実装記録だ。

## 対象読者・前提環境

**対象読者**
- AndroidアプリにVAD + ASRを組み込んでいるエンジニア
- 音声コマンド認識でタイミング依存のバグに悩んでいるエンジニア
- Deferred Dispatch / タイマーベースの非同期処理に興味があるエンジニア

**前提環境**

| 項目 | バージョン / 値 |
|------|---------------|
| Android | API 31以上（Android 12+） |
| Kotlin | 1.9系 |
| Vosk Android | 0.3.47 |
| Silero VAD | v5（ONNX形式） |
| ONNX Runtime Android | 1.17.3 |
| テストデバイス | Pixel 7（arm64-v8a） |

**システム構成の前提**

```
// 概念コード — 本番実装とは異なります
[マイク入力 16kHz PCM]
        ↓
AudioRouter（ByteArray → ShortArray変換・分配）
        ├── VoskEngine（HandlerThread）  ← コマンド検出専用（オンデバイス）
        └── SpeechDetector（Silero VAD）← 発話区間検出専用（オンデバイス）
                ↓（発話終了検知）
           WAVEncoder → WebSocket → サーバー（質問のみ）
```

VoskとVADは**独立した非同期コンポーネント**として動作する。このことが今回の競合の根本原因になる。

---

## 問題・課題の概要

### 現象：「もう一度」が新しい質問になる

音声AIインカムのフォローアップ機能はこう動く。

1. メインの質問（例：「褥瘡の予防策は？」）に対してシステムが音声で回答する
2. 回答後、`LISTENING_FOLLOWUP` 状態に入る
3. この状態で「もう一度」と言えば同じ回答を再生、「次」と言えば次の項目に進む
4. それ以外の発話はサーバーへ新規質問として送信する

フィールドテストで「もう一度と言ったのに別の答えが来た」「先輩への返事が質問として飛んでいった」という報告が複数入った。Logcatを調べると全件が `LISTENING_FOLLOWUP` 状態中に発生していた。

### Logcatで計測した344ms差

問題を再現させてタイムスタンプを並べた。

```
// 概念コード — 本番実装とは異なります
14:50:03.400  SpeechDetector: 発話開始を検出しました
14:50:03.886  SpeechDetector: 発話終了を検出しました      ← VADが先に発火
14:50:03.887  VoiceAssistantService: handleSpeechComplete() → WAV送信開始
14:50:03.888  VoiceAssistantService: 状態遷移: LISTENING_FOLLOWUP → PROCESSING
14:50:04.230  VoskEngine: Vosk最終結果: {"text": "もう一度"} ← 344ms遅れて到着
14:50:06.800  サーバー返答: "もう一度。" ← サーバーは"もう一度"を質問として処理
```

`handleSpeechComplete()` が `14:50:03.887` にWAVを送信した。Voskが「もう一度」という認識結果を返したのは `14:50:04.230`、**344ms後**だった。

---

## 競合が起きるメカニズム

### スレッドモデルの全体像

```
// 概念コード — 本番実装とは異なります

AudioRecord スレッド（THREAD_PRIORITY_URGENT_AUDIO）
├─ audioRecord.read() → shortBuffer
├─ voskEngine.processFrame(shortBuffer)  → voskHandlerThread へ post
└─ speechDetector.processFrame(floatBuffer)  ← 同期実行（このスレッドで直接処理）
        ↓ VAD が発話終了を検出
   mainHandler.post { handleSpeechComplete(wavData) }  ← メインスレッドへ dispatch

voskHandlerThread（HandlerThread）
└─ recognizer.acceptWaveForm(audioData, size)
        ↓ Vosk 最終結果確定
   mainHandler.post { onCommandDetected(command) }  ← メインスレッドへ dispatch

メインスレッド（Looper）
├─ handleSpeechComplete()   ← SpeechDetector コールバック
└─ onCommandDetected()      ← Vosk コールバック
```

**SpeechDetectorが先に発火する理由** — SpeechDetectorは15フレーム × 32msの無音を検出したとき発話終了と判定する（約480ms）。この時点でWAVデータは揃っている。一方Voskは音響モデルへの入力 → デコーダー演算 → 後処理というパイプラインを経るため、内部遅延が **200〜350ms** 常態だ。さらにvoskHandlerThread から mainHandler への `post()` で数十ms上乗せされる。

合算すると、SpeechDetectorが発火してからVoskの最終結果が届くまでに **250〜400ms** かかる。`handleSpeechComplete()` が呼ばれた時点でVoskはまだ処理の途中なのだ。

---

## 解決アプローチの概要

### 失敗アプローチ1：boolean フラグ

```kotlin
// 概念コード — 本番実装とは異なります
// NG: フラグはhandleSpeechCompleteの「後」にしか立てられない

var commandRecognizedDuringFollowup = false

fun handleSpeechComplete(wavData: ByteArray) {
    // チェック時点では常に false — Vosk はまだ結果を返していない
    if (commandRecognizedDuringFollowup) {
        commandRecognizedDuringFollowup = false
        return
    }
    sendToServer(wavData)  // Vosk の結果を待たずに送信してしまう
}

// Vosk コールバック（voskHandlerThread → mainHandler）
fun onCommandDetected(command: VoskCommand) {
    // handleSpeechComplete が終わった「後」に呼ばれる
    commandRecognizedDuringFollowup = true  // 遅すぎる
}
```

`handleSpeechComplete()` は `onCommandDetected()` より必ず先に実行される。フラグは`handleSpeechComplete()`の**後**に立てられるため、チェック時点では常に `false` だ。これは**タイミングの問題**であってコードの書き方の問題ではない。フラグ手法そのものがこの競合状態に合わない。

### 失敗アプローチ2：CountDownLatch によるブロッキング

```kotlin
// 概念コード — 本番実装とは異なります
// NG: audioスレッドのブロッキングは録音データ欠落を引き起こす

fun handleSpeechComplete(wavData: ByteArray) {
    val latch = CountDownLatch(1)
    // Vosk 結果を最大 500ms 待つ
    // 問題: このメソッドは audioRecord スレッドから呼ばれる場合がある
    latch.await(500, TimeUnit.MILLISECONDS)  // audioスレッドが止まる → バッファ欠落
    ...
}
```

`handleSpeechComplete()` の呼び出しスタックが audioスレッドを経由する構成では、このスレッドをブロックすると `audioRecord.read()` が止まる。その間のマイク入力が欠落し、音声認識精度が低下する。音声が歪んで聞こえるという副作用もテストで確認された。

### 採用：Deferred Dispatch パターン

| アプローチ | 仕組み | 結果 |
|-----------|--------|------|
| boolean フラグ | Vosk結果でフラグON → handleSpeechCompleteでチェック | 失敗（フラグセットが常に遅い） |
| synchronized ブロック | Vosk完了まで handleSpeechComplete をブロック | 失敗（audioスレッド停止 → 録音欠落） |
| **Deferred Dispatch** | **WAV保持 → 400msデッドライン → Vosk到着で即判定** | **成功（最大400ms遅延、体感変化なし）** |

Deferred Dispatch の考え方：
1. WAVデータが揃ったとき、**すぐ送信せず保留する**
2. Voskがデッドライン内にコマンドを返した → **即座にコマンド処理し、WAVを破棄する**
3. デッドラインを過ぎてもVoskの結果が来ない → **新規質問として送信する**

---

## 実装詳細

### handleSpeechComplete — 保留とデッドライン登録

```kotlin
// 概念コード — 本番実装とは異なります

class VoiceAssistantService : Service() {

    // Deferred Dispatch の核となる3変数
    private var pendingFollowupWavData: ByteArray? = null          // 保留中のWAVデータ
    private var followupDecisionRunnable: Runnable? = null         // デッドライン判定Runnable
    private val followupVoskDeadlineMs: Long = 400L                // Vosk到着待ちデッドライン

    private val mainHandler = Handler(Looper.getMainLooper())

    // SpeechDetector コールバック（メインスレッドで呼ばれる）
    private fun handleSpeechComplete(wavData: ByteArray) {
        if (pipelineState != PipelineState.LISTENING_FOLLOWUP) {
            // フォローアップ待機中でなければ即座に送信（通常フロー）
            sendToServer(wavData)
            return
        }

        // === Deferred Dispatch 開始 ===

        // 既存のデッドラインRunnableがあればキャンセルして上書き
        // （二重発火しないように必ずremoveしてから再登録）
        followupDecisionRunnable?.let { mainHandler.removeCallbacks(it) }

        // WAVデータをメモリに保留する
        pendingFollowupWavData = wavData

        val decisionRunnable = Runnable {
            // デッドライン到来 → Vosk がデッドライン内に間に合わなかった
            val wav = pendingFollowupWavData ?: return@Runnable
            pendingFollowupWavData = null
            followupDecisionRunnable = null
            // 新規質問として送信する
            transitionTo(PipelineState.PROCESSING)
            sendToServer(wav)
            Log.i(TAG, "Deferred Dispatch: デッドライン到来 → 新規質問として送信")
        }

        followupDecisionRunnable = decisionRunnable
        mainHandler.postDelayed(decisionRunnable, followupVoskDeadlineMs)
        Log.d(TAG, "Deferred Dispatch: WAV保留 — ${followupVoskDeadlineMs}ms 待機開始")
    }
}
```

### Voskコールバック — デッドライン内到着時の処理

```kotlin
// 概念コード — 本番実装とは異なります

// VoskEngine からメインスレッドへ post されるコールバック
private fun onCommandDetected(command: VoskCommand) {
    if (pipelineState != PipelineState.LISTENING_FOLLOWUP) return

    // Deferred Dispatch: デッドラインRunnableをキャンセル
    followupDecisionRunnable?.let {
        mainHandler.removeCallbacks(it)
        followupDecisionRunnable = null
    }

    // 保留中のWAVデータを破棄（送信しない）
    pendingFollowupWavData = null

    Log.i(TAG, "Deferred Dispatch: Vosk到着 コマンド=$command → WAV破棄")

    // コマンドに応じた処理
    when (command) {
        VoskCommand.REPEAT -> {
            // 「もう一度」: キャッシュ済み音声を再生
            replayLastAnswer()
        }
        VoskCommand.NEXT -> {
            // 「次」: サーバーへ次コマンドを送信
            transitionTo(PipelineState.PROCESSING)
            requestNextItem()
        }
        VoskCommand.WAKE_WORD -> {
            // 「ウェイクワード」: フォローアップを終了して新しい質問待機へ
            cleanupDeferredDispatch()  // ← 下記のクリーンアップ関数
            transitionTo(PipelineState.ARMED)
        }
        else -> {
            // 認識されたが既知コマンド以外 → WAV送信に戻す
            // （pendingFollowupWavData は既に null なのでデッドライン側で送信された）
        }
    }
}
```

### 3箇所のクリーンアップ

Deferred Dispatch の実装で**最もバグが出やすい部分**がクリーンアップだ。`followupDecisionRunnable` は `mainHandler` に登録されたタイマーであり、残留すると「関係のない場面でタイマーが発火する」という再現困難なバグになる。

クリーンアップが必要な箇所は3つある。

```kotlin
// 概念コード — 本番実装とは異なります

// === クリーンアップ関数（3箇所で呼ぶ共通処理）===
private fun cleanupDeferredDispatch() {
    followupDecisionRunnable?.let {
        mainHandler.removeCallbacks(it)
    }
    followupDecisionRunnable = null
    pendingFollowupWavData = null
}

// --- クリーンアップ箇所 1: ウェイクワード待機への復帰時 ---
private fun returnToWakeWordListening() {
    cleanupDeferredDispatch()  // ← 必須
    transitionTo(PipelineState.LISTENING_WAKE_WORD)
    Log.d(TAG, "ウェイクワード待機に戻りました")
}

// --- クリーンアップ箇所 2: フォローアップモードへの再入時 ---
// 「再びフォローアップに入る」ときに古い保留データが残っていると競合する
private fun enterFollowupMode() {
    cleanupDeferredDispatch()  // ← 必須（前回の保留データをリセット）
    transitionTo(PipelineState.LISTENING_FOLLOWUP)
    Log.d(TAG, "フォローアップモード開始")
}

// --- クリーンアップ箇所 3: Service 終了時（onDestroy） ---
// Service が破棄されても mainHandler が生きているとRunnableが残留する
// → メモリリーク + 予期しない動作の両方を引き起こす
override fun onDestroy() {
    cleanupDeferredDispatch()  // ← 必須
    voskEngine.release()
    speechDetector.close()
    super.onDestroy()
    Log.d(TAG, "VoiceAssistantService: onDestroy — クリーンアップ完了")
}
```

**なぜ3箇所全て必要か**

| クリーンアップ箇所 | 省略した場合のバグ |
|-----------------|----------------|
| ウェイクワード待機への復帰時 | 旧デッドラインが発火 → 別の状態でWAV送信が走る |
| フォローアップモードへの再入時 | 前回の `pendingFollowupWavData` が残留 → 古いWAVが送信される |
| `onDestroy` | Service 破棄後も Runnable が残留 → メモリリーク・予期しない動作 |

---

## シーケンス図：正常系と異常系

### 正常系（Vosk がデッドライン内に到着）

```
SpeechDetector          VoiceAssistantService（main）          VoskEngine
     |                            |                                 |
     |-- handleSpeechComplete --> |                                 |
     |   (wavData)                |-- pendingFollowupWavData=wav    |
     |                            |-- postDelayed(runnable, 400ms)  |
     |                            |                                 |-- recognizer.result
     |                            |                                 |   "もう一度" 確定
     |                            |<-- onCommandDetected(REPEAT) ---|
     |                            |   (344ms後、デッドライン内)      |
     |                            |-- removeCallbacks(runnable)     |
     |                            |-- pendingFollowupWavData=null   |
     |                            |-- replayLastAnswer()            |
     |                            |                                 |
```

### 異常系（Vosk がデッドラインに間に合わない）

```
SpeechDetector          VoiceAssistantService（main）          VoskEngine
     |                            |                                 |
     |-- handleSpeechComplete --> |                                 |
     |   (wavData)                |-- pendingFollowupWavData=wav    |
     |                            |-- postDelayed(runnable, 400ms)  |
     |                            |                                 |
     |                          (400ms経過)                          |
     |                            |-- runnable 発火                 |
     |                            |-- sendToServer(wav)             |
     |                            |-- pendingFollowupWavData=null   |
     |                            |                                 |-- (遅れてVosk結果到着)
     |                            |<-- onCommandDetected(...) ------|
     |                            |   pipelineState != FOLLOWUP     |
     |                            |   → 何もしない（無視）           |
```

異常系でVosk結果が後から届いた場合、`pipelineState` がすでに `PROCESSING` になっているため、`onCommandDetected()` の先頭ガード `if (pipelineState != LISTENING_FOLLOWUP) return` によって無視される。二重処理は発生しない。

---

## ハマりどころ・注意点

### 1. `removeCallbacks` はRunnable参照で一致確認する

```kotlin
// 概念コード — 本番実装とは異なります

// NG: ラムダをその場で渡すと別オブジェクトになりremoveが効かない
mainHandler.postDelayed({ sendToServer(wav) }, 400L)
mainHandler.removeCallbacks { sendToServer(wav) }  // 別オブジェクト → 削除されない！

// OK: Runnableを変数に保持してからremoveする
val runnable = Runnable { sendToServer(wav) }
mainHandler.postDelayed(runnable, 400L)
mainHandler.removeCallbacks(runnable)  // 同じオブジェクト参照 → 正しく削除される
```

### 2. `pendingFollowupWavData` の null チェック順序

```kotlin
// 概念コード — 本番実装とは異なります

// デッドラインRunnable内でのパターン
val decisionRunnable = Runnable {
    // null チェックを先に行う
    // （cleanupDeferredDispatch() が先に呼ばれた場合は null になっている）
    val wav = pendingFollowupWavData ?: return@Runnable
    pendingFollowupWavData = null
    followupDecisionRunnable = null
    sendToServer(wav)
}
```

`cleanupDeferredDispatch()` が `removeCallbacks()` の前後タイミングで呼ばれる可能性があるため、Runnable 内でも `pendingFollowupWavData` が null かどうかを確認する。

### 3. デッドライン値はシステム全体のレイテンシを考慮して決める

400msはこのシステム固有の実測値から決めた。VAD・Voskのパラメータ（フレーム数・モデルサイズ）やデバイス性能が異なる環境では、再計測が必要だ。

---

## パフォーマンス計測結果

### VAD→Vosk タイミング差の実測

| 計測回数 | VAD発話終了 | Vosk最終結果到着 | タイミング差 |
|---------|-----------|----------------|------------|
| 1 | T+0ms | T+344ms | **344ms** |
| 2 | T+0ms | T+298ms | 298ms |
| 3 | T+0ms | T+371ms | 371ms |
| 4 | T+0ms | T+263ms | 263ms |
| 5 | T+0ms | T+389ms | 389ms |
| 平均 | — | — | **333ms** |
| 最大 | — | — | **389ms** |

### タイミング差の内訳

| 要素 | 計測値 |
|------|--------|
| VAD 無音検出（15フレーム × 32ms） | 約480ms |
| Vosk 内部遅延（音響モデル処理） | 200〜350ms |
| voskHandlerThread → mainHandler post 遅延 | 数十ms |
| SpeechDetector発火からVosk最終結果到着まで | 250〜400ms |

250〜400msのレンジをカバーするには400msのデッドラインで十分だ。これより短いと取りこぼしが残る。これより長くすると新規質問の送信が遅れてユーザーが「固まった？」と感じる可能性が出てくる。400msは**許容範囲ギリギリの最小値**として決めた。

### Deferred Dispatch 導入前後の改善結果

| 指標 | 導入前 | 導入後 |
|------|--------|--------|
| 「もう一度」誤送信率 | 約20%（5回に1回） | **0%** |
| 「次」コマンド誤送信 | 数件/テストセッション | **0件** |
| 新規質問の送信遅延 | 0ms（遅延なし） | 最大400ms（体感変化なし） |
| Voskコールバック平均処理時間 | — | 0.3ms（removeCallbacks含む） |

---

## まとめ・今後の課題

### まとめ

VADとVosk（ASR）を組み合わせた音声パイプラインで発生するタイミング競合は、boolean フラグや synchronized ブロックでは解決できない。これらは「タイミングが合えばうまくいく」設計であり、そもそも競合の存在を前提にしていない。

Deferred Dispatch パターンは「タイミングに依存しない」設計だ。WAVデータを保留してデッドラインを登録し、Voskが先に到着したらコマンド処理 + WAV破棄、デッドラインが先に到着したら新規質問として送信する。この判断分岐により、Voskの到着タイミングが不定でも正しく動作する。

実装の要点を3点に絞る。

- **WAVデータの保留変数・デッドラインRunnable・デッドライン値の3変数がDeferred Dispatchの核**
- **タイマー系Runnableのクリーンアップは3箇所（復帰時・再入時・onDestroy）で行う**
- **Runnableは変数に保持してから `removeCallbacks` で削除する（ラムダをその場で渡すと削除できない）**

### 今後の課題

- デッドライン値（400ms）の動的調整：Voskの内部遅延はデバイス性能・モデルサイズで変化する。過去N回の計測平均から動的に設定する仕組みが望ましい
- フォローアップ状態でのタイムアウト：`LISTENING_FOLLOWUP` に入ったまま無音が続く場合のフォールバック処理（現在は30秒タイムアウト）

---

## 参考リンク

- [Vosk API 公式ドキュメント](https://alphacephei.com/vosk/)
- [Vosk Android サンプル (alphacep/vosk-android-demo)](https://github.com/alphacep/vosk-android-demo)
- [Silero VAD GitHub (snakers4/silero-vad)](https://github.com/snakers4/silero-vad)
- [Android Handler & Looper 公式ドキュメント](https://developer.android.com/reference/android/os/Handler)
- [Handler.removeCallbacks(Runnable) リファレンス](https://developer.android.com/reference/android/os/Handler#removeCallbacks(java.lang.Runnable))

---

> **背景・ストーリーはnoteに**
> VADとVoskの344ms差に気づいた経緯、フィールドテストでの担当者フィードバック、失敗したアプローチの試行錯誤をストーリー形式で書いています。
> → https://note.com/yamashita_aidev/n/nb71fd035a6cd
