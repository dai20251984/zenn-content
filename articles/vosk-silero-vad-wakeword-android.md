---
title: "Vosk文法モードとSilero VAD v5でウェイクワードをオンデバイス検出する"
emoji: "🎙️"
type: "tech"
topics: ["android", "kotlin", "llm", "rag", "介護テック"]
published: true
---

# Vosk文法モードとSilero VAD v5でウェイクワードをオンデバイス検出する

介護現場向けハンズフリー音声AIインカムを開発する中で、ウェイクワード検出と短いコマンド認識をオンデバイスに完全移行した。この記事では、その実装の技術的な核心部分——Voskの文法モードの内部動作、Silero VAD v5のONNXパラメータチューニング、HandlerThreadによるスレッド設計、6ステート状態機械の設計判断——を詳しく記録する。

## この記事で扱うこと

- Voskの文法モードが内部でどう動くか（グラフ構造と`[unk]`の役割）
- Silero VAD v5のONNXモデル構造と、介護施設騒音環境向けパラメータチューニングの実験記録
- `HandlerThread`パターンと`Coroutine(Dispatchers.IO)`の比較——なぜCoroutineでクラッシュするか
- 6ステート状態機械の設計判断とその代替案との比較
- `build.gradle`設定と`assets`フォルダへのモデル配置

## システム構成の前提

音声AIシステムの全体構成はこうなっている：

```
[マイク入力 16kHz PCM]
        ↓
AudioRouter（ByteArrayからShortArrayへ変換・分配）
        ├── VoskEngine（HandlerThread）  ← コマンド検出専用
        └── SpeechDetector（Silero VAD）← 発話区間検出専用
                ↓（発話終了検知 + ウェイクワード確認済み）
           WAVEncoder → WebSocket → FastAPIサーバー（長文質問のみ）
                ↓
           Claude API（RAG付き） → TTS → スピーカー
```

重要な設計原則：**コマンド認識（ウェイクワード・「次」・「もう一度」）はデバイス上で完結させ、Whisper APIには「本当の質問」だけを送る**。

---

## 1. build.gradle と依存関係設定

まずプロジェクト構成から。`build.gradle.kts` の関連部分：

```kotlin
// app/build.gradle.kts

android {
    defaultConfig {
        // ONNXランタイムのABI フィルタリング（不要なアーキテクチャを除外してAPKサイズ削減）
        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")  // x86はエミュレータ用
        }
    }

    // assetsフォルダへのアクセスを有効化（Voskモデル展開に必要）
    sourceSets {
        getByName("main") {
            assets.srcDirs("src/main/assets")
        }
    }
}

dependencies {
    // Vosk Android ライブラリ（JARとネイティブライブラリを内包）
    implementation("com.alphacephei:vosk-android:0.3.47")

    // ONNX Runtime Android（Silero VAD v5用）
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.17.3")

    // コルーチン（WebSocket通信用。Voskには使わない）
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.0")

    // JSON処理（Voskの認識結果パース用）
    implementation("org.json:json:20231013")
}
```

`assets`フォルダのディレクトリ構造：

```
src/main/assets/
├── vosk-model-small-ja/      ← Vosk日本語スモールモデル（約50MB）
│   ├── am/
│   ├── conf/
│   ├── graph/
│   │   └── phones/
│   ├── ivector/
│   └── ...
└── silero_vad_v5.onnx        ← Silero VAD v5モデル（約2MB）
```

Voskモデルは [alphacep/vosk-model-small-ja](https://alphacephei.com/vosk/models) からダウンロードしてそのままassetsに置く。APKに含まれるため、`abiFilters`でサイズを絞ることが重要になる。

---

## 2. Voskの文法モード——内部で何が起きているか

### 全語彙モードと文法モードの違い

Voskは内部でWeighted Finite-State Transducer（WFST）を使って音声認識を行う。全語彙モードでは、発音辞書に登録されている全単語が認識候補になる。日本語の場合、数万単語が候補に含まれる。

文法モード（Grammar mode）では、WFST グラフを**指定した単語リストだけに制限した部分グラフ**に置き換える。認識候補が数万から数十語に絞られるため：

- **誤検知率が大幅に低下する**（近い発音の無関係な語への誤認識がなくなる）
- **CPU負荷が下がる**（探索空間が小さい）
- **認識速度が向上する**

文法の指定は以下のJSONフォーマットで行う：

```kotlin
private fun buildGrammar(): String {
    // 認識対象の全バリアントを収集する
    val allVariants: List<String> = buildList {
        addAll(config.wakePhraseVariants)       // ウェイクワードの表記ゆれ
        addAll(config.nextCommandVariants)       // 「次」系のバリアント
        addAll(config.repeatCommandVariants)     // 「もう一度」系のバリアント
    }

    // JSON配列文字列に変換し、末尾に[unk]を追加
    val quotedEntries = allVariants.map { "\"${it.replace("\"", "\\\"")}\"" }
    val withUnk = quotedEntries + "\"[unk]\""
    return "[${withUnk.joinToString(", ")}]"
}

// 生成されるJSON例:
// ["アシスト", "あしすと", "アシスト ",
//  "次", "つぎ", "続き", "つづき",
//  "もう一度", "もう一回", "もう 一度", "もう 一回",
//  "[unk]"]
```

### `[unk]`が必須な理由——内部グラフの観点から

`[unk]`（unknown）は文法モード特有の特殊トークンだ。これを省略したときに何が起きるかを理解するには、WFSTの構造を把握する必要がある。

文法なしの全語彙モードでは、音声フレームが入力されるたびにVoskはビーム探索を進め、一定フレーム数ごとに「最尤仮説」を確定させる。常に何らかの認識結果が返ってくる。

文法モードで`[unk]`を省いた場合、**文法に登録されていない音声が入力されると探索が行き詰まる**。WFSTのグラフ上にマッチするパスが存在しないため、Voskは認識結果を保留し続ける。結果として：

1. `recognizer.result` / `recognizer.partialResult` が空文字列を返し続ける
2. 次のウェイクワードが来ても、保留状態のため認識されない
3. 見かけ上「認識できなくなった」状態になる

`[unk]`を追加することで、「文法に登録されていない任意の音声」を受け取るパスがグラフに追加される。文法外の音声は`"[unk]"`として安全に返ってきて、探索が詰まらなくなる。

```kotlin
// [unk]なし → 文法外の音声で詰まる
val badGrammar = """["アシスト", "次", "もう一度"]"""

// [unk]あり → 文法外は"[unk]"として処理され、次の音声を待てる
val goodGrammar = """["アシスト", "次", "もう一度", "[unk]"]"""

recognizer = Recognizer(model, sampleRate.toFloat(), goodGrammar)
```

### 発音バリアント設計

日本語の音声認識では、同じ単語でも複数の発音・表記がVoskから返ってくる可能性がある。以下の3つの要因を考慮してバリアントを設計した：

| 要因 | 例 | 対策 |
|------|-----|------|
| 分かち書き挿入 | `"もう一度"` → `"もう 一度"` | スペース入りのバリアントも登録 |
| ひらがな/カタカナ | `"アシスト"` → `"あしすと"` | 両方登録 |
| 末尾スペース | `"アシスト"` → `"アシスト "` | 末尾スペース付きも登録 |

さらにコマンド分類時に文字列正規化を行い、バリアント漏れを吸収する：

```kotlin
internal fun normalizeText(text: String): String {
    return text
        .replace(" ", "")   // 半角スペース除去
        .replace("　", "")  // 全角スペース除去
        .trim()
}

internal fun isWakeWordMatch(recognizedText: String): Boolean {
    val normalized = normalizeText(recognizedText)
    // ウェイクフレーズ本体との比較
    if (normalized == normalizeText(config.wakePhrase)) return true
    // 全バリアントとの比較
    return config.wakePhraseVariants.any { variant ->
        normalized == normalizeText(variant)
    }
}
```

---

## 3. Silero VAD v5——ONNXモデル構造とパラメータチューニング

### モデルアーキテクチャの概要

Silero VAD v5はTemporal Convolutional Network（TCN）ベースの軽量音声活動検出モデルだ。ONNXフォーマットで配布されており、ファイルサイズは約2MBと小さい。

入出力仕様：

```
入力:
  - input:  Float32[1, 512]  // 1チャンネル、512サンプル（32ms @ 16kHz）
  - sr:     Int64[1]         // サンプルレート（16000 固定）
  - h:      Float32[2, 1, 64] // 隠れ状態（ゼロ初期化して渡す）
  - c:      Float32[2, 1, 64] // セル状態（ゼロ初期化して渡す）

出力:
  - output: Float32[1, 1]   // 音声確率（0.0〜1.0）
  - hn:     Float32[2, 1, 64] // 更新後の隠れ状態
  - cn:     Float32[2, 1, 64] // 更新後のセル状態
```

**重要：v5はフレームサイズが512サンプル固定**（v4以前の512/256から変更）。また、隠れ状態`h`と`c`を呼び出しごとに引き継ぐことで、フレーム間の文脈を保持する。

### Androidでの初期化

```kotlin
class SileroVAD(
    context: Context,
    private val config: VADConfig = VADConfig()
) {
    // ONNX セッションオプション
    private val sessionOptions = OrtEnvironment.getEnvironment().let { env ->
        OrtSession.SessionOptions().apply {
            // Android向け最適化: NNAPI (Neural Networks API) を有効化
            // 対応デバイスでは NPU/DSP に処理をオフロードできる
            try {
                addNnapi()
            } catch (e: OrtException) {
                // NNAPIが利用できないデバイスではCPUフォールバック
                Log.d(TAG, "NNAPI利用不可、CPUで実行します: ${e.message}")
            }
            // スレッド数を制限（他処理への影響を抑える）
            setIntraOpNumThreads(2)
        }
    }

    private val ortSession: OrtSession
    private val ortEnvironment = OrtEnvironment.getEnvironment()

    // LSTM 隠れ状態（フレーム間で引き継ぐ）
    private var hiddenState = Array(2) { Array(1) { FloatArray(64) } }
    private var cellState = Array(2) { Array(1) { FloatArray(64) } }

    init {
        // assetsからONNXモデルを読み込む
        val modelBytes = context.assets.open("silero_vad_v5.onnx").readBytes()
        ortSession = ortEnvironment.createSession(modelBytes, sessionOptions)
    }

    @Synchronized  // マルチスレッド呼び出しに対してスレッドセーフを保証
    fun processFrame(audioFloat: FloatArray): Float {
        require(audioFloat.size == config.frameSize) {
            // フレームサイズ不一致は致命的エラー——早期に検出する
            "フレームサイズ不正: 期待=${config.frameSize}, 実際=${audioFloat.size}"
        }

        // 入力テンソルの準備
        val inputTensor = OnnxTensor.createTensor(
            ortEnvironment,
            arrayOf(audioFloat),  // shape: [1, 512]
        )
        val srTensor = OnnxTensor.createTensor(
            ortEnvironment,
            longArrayOf(config.sampleRate.toLong()),
        )
        val hTensor = OnnxTensor.createTensor(ortEnvironment, hiddenState)
        val cTensor = OnnxTensor.createTensor(ortEnvironment, cellState)

        val inputs = mapOf(
            "input" to inputTensor,
            "sr" to srTensor,
            "h" to hTensor,
            "c" to cTensor,
        )

        val results = ortSession.run(inputs)
        val probability = (results["output"]?.value as Array<*>)[0].let {
            (it as FloatArray)[0]
        }

        // 隠れ状態を次フレームへ引き継ぐ
        hiddenState = results["hn"]?.value as Array<Array<FloatArray>>
        cellState = results["cn"]?.value as Array<Array<FloatArray>>

        // リソース解放（必須）
        results.close()
        inputTensor.close()
        srTensor.close()
        hTensor.close()
        cTensor.close()

        return probability
    }

    fun reset() {
        // 発話検出のセッション終了時に隠れ状態をリセット
        hiddenState = Array(2) { Array(1) { FloatArray(64) } }
        cellState = Array(2) { Array(1) { FloatArray(64) } }
    }
}
```

### パラメータチューニングの実験記録

チューニングは3段階の環境で行った：

**実験環境**

| フェーズ | 環境 | 特徴 |
|---------|------|------|
| Phase 1 | 静かなオフィス | 背景雑音なし（SNR > 30dB） |
| Phase 2 | テレビ音あり | ニュース番組60dB、話者距離1m |
| Phase 3 | 介護施設想定 | テレビ + 空調 + 複数人会話 |

**パラメータと結果**

| threshold | minSpeechFrames | 環境 | 誤検知率 | 未検出率 | 総合判定 |
|-----------|----------------|------|---------|---------|---------|
| 0.50 | 3 | Phase 1 | 2% | 1% | OK |
| 0.50 | 3 | Phase 2 | **31%** | 1% | NG（テレビ誤検知多発） |
| 0.50 | 5 | Phase 2 | 18% | 2% | NG |
| 0.65 | 5 | Phase 2 | 8% | 3% | 改善 |
| 0.65 | 5 | Phase 3 | 11% | 4% | 許容範囲 |
| 0.70 | 5 | Phase 3 | 7% | **12%** | NG（未検出増加） |

**最終採用値：`threshold=0.65`, `minSpeechFrames=5`**

```kotlin
data class VADConfig(
    // 音声確率の閾値。高いほど誤検知が減るが未検出が増える
    // 0.65: テレビ音あり環境での誤検知と未検出のバランス点
    val threshold: Float = 0.65f,

    // 「発話開始」と判定するために閾値を超え続けるフレーム数
    // 5フレーム × 32ms = 160ms 連続で音声と判定されないと発話開始にしない
    // テレビの音は0.65を超えても断続的なため、これで除外できる
    val minSpeechFrames: Int = 5,

    // 「発話終了」と判定するために閾値を下回り続けるフレーム数
    // 15フレーム × 32ms = 480ms の無音で発話終了
    // 介護業務中の「間」を取りながら話す発話パターンに対応
    val minSilenceFrames: Int = 15,

    // 強制終了タイムアウト（誤検知で録音が終わらなくなる事態を防ぐ）
    val maxSpeechDurationMs: Long = 30_000L,

    val sampleRate: Int = 16_000,

    // Silero VAD v5 のフレームサイズ（変更不可）
    val frameSize: Int = 512
)
```

**推論速度の計測結果**（Pixel 7、arm64-v8a）：

| 実行方式 | 1フレームあたりの推論時間 |
|---------|----------------------|
| CPU（スレッド数2） | 0.8ms |
| NNAPI有効 | 0.3ms |

16kHz / 512サンプル = 1フレーム32ms。推論時間は0.3〜0.8msのため、リアルタイム処理には十分な余裕がある。

---

## 4. HandlerThreadパターン——なぜCoroutineではダメか

### Voskがスレッドセーフでない理由

Voskのネイティブライブラリ（kaldi-basedのC++コード）は、`Recognizer`インスタンスのメソッドに対して並行アクセスを想定していない。具体的には：

- `acceptWaveForm()` の内部状態（デコードグラフ、ビームサーチの仮説リスト）が可変
- C++側でmutexによる保護が実装されていない

Kotlinのコルーチンでこれを呼ぶとどうなるか：

```kotlin
// NG: Coroutine(Dispatchers.IO)でVoskを呼ぶ
// Dispatchers.IOはスレッドプールを使う。
// 連続してフレームを投入すると複数スレッドからacceptWaveFormが呼ばれる
scope.launch(Dispatchers.IO) {
    recognizer.acceptWaveForm(audioData, audioData.size)  // クラッシュする
}
```

`Dispatchers.IO`はデフォルトでスレッドプールを使うため、複数の`launch`が同時に実行される可能性がある。`acceptWaveForm`が並行呼び出しされると：

```
FATAL EXCEPTION: DefaultDispatcher-worker-2
java.lang.IllegalStateException: Native recognizer has been released
  at org.vosk.Recognizer.acceptWaveForm(Native Method)
  ...
```

`SingleThreaded`なCoroutineコンテキストを作れば動くかもしれないが、それは事実上HandlerThreadと同等になる。Androidのオーディオ処理では、確立されたパターンであるHandlerThreadを使う方が意図が明確だ。

### HandlerThreadの実装

```kotlin
class VoskEngine(
    private val config: WakeWordConfig,
    private val onCommandDetected: (VoskCommand) -> Unit,
    private val onError: (Exception) -> Unit,
) {
    // HandlerThread: Androidのメッセージキューを持つ専用スレッド
    // 投入されたRunnableを1つずつ順番に処理する（シリアライズ）
    private val voskHandlerThread = HandlerThread(
        "VoskWorkerThread",
        Process.THREAD_PRIORITY_DEFAULT  // URGENT_AUDIOはマイクスレッドのみに付与
    ).also { it.start() }

    private val voskHandler = Handler(voskHandlerThread.looper)

    // メインスレッドへのコールバック用
    private val mainHandler = Handler(Looper.getMainLooper())

    fun processFrame(audioData: ShortArray) {
        // voskHandlerに投入 → キューに積まれ、順番に実行される
        voskHandler.post {
            try {
                val finalResultDetected = recognizer?.acceptWaveForm(
                    audioData,
                    audioData.size
                ) ?: return@post

                if (finalResultDetected) {
                    // 最終確定結果（発話終了後）
                    val json = JSONObject(recognizer?.result ?: return@post)
                    val text = json.optString("text", "")
                    if (text.isNotBlank()) handleRecognizedText(text)
                } else {
                    // 部分結果（発話中）— デバッグ用途にのみ使う
                    // val partial = JSONObject(recognizer?.partialResult ?: "")
                    // Log.v(TAG, "部分認識: ${partial.optString("partial")}")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Vosk推論エラー", e)
                // エラーをメインスレッドに通知
                mainHandler.post { onError(e) }
            }
        }
    }

    private fun handleRecognizedText(text: String) {
        // このメソッドはvoskHandlerThread上で実行される
        val command = classifyCommand(text) ?: return
        // コールバックはメインスレッドで呼ぶ
        mainHandler.post { onCommandDetected(command) }
    }

    fun release() {
        voskHandler.post {
            // Voskリソースの解放もHandlerThread上で行う
            recognizer?.close()
            recognizer = null
            model?.close()
            model = null
            // HandlerThread自体を停止
            voskHandlerThread.quitSafely()
        }
    }
}
```

**HandlerThreadのメリット（まとめ）**

| 側面 | HandlerThread | Coroutine(Dispatchers.IO) |
|------|--------------|--------------------------|
| 実行保証 | 1スレッドで直列実行（確定的） | スレッドプール（並行実行の可能性） |
| Vosk安全性 | 安全 | 設定次第で危険 |
| キューサイズ制御 | `removeCallbacksAndMessages(null)` でクリア可能 | キャンセルが複雑 |
| Androidイディオム | 確立されたパターン | 問題ないが意図が伝わりにくい |

---

## 5. 6ステート状態機械の設計判断

### 状態一覧と遷移

```
INITIALIZING ─────────────────────────────────────────────────────────┐
     ↓ モデルロード完了                                                 │
LISTENING_WAKE_WORD ←──────────────────────────────────────────────── │
     ↓ ウェイクワード検出                                               │
ARMED ─────────────────────────────── タイムアウト(10秒) ─────────────→ (LISTENING_WAKE_WORDへ)
     ↓ VAD: 発話開始検知                                                │
PROCESSING ─────────────────────────── エラー ───────────────────────→ (LISTENING_WAKE_WORDへ)
     ↓ サーバーから応答受信                                             │
PLAYING_ANSWER ──────────────────────── 「次」コマンド ──────────────→ PROCESSING
     ↓ 再生完了                                                         │
LISTENING_FOLLOWUP ─────────── タイムアウト(30秒)/ウェイクワード ─────→ (LISTENING_WAKE_WORDへ)
     ↓ 「次」コマンド                                                   │
     └──────────────────────────────────────────────────────────────→ PROCESSING
```

### なぜ6ステートか——設計の検討過程

初期設計では3ステートを検討していた：

**3ステート案（却下）**

```
IDLE → LISTENING → PROCESSING
```

問題点：
- `ARMED`がない → ウェイクワード検出後、ユーザーが発話しているかどうかを判断する猶予がない。ウェイクワード直後の無音をVADが発話開始と誤判定した場合、空のPCMをサーバーに送ってしまう
- `PLAYING_ANSWER`がない → 音声再生中に次のウェイクワードを受け付けてしまい、会話が重なる
- `LISTENING_FOLLOWUP`がない → 回答後の「次」コマンドを受け付けるフェーズがなく、毎回ウェイクワードから言い直しが必要

**イベント駆動案（却下）**

イベントハンドラーのif-else連鎖で実装する案：

```kotlin
// NG: 状態管理をフラグで行うアンチパターン
var isArmed = false
var isProcessing = false
var isPlaying = false

fun handleAudioFrame(...) {
    if (!isArmed && !isProcessing && !isPlaying) {
        // IDLE状態の処理
    } else if (isArmed && !isProcessing) {
        // ARMED状態の処理
    } else if (isProcessing) {
        // 処理中
    }
    // 組み合わせが増えると指数的に複雑化する
}
```

フラグが増えると状態の組み合わせが指数的に増える。`isArmed && isProcessing && isPlaying`のような「あり得ない状態」が発生したときのハンドリングが困難になる。

**6ステートのsealedクラス実装**

```kotlin
sealed class AudioPipelineState {
    // 初期化中（モデルロード完了待ち）
    object Initializing : AudioPipelineState()

    // ウェイクワード待機中（通常の待機状態）
    object ListeningWakeWord : AudioPipelineState()

    // ウェイクワード検出済み（発話開始待ち）
    // タイムアウトタイムスタンプを保持
    data class Armed(val armedAtMs: Long = SystemClock.elapsedRealtime()) : AudioPipelineState()

    // 発話処理中（WebSocket送信 → サーバー処理 → TTS待ち）
    object Processing : AudioPipelineState()

    // 回答再生中
    // チャンクインデックスを保持（「次」コマンドで使う）
    data class PlayingAnswer(val currentChunkIndex: Int = 0) : AudioPipelineState()

    // フォローアップ待機中（回答後、「次」か再ウェイクワードを待つ）
    data class ListeningFollowup(val enteredAtMs: Long = SystemClock.elapsedRealtime()) : AudioPipelineState()
}
```

`when`式による状態遷移：

```kotlin
private fun handleVoskCommand(command: VoskCommand) {
    val current = currentState
    val next: AudioPipelineState? = when (current) {
        // ARMED状態でウェイクワードが来ても無視（冷却期間）
        is AudioPipelineState.Armed -> null

        is AudioPipelineState.ListeningWakeWord -> when (command) {
            VoskCommand.WAKE_WORD -> {
                startArmedTimeout()
                AudioPipelineState.Armed()
            }
            // ウェイクワード待機中に「次」「もう一度」が来ても無視
            else -> null
        }

        is AudioPipelineState.ListeningFollowup -> when (command) {
            VoskCommand.NEXT -> {
                startProcessing(followup = true)
                AudioPipelineState.Processing
            }
            VoskCommand.WAKE_WORD -> {
                // フォローアップ中に再びウェイクワード → 新しい質問
                startArmedTimeout()
                AudioPipelineState.Armed()
            }
            else -> null
        }

        // その他の状態ではコマンドを無視
        else -> null
    }

    next?.let { transitionTo(it) }
}
```

`sealed class`にすることで、`when`式の網羅性チェックがコンパイル時に行われる。新しい状態を追加したときに、処理漏れがコンパイルエラーとして検出される。

---

## 6. AudioRecordからSilero VADへのデータフロー

Silero VADはFloat32の正規化済み音声データを要求する。`AudioRecord`のデフォルト出力はInt16（ShortArray）だ。変換と分配のコードを示す：

```kotlin
class AudioRouter(
    private val voskEngine: VoskEngine,
    private val speechDetector: SileroVAD,
    private val sampleRate: Int = 16_000,
    private val channelConfig: Int = AudioFormat.CHANNEL_IN_MONO,
    private val audioFormat: Int = AudioFormat.ENCODING_PCM_16BIT,
) {
    private val bufferSize = AudioRecord.getMinBufferSize(
        sampleRate,
        channelConfig,
        audioFormat
    ).let { minSize ->
        // VADのフレームサイズ(512)の倍数に切り上げ、最低4096確保
        maxOf(minSize, 4096).let { size ->
            val frameBytes = 512 * 2  // 512 samples × 2 bytes/sample (Int16)
            ((size + frameBytes - 1) / frameBytes) * frameBytes
        }
    }

    private val audioRecord = AudioRecord(
        MediaRecorder.AudioSource.VOICE_COMMUNICATION,  // エコーキャンセラー有効
        sampleRate,
        channelConfig,
        audioFormat,
        bufferSize
    )

    fun startRecording() {
        // マイクスレッドは最高優先度で実行する
        val recordingThread = Thread {
            Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO)
            audioRecord.startRecording()

            val shortBuffer = ShortArray(512)  // VADの1フレーム = 512 samples

            while (!Thread.currentThread().isInterrupted) {
                val read = audioRecord.read(shortBuffer, 0, shortBuffer.size)
                if (read <= 0) continue

                // Vosk用: ShortArrayをそのまま渡す（型変換不要）
                voskEngine.processFrame(shortBuffer.copyOf(read))

                // Silero VAD用: Int16 → Float32に正規化（-1.0〜1.0の範囲に収める）
                val floatBuffer = FloatArray(read) { i ->
                    shortBuffer[i] / 32768.0f  // Int16最大値で除算
                }
                speechDetector.processFrame(floatBuffer)
            }
        }
        recordingThread.start()
    }
}
```

**`VOICE_COMMUNICATION`の選択理由**

`AudioSource`には複数の選択肢がある：

| AudioSource | 特徴 | 本システムでの評価 |
|-------------|------|-----------------|
| `MIC` | 生のマイク入力 | ノイズ対策は自前で行う必要あり |
| `VOICE_COMMUNICATION` | エコーキャンセラー・ノイズサプレッサー有効 | 採用（ハンズフリー想定） |
| `VOICE_RECOGNITION` | 音声認識向け最適化、AGC有効 | 音量正規化が行われSilero VADの閾値がずれる |
| `CAMCORDER` | ビデオ録画向け | 不適 |

`VOICE_RECOGNITION`はAutomatic Gain Control（AGC）が有効なため、音量が自動調整される。これはSilero VADの確率値に影響し、チューニングした閾値が意味をなさなくなる。`VOICE_COMMUNICATION`を選んでAGCをオフにした状態で使用している。

---

## 7. モデル初期化とライフサイクル管理

```kotlin
class VoskEngine(...) {

    private var model: Model? = null
    private var recognizer: Recognizer? = null

    fun initialize(onReady: () -> Unit, onError: (IOException) -> Unit) {
        // StorageServiceによるモデル展開はUIスレッドをブロックするためHandlerThread上で行う
        voskHandler.post {
            StorageService.unpack(
                context,
                config.modelAssetPath,   // "vosk-model-small-ja"
                MODEL_OUTPUT_DIR,        // 展開先ディレクトリ名
                { loadedModel ->
                    // コールバック引数は Model オブジェクト（Stringパスではない）
                    model = loadedModel
                    val grammar = buildGrammar()
                    recognizer = Recognizer(loadedModel, config.sampleRate.toFloat(), grammar)
                    mainHandler.post { onReady() }
                },
                { ioException ->
                    mainHandler.post {
                        onError(IOException("Voskモデル展開失敗: ${ioException.message}", ioException))
                    }
                }
            )
        }
    }

    fun updateGrammar(newConfig: WakeWordConfig) {
        // 文法の動的更新（ウェイクワード変更時）
        voskHandler.post {
            val newGrammar = buildGrammar()
            // Recognizerを再生成することで文法を更新する
            // (VoskはRecognizer.setGrammar()を提供していないため)
            recognizer?.close()
            recognizer = model?.let { m ->
                Recognizer(m, newConfig.sampleRate.toFloat(), newGrammar)
            }
        }
    }
}
```

**注意点：`StorageService.unpack()`のコールバック引数の型**

公式サンプルを見ると、コールバックの引数が `Model` オブジェクトだ。`String`（パス）ではない。Javaのサンプルコードをそのまま読むと型を間違えやすい：

```java
// Javaサンプル（公式より）
StorageService.unpack(this, "model-small-ja", "model",
    model -> {
        // model は org.vosk.Model オブジェクト
        setModel(model);
    }, exception -> { ... }
);
```

Kotlinでは型推論があるため実際にはコンパイルエラーで気づけるが、初見では`String`だと思い込みがちな箇所だ。

---

## 8. テスト戦略

ビジネスロジック（コマンド分類・状態遷移）は`internal`修飾子を使ってテスト可能にした：

```kotlin
// VoskEngineTest.kt
class VoskEngineTest {

    private val defaultConfig = WakeWordConfig(
        wakePhrase = "テスト",
        wakePhraseVariants = listOf("テスト", "てすと", "テスト "),
        nextCommandVariants = listOf("次", "つぎ"),
        repeatCommandVariants = listOf("もう一度", "もう 一度"),
    )

    @Test
    fun `wakeWordMatch_正規化前のスペース入りバリアントを正しく認識する`() {
        val engine = VoskEngine(defaultConfig, {}, {})
        // Voskが"テスト "（末尾スペース）を返してきた場合
        assertTrue(engine.isWakeWordMatch("テスト "))
    }

    @Test
    fun `wakeWordMatch_全角スペースを含む入力を正しく正規化する`() {
        val engine = VoskEngine(defaultConfig, {}, {})
        assertTrue(engine.isWakeWordMatch("テ　スト"))  // 全角スペース挿入
    }

    @Test
    fun `classifyCommand_ウェイクワードが最優先で分類される`() {
        val config = WakeWordConfig(
            wakePhrase = "次",  // ウェイクワードとNEXTコマンドが衝突するエッジケース
            nextCommandVariants = listOf("次"),
        )
        val engine = VoskEngine(config, {}, {})
        // ウェイクワードが優先されるべき
        assertEquals(VoskCommand.WAKE_WORD, engine.classifyCommand("次"))
    }

    @Test
    fun `classifyCommand_unkトークンはnullを返す`() {
        val engine = VoskEngine(defaultConfig, {}, {})
        assertNull(engine.classifyCommand("[unk]"))
    }

    @Test
    fun `buildGrammar_unkが末尾に含まれる`() {
        val engine = VoskEngine(defaultConfig, {}, {})
        val grammar = engine.buildGrammarForTest()
        assertTrue(grammar.contains("[unk]"))
        assertTrue(grammar.endsWith(", \"[unk]\"]"))
    }
}
```

状態遷移のテスト：

```kotlin
// AudioPipelineStateMachineTest.kt
class AudioPipelineStateMachineTest {

    @Test
    fun `LISTENING_WAKE_WORD状態でウェイクワードを受け取るとARMEDに遷移する`() {
        val sm = AudioPipelineStateMachine()
        sm.setState(AudioPipelineState.ListeningWakeWord)
        sm.handleVoskCommand(VoskCommand.WAKE_WORD)
        assertIs<AudioPipelineState.Armed>(sm.currentState)
    }

    @Test
    fun `LISTENING_WAKE_WORD状態でNEXTコマンドは無視される`() {
        val sm = AudioPipelineStateMachine()
        sm.setState(AudioPipelineState.ListeningWakeWord)
        sm.handleVoskCommand(VoskCommand.NEXT)
        // 状態が変化しないこと
        assertIs<AudioPipelineState.ListeningWakeWord>(sm.currentState)
    }

    @Test
    fun `ARMED状態でVAD発話開始イベントが来るとPROCESSINGに遷移する`() {
        val sm = AudioPipelineStateMachine()
        sm.setState(AudioPipelineState.Armed())
        sm.handleVadEvent(VadEvent.SpeechStarted)
        assertIs<AudioPipelineState.Processing>(sm.currentState)
    }
}
```

---

## まとめと振り返り

この実装を通じて得た教訓を3点に絞る。

**1. ライブラリのスレッドセーフ保証を最初に確認する**

Voskのドキュメントには "Not thread-safe" と記載がある。Kotlinに初めて触れた状態でCoroutineを当てにかけて実装し、数時間後にクラッシュで気づいた。使うライブラリのスレッドモデルは、実装を始める前に確認する。

**2. ONNXモデルのパラメータは本番想定環境でチューニングする**

デフォルト値（`threshold=0.5`）は汎用的な環境向けの値であり、介護施設のように常にテレビが流れている環境では誤検知が多発した。「実装が終わった」と思ってからパラメータチューニングに半日かかった。本番想定の騒音を最初から再現してチューニングする方が効率的だった。

**3. 状態機械は最初から設計する——後付けでは痛い目を見る**

最初のプロトタイプはフラグベースで実装した。機能追加のたびにフラグが増え、ある日「ウェイクワード後、VAD誤検知で無音データがサーバーに送られる」バグを追うのに丸1日かかった。6ステートの状態機械に書き直してからは、「どの状態でどのイベントを無視するか」がコードで一目瞭然になった。

---

## 参考リンク

- [Vosk API 公式ドキュメント](https://alphacephei.com/vosk/)
- [Vosk Android サンプル (alphacep/vosk-android-demo)](https://github.com/alphacep/vosk-android-demo)
- [Silero VAD GitHub (snakers4/silero-vad)](https://github.com/snakers4/silero-vad)
- [ONNX Runtime Android ドキュメント](https://onnxruntime.ai/docs/get-started/with-java.html)

---

> **関連記事（note）**
> [介護現場向けハンズフリー音声AIインカムを作った — 全体設計と技術選定の理由](https://note.com/yamashita_aidev/n/n7a3245355187)
> 実装の背景・設計思想・「なぜVoskを選んだか」はnote記事で詳しく書いています。
