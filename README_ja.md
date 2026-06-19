<table>
	<thead>
		<tr>
			<th style="text-align:center"><a href="README.md">English</a></th>
			<th style="text-align:center">日本語</th>
		</tr>
	</thead>
</table>

# QML Benchmark ノートブック - 詳細技術解説（Qiskit v2.x 対応）

`QML_benchmark.ipynb` は、6 種類の量子機械学習モデルを **CPU / GPU(cuStateVec) / 実機 IBM Quantum**
の各バックエンドで比較するベンチマークです。元は **Qiskit 1.x + Google Colab** 向けでしたが、
最新の **Qiskit v2.x** スタックへ全面移行し、**ローカルの Python 3.12 仮想環境(.venv)** で実行できます。

## 📚 目次
1. [Qiskit v2.x への移行点](#移行点)
2. [動作確認済み環境](#環境)
3. [全体アーキテクチャ](#全体アーキテクチャ)
4. [主要クラスの解説](#主要クラスの解説)
5. [量子機械学習アルゴリズム詳細](#量子機械学習アルゴリズム詳細)
6. [バックエンド管理機能](#バックエンド管理機能)
7. [結果出力とロギング](#結果出力)

<a id="移行点"></a>

## 🔄 Qiskit v2.x への移行点

Qiskit 2.x で削除・非推奨となった API を以下のように移行しています。

| 旧 (Qiskit 1.x) | 新 (Qiskit 2.x：本ノートブック) |
|---|---|
| `from qiskit.primitives import Sampler, Estimator`（V1） | **削除済み** → V2 プリミティブを使用 |
| `qiskit_aer.primitives.Sampler/Estimator`（V1） | `SamplerV2` / `EstimatorV2`（`options={'backend_options': {...}}`） |
| `ZZFeatureMap` / `ZFeatureMap` / `RealAmplitudes` / `EfficientSU2`（クラス） | `zz_feature_map` / `z_feature_map` / `real_amplitudes` / `efficient_su2`（関数） |
| `FidelityQuantumKernel(..., pass_manager=...)` | `ComputeUncompute(sampler=..., pass_manager=...)` を `fidelity=` に渡す |
| `channel='ibm_quantum'` | `channel='ibm_quantum_platform'`（+ CRN を `instance` に指定） |
| Colab の `!pip install` / `display()` | ローカル `.venv` ＋ `logging` ＋ `results/` への成果物出力 |

> Qiskit 2.x では関数ビルダー（`zz_feature_map` 等）は `BlueprintCircuit` ではなく通常の
> `QuantumCircuit` を返します。`.parameters` / `.num_qubits` などの参照はそのまま動作します。

<a id="環境"></a>

## 🖥️ 動作確認済み環境

- **OS**: AlmaLinux 9.7 ／ **CPU**: Intel Core i5-13600K（14 コア）／ **メモリ**: 128 GB DDR5
- **GPU**: NVIDIA RTX PRO 6000 Blackwell（96 GB, `sm_120`, ドライバ 580.x）／ **CUDA**: 13.0
- **Python**: 3.12（`.venv` 仮想環境）

| パッケージ | バージョン |
|---|---|
| qiskit | 2.4.2 |
| qiskit-machine-learning | 0.9.0 |
| qiskit-aer / qiskit-aer-gpu-cu11 | 0.17.2 |
| qiskit-ibm-runtime | 0.47.0 |
| torch | 2.11.0+cu128 |
| scikit-learn / matplotlib / pandas / numpy | 1.9.0 / 3.11.0 / 3.0.3 / 2.4.6 |

> **GPU ビルドについて**：公開されている最新の `qiskit-aer-gpu-cu11==0.17.2` は CPU 版 `qiskit-aer`
> 0.17.2 と同一バージョンで Qiskit 2.x と互換です。同梱の CUDA 11.8 / cuStateVec 1.6.0 は Blackwell
> より前の世代ですが、NVIDIA 580 ドライバ経由で `sm_120` 上でも正しく動作します
> （`AerSimulator().available_devices()` が `('CPU', 'GPU')` を返し、GPU シミュレーションが正しい結果を返す）。

### インストール（ローカル Python 3.12 venv）

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U qiskit qiskit-machine-learning qiskit-ibm-runtime \
               scikit-learn matplotlib pylatexenc pandas ipykernel nbconvert jupyter
pip install "qiskit-aer-gpu-cu11==0.17.2"                       # GPU(NVIDIA)
pip install torch --index-url https://download.pytorch.org/whl/cu128   # PyTorch(sm_120)
```

IBM 実機を使う場合は、リポジトリ直下（ノートブックの 1 つ上の階層）に `apikey.json` を置きます。

```json
{ "apikey": "<IBM_QUANTUM_API_KEY>", "crn": "<IBM_QUANTUM_PLATFORM_CRN>" }
```

`apikey` をトークン、`crn` を `instance` として `ibm_quantum_platform` チャネルで読み込みます。
ファイルが無い場合 IBM バックエンドはスキップされます。

<a id="全体アーキテクチャ"></a>

## 🏗️ 全体アーキテクチャ

```
┌─────────────────────────────────────────┐
│         QuantumModels クラス             │
│  (量子MLモデルの統合管理フレームワーク)    │
├─────────────────────────────────────────┤
│  ┌──────────┐ ┌──────────┐ ┌──────────┐│
│  │   VQC    │ │   QSVM   │ │   QNN    ││
│  └──────────┘ └──────────┘ └──────────┘│
│  ┌──────────┐ ┌──────────┐ ┌──────────┐│
│  │   QCNN   │ │   QRNN   │ │   QGAN   ││
│  └──────────┘ └──────────┘ └──────────┘│
├─────────────────────────────────────────┤
│      バックエンド抽象化（V2 プリミティブ）  │
├─────────────────────────────────────────┤
│  ┌──────────┐ ┌──────────┐ ┌──────────┐│
│  │CPU (Aer) │ │GPU(cuSV) │ │IBM 実機  ││
│  └──────────┘ └──────────┘ └──────────┘│
└─────────────────────────────────────────┘
```

### 主要な設定パラメータ（`CONFIG`）

```python
CONFIG = {
    'feature_dim': 2,
    'num_qubits': 2,            # 全モデル共通の固定量子ビット数
    'training_size': 80,
    'test_size': 20,
    'batch_size': 20,          # 並列処理用バッチサイズ
    'max_iterations': 6,       # デモ用に小さく設定（本格学習では増やす）
    'ibm_token':    '<apikey.json から読込>',
    'ibm_instance': '<apikey.json の CRN>',
    'ibm_channel': 'ibm_quantum_platform',
    'use_gpu': torch.cuda.is_available(),
    'num_parallel_jobs': 4,
    # --- IBM 実機専用ノブ（実機パスを高速・低コストに保つ）---
    'ibm_models': ['VQC', 'QSVM'],   # 実機で実行するモデル
    'ibm_max_iterations': 1,         # SPSA をステップ固定（自動較正を省略）
    'ibm_train_subset': 8,           # 変分モデルの実機学習サンプル数
    'ibm_qsvm_subset': 8,            # QSVM カーネルの実機学習サンプル数
}
```

> バックエンドは環境変数 `QML_BACKENDS` で上書き可能です
> （例：`QML_BACKENDS=cpu` / `cpu,gpu` / `cpu,gpu,ibm_quantum_platform`）。
> 未指定時は CPU＋（GPU があれば GPU）＋（`apikey.json` があれば IBM 実機）を自動選択します。

<a id="主要クラスの解説"></a>

## 📦 主要クラスの解説

### 1. QuantumModels クラス

すべての量子機械学習モデルを統合管理する中央コントローラーです。バックエンド種別に応じて
**V2 プリミティブ**（`SamplerV2` / `EstimatorV2`）を構築します。

```python
class QuantumModels:
    def __init__(self, data, backend_type='cpu'):
        self.data = data
        self.backend_type = backend_type
        self.results = {}
        self._setup_backend()

    def _setup_backend(self):
        if self.backend_type == 'cpu':
            self.sampler   = AerSampler(options={'backend_options': {'device': 'CPU', ...}})
            self.estimator = AerEstimator(options={'backend_options': {'device': 'CPU', ...}})
        elif self.backend_type == 'gpu':
            self.sampler   = AerSampler(options={'backend_options': {'device': 'GPU',
                                                 'cuStateVec_enable': True, ...}})
            self.estimator = AerEstimator(options={'backend_options': {'device': 'GPU', ...}})
        elif self.backend_type == 'ibm_quantum_platform':
            self.sampler      = SamplerV2(mode=self.backend)
            self.estimator    = EstimatorV2(mode=self.backend)
            self.pass_manager = generate_preset_pass_manager(backend=self.backend,
                                                             optimization_level=1)
```

**主要メソッド**: `train_vqc()` / `train_qsvm()` / `train_qnn()` / `train_qcnn()` /
`train_qrnn()` / `train_qgan()` / `run_all_models()`。

### 2. TrainingProgressCallback クラス

学習進行状況の監視・記録を担当します。SPSA（`(nfev, params, fval, ...)`）と VQC（`(weights, fval)`）
の両方の引数形式に対応し、損失値を `logging` 経由で出力します。

<a id="量子機械学習アルゴリズム詳細"></a>

## 🧮 量子機械学習アルゴリズム詳細

### 1. VQC (Variational Quantum Classifier)

```python
feature_map = zz_feature_map(feature_dimension=CONFIG['feature_dim'], reps=2, entanglement='linear')
ansatz      = real_amplitudes(num_qubits=CONFIG['num_qubits'], reps=3)

vqc = VQC(feature_map=feature_map, ansatz=ansatz,
          optimizer=SPSA(maxiter=CONFIG['max_iterations'], callback=callback),
          sampler=self.sampler,              # V2 Sampler
          callback=callback)
# IBM 実機では pass_manager=self.pass_manager を追加
```

- **特徴マップ / アンザッツ**は関数ビルダーで生成（Qiskit 2.x）。
- シミュレータでは `batch_size` 単位で `fit` を繰り返すバッチ学習を実施。

### 2. QSVM (Quantum Support Vector Machine)

```python
feature_map = z_feature_map(feature_dimension=CONFIG['feature_dim'], reps=2)

# CPU: 既定（厳密 statevector）フィデリティが最速
quantum_kernel = FidelityQuantumKernel(feature_map=feature_map, evaluate_duplicates='off_diagonal')

# GPU / IBM 実機: フィデリティにバックエンド固有の V2 Sampler を割り当てる
fidelity = ComputeUncompute(sampler=self.sampler, pass_manager=self.pass_manager)  # IBM のみ pass_manager
quantum_kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity,
                                       evaluate_duplicates='off_diagonal')

qsvc = QSVC(quantum_kernel=quantum_kernel)
```

> **重要**：Qiskit-ML 0.9 の `FidelityQuantumKernel` は `pass_manager` 引数を持ちません。
> 実機向けのトランスパイルは `ComputeUncompute` 側に `pass_manager` を渡して実現します。

### 3. QNN (Quantum Neural Network)

```python
qc = QuantumCircuit(CONFIG['num_qubits'])
qc.compose(zz_feature_map(CONFIG['num_qubits']), inplace=True)
qc.compose(real_amplitudes(CONFIG['num_qubits'], reps=3), inplace=True)

qnn = EstimatorQNN(circuit=qc,
                   observables=SparsePauliOp.from_list([("Z"*CONFIG['num_qubits'], 1)]),
                   input_params=feature_map.parameters,
                   weight_params=ansatz.parameters,
                   estimator=self.estimator)        # V2 Estimator（IBM では pass_manager も指定）

classifier = NeuralNetworkClassifier(qnn, optimizer=SPSA(maxiter=CONFIG['max_iterations']), ...)
```

### 4. QCNN (Quantum Convolutional Neural Network)

最低 4 量子ビットで畳み込み層・プーリング層を構成し、`EstimatorQNN` + SPSA で学習します
（`Parameter` / `ParameterVector` を直接使用するため API 変更の影響はありません）。

### 5. QRNN (Quantum Recurrent Neural Network)

`num_qubits + 1` 量子ビット（隠れ状態用 1 ビット）で再帰セルを構成し、`EstimatorQNN` + SPSA で学習します。

### 6. QGAN (Quantum Generative Adversarial Network)

```python
generator = QuantumCircuit(num_qubits)              # 量子生成器
generator.compose(efficient_su2(num_qubits, reps=2), inplace=True)
gen_qnn  = SamplerQNN(circuit=generator, input_params=latent, weight_params=...,
                      sampler=self.sampler)          # V2 Sampler
gen_torch = TorchConnector(gen_qnn)                  # PyTorch 連携
# 識別器は古典 NN（nn.Module）。GPU バックエンド時は識別器とテンソルを CUDA(sm_120) へ転送。
```

- 実機（`ibm_quantum_platform`）では回路評価が多すぎるため QGAN はスキップします。

<a id="バックエンド管理機能"></a>

## 🖥️ バックエンド管理機能

### CPU バックエンド

```python
AerSimulator(method='statevector', device='CPU',
             max_parallel_threads=CONFIG['num_parallel_jobs'],
             max_parallel_experiments=CONFIG['num_parallel_jobs'])
```

### GPU バックエンド（cuStateVec）

```python
def gpu_available():
    return 'GPU' in AerSimulator().available_devices()

# フォールバックなし：GPU が無ければ例外を送出（隠蔽しない）
AerSimulator(method='statevector', device='GPU', cuStateVec_enable=True,
             batched_shots_gpu=True)
```

### IBM Quantum バックエンド（実機）

```python
service = QiskitRuntimeService(channel='ibm_quantum_platform',
                               token=CONFIG['ibm_token'],
                               instance=CONFIG['ibm_instance'])   # CRN
backend = service.least_busy(simulator=False, operational=True)
sampler   = SamplerV2(mode=backend)
estimator = EstimatorV2(mode=backend)
pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=1)
```

<a id="結果出力"></a>

## 🔧 結果出力とロギング

実行のたびに `results/` ディレクトリを**作り直し**、`print` の代わりに Python `logging`
（コンソール＋ファイル）で進捗を記録します。

```
results/
├── benchmark.log                       # 実行ログ（全体）
├── dataset_distribution.png            # データセット散布図
├── vqc_circuit_*.png / qcnn_circuit_*.png / qrnn_circuit_*.png / qgan_training_*.png
├── quantum_ml_comparison_detailed.png  # 精度／時間／損失／総合比較
├── quantum_ml_results.json             # 機械可読の結果
└── summary.csv, summary.txt            # 結果サマリ表
```

## 📊 ベンチマーク結果（CPU vs GPU・本環境）

`max_iterations=6`、学習 80／テスト 20 の代表的な結果です。QNN/QCNN/QRNN の精度が低いのは、
デモを数秒で完了させるため SPSA の反復を 3〜6 回に抑えているためです。

| モデル | CPU 精度 | CPU 時間(s) | GPU 精度 | GPU 時間(s) |
|------|:------:|:----:|:------:|:----:|
| VQC  | 0.50 | 11.3 | 0.83 | 18.1 |
| QSVM | 0.85 | 9.4  | 0.88 | 23.7 |
| QNN  | 0.45 | 0.7  | 0.28 | 2.7  |
| QCNN | 0.23 | 0.4  | 0.28 | 2.0  |
| QRNN | 0.28 | 0.5  | 0.25 | 1.8  |
| QGAN | 0.79† | 19.2 | 0.74† | 27.5 |

† QGAN は生成モデルのため分類精度はありません。値は**生成品質スコア** = `1 − TVD`（生成分布と
固定目標分布の全変動距離）で `[0,1]`・高いほど良く、他モデルの精度と直接比較できます
（`tvd` / `kl_divergence` / `final_loss` は `quantum_ml_results.json` に記録）。QNN/QCNN/QRNN の
精度が低いのは SPSA 反復を 3〜6 回に抑えたデモ設定のためです。

> **本スケールでは GPU は CPU より遅い**：全モデルが 2〜4 量子ビットのため GPU の起動オーバーヘッドが
> 支配的になります（速度比 0.2〜0.7 倍）。GPU が有利になるのは大きな状態ベクトルの場合で、別途
> 24 量子ビットの `efficient_su2` 回路では **GPU が約 2.4 倍高速**（0.33 秒 vs 0.81 秒）でした。
> `num_qubits` を増やすと GPU の優位性が現れます。

### IBM 実機（リアルハードウェア）

`ibm_quantum_platform` チャネル経由で実機実行を検証しました（観測バックエンド：`ibm_fez` /
`ibm_marrakesh` / `ibm_kingston`）。

| モデル | バックエンド | 実機精度 | 実機時間(s) |
|------|---------|:------:|:----:|
| VQC  | ibm_fez | 0.60 | 57.9 |
| QSVM | ibm_fez | 0.675 | 42.9 |

**なぜ 2 モデル・縮小設定なのか**：実機では SPSA の各評価が個別の投入ジョブになります。本キーは
**IBM Open プラン**で **Session を使用できない**ため、ジョブを連続実行できず 1 件ごとに数分の
キュー/往復が発生します。設定そのままのフルベンチマークは **VQC だけで約 1 時間**を要し、
**3 時間の QPU クォータ**を使い切る見込みでした。そこで実機パスを高速かつ忠実な疎通確認に保つため、
`CONFIG` に実機専用のノブを用意しています（上記アーキテクチャ節の `CONFIG` 参照）：
代表 2 モデル（`ibm_models` = `VQC`(変分) / `QSVM`(カーネル)）のみを実行し、SPSA のステップ幅を固定して
約 25 回の自動キャリブレーションを省略（`ibm_max_iterations=1`）、学習は少数の層化サブセット
（`ibm_train_subset` / `ibm_qsvm_subset = 8`）で行います。これにより両モデルが実機で**合計約 10 分**で
完了します。実機で他モデルや大きな問題を動かす場合は、これらの `CONFIG` を編集してください
（有料プランではコードが自動的に `Session` を使い、往復が大幅に高速化します）。

## 📄 ライセンス

本プロジェクトは MIT ライセンスです（[LICENSE](LICENSE) を参照）。
