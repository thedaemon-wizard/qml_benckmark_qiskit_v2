<table>
	<thead>
		<tr>
			<th style="text-align:center"><a href="README.md">English</a></th>
			<th style="text-align:center">日本語</th>
		</tr>
	</thead>
</table>
# QML Benchmark ノートブック - 詳細技術解説

## 📚 目次
1. [全体アーキテクチャ](#全体アーキテクチャ)
2. [主要クラスの解説](#主要クラスの解説)
3. [量子機械学習アルゴリズム詳細](#量子機械学習アルゴリズム詳細)
4. [バックエンド管理機能](#バックエンド管理機能)
5. [ユーティリティ関数](#ユーティリティ関数)

## 🏗️ 全体アーキテクチャ

### システム構成図
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
│         バックエンド抽象化レイヤー         │
├─────────────────────────────────────────┤
│  ┌──────────┐ ┌──────────┐ ┌──────────┐│
│  │CPU (Aer) │ │GPU(cuQ)  │ │IBM Cloud ││
│  └──────────┘ └──────────┘ └──────────┘│
└─────────────────────────────────────────┘
```

### 主要な設定パラメータ
```python
CONFIG = {
    'n_qubits': 4,           # 量子ビット数
    'n_samples': 100,        # データセットサイズ
    'max_iterations': 20,    # 最大イテレーション数
    'batch_size': 20,        # バッチサイズ（並列処理用）
    'use_gpu': True,         # GPU使用フラグ
    'ibm_token': '',         # IBM Quantum APIトークン
}
```

## 📦 主要クラスの解説

### 1. QuantumModels クラス

**役割**: すべての量子機械学習モデルを統合管理する中央コントローラー

```python
class QuantumModels:
    """量子機械学習モデルの統合管理クラス"""
    
    def __init__(self, data, backend_type='cpu', config=None):
        """
        Args:
            data: (X_train, X_test, y_train, y_test) のタプル
            backend_type: 'cpu', 'gpu', 'ibm_quantum' のいずれか
            config: 設定辞書
        """
```

**主要メソッド**:
- `_setup_backend()`: バックエンドの初期化と設定
- `_create_batches()`: データのバッチ分割（並列処理用）
- `train_vqc()`: VQCアルゴリズムの実行
- `train_qsvm()`: QSVMアルゴリズムの実行
- `train_qnn()`: QNNアルゴリズムの実行
- `train_qcnn()`: QCNNアルゴリズムの実行
- `train_qrnn()`: QRNNアルゴリズムの実行
- `train_qgan()`: QGANアルゴリズムの実行
- `run_all_models()`: すべてのモデルを実行

### 2. TrainingProgressCallback クラス

**役割**: 学習進行状況の監視とロギング

```python
class TrainingProgressCallback:
    """学習進行状況監視用コールバッククラス"""
    
    def __init__(self, model_name, backend_type):
        self.model_name = model_name
        self.backend_type = backend_type
        self.iteration = 0
        self.start_time = time.time()
```

**機能**:
- リアルタイム学習進捗表示
- 損失値の追跡
- 実行時間の測定
- イテレーション毎の状態記録

## 🧮 量子機械学習アルゴリズム詳細

### 1. VQC (Variational Quantum Classifier)

**概要**: ハイブリッド量子-古典分類器。変分量子回路を使用して分類タスクを実行。

**実装詳細**:
```python
def train_vqc(self):
    # 特徴マップ: ZZFeatureMap
    feature_map = ZZFeatureMap(
        feature_dimension=feature_dimension,
        reps=2,  # 繰り返し数
        entanglement='linear'  # エンタングルメント構造
    )
    
    # アンザッツ: RealAmplitudes
    ansatz = RealAmplitudes(
        num_qubits=n_qubits,
        reps=2,  # 回路の深さ
        entanglement='linear'
    )
    
    # オプティマイザ: SPSA（ノイズ耐性）
    optimizer = SPSA(
        maxiter=CONFIG['max_iterations'],
        learning_rate=0.01,
        perturbation=0.1
    )
```

**特徴**:
- **並列バッチ処理**: 複数のデータポイントを同時処理
- **ノイズ耐性**: SPSAオプティマイザによる勾配フリー最適化
- **効率的なエンタングルメント**: 線形エンタングルメントによる計算効率化

### 2. QSVM (Quantum Support Vector Machine)

**概要**: 量子カーネル法を使用したSVM。古典的なSVMに量子特徴マップを組み合わせる。

**実装詳細**:
```python
def train_qsvm(self):
    # 量子特徴マップ
    feature_map = ZZFeatureMap(
        feature_dimension=feature_dimension,
        reps=2,
        entanglement='full'  # 完全エンタングルメント
    )
    
    # 量子カーネル
    quantum_kernel = FidelityQuantumKernel(
        feature_map=feature_map,
        fidelity=ComputeUncompute(sampler=sampler),
        evaluate_duplicates='off_diagonal'  # カーネル計算最適化
    )
    
    # QSVC分類器
    qsvc = QSVC(quantum_kernel=quantum_kernel)
```

**特徴**:
- **並列カーネル計算**: カーネル行列の要素を並列計算
- **フィデリティベース**: 量子状態の重なりを利用
- **最適化されたカーネル評価**: 重複計算の回避

### 3. QNN (Quantum Neural Network)

**概要**: 量子回路を用いたニューラルネットワーク。古典的なNNの構造を量子版に拡張。

**実装詳細**:
```python
def train_qnn(self):
    # パラメータ化された量子回路
    ansatz = RealAmplitudes(
        CONFIG['num_qubits'], 
        reps=3  # より深い回路
    )
    
    # EstimatorQNN
    qnn = EstimatorQNN(
        circuit=ansatz,
        input_params=[],  # 入力パラメータ
        weight_params=ansatz.parameters,  # 重みパラメータ
        input_gradients=False
    )
    
    # ニューラルネットワーク分類器
    classifier = NeuralNetworkClassifier(
        neural_network=qnn,
        optimizer=SPSA(maxiter=CONFIG['max_iterations']),
        callback=callback
    )
```

**特徴**:
- **深い量子回路**: reps=3による表現力の向上
- **Estimatorベース**: 期待値計算の効率化
- **SPSAオプティマイザ**: ノイズ環境での安定した学習

### 4. QCNN (Quantum Convolutional Neural Network)

**概要**: 量子畳み込みニューラルネットワーク。画像認識タスクに適した構造。

**実装詳細**:
```python
def train_qcnn(self):
    # 固定量子ビット数での実装
    n_qubits_fixed = 4
    
    # 畳み込み層とプーリング層の構築
    qc = QuantumCircuit(n_qubits_fixed)
    
    # エンコーディング層
    for i in range(n_qubits_fixed):
        qc.ry(params[i], i)
    
    # 畳み込み層
    for i in range(n_qubits_fixed - 1):
        qc.cx(i, i + 1)
        qc.rz(params[n_qubits_fixed + i], i + 1)
    
    # プーリング層（量子測定）
    # 部分的な測定により次元削減
```

**特徴**:
- **固定量子ビット**: リソース制約下での効率的な実装
- **階層的構造**: 畳み込み層とプーリング層の組み合わせ
- **パラメータ共有**: CNNの特徴を量子版で実現

### 5. QRNN (Quantum Recurrent Neural Network)

**概要**: 量子回路を用いた再帰型ニューラルネットワーク。時系列データ処理に特化。

**実装詳細**:
```python
def train_qrnn(self):
    # 固定量子ビット数
    n_qubits_fixed = 4
    
    # 再帰的な量子回路構造
    qc = QuantumCircuit(n_qubits_fixed)
    
    # 時系列ステップごとの処理
    for step in range(sequence_length):
        # 入力エンコーディング
        encode_input(qc, input_data[step])
        
        # 再帰的接続（前の状態との相互作用）
        apply_recurrent_layer(qc, hidden_state)
        
        # 状態更新
        hidden_state = measure_and_update(qc)
```

**特徴**:
- **メモリ効果**: 量子状態による情報保持
- **時系列処理**: シーケンシャルデータの扱い
- **固定アーキテクチャ**: 効率的な実装

### 6. QGAN (Quantum Generative Adversarial Network)

**概要**: 量子回路を用いたGAN。生成器と識別器の敵対的学習。

**実装詳細**:
```python
def train_qgan(self):
    # 生成器（量子回路）
    generator_circuit = QuantumCircuit(n_qubits)
    generator_circuit.append(
        RealAmplitudes(n_qubits, reps=2),
        range(n_qubits)
    )
    
    # 識別器（古典ニューラルネットワーク）
    class Discriminator(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
    
    # 敵対的学習ループ
    for epoch in range(num_epochs):
        # 生成器の学習
        train_generator()
        # 識別器の学習
        train_discriminator()
```

**特徴**:
- **ハイブリッド構造**: 量子生成器 + 古典識別器
- **敵対的学習**: MinMax最適化
- **バッチ処理**: 効率的な学習

## 🖥️ バックエンド管理機能

### 1. CPU バックエンド (`get_cpu_backend`)

```python
def get_cpu_backend():
    """CPU シミュレータバックエンド"""
    backend = AerSimulator(
        method='statevector',  # 状態ベクトル法
        max_parallel_threads=multiprocessing.cpu_count(),
        max_parallel_experiments=CONFIG['batch_size'],
        statevector_parallel_threshold=12
    )
    return backend
```

**最適化ポイント**:
- マルチスレッド並列化
- バッチ並列実行
- 状態ベクトル並列化閾値の調整

### 2. GPU バックエンド (`get_gpu_backend`)

```python
def get_gpu_backend():
    """GPU 加速バックエンド（cuQuantum）"""
    backend = AerSimulator(
        method='statevector',
        device='GPU',  # GPU使用
        cuStateVec_enable=True  # cuQuantum有効化
    )
    return backend
```

**最適化ポイント**:
- NVIDIA cuQuantumライブラリの活用
- GPU並列計算
- 大規模量子回路のシミュレーション

### 3. IBM Quantum バックエンド (`get_ibm_backend`)

```python
def get_ibm_backend(service=None):
    """IBM Quantum プラットフォーム"""
    service = QiskitRuntimeService(
        token=CONFIG['ibm_token'],
        channel='ibm_quantum'
    )
    backend = service.get_backend('ibm_qasm_simulator')
    return backend
```

**特徴**:
- 実量子ハードウェアへのアクセス
- クラウド実行
- ノイズモデルの考慮

## 🔧 ユーティリティ関数

### 1. データ準備関数 (`prepare_data`)

```python
def prepare_data():
    """量子データセットの準備"""
    # データ生成またはロード
    training_features, training_labels = generate_dataset()
    test_features, test_labels = generate_test_set()
    
    # 正規化
    features = normalize_features(features)
    
    # 量子回路用に次元調整
    features = reduce_dimensions(features, n_qubits)
    
    return X_train, X_test, y_train, y_test
```

**処理内容**:
- データの生成/ロード
- 特徴量の正規化
- 次元削減（量子ビット数に合わせる）
- 訓練/テストデータの分割

### 2. バッチ作成関数 (`_create_batches`)

```python
def _create_batches(self, features, labels):
    """並列処理用バッチ作成"""
    n_samples = features.shape[0]
    batch_size = CONFIG['batch_size']
    
    batches = []
    for i in range(0, n_samples, batch_size):
        batch_features = features[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        batches.append((batch_features, batch_labels))
    
    return batches
```

**機能**:
- データのバッチ分割
- メモリ効率的な処理
- 並列実行の準備

### 3. 結果可視化関数 (`plot_results`)

```python
def plot_results(all_results):
    """結果の可視化"""
    # 精度比較グラフ
    plot_accuracy_comparison()
    
    # 実行時間比較
    plot_runtime_comparison()
    
    # 収束曲線
    plot_convergence_curves()
    
    # ヒートマップ
    plot_performance_heatmap()
```

**出力内容**:
- モデル別精度比較
- バックエンド別実行時間
- 学習曲線
- 総合パフォーマンスマップ

## 📊 パフォーマンス最適化技術

### 1. 並列化戦略
- **データ並列**: バッチ単位での並列処理
- **回路並列**: 複数の量子回路を同時実行
- **GPU並列**: cuQuantumによる大規模並列化

### 2. メモリ管理
- **バッチ処理**: メモリ使用量の制御
- **遅延評価**: 必要時のみ計算実行
- **ガベージコレクション**: 明示的なメモリ解放

### 3. 量子回路最適化
- **トランスパイル最適化**: 回路の簡約化
- **ゲート融合**: 連続するゲートの統合
- **並列ゲート実行**: 独立したゲートの同時実行

## 🎯 使用上の推奨事項

### 初心者向け設定
```python
CONFIG = {
    'n_qubits': 2,      # 少ない量子ビット
    'n_samples': 50,    # 小規模データセット
    'max_iterations': 5, # 少ないイテレーション
    'batch_size': 10,   # 小さいバッチサイズ
}
```

### 本格的な実験向け設定
```python
CONFIG = {
    'n_qubits': 6,       # より多い量子ビット
    'n_samples': 1000,   # 大規模データセット
    'max_iterations': 50, # 十分なイテレーション
    'batch_size': 50,    # 大きいバッチサイズ
}
```

### GPU利用時の推奨設定
```python
CONFIG = {
    'use_gpu': True,
    'batch_size': 100,   # GPUメモリに応じて調整
    'n_qubits': 8,       # GPUで扱える範囲で最大化
}
```