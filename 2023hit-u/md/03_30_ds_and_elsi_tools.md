## 6. 線形回帰

- 線形回帰を爆発的に有用にしたコンピュータ
- 大量のデータから相関性を導くことは容易
- 因果関係と相関関係
- RCT
- 次元の罠

## 6.1 線形代数
線形代数は20世紀後半に重要分野として基礎教育課程にとりいれられた。(1970年代以後).
コンピュータの登場が大量の統計情報の収集とその分析を可能にした。
線形回帰、線形計画法、線形予測はコンピュータ時代がもたらしたデータ処理手法の三種の神器。

線形代数は以下のような体系だった。

### ベクトル
$$
\mathbf{v} = \begin{pmatrix}
v_1 \\
v_2 \\
\vdots \\
v_n
\end{pmatrix}
$$

### 行列
$$
\mathbf{A} = \begin{pmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{pmatrix}
$$

### 行列の積
$$
\mathbf{C} = \mathbf{A} \mathbf{B}
$$

$$
\mathbf{C}_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj}
$$

### 行列の逆行列
$$
\mathbf{A}^{-1} \mathbf{A} = \mathbf{I}
$$

### 行列式
$$
\det(\mathbf{A}) = \sum_{\sigma \in S_n} (\text{sgn}(\sigma) \prod_{i=1}^{n} a_{i,\sigma(i)})
$$

### 固有値と固有ベクトル
$$
\mathbf{A} \mathbf{v} = \lambda \mathbf{v}
$$


## 6.2. 線形回帰

### 線形回帰の基本式
$$
y = \beta_0 + \beta_1 x + \epsilon
$$

### 最小二乗法によるパラメータ推定
$$
\hat{\beta} = (X^T X)^{-1} X^T y
$$

### 決定係数 (R^2)
$$
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
$$

## 6.3 線形予測

### 線形予測の基本式
線形予測の基本式は、以下のように表されます。

$$
\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n
$$

ここで、
$$
\hat{y}    は予測される値
 $$
 $$
\beta_0 は切片
$$
$$
\beta_1, \beta_2, \ldots, \beta_n は各説明変数 x_1, x_2, \ldots, x_n に対応
$$

## 6.4 線形計画法

### 線形計画法の基本式

線形計画法の目的は、以下の線形関数を最大化または最小化することです。

$$
\text{maximize or minimize } \mathbf{c}^T \mathbf{x}
$$

ここで、
$$
- \mathbf{c} は係数ベクトル
$$
$$
- \mathbf{x} は変数ベクトル
$$

### 制約条件

線形計画法には、以下のような制約条件が付きます。

$$
\mathbf{A} \mathbf{x} \leq \mathbf{b}
$$

ここで、
$$
- \mathbf{A} は係数行列
$$
$$
- \mathbf{b} は定数ベクトル
$$

### 非負制約

変数は非負であることが求められます。

$$
\mathbf{x} \geq 0
$$

### 例

例えば、以下のような線形計画問題を考えます。

$$
\text{maximize } z = 3x_1 + 2x_2
$$

制約条件は以下の通りです。

$$
\begin{cases}
2x_1 + x_2 \leq 20 \\
4x_1 + 3x_2 \leq 42 \\
x_1 + 2x_2 \leq 18 \\
x_1, x_2 \geq 0
\end{cases}
$$

## 7. LLM の発展を導いたwordvecと共起行列の処理

### Word2Vec
Word2Vecは、単語をベクトル空間に埋め込むための手法です。以下は、Skip-gramモデルの数式です。

#### Skip-gramモデル
Skip-gramモデルは、ある単語 \( w_t \) を与えられたときに、その周辺の単語 \( w_{t-k}, \ldots, w_{t+k} \) を予測するモデルです。

$$
\max \sum_{t=1}^{T} \sum_{-k \leq j \leq k, j \neq 0} \log P(w_{t+j} | w_t)
$$

ここで、\( P(w_{t+j} | w_t) \) は条件付き確率であり、以下のように定義されます。

$$
P(w_{t+j} | w_t) = \frac{\exp(\mathbf{v}_{w_{t+j}} \cdot \mathbf{v}_{w_t})}{\sum_{w=1}^{W} \exp(\mathbf{v}_w \cdot \mathbf{v}_{w_t})}
$$

### 共起行列
共起行列は、単語の共起情報を行列形式で表現したものです。以下は、共起行列 \( \mathbf{C} \) の定義です。

#### 共起行列の定義
$$
共起行列  \mathbf{C}  の要素
$$

$$ C_{ij} = \text{count}(w_i, w_j) $$


### Transformer

### Transformerモデルの数式

#### 入力埋め込み
入力シーケンス \( x \) を埋め込みベクトル \( E \) に変換します。
$$
E = \text{Embedding}(x)
$$

#### ポジショナルエンコーディング
位置情報を埋め込みベクトルに追加します。
$$
E' = E + \text{PositionalEncoding}(x)
$$

#### 自己注意機構 (Self-Attention)
自己注意機構は、クエリ \( Q \)、キー \( K \)、バリュー \( V \) を用いて計算されます。
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

#### マルチヘッド注意機構 (Multi-Head Attention)
複数の自己注意機構を並列に計算し、それらを結合します。
$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \ldots, \text{head}_h)W^O
$$
ここで、各ヘッドは以下のように計算されます。
$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

#### フィードフォワードネットワーク (Feed-Forward Network)
各位置に対して独立に適用されるフィードフォワードネットワークです。
$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

#### エンコーダ層 (Encoder Layer)
エンコーダ層は、マルチヘッド注意機構とフィードフォワードネットワークから構成されます。
$$
\text{EncoderLayer}(x) = \text{FFN}(\text{MultiHead}(x, x, x))
$$

#### デコーダ層 (Decoder Layer)
デコーダ層は、マルチヘッド注意機構、エンコーダ-デコーダ注意機構、フィードフォワードネットワークから構成されます。
$$
\text{DecoderLayer}(x, \text{enc\_output}) = \text{FFN}(\text{MultiHead}(x, x, x) + \text{MultiHead}(x, \text{enc\_output}, \text{enc\_output}))
$$

## 8. AIとシャーロックリスク

- 膨大な情報が収集できるようになった
- AI(DL) による分析手法は絶大な効果を示している
- 予想外の情報が取得可能になる可能性がある

[医療情報ビッグデータ分析におけるシャーロックリスクの考察とその形式化](./pdf/IPSJ-EIP20088015.pdf)


[大量ウェアラブルデバイスと大規模生体情報時代におけるAI機械学習のシャーロック問題への対策としてのプライバシーエージェントのDockerベクトル化](./pdf/IPSJ-EIP22095001.pdf)

## 9. まとめ

統計、データサイエンス、AIは不確定性を扱う体系
現代の生活で得られる情報の多くが、統計、データサイエンス、AIの理科を必要とする
理論だけでなく、実例との関係をよく理解することが大事

## 推奨参考書

- [統計学の極意](https://www.amazon.co.jp/dp/4794226926)

  元・英国統計学会会長による統計学入門書の最新決定版。
- [統計学を哲学する](https://www.amazon.co.jp/dp/4815810036/)

   書名には統計学とあるが、実際は統計、データサイエンス、人工知能までカバーしている。これらの「計算の意味」を具体的にわかりやすく解説している。

