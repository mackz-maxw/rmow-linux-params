# rmow-linux-params

Linux 系统参数自适应优化的最小可行实现（Rust）。包含：
- 基于贝叶斯优化的核心决策引擎 [`optimizer::BayesOptimizer`](src/optimizer.rs)，利用高斯过程回归对黑盒性能函数建模
- 自动化闭环（最小版本）：参数生成 →（Dry-run）配置施加 → 性能采样 → 反馈优化器，入口见 [src/main.rs](src/main.rs)
- 数据可视化（最小版本）：二维参数空间的 ASCII 热力图，辅助观察参数-性能关联

## 背景与目标
研发一个智能化性能优化平台，通过数据驱动的方式，自动、持续地寻找并施加最优系统参数配置，以最大化硬件资源利用率和系统吞吐量。目标系统可扩展到 Nginx 等关键组件，期望在标准压力测试下的关键指标（如 BIC 评分）提升约 30%。

## 数学概述
- 黑盒目标函数 $f(x)$（如 QPS/延迟/CPU 的综合指标）通过高斯过程回归（GP）建模：

  $$
  f(\cdot) \sim \mathcal{GP}\left(0, k(\cdot, \cdot)\right),\quad
  k(x, x') = \sigma_f^2 \exp\left(-\frac{1}{2}\sum_i \frac{(x_i - x'_i)^2}{\ell^2}\right)
  $$

- 后验预测（给定观测 $(X, y)$）：

  $$
  \mu(x_*) = k_*^\top K^{-1} y,\quad
  \sigma^2(x_*) = k(x_*, x_*) - k_*^\top K^{-1} k_*
  $$

- 采集函数（最小实现使用 UCB）：

  $$
  a_{\mathrm{UCB}}(x) = \mu(x) + \kappa \cdot \sigma(x)
  $$

  其中 $\kappa$ 控制探索-利用的权衡。

核心实现见 [`optimizer::GaussianProcess`](src/optimizer.rs) 与 [`optimizer::BayesOptimizer`](src/optimizer.rs)。

## 快速开始
- 构建与运行：
  ```
  cargo run
  ```
- 输出包含每次迭代的得分与最佳参数；当维度为 2 时，终端打印 20x20 的 ASCII 热力图（越亮越好）。

## 项目结构
- [src/main.rs](src/main.rs)：最小自动化闭环与热力图
- [src/optimizer.rs](src/optimizer.rs)：高斯过程 + UCB 采集函数
- [src/params.rs](src/params.rs)：参数空间定义与归一化映射
- [Cargo.toml](Cargo.toml)：依赖与构建配置

## 开发计划（里程碑）
- M0 最小可行（当前）
  - GP + UCB
  - Dry-run 施配与合成目标函数
  - 2D ASCII 热力图
- M1 实机指标接入
  - 采集器：QPS、P99 延迟、CPU 利用率（/proc、eBPF 或压测端点）
  - 真实施配器：sysctl、tuned、cgroup、内核参数（安全回滚/回退）
  - 采集函数：EI/PI/UCB 可选，支持批量候选
- M2 可视化看板
  - 实时热力图与趋势图（Web UI + WebSocket）
  - 试验管理、配置档案与可解释性分析
- M3 生产闭环
  - 无人值守在线优化、灰度与安全保护（阈值、熔断、回退）
  - 适配 Nginx/Redis/Kafka 等场景的参数模板
- M4 评测与度量
  - 集成标准压力测试，统一 BIC/吞吐/延迟指标汇总
  - 成果复现实验与对比报告（目标：关键指标提升约 30%）

## 说明
- 当前版本不会修改系统参数，仅打印拟施加的配置（Dry-run）。
- 参数空间与目标函数在 [`params::ParameterSpace`](src/params.rs) 与 [src/main.rs](src/main.rs) 中可按需扩展。

<details>
<summary>

## 高斯过程详解
</summary>
使用向奶奶解释法让ai工具生成的解释，方便理解

### 一、这段代码总体在做什么？
> **给我看过的一些“样本点”（输入和结果），用一种聪明的数学方法去“猜”还没见过的新点的结果，而且还能告诉我自己猜得有多准、心里有多没底。**  
> 这套聪明方法就叫“高斯过程（Gaussian Process）”。

再具体一点：

- `GaussianProcess`：这是“高斯过程模型”，负责：
  - 把你已经看到的点（输入 `x` 和输出 `y`）记下来；
  - 根据这些点，算出一个“整体的规律”；
  - 当来一个新问题 `x*` 时，给出：
    - 预测的结果 `mu`（均值，相当于“我觉得大概是多少”）；
    - 预测的不确定性 `sigma`（标准差，相当于“我有多没把握”）。

- `BayesOptimizer`（贝叶斯优化器）：在高斯过程的基础上做一件事：
  - 给它一片“参数空间”，就是“这些输入的范围我可以试”；
  - 它会在里面随机试很多候选点，用高斯过程去估计哪个点“最有希望、最有潜力”，选出一个点建议你去真正实验、真正计算。  
  这在调参（比如调机器学习模型超参数）里非常常见。

---

### 二、用生活比喻先解释高斯过程
- 想象在看一个“山丘的高度图”：  
  横轴是位置（x），竖轴是高度（y）。
- 现在实际只量了几处高度，比如：
  - 在 0 米处，量到高度 1 米；
  - 在 1 米处，量到高度 -1 米；
- 其他地方没量。  
  高斯过程就是一种**根据已经量的这些点，来“画出整条曲线”的方法**：
  - 画的时候不仅画一条线（最可能的高度），
  - 还在每个位置标出“我这里不确定的程度”（方差/标准差）。

直觉上：

- 离已经量过的点很近的地方，大家会比较有把握——线画得稳，方差小。
- 离所有点都很远的地方，完全没什么依据——方差大，表示“不太知道”。

---

### 三、代码里的“核函数”是啥？

在代码里：

```rust
fn kernel(&self, a: &[f64], b: &[f64]) -> f64 {
    let mut sq = 0.0;
    for i in 0..a.len() {
        let d = (a[i] - b[i]) / self.length_scale;
        sq += d * d;
    }
    self.sigma_f.powi(2) * (-0.5 * sq).exp()
}
```

这就是高斯过程最核心的东西之一：**核函数（kernel）**，这里用的是常见的 **RBF / 高斯核**。

可以这样讲：

- 把每个输入 `x` 看成一个点；
- 核函数 `k(a, b)` 就是在回答一个问题：
  > “点 a 和点 b 的函数值，应该有多相似？”
- 如果 a 和 b 很接近，那 `k(a, b)` 就会很大（接近 `sigma_f^2`）；
- 如果 a 和 b 很远，那 `k(a, b)` 就会很小（接近 0）。

数学形式稍微专业一点：

\[
k(a,b) = \sigma_f^2 \exp\left(-\frac{1}{2} \sum_i \left(\frac{a_i-b_i}{\ell}\right)^2 \right)
\]

其中：

- \(\ell\) 是 `length_scale`（长度尺度）：  
  决定函数“变化得快不快”。  
  - \(\ell\) 小：函数变化很快，稍微挪一点，值就大变；
  - \(\ell\) 大：函数很平滑，挪一点儿，高度差不大。

- \(\sigma_f\)：函数值的大致“纵向尺度”，值能有多大波动。

---

### 四、训练（`fit`）在做什么？

核心函数：

```rust
fn fit(&mut self) {
    let n = self.x.len();
    ...
    let mut k = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let val = self.kernel(&self.x[i], &self.x[j]);
            k[(i, j)] = val;
        }
    }
    // 加噪对角稳定
    for i in 0..n {
        k[(i, i)] += self.noise;
    }
    let chol = Cholesky::new(k);
    self.chol = chol;
    self.y_vec = Some(DVector::from_vec(self.y.clone()));
}
```

#### 1. 构建核矩阵 K

- 已经观测到的输入点是 `x[0], x[1], ..., x[n-1]`；
- 定义一个 n×n 的矩阵 K，里面每个元素：

  \[
  K_{ij} = k(x_i, x_j)
  \]

- 你可以说：这是一个“相似度表格”：
  - 行和列都是观测过的点；
  - 每个格子是这两个点的“相关程度”。

#### 2. 在对角线加上一点噪声 `noise`

```rust
for i in 0..n {
    k[(i, i)] += self.noise;
}
```

- 这是往每个观测点上加一点“小误差余地”，也叫“观测噪声”；
- 数学上也是一个数值稳定的小技巧，防止矩阵太“完美”导致求逆不稳定。

#### 3. 做 Cholesky 分解

```rust
let chol = Cholesky::new(k);
self.chol = chol;
```

- 我们最终需要用到 \(K^{-1}\)（矩阵的逆）；
- 直接算逆既慢又不稳定；
- Cholesky 分解是：

  \[
  K = L L^T
  \]

  其中 L 是下三角矩阵，用它来解方程要简单很多。

---

### 五、预测（`predict`）的数学原理

核心函数：

```rust
pub fn predict(&self, x_star: &[f64]) -> (f64, f64) { ... }
```

给定一个新点 `x_star`，想要知道：

- 预测均值 `mu(x_star)`；
- 预测标准差 `sigma(x_star)`。

#### 1. 构建向量 `k_star` 和标量 `k_ss`

```rust
let mut k_star = DVector::<f64>::zeros(n);
for i in 0..n {
    k_star[i] = self.kernel(&self.x[i], x_star);
}
let k_ss = self.kernel(x_star, x_star);
```

- `k_star` 是一个 n 维向量，第 i 个元素：
  \[
  (k_*)_i = k(x_i, x_*)
  \]
- `k_ss = k(x_*, x_*)`，代表新点和自己之间的相关性，其实就是 \(\sigma_f^2\)。

#### 2. 预测均值公式

代码：

```rust
let w = chol.solve(&k_star);
let mu = w.dot(y);
```

数学上相当于：

- 先算：
  \[
  w = K^{-1} k_*
  \]
- 再算：
  \[
  \mu(x_*) = k_*^T K^{-1} y = w^T y
  \]

直观解释：

- \(K^{-1} y\) 可以看成“各个观测点对整体形状的影响权重”；
- 再乘上 \(k_*\)，就是拿这些影响权重来组合，生成对新点的预测。

> 可以想成：  
> 每个老点（已经观察过的）都会对新点出一份“主意”，  
> 它出多大力（权重）取决于：
> - 它自己在整体中的重要性（由 \(K^{-1}\) 决定）；
> - 它和新点有多相似（由 `k_star` 决定）。  
> 所有老点的意见加起来，就是新点的预测值。

#### 3. 预测方差公式

代码：

```rust
let var = k_ss - k_star.dot(&w);
let sigma = if var > 0.0 { var.sqrt() } else { 0.0 };
```

数学上：

\[
\text{var}(x_*) = k_{ss} - k_*^T K^{-1} k_*
\]

- `k_ss` 是“在什么都不知道的情况下，新点的先验不确定度”；
- 减去 `k_*^T K^{-1} k_*`，就是“因为看到了这些观测点，我们对新点的了解增加了多少”。

所以：

- 在离训练点很近的地方，`k_*` 很大，`k_*^T K^{-1} k_*` 也大 → 方差变小；
- 在非常远的地方，`k_*` 接近 0 → 方差接近 `k_ss`（最大不确定）。

---

### 六、BayesOptimizer 在干啥（UCB 策略）

```rust
// UCB: a(x) = mu(x) + kappa * sigma(x)
pub fn suggest<R: Rng>(&self, rng: &mut R, candidates: usize) -> Vec<f64> {
    let mut best_x = self.space.sample_random(rng);
    let mut best_a = f64::NEG_INFINITY;
    for _ in 0..candidates {
        let x = self.space.sample_random(rng);
        let (mu, sigma) = self.gp.predict(&x);
        let a = mu + self.kappa * sigma;
        if a > best_a {
            best_a = a;
            best_x = x;
        }
    }
    best_x
}
```

- 现在我们要“选下一步去哪里试一试”；
- 每个候选点 `x` 上，我们有：
  - `mu(x)`：觉得大概能有多好（期望收益）；
  - `sigma(x)`：不确定度，表示“这里我们了解得多不多”。

UCB 策略（Upper Confidence Bound，上置信界）：

\[
a(x) = \mu(x) + \kappa \cdot \sigma(x)
\]

- `mu` 高 → 看起来已经很好；
- `sigma` 高 → 我们还不太确定，可能隐藏着更大的惊喜；
- `kappa` 控制“敢不敢冒险”：
  - 大 `kappa`：更重视“不确定的地方”，更爱探索；
  - 小 `kappa`：更重视“看起来已经很好”的地方，更爱利用。

`BayesOptimizer::suggest` 做的事：

1. 在参数空间里随机抽很多个候选点；
2. 用高斯过程算出每个点的 `mu` 和 `sigma`；
3. 算出它们的 `a(x)`；
4. 选 `a(x)` 最大的点，作为“下一步最值得尝试的地方”。

---

### 七、测试在验证什么？

测试模块：

```rust
#[cfg(test)]
mod tests { ... }
```

这里有两个测试，分别是在检查：

1. 你的实现是否和教科书里的“闭式解公式”完全一致；
2. 在训练点附近，预测的不确定度是不是足够小。

#### 1. `predict_matches_closed_form`

这段测试大意：

```rust
let gp = make_simple_gp(); // x=0->1.0, x=1->-1.0

// 手工搭一遍 K、k_*、k_ss、mu、var
...
let alpha = chol.solve(&y);
let mu_manual = k_star.dot(&alpha);
...
let v = chol.solve(&k_star);
let var_manual = k_ss - k_star.dot(&v);

let (mu_gp, sigma_gp) = gp.predict(&x_star);

assert!((mu_gp - mu_manual).abs() < 1e-8);
assert!((sigma_gp.powi(2) - var_manual).abs() < 1e-8);
```

解释：

- 先用你的 `GaussianProcess` 训练好一个简单的一维模型（在 0 和 1 两个点上有观测）；
- 再**不用你的 `predict` 函数**，而是：
  1. 按照数学推导的标准公式，手工构造：
     - K、`k_star`、`k_ss`；
     - 通过 Cholesky 自己算出 `mu_manual` 和 `var_manual`；
  2. 然后调用你写的 `gp.predict`，得到 `mu_gp` 和 `sigma_gp`；
  3. 比较：
     - `mu_gp` 和 `mu_manual` 的差是否小于 `1e-8`（非常非常小）；
     - `sigma_gp^2` 与 `var_manual` 的差是否也小于 `1e-8`。

也就是说，这个测试在验证：

> 你的实现是不是完全遵循教科书里的高斯过程回归公式，  
> 没有把公式写错、索引写错、矩阵操作写错。

#### 2. `variance_near_observed_is_small`

```rust
let gp = make_simple_gp();
let (_mu0, s0) = gp.predict(&[0.0]);
let (_mu1, s1) = gp.predict(&[1.0]);
assert!(s0 < 1e-3, "sigma at training point should be ~0, got {}", s0);
assert!(s1 < 1e-3, "sigma at training point should be ~0, got {}", s1);
```

解释：

- 在训练点本身（x=0 和 x=1）上，我们已经有观测值，而且噪声非常小（`1e-9`）；
- 理论上的高斯过程告诉我们：在这些点上，预测的不确定度应该“几乎为 0”；
- 测试就检查：  
  在 `x=0` 和 `x=1` 处预测出来的 `sigma` 是否足够小（小于 `1e-3`）。

这个测试在验证：

> 高斯过程的“记忆”是不是正确：  
> 既然你在这里真正见过数据，就不该还对这里很迷茫。

---

### 八、如果你想对奶奶一句话总结

你可以最后这样说：

> 奶奶，我这段代码是在写一个聪明的“曲线猜测器”。  
> 它看过我给它的一些点的数据后，  
> 会帮我在新的地方猜结果，  
> 还会告诉我自己有多没底气。  
> 然后用这些“有把握程度”，帮我在一大堆可能性里选出最值得去试的地方。  
> 为了确保它没写错，我用数学公式手工算了一遍，  
> 再拿它自己算的结果做对比，确认两边一模一样，  
> 还确认在它真正见过的数据点那儿，它的“没底气指数”非常低，  
> 说明这套东西工作得很靠谱。

</details>