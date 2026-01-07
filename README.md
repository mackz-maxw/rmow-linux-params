# rmow-linux-params

Linux 系统参数自适应优化的最小可行实现（Rust）。包含：
- 基于贝叶斯优化的核心决策引擎 [`optimizer::BayesOptimizer`](src/optimizer.rs)，利用高斯过程回归对黑盒性能函数建模
- 自动化闭环（最小版本）：参数生成 →（Dry-run）配置施加 → 性能采样 → 反馈优化器，入口见 [src/main.rs](src/main.rs)
- 数据可视化（最小版本）：二维参数空间的 ASCII 热力图，辅助观察参数-性能关联

## 背景与目标
研发一个智能化性能优化平台，通过数据驱动的方式，自动、持续地寻找并施加最优系统参数配置，以最大化硬件资源利用率和系统吞吐量。目标系统可扩展到 Nginx 等关键组件，期望在标准压力测试下的关键指标（如 BIC 评分）提升约 30%。

## 数学概述
- 黑盒目标函数 $f(x)$（如 QPS/延迟/CPU 的综合指标）通过高斯过程回归（GP）建模：
  $
  f(\cdot) \sim \mathcal{GP}\left(0, k(\cdot, \cdot)\right),\quad
  k(x, x') = \sigma_f^2 \exp\left(-\frac{1}{2}\sum_i \frac{(x_i - x'_i)^2}{\ell^2}\right)
  $
- 后验预测（给定观测 $(X, y)$）：
  $
  \mu(x_*) = k_*^\top K^{-1} y,\quad
  \sigma^2(x_*) = k(x_*, x_*) - k_*^\top K^{-1} k_*
  $
- 采集函数（最小实现使用 UCB）：
  $
  a_{\mathrm{UCB}}(x) = \mu(x) + \kappa \cdot \sigma(x)
  $
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