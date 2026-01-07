mod optimizer;
mod params;

use optimizer::BayesOptimizer;
use params::{ParameterSpace};
use rand::{SeedableRng, rngs::StdRng};

fn main() {
    // 初始化参数空间（示例选择两个常见 Linux 网络参数）
    let space = ParameterSpace::new(vec![
        // net.core.somaxconn：监听队列长度
        ("net.core.somaxconn", 128.0, 65535.0, true),
        // net.ipv4.tcp_fin_timeout：FIN-WAIT-2 超时
        ("net.ipv4.tcp_fin_timeout", 10.0, 120.0, true),
    ]);

    // 初始化优化器（UCB：mu + kappa * sigma）
    let mut rng = StdRng::seed_from_u64(42);
    let mut opt = BayesOptimizer::new(space.clone(), 2.0);

    // 最小闭环：生成参数 →（伪）施加 → 采样性能 → 回传优化器
    let iterations = 20;
    let mut best_score = f64::MIN;
    let mut best_cfg = Vec::new();

    for i in 0..iterations {
        let x_norm = if i == 0 {
            space.sample_random(&mut rng)
        } else {
            opt.suggest(&mut rng, 200)
        };

        // 将归一化参数映射到真实系统参数值
        let params_real = space.to_real(&x_norm);

        // 施加配置（最小可行：打印 Dry-run，不进行真实 sysctl 修改）
        println!("[dry-run] apply:");
        for (p, v) in space.named_values(&params_real) {
            println!("  {} = {}", p, v);
        }

        // 采样性能（最小可行：使用可控的合成目标函数，模拟 QPS/延迟/CPU 的关联）
        let score = mock_system_objective(&params_real);

        // 写入观测
        opt.observe(x_norm.clone(), score);

        // 记录最佳
        if score > best_score {
            best_score = score;
            best_cfg = params_real.clone();
        }

        println!("iter={} score={:.6}", i + 1, score);
    }

    // 输出最佳配置
    println!("\nBest score={:.6}", best_score);
    for (p, v) in space.named_values(&best_cfg) {
        println!("  {} = {}", p, v);
    }

    // 生成 2D 热力图（若维度为2）
    if space.dim() == 2 {
        println!("\nHeatmap (coarse, 20x20): higher is better");
        render_heatmap(&space, |vals| mock_system_objective(vals), 20);
    }
}

// 合成目标函数：模拟性能指标相关性（越大越好）
// 以 net.core.somaxconn 在 ~40000 处、tcp_fin_timeout 在 ~60 处为近似最优。
fn mock_system_objective(vals: &Vec<f64>) -> f64 {
    let s = vals[0]; // somaxconn
    let t = vals[1]; // fin_timeout

    // 模拟 QPS 峰值与延迟/CPU 约束的折中
    // 目标：最大化
    let qps_peak = -((s - 40000.0).powi(2)) / (2.0 * 9.0e8) + 1.0;
    let latency_penalty = -((t - 60.0).powi(2)) / (2.0 * 400.0);
    let cpu_penalty = -((s / 65535.0) * (t / 120.0)) * 0.05;

    qps_peak + latency_penalty + cpu_penalty
}

// 简单 2D 热力图渲染（ASCII）
fn render_heatmap<F>(space: &ParameterSpace, f: F, grid: usize)
where
    F: Fn(&Vec<f64>) -> f64,
{
    use std::cmp::Ordering;

    let mut vals = Vec::new();
    for iy in 0..grid {
        for ix in 0..grid {
            let x0 = ix as f64 / (grid - 1) as f64;
            let x1 = iy as f64 / (grid - 1) as f64;
            let real = space.to_real(&vec![x0, x1]);
            let s = f(&real);
            vals.push(s);
        }
    }
    let min = vals
        .iter()
        .cloned()
        .fold(f64::INFINITY, |a, b| if b < a { b } else { a });
    let max = vals
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, |a, b| if b > a { b } else { a });

    let shades = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];
    for iy in 0..grid {
        let mut line = String::new();
        for ix in 0..grid {
            let s = vals[iy * grid + ix];
            let norm = if (max - min).partial_cmp(&0.0) == Some(Ordering::Equal) {
                0.0
            } else {
                (s - min) / (max - min)
            };
            let idx = (norm * (shades.len() as f64 - 1.0)).round() as usize;
            line.push(shades[idx]);
        }
        println!("{}", line);
    }
}
