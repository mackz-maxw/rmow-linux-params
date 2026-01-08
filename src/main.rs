mod optimizer;
mod params;

use optimizer::BayesOptimizer;
use params::ParameterSpace;
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

#[cfg(test)]
mod tests {
    use super::*;
    use optimizer::GaussianProcess;
    use nalgebra::{DMatrix, DVector, Cholesky};

    fn make_simple_gp() -> GaussianProcess {
        let mut gp = GaussianProcess::new(0.5, 1.0, 1e-9);
        // 1D 简单数据：x=0 -> 1.0, x=1 -> -1.0
        gp.add_observation(vec![0.0], 1.0);
        gp.add_observation(vec![1.0], -1.0);
        gp
    }

    #[test]
    fn test_predict_matches_closed_form() {
        let gp = make_simple_gp();

        // 手工计算后验
        let x_train = vec![vec![0.0], vec![1.0]];
        let y = DVector::from_vec(vec![1.0, -1.0]);

        let n = x_train.len();
        let mut k = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                k[(i, j)] = gp.kernel(&x_train[i], &x_train[j]);
            }
        }
        for i in 0..n {
            k[(i, i)] += 1e-9; // noise
        }

        let chol = Cholesky::new(k).expect("Cholesky failed");

        // 测试点
        let x_star = vec![0.25];
        let mut k_star = DVector::<f64>::zeros(n);
        for i in 0..n {
            k_star[i] = gp.kernel(&x_train[i], &x_star);
        }
        let k_ss = gp.kernel(&x_star, &x_star);

        // mu = k_*^T K^{-1} y
        let alpha = chol.solve(&y);
        let mu_manual = k_star.dot(&alpha);

        // var = k_ss - k_*^T K^{-1} k_*
        let v = chol.solve(&k_star);
        let var_manual = k_ss - k_star.dot(&v);

        let (mu_gp, sigma_gp) = gp.predict(&x_star);

        let mu_diff = (mu_gp - mu_manual).abs();
        let var_diff = (sigma_gp.powi(2) - var_manual).abs();
        
        println!("  测试点 x=0.25:");
        println!("    均值(GP实现)={:.8}, 均值(闭式解)={:.8}, 误差={:.2e}", mu_gp, mu_manual, mu_diff);
        println!("    方差(GP实现)={:.8}, 方差(闭式解)={:.8}, 误差={:.2e}", sigma_gp.powi(2), var_manual, var_diff);
        
        assert!(mu_diff < 1e-8, "均值误差过大: {:.2e}", mu_diff);
        assert!(var_diff < 1e-8, "方差误差过大: {:.2e}", var_diff);
    }

    #[test]
    fn test_variance_near_observed_is_small() {
        let gp = make_simple_gp();
        let (_mu0, s0) = gp.predict(&[0.0]);
        let (_mu1, s1) = gp.predict(&[1.0]);
        
        println!("  训练点 x=0.0: sigma={:.2e}", s0);
        println!("  训练点 x=1.0: sigma={:.2e}", s1);
        
        assert!(s0 < 1e-3, "训练点 x=0 的 sigma 应该接近0, 实际: {:.2e}", s0);
        assert!(s1 < 1e-3, "训练点 x=1 的 sigma 应该接近0, 实际: {:.2e}", s1);
    }

    #[test]
    fn test_visualize_gp_1d() {
        let gp = make_simple_gp();
        
        println!("\n  训练数据: (x=0.0, y=1.0), (x=1.0, y=-1.0)");
        println!("  超参数: length_scale=0.5, sigma_f=1.0, noise=1e-9\n");
        
        // 在 [-0.5, 1.5] 范围画曲线
        let n_points = 50;
        let x_min = -0.5;
        let x_max = 1.5;
        
        println!("  高斯过程预测曲线 (x 从 {:.1} 到 {:.1}):\n", x_min, x_max);
        println!("   x     |   μ(x)  |  σ(x)   | 可视化 [μ-2σ --- μ --- μ+2σ]");
        println!("  -------|---------|---------|{}", "-".repeat(50));
        
        for i in 0..n_points {
            let x = x_min + (x_max - x_min) * (i as f64) / ((n_points - 1) as f64);
            let (mu, sigma) = gp.predict(&[x]);
            
            // 画一个简单的 ASCII 图：显示均值和 ±2σ 区间
            let lower = mu - 2.0 * sigma;
            let upper = mu + 2.0 * sigma;
            
            // 映射到字符位置 (假设 y 范围 [-2, 2])
            let y_min = -2.0;
            let y_max = 2.0;
            let width = 50;
            
            let pos_mu = ((mu - y_min) / (y_max - y_min) * (width as f64)).round() as i32;
            let pos_lower = ((lower - y_min) / (y_max - y_min) * (width as f64)).round() as i32;
            let pos_upper = ((upper - y_min) / (y_max - y_min) * (width as f64)).round() as i32;
            
            let mut line = vec![' '; width];
            
            // 画置信区间
            for j in 0..width {
                let j_i32 = j as i32;
                if j_i32 >= pos_lower.max(0) && j_i32 <= pos_upper.min(width as i32 - 1) {
                    line[j] = '·';
                }
            }
            
            // 画均值点
            if pos_mu >= 0 && pos_mu < width as i32 {
                line[pos_mu as usize] = '█';
            }
            
            // 标记训练点
            let marker = if (x - 0.0).abs() < 0.02 || (x - 1.0).abs() < 0.02 { "*" } else { " " };
            
            println!("  {:.3}{} | {:7.3} | {:7.3} | {}", 
                     x, marker, mu, sigma, line.iter().collect::<String>());
        }
        
        println!("\n  图例: * = 训练点位置, █ = 预测均值 μ(x), · = 95%置信区间 [μ-2σ, μ+2σ]");
    }
}
