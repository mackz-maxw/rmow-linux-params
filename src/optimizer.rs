use nalgebra::{DMatrix, DVector, Cholesky};
use rand::Rng;

use crate::params::ParameterSpace;

pub struct GaussianProcess {
    length_scale: f64,
    sigma_f: f64,
    noise: f64,
    x: Vec<Vec<f64>>, // 归一化输入
    y: Vec<f64>,      // 观测目标
    chol: Option<Cholesky<f64, DMatrix<f64>>>,
    y_vec: Option<DVector<f64>>,
}

impl GaussianProcess {
    pub fn new(length_scale: f64, sigma_f: f64, noise: f64) -> Self {
        Self {
            length_scale,
            sigma_f,
            noise,
            x: Vec::new(),
            y: Vec::new(),
            chol: None,
            y_vec: None,
        }
    }

    fn kernel(&self, a: &[f64], b: &[f64]) -> f64 {
        let mut sq = 0.0;
        for i in 0..a.len() {
            let d = (a[i] - b[i]) / self.length_scale;
            sq += d * d;
        }
        self.sigma_f.powi(2) * (-0.5 * sq).exp()
    }

    pub fn add_observation(&mut self, x: Vec<f64>, y: f64) {
        self.x.push(x);
        self.y.push(y);
        self.fit();
    }

    fn fit(&mut self) {
        let n = self.x.len();
        if n == 0 {
            self.chol = None;
            self.y_vec = None;
            return;
        }
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

    // 返回 (mu, sigma)
    pub fn predict(&self, x_star: &[f64]) -> (f64, f64) {
        let n = self.x.len();
        if n == 0 || self.chol.is_none() || self.y_vec.is_none() {
            // 暂无数据：返回不确定的大方差
            return (0.0, self.sigma_f);
        }

        let chol = self.chol.as_ref().unwrap();
        let y = self.y_vec.as_ref().unwrap();

        let mut k_star = DVector::<f64>::zeros(n);
        for i in 0..n {
            k_star[i] = self.kernel(&self.x[i], x_star);
        }
        let k_ss = self.kernel(x_star, x_star) + self.noise;

        // mu = k*^T * K^{-1} * y  => w = chol.solve(k*), mu = w^T y
        let w = chol.solve(&k_star);
        let mu = w.dot(y);

        // var = k_ss - k*^T * K^{-1} * k*
        let var = k_ss - k_star.dot(&w);
        let sigma = if var > 0.0 { var.sqrt() } else { 0.0 };
        (mu, sigma)
    }
}

pub struct BayesOptimizer {
    pub gp: GaussianProcess,
    pub kappa: f64,
    pub space: ParameterSpace,
}

impl BayesOptimizer {
    pub fn new(space: ParameterSpace, kappa: f64) -> Self {
        Self {
            gp: GaussianProcess::new(length_scale: 0.2, sigma_f: 1.0, noise: 1e-6),
            kappa,
            space,
        }
    }

    pub fn observe(&mut self, x_norm: Vec<f64>, y: f64) {
        self.gp.add_observation(x_norm, y);
    }

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
}