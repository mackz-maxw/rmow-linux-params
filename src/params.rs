use rand::Rng;

#[derive(Clone)]
pub struct Parameter {
    pub name: String,
    pub min: f64,
    pub max: f64,
    pub integer: bool,
}

#[derive(Clone)]
pub struct ParameterSpace {
    params: Vec<Parameter>,
}

impl ParameterSpace {
    pub fn new(entries: Vec<(&str, f64, f64, bool)>) -> Self {
        let params = entries
            .into_iter()
            .map(|(n, min, max, integer)| Parameter {
                name: n.to_string(),
                min,
                max,
                integer,
            })
            .collect();
        Self { params }
    }

    pub fn dim(&self) -> usize {
        self.params.len()
    }

    pub fn sample_random<R: Rng>(&self, rng: &mut R) -> Vec<f64> {
        (0..self.params.len())
            .map(|_| rng.r#gen::<f64>())
            .collect()
    }

    // 将 [0,1]^d 映射到实际参数值
    pub fn to_real(&self, x: &Vec<f64>) -> Vec<f64> {
        self.params
            .iter()
            .zip(x.iter())
            .map(|(p, &u)| {
                let v = p.min + u.clamp(0.0, 1.0) * (p.max - p.min);
                if p.integer {
                    v.round()
                } else {
                    v
                }
            })
            .collect()
    }

    // 返回带参数名的键值对
    pub fn named_values(&self, vals: &Vec<f64>) -> Vec<(String, f64)> {
        self.params
            .iter()
            .zip(vals.iter())
            .map(|(p, &v)| (p.name.clone(), v))
            .collect()
    }
}