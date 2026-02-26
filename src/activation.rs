pub enum Activation {
    Linear,
    Sigmoid,
    Tanh,
    ReLU,
}

impl Activation {
    pub fn forward(&self, input: f64) -> f64 {
        match self {
            Activation::Linear => input,
            Activation::Sigmoid => 1.0 / (1.0 + (-input).exp()),
            Activation::Tanh => (2.0 / (1.0 + (-2.0 * input).exp())) - 1.0,
            Activation::ReLU => input.max(0.0),
        }
    }
    pub fn derivative(&self, input: f64) -> f64 {
        match self {
            Activation::Linear => 1.0,
            Activation::Sigmoid => {
                let s = 1.0 / (1.0 + (-input).exp());
                s * (1.0 - s)
            }
            Activation::Tanh => {
                let t = (2.0 / (1.0 + (-2.0 * input).exp())) - 1.0;
                1.0 - t * t
            }
            Activation::ReLU => {
                if input > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}
