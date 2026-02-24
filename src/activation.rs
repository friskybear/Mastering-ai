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
}