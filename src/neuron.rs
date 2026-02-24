

use crate::{activation::Activation, vec::{VecMath, Vector}};

pub struct Neuron {
    weights: Vector,
    bias: f64,
}

impl Neuron {
    pub fn new(weights: Vector, bias: f64) -> Self {
        Neuron { weights, bias }
    }

    pub fn forward(&self, inputs: &Vector) -> f64 {
        assert_eq!(
            inputs.len(),
            self.weights.len(),
            "Neuron::forward: input/weight dimension mismatch"
        );
        inputs.dot(&self.weights) + self.bias
    }
}

pub struct LayerNeuron {
    pub neuron: Neuron,
    pub activation: Activation,
}

impl LayerNeuron {
    pub fn new(weights: Vector, bias: f64, activation: Activation) -> Self {
        LayerNeuron { neuron: Neuron::new(weights, bias), activation }
    }

    pub fn forward(&self, inputs: &Vector) -> (f64, f64) {
        let z = self.neuron.forward(inputs);
        let a = self.activation.forward(z);
        (z, a)
    }
}