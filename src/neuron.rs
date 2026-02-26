use crate::{
    activation::Activation,
    vec::{VecMath, Vector},
};

pub struct Neuron {
    pub weights: Vector,
    pub bias: f64,
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
        LayerNeuron {
            neuron: Neuron::new(weights, bias),
            activation,
        }
    }

    pub fn forward(&self, inputs: &Vector) -> (f64, f64) {
        let z = self.neuron.forward(inputs);
        let a = self.activation.forward(z);
        (z, a)
    }
}

pub struct LayerGradients {
    pub dweights: Vec<Vector>,
    pub dbiases: Vec<f64>,
    pub dinputs: Vector,
}

pub struct DenseLayer {
    pub neurons: Vec<LayerNeuron>,
}

impl DenseLayer {
    pub fn new(neurons: Vec<LayerNeuron>) -> Self {
        assert!(!neurons.is_empty(), "DenseLayer cannot be empty");
        Self { neurons }
    }
    pub fn forward(&self, input: &Vector) -> (Vec<f64>, Vector) {
        let mut zs = Vec::with_capacity(self.neurons.len());
        let mut activations = Vec::with_capacity(self.neurons.len());

        for neuron in &self.neurons {
            let (z, a) = neuron.forward(input);
            zs.push(z);
            activations.push(a);
        }

        (zs, Vector::from(activations))
    }
    pub fn backward(
        &self,
        input: &Vector,
        zs: &[f64],        // pre-activation values
        dloss_da: &Vector, // gradient from next layer (∂L/∂a)
    ) -> LayerGradients {
        let num_inputs = input.len();
        let num_neurons = self.neurons.len();

        let mut dweights = Vec::with_capacity(num_neurons);
        let mut dbiases = Vec::with_capacity(num_neurons);
        let mut dinputs = vec![0.0; num_inputs];

        for (j, neuron) in self.neurons.iter().enumerate() {
            // Step 1: δ = dL/da * f'(z)
            let dz = dloss_da.as_slice()[j] * neuron.activation.derivative(zs[j]);

            // Step 2: ∂L/∂w = δ * input
            let dw = input.scale(dz);
            dweights.push(dw);

            // Step 3: ∂L/∂b = δ
            dbiases.push(dz);

            // Step 4: accumulate ∂L/∂input = sum_j w_ij * δ_j
            for i in 0..num_inputs {
                dinputs[i] += neuron.neuron.weights.as_slice()[i] * dz;
            }
        }

        LayerGradients {
            dweights,
            dbiases,
            dinputs: Vector::from(dinputs),
        }
    }
}
