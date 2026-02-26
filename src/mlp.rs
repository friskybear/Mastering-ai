use crate::{
    neuron::{DenseLayer, LayerGradients},
    vec::{VecMath, Vector},
};

pub struct MLP {
    layers: Vec<DenseLayer>,
}

impl MLP {
    pub fn new(layers: Vec<DenseLayer>) -> Self {
        assert!(!layers.is_empty(), "MLP cannot be empty");
        Self { layers }
    }

    pub fn forward(&self, input: &Vector) -> (Vec<Vec<f64>>, Vector) {
        let mut output = input.clone();
        let mut zs = Vec::with_capacity(self.layers.len());

        for layer in &self.layers {
            let (layer_zs, layer_activations) = layer.forward(&output);
            zs.push(layer_zs);
            output = layer_activations;
        }

        (zs, output)
    }

    /// Forward pass that caches all intermediate values needed for backprop.
    /// Returns `(output, activations, zs)` where:
    /// - `output`      — final activated output of the network
    /// - `activations` — inputs to each layer (length == num_layers);
    ///                   `activations[l]` is the vector fed into layer `l`
    /// - `zs`          — pre-activation values for each layer (length == num_layers)
    pub fn forward_with_cache(&self, input: &Vector) -> (Vector, Vec<Vector>, Vec<Vec<f64>>) {
        let num_layers = self.layers.len();
        let mut activations = Vec::with_capacity(num_layers);
        let mut zs = Vec::with_capacity(num_layers);

        let mut current = input.clone();

        for layer in &self.layers {
            // Cache the input to this layer before transforming it
            activations.push(current.clone());

            let (layer_zs, layer_output) = layer.forward(&current);
            zs.push(layer_zs);
            current = layer_output;
        }

        // `current` is now the final network output
        (current, activations, zs)
    }

    pub fn backward(
        &self,
        inputs: &[Vector], // store activations for each layer including input
        zs: &[Vec<f64>],   // pre-activation values per layer
        dloss_da: &Vector,
    ) -> Vec<LayerGradients> {
        let mut gradients = Vec::with_capacity(self.layers.len());
        let mut dloss_dz = dloss_da.clone();

        for l in (0..self.layers.len()).rev() {
            let layer_input = &inputs[l]; // input to this layer
            let g = self.layers[l].backward(layer_input, &zs[l], &dloss_dz);
            dloss_dz = g.dinputs.clone();
            gradients.push(g);
        }

        gradients.reverse(); // so layer 0 is first
        gradients
    }

    pub fn sgd_step(&mut self, inputs: &Vector, target: &Vector, learning_rate: f64) {
        // --- 1. Forward pass ---
        let (outputs, activations, zs) = self.forward_with_cache(inputs);

        // --- 2. Compute loss gradient w.r.t output ---
        // general
        let dloss_da = outputs.sub(target).scale(2.0 / outputs.len() as f64);

        // --- 3. Backward pass ---
        let gradients = self.backward(&activations, &zs, &dloss_da);

        // --- 4. Update weights & biases ---
        for (layer, grad) in self.layers.iter_mut().zip(gradients.iter()) {
            for (neuron, (dw, db)) in layer
                .neurons
                .iter_mut()
                .zip(grad.dweights.iter().zip(grad.dbiases.iter()))
            {
                // w = w - η * dw
                neuron.neuron.weights = neuron.neuron.weights.sub(&dw.scale(learning_rate));
                // b = b - η * db
                neuron.neuron.bias -= learning_rate * db;
            }
        }
    }
}
