use crate::{
    neuron::{DenseLayer, LayerGradients},
    vec::{VecMath, Vector},
};

pub trait Optimizer {
    fn step(&mut self, layers: &mut [DenseLayer], grads: &[LayerGradients]);
}

// ---------------------------------------------------------------------------
// SGD  (optionally with momentum and / or weight decay)
// ---------------------------------------------------------------------------
//
// Update rules:
//
//   Without momentum:
//     w = w - lr * (g + λ·w)
//
//   With momentum (β):
//     v = β·v + g           (velocity accumulates scaled gradient)
//     w = w - lr * (v + λ·w)
//
// Weight decay (λ) is applied to weights only, never to biases —
// this matches every standard reference implementation.
// ---------------------------------------------------------------------------

pub struct SGD {
    pub lr: f64,

    /// Momentum coefficient β ∈ [0, 1).  None means plain SGD.
    pub momentum: Option<f64>,

    /// L2 weight-decay coefficient λ ≥ 0.  None / 0.0 means no decay.
    pub decay: Option<f64>,

    /// Velocity buffers — one `LayerGradients` per layer.
    /// Initialized lazily on the first call to `step`.
    velocity: Vec<LayerVelocity>,
}

/// Per-layer velocity state (mirrors the shape of `LayerGradients`).
struct LayerVelocity {
    /// One velocity vector per neuron (same shape as dweights).
    weights: Vec<Vector>,
    /// One velocity scalar per neuron (same shape as dbiases).
    biases: Vec<f64>,
}

impl LayerVelocity {
    fn zero_like(grad: &LayerGradients) -> Self {
        Self {
            weights: grad
                .dweights
                .iter()
                .map(|w| Vector::zeros(w.len()))
                .collect(),
            biases: vec![0.0; grad.dbiases.len()],
        }
    }
}

impl SGD {
    /// Plain SGD — identical behaviour to the old `SGD::new`.
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            momentum: None,
            decay: None,
            velocity: Vec::new(),
        }
    }

    /// Enable momentum.  Typical value: 0.9.
    pub fn with_momentum(mut self, beta: f64) -> Self {
        assert!(
            (0.0..1.0).contains(&beta),
            "momentum beta must be in [0, 1), got {}",
            beta
        );
        self.momentum = Some(beta);
        self
    }

    /// Enable L2 weight decay.  Typical value: 1e-4.
    pub fn with_decay(mut self, lambda: f64) -> Self {
        assert!(lambda >= 0.0, "decay lambda must be >= 0, got {}", lambda);
        self.decay = Some(lambda);
        self
    }
}

impl Optimizer for SGD {
    fn step(&mut self, layers: &mut [DenseLayer], grads: &[LayerGradients]) {
        // ----- lazy velocity initialisation --------------------------------
        if self.momentum.is_some() && self.velocity.is_empty() {
            self.velocity = grads.iter().map(LayerVelocity::zero_like).collect();
        }

        let beta = self.momentum.unwrap_or(0.0);
        let lambda = self.decay.unwrap_or(0.0);
        let lr = self.lr;
        let use_momentum = self.momentum.is_some();

        for (l, (layer, grad)) in layers.iter_mut().zip(grads.iter()).enumerate() {
            for (n, (neuron, (dw, db))) in layer
                .neurons
                .iter_mut()
                .zip(grad.dweights.iter().zip(grad.dbiases.iter()))
                .enumerate()
            {
                // ---- weights ------------------------------------------------
                if use_momentum {
                    let vel = &mut self.velocity[l].weights[n];
                    // v = β·v + g
                    *vel = vel.scale(beta).add(dw);
                    // effective gradient = v + λ·w  (decay on weight, not velocity)
                    let effective = vel.add(&neuron.neuron.weights.scale(lambda));
                    neuron.neuron.weights = neuron.neuron.weights.sub(&effective.scale(lr));
                } else {
                    // effective gradient = g + λ·w
                    let effective = dw.add(&neuron.neuron.weights.scale(lambda));
                    neuron.neuron.weights = neuron.neuron.weights.sub(&effective.scale(lr));
                }

                // ---- bias  (no weight decay, no momentum on bias) -----------
                if use_momentum {
                    let vel_b = &mut self.velocity[l].biases[n];
                    // v_b = β·v_b + db
                    *vel_b = beta * *vel_b + db;
                    neuron.neuron.bias -= lr * *vel_b;
                } else {
                    neuron.neuron.bias -= lr * db;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// BatchGradients  (unchanged)
// ---------------------------------------------------------------------------

pub struct BatchGradients {
    pub layers: Vec<LayerGradients>,
}

impl BatchGradients {
    pub fn zero_like(layers: usize) -> Self {
        Self {
            layers: (0..layers).map(|_| LayerGradients::zero()).collect(),
        }
    }

    pub fn add(&mut self, other: &[LayerGradients]) {
        for (a, b) in self.layers.iter_mut().zip(other.iter()) {
            a.add_inplace(b);
        }
    }

    pub fn scale(&mut self, factor: f64) {
        for g in &mut self.layers {
            g.scale_inplace(factor);
        }
    }
}
