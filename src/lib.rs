pub mod activation;
pub mod loss;
pub mod matrix;
pub mod mlp;
pub mod neuron;
pub mod optimizer;
pub mod util;
pub mod vec;

pub use activation::Activation;
pub use loss::{Loss, LossFunction, binary_cross_entropy, mae, mse};
pub use matrix::Matrix;
pub use mlp::MLP;
pub use neuron::{DenseLayer, LayerGradients, LayerNeuron, Neuron};
pub use optimizer::{BatchGradients, Optimizer, SGD};
pub use util::{
    cosine_distance, euclidean_distance, gradient_descent_step, numerical_derivative,
    numerical_gradient, shuffle, train_test_split,
};
pub use vec::{VecMath, Vector, normalize_dataset, normalize_sample};
