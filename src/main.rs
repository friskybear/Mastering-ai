mod activation;
mod matrix;
mod neuron;
mod util;
mod vec;
use matrix::Matrix;
mod loss;
use neuron::LayerNeuron;
use util::*;

use vec::{VecMath, Vector};

use crate::activation::Activation;
fn main() {
    let neuron = LayerNeuron::new(Vector::from(vec![0.5, -0.2]), 0.1, Activation::Sigmoid);

    let x = Vector::from(vec![1.0, 2.0]);

    let (z, y) = neuron.forward(&x);

    println!("z = {z}, y = {y}");
}
