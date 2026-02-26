use crate::vec::{VecMath, Vector};

/// Fisher-Yates shuffle of any slice, using rand::random as the source of randomness.
pub fn shuffle<T>(data: &mut [T]) {
    let n = data.len();
    for i in (1..n).rev() {
        let j = (rand::random::<u64>() as usize) % (i + 1);
        data.swap(i, j);
    }
}

/// Split a slice into two owned Vecs at `split_index`.
/// Returns (first, second) where first has `split_index` elements.
pub fn train_test_split<T: Clone>(data: &[T], train_ratio: f64) -> (Vec<T>, Vec<T>) {
    assert!(
        (0.0..=1.0).contains(&train_ratio),
        "train_ratio must be in [0, 1]"
    );
    let split = (data.len() as f64 * train_ratio).round() as usize;
    let split = split.clamp(1, data.len().saturating_sub(1));
    (data[..split].to_vec(), data[split..].to_vec())
}

pub fn euclidean_distance(predicted: &[f64], actual: &[f64]) -> f64 {
    assert!(!predicted.is_empty(), "euclidean_distance: empty input");
    assert_eq!(
        predicted.len(),
        actual.len(),
        "euclidean_distance: length mismatch"
    );
    predicted
        .iter()
        .zip(actual)
        .map(|(&yhat, &y)| (yhat - y).powi(2))
        .sum::<f64>()
        .sqrt()
}
pub fn cosine_distance(predicted: &[f64], actual: &[f64]) -> f64 {
    assert!(!predicted.is_empty(), "cosine_distance: empty input");
    assert_eq!(
        predicted.len(),
        actual.len(),
        "cosine_distance: length mismatch"
    );

    let dot_product: f64 = predicted.iter().zip(actual).map(|(&a, &b)| a * b).sum();
    let norm_a = predicted.iter().map(|&x| x.powi(2)).sum::<f64>().sqrt();
    let norm_b = actual.iter().map(|&x| x.powi(2)).sum::<f64>().sqrt();

    // Prevent division by zero
    if norm_a < f64::EPSILON || norm_b < f64::EPSILON {
        return 1.0;
    }

    1.0 - (dot_product / (norm_a * norm_b))
}

// Numerical derivative of f at point x using centered finite difference
// h is the step size
pub fn numerical_derivative(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
    (f(x + h) - f(x - h)) / (2.0 * h)
}

// Numerical gradient of f: R^n -> R at point x
// Returns a Vec<f64> of partial derivatives, one per input dimension
// Perturbs each dimension independently, holds others fixed
pub fn numerical_gradient<F>(f: F, x: &[f64], h: f64) -> Vector
where
    F: Fn(&[f64]) -> f64,
{
    let mut gradient = vec![0.0; x.len()];
    let mut x_copy = x.to_vec();

    for i in 0..x.len() {
        let original_val = x_copy[i];

        // Central difference: (f(x + h) - f(x - h)) / 2h
        x_copy[i] = original_val + h;
        let upper = f(&x_copy);

        x_copy[i] = original_val - h;
        let lower = f(&x_copy);

        gradient[i] = (upper - lower) / (2.0 * h);

        // Reset for next dimension
        x_copy[i] = original_val;
    }
    Vector(gradient)
}

// Gradient descent step for f: R^n -> R at point x
// Returns a Vector representing the updated point

pub fn gradient_descent_step<F>(f: F, x: &Vector, learning_rate: f64, h: f64) -> Vector
where
    F: Fn(&[f64]) -> f64,
{
    let gradient = numerical_gradient(f, x.as_slice(), h);
    x.sub(&gradient.scale(learning_rate))
}
