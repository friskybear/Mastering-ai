use std::slice::SliceIndex;

#[derive(Debug, Clone)]
pub struct Vector(pub Vec<f64>);
impl Vector {
    pub fn new(vec: Vec<f64>) -> Self {
        Vector(vec)
    }
    pub fn zeros(size: usize) -> Self {
        Vector(vec![0.0; size])
    }
    pub fn ones(size: usize) -> Self {
        Vector(vec![1.0; size])
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn rand(size: usize) -> Self {
        Vector(
            (0..size)
                .map(|_| rand::random::<f64>() * 2.0 - 1.0)
                .collect(),
        )
    }

    /// He initialization: N(0, sqrt(2 / fan_in))
    /// Recommended for ReLU activations.
    pub fn rand_he(size: usize, fan_in: usize) -> Self {
        let std = (2.0 / fan_in as f64).sqrt();
        Vector(
            (0..size)
                .map(|_| {
                    // Box-Muller transform: two uniform samples -> one normal sample
                    let u1: f64 = rand::random::<f64>().max(f64::EPSILON);
                    let u2: f64 = rand::random::<f64>();
                    let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    normal * std
                })
                .collect(),
        )
    }

    /// Xavier / Glorot initialization: U(-sqrt(6 / (fan_in + fan_out)), +sqrt(...))
    /// Recommended for Tanh and Sigmoid activations.
    pub fn rand_xavier(size: usize, fan_in: usize, fan_out: usize) -> Self {
        let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
        Vector(
            (0..size)
                .map(|_| rand::random::<f64>() * 2.0 * limit - limit)
                .collect(),
        )
    }
    pub fn get<I>(&self, index: I) -> Option<&I::Output>
    where
        I: SliceIndex<[f64]>,
    {
        self.0.get(index)
    }

    pub fn get_mut<I>(&mut self, index: I) -> Option<&mut I::Output>
    where
        I: SliceIndex<[f64]>,
    {
        self.0.get_mut(index)
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|&x| x.abs() < std::f64::EPSILON)
    }
    pub fn is_one(&self) -> bool {
        self.0.iter().all(|&x| x == 1.0)
    }
    pub fn as_slice(&self) -> &[f64] {
        &self.0
    }
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.0
    }
    pub fn as_mut_vec(&mut self) -> &mut Vec<f64> {
        &mut self.0
    }
    pub fn as_mut_vec_mut(&mut self) -> &mut Vec<f64> {
        &mut self.0
    }
}
impl From<Vec<f64>> for Vector {
    fn from(vec: Vec<f64>) -> Self {
        Vector(vec)
    }
}

impl From<&[f64]> for Vector {
    fn from(slice: &[f64]) -> Self {
        Vector(slice.to_vec())
    }
}
impl<'a> Into<&'a [f64]> for &'a Vector {
    fn into(self) -> &'a [f64] {
        &self.0
    }
}

impl<'a> IntoIterator for &'a Vector {
    type Item = &'a f64;
    type IntoIter = std::slice::Iter<'a, f64>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}
impl FromIterator<f64> for Vector {
    fn from_iter<T: IntoIterator<Item = f64>>(iter: T) -> Self {
        Vector(iter.into_iter().collect())
    }
}

impl Vector {
    // Return an iterator over references to the elements
    pub fn iter(&self) -> std::slice::Iter<'_, f64> {
        self.0.iter()
    }
}
impl AsRef<[f64]> for Vector {
    fn as_ref(&self) -> &[f64] {
        &self.0
    }
}

/// Min-max normalizes a dataset of vectors in-place, per feature.
/// Each feature i is scaled to [0, 1] using the min and max observed across all samples.
/// Returns the (mins, maxs) per feature so the same transform can be applied to new data.
pub fn normalize_dataset(data: &mut [Vector]) -> (Vector, Vector) {
    assert!(!data.is_empty(), "normalize_dataset: empty dataset");
    let n_features = data[0].len();

    let mut mins = vec![f64::INFINITY; n_features];
    let mut maxs = vec![f64::NEG_INFINITY; n_features];

    // Pass 1: find min/max per feature
    for sample in data.iter() {
        assert_eq!(
            sample.len(),
            n_features,
            "normalize_dataset: inconsistent feature count"
        );
        for (i, &v) in sample.iter().enumerate() {
            if v < mins[i] {
                mins[i] = v;
            }
            if v > maxs[i] {
                maxs[i] = v;
            }
        }
    }

    // Pass 2: scale each feature to [0, 1]
    for sample in data.iter_mut() {
        for i in 0..n_features {
            let range = maxs[i] - mins[i];
            sample.0[i] = if range < f64::EPSILON {
                0.0 // constant feature — set to 0
            } else {
                (sample.0[i] - mins[i]) / range
            };
        }
    }

    (Vector::from(mins), Vector::from(maxs))
}

/// Apply a previously computed min-max normalization to a single vector.
pub fn normalize_sample(sample: &mut Vector, mins: &Vector, maxs: &Vector) {
    assert_eq!(
        sample.len(),
        mins.len(),
        "normalize_sample: dimension mismatch"
    );
    for i in 0..sample.len() {
        let range = maxs.0[i] - mins.0[i];
        sample.0[i] = if range < f64::EPSILON {
            0.0
        } else {
            (sample.0[i] - mins.0[i]) / range
        };
    }
}

pub trait VecMath {
    // Computes sum(self[i] * other[i])
    fn dot(&self, other: &Vector) -> f64;
    // Computes self[i] + other[i]
    fn add(&self, other: &Vector) -> Vector;
    // Computes self[i] - other[i]
    fn sub(&self, other: &Vector) -> Vector;
    // Computes self[i] * scaler
    fn scale(&self, scaler: f64) -> Vector;
    // norm of vector
    fn norm(&self) -> f64;
}

impl VecMath for Vector {
    fn dot(&self, other: &Vector) -> f64 {
        assert_eq!(self.len(), other.len());
        self.iter().zip(other.iter()).map(|(a, b)| a * b).sum()
    }

    fn add(&self, other: &Vector) -> Vector {
        assert_eq!(self.len(), other.len());
        self.iter().zip(other.iter()).map(|(a, b)| a + b).collect()
    }

    fn sub(&self, other: &Vector) -> Vector {
        assert_eq!(self.len(), other.len());
        self.iter().zip(other.iter()).map(|(a, b)| a - b).collect()
    }

    fn scale(&self, scaler: f64) -> Vector {
        self.iter().map(|&x| x * scaler).collect()
    }

    fn norm(&self) -> f64 {
        // self.iter().map(|&x| x * x).sum::<f64>().sqrt()
        self.dot(self).sqrt()
    }
}
