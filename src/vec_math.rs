use std::slice::SliceIndex;

use itertools::Itertools;
#[derive(Debug, Clone)]
pub struct Vector(pub Vec<f64>);
impl Vector {
    pub fn new(vec: Vec<f64>) -> Self {
        Vector(vec)
    }
    pub fn len(&self) -> usize {
        self.0.len()
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
