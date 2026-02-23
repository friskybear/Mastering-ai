use crate::vec::{VecMath, Vector};

#[derive(Debug, Clone)]
pub struct Matrix {
    row: usize,
    col: usize,
    data: Vector,
}

impl Matrix {
    pub fn new(row: usize, col: usize, data: Vector) -> Self {
        Matrix { row, col, data }
    }
    pub fn zeros(row: usize, col: usize) -> Self {
        Matrix {
            row,
            col,
            data: Vector::new(vec![0.0; row * col]),
        }
    }
    pub fn ones(row: usize, col: usize) -> Self {
        Matrix {
            row,
            col,
            data: Vector::new(vec![1.0; row * col]),
        }
    }
    pub fn get(&self, row: usize, col: usize) -> f64 {
        if row >= self.row || col >= self.col {
            panic!("Index out of bounds");
        }
        self.data.as_slice()[row * self.col + col]
    }
    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        if row >= self.row || col >= self.col {
            panic!("Index out of bounds");
        }
        self.data.as_mut_slice()[row * self.col + col] = value;
    }
    pub fn rows(&self) -> usize {
        self.row
    }
    pub fn cols(&self) -> usize {
        self.col
    }
    pub fn row(&self, row: usize) -> Vector {
        self.data.as_slice()[row * self.col..(row + 1) * self.col]
            .iter()
            .copied()
            .collect()
    }
    pub fn col(&self, col: usize) -> Vector {
        self.data.as_slice()[col..self.row * self.col]
            .chunks(self.col)
            .map(|chunk| chunk[col])
            .collect()
    }
    pub fn transpose(&self) -> Matrix {
        let mut transposed = Matrix::zeros(self.col, self.row);
        for i in 0..self.row {
            for j in 0..self.col {
                transposed.set(j, i, self.get(i, j));
            }
        }
        transposed
    }
    pub fn mat_mul_vec(&self, vec: &Vector) -> Vector {
        assert_eq!(self.cols(), vec.len());
        let mut result = Vector::new(vec![0.0; self.rows()]);
        for i in 0..self.rows() {
            result.as_mut_slice()[i] = self.row(i).dot(vec);
        }
        result
    }
}

impl std::ops::Mul for Matrix {
    type Output = Matrix;

    fn mul(self, other: Matrix) -> Matrix {
        assert_eq!(self.cols(), other.rows());
        let mut result = Matrix::zeros(self.rows(), other.cols());
        for i in 0..self.rows() {
            for j in 0..other.cols() {
                result.set(i, j, self.row(i).dot(&other.col(j)));
            }
        }
        result
    }
}

impl std::ops::Mul<&Matrix> for Matrix {
    type Output = Matrix;

    fn mul(self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols(), other.rows());
        let mut result = Matrix::zeros(self.rows(), other.cols());
        for i in 0..self.rows() {
            for j in 0..other.cols() {
                result.set(i, j, self.row(i).dot(&other.col(j)));
            }
        }
        result
    }
}






