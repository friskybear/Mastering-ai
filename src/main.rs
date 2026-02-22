mod matrix;
mod vec_math;
mod util;
use util::*;
use matrix::Matrix;

use vec_math::{VecMath, Vector};
fn main() {
    print!("{:?}", numerical_gradient(|v| v[0]*v[0] + v[1]*v[1], &[3.0, 4.0], 1e-5));
}
