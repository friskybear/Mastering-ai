use crate::vec::{VecMath, Vector};

pub enum Loss {
    MAE,
    MSE,
    BCE,
}
pub struct LossFunction {
    loss: Loss,
}

impl LossFunction {
    pub fn new(loss: Loss) -> Self {
        LossFunction { loss }
    }

    pub fn calculate(&self, predicted: &[f64], actual: &[f64]) -> f64 {
        match self.loss {
            Loss::MAE => mae(predicted, actual),
            Loss::MSE => mse(predicted, actual),
            Loss::BCE => binary_cross_entropy(predicted, actual),
        }
    }
    pub fn batch_loss(loss_fn: &LossFunction, predicted: &[Vector], actual: &[Vector]) -> f64 {
        assert_eq!(predicted.len(), actual.len(), "batch size mismatch");
        let n = predicted.len() as f64;
        predicted
            .iter()
            .zip(actual)
            .map(|(p, a)| loss_fn.calculate(p.as_slice(), a.as_slice()))
            .sum::<f64>()
            / n
    }
    pub fn gradient(&self, predicted: &Vector, actual: &Vector) -> Vector {
        let n = predicted.len() as f64;
        match self.loss {
            // d/dŷ [ (1/n) * sum((ŷ - y)^2) ] = (2/n) * (ŷ - y)
            Loss::MSE => predicted.sub(actual).scale(2.0 / n),
            // d/dŷ [ (1/n) * sum(|ŷ - y|) ] = (1/n) * sign(ŷ - y)
            Loss::MAE => {
                let raw: Vec<f64> = predicted
                    .iter()
                    .zip(actual.iter())
                    .map(|(&yhat, &y)| (yhat - y).signum() / n)
                    .collect();
                Vector::from(raw)
            }
            // d/dŷ [ -(1/n) * sum(y*ln(ŷ) + (1-y)*ln(1-ŷ)) ]
            // = (1/n) * (-(y/ŷ) + (1-y)/(1-ŷ))
            Loss::BCE => {
                let raw: Vec<f64> = predicted
                    .iter()
                    .zip(actual.iter())
                    .map(|(&yhat, &y)| {
                        let yhat = yhat.clamp(std::f64::EPSILON, 1.0 - std::f64::EPSILON);
                        (-(y / yhat) + (1.0 - y) / (1.0 - yhat)) / n
                    })
                    .collect();
                Vector::from(raw)
            }
        }
    }
}

pub fn mae(predicted: &[f64], actual: &[f64]) -> f64 {
    assert!(!predicted.is_empty(), "mae: empty input");
    assert_eq!(predicted.len(), actual.len(), "mae: length mismatch");
    predicted
        .iter()
        .zip(actual)
        .map(|(&yhat, &y)| (yhat - y).abs())
        .sum::<f64>()
        / predicted.len() as f64
}
pub fn mse(predicted: &[f64], actual: &[f64]) -> f64 {
    assert!(!predicted.is_empty(), "mse: empty input");
    assert_eq!(predicted.len(), actual.len(), "mse: length mismatch");
    predicted
        .iter()
        .zip(actual)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / (predicted.len() as f64)
}
pub fn binary_cross_entropy(predicted: &[f64], actual: &[f64]) -> f64 {
    assert!(!predicted.is_empty(), "bce: empty input");
    assert_eq!(predicted.len(), actual.len(), "bce: length mismatch");
    predicted
        .iter()
        .zip(actual.iter())
        .map(|(&yhat, &y)| {
            let yhat = yhat.clamp(std::f64::EPSILON, 1.0 - std::f64::EPSILON);
            y * yhat.ln() + (1.0 - y) * (1.0 - yhat).ln()
        })
        .sum::<f64>()
        * -1.0
        / predicted.len() as f64
}
