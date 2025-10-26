//Copyright (c) 2025, tree-chutes

use core::panic;

use super::{cross_entropy::CrossEntropy, mean_squares::MeanSquares};
use num_traits::Float;

pub enum LossFunctions {
    MeanSquares,
    CrossEntropy
}

pub trait Loss<F: Float> {
    fn forward(&self, t: &[F], p: &[F]) -> Vec<F>;
    fn backward(&self, loss: &[F], dl: &mut [F]) -> Vec<F>;
    fn resize(&self, _t: Vec<F>) -> Vec<F> {
        panic!()
    }
}

pub fn loss_function_factory<F: Float + 'static>(
    l: LossFunctions,
    t: Vec<F>,
    s: F
) -> (Vec<F>, Box<dyn Loss<F>>) {
    match l {
        LossFunctions::MeanSquares => {
            let mut l1 = MeanSquares { one: s, len: t.len() };
            (l1.resize(t), Box::new(l1))
        }
        LossFunctions::CrossEntropy => {
            let mut l1 = CrossEntropy{len: t.len(), zero: s}; 
            (l1.resize(t), Box::new(l1))
        }
    }
}
