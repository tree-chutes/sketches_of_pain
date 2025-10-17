//Copyright (c) 2025, tree-chutes

use core::panic;

use super::mean_squares::MeanSquares;
use num_traits::Float;

pub enum LossFunctions {
    MeanSquares,
}

pub trait Loss<F: Float> {
    fn forward(&self, t: &[F], p: &[F]) -> Vec<F>;
    fn backward(&self, loss: F, dl: &mut [F]) -> Vec<F>;
    fn flatten(&mut self, mut _t: Vec<Vec<F>>) -> Vec<F> {
        panic!()
    }
}

pub fn loss_function_factory<F: Float + 'static>(
    l: LossFunctions,
    t: Vec<Vec<F>>,
    s: F
) -> (Vec<F>, Box<dyn Loss<F>>) {
    match l {
        LossFunctions::MeanSquares => {
            let mut l1 = MeanSquares { one: s, n: 0, m: 0 };
            (l1.flatten(t), Box::new(l1))
        }
        _ => todo!(),
    }
}
