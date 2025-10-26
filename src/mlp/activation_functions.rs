//Copyright (c) 2025, tree-chutes

use num_traits::Float;
use super::softmax::Softmax;
pub enum Activations {
    Softmax
}


pub trait Activation<F: Float>{
    fn forward(&self, input: &[F])-> Vec<F>;
    fn backward(&self, truth: &[F], z: &[F]) -> Vec<F>;
}


pub fn activation_function_factory<F: Float + 'static>(a: Activations, l: usize, s: F) -> Box<dyn Activation<F>>{
    assert!(l != 0);
    match a {
        Activations::Softmax =>{
            Box::new(Softmax::<F>{len: l, zero: s})
        }
    }
}
