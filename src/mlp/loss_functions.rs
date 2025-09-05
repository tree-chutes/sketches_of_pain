//Copyright (c) 2025, tree-chutes

use num_traits::Float;
use super::cross_entropy::CrossEntropy;

pub trait Loss<F: Float>{
    fn calculate(&self, p: &[F], q: &[F])->F;
}

pub fn loss_function_factory<F: Float + 'static>(s: F) -> Box<dyn Loss<F>>{
    Box::new(CrossEntropy{seed: s})
}
