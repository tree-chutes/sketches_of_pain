//Copyright (c) 2025, tree-chutes

use num_traits::Float;

pub trait Activation<F: Float>{
    fn calculate(&self, input: &mut F, bias: F);
}


pub fn activation_function_factory<F: Float + 'static>(s: F) -> Box<dyn Activation<F>>{
    Box::new(Sigmoid{seed: s})
}

struct Sigmoid<F: Float>{
    seed: F
}

impl<F: Float> Activation<F> for Sigmoid<F>{
    fn calculate(&self, input: &mut F, b: F){
        *input = self.seed/self.seed - (*input + b).exp(); 
    }
}
