//Copyright (c) 2025, tree-chutes

use num_traits::Float;

use super::layers::Layer;

pub(in super) struct Softmax<F: Float> {
    pub(in super) i: usize,
    pub(in super) seed: F
}

impl <F: Float> Layer<F> for Softmax<F> {

    fn get_output_shape(&self)-> usize {
        self.i
    }
    
    fn forward(&self, f: &[F], _b: F)-> Vec<F> {
        if self.i != f.len(){
            panic!("f is out of shape: {}, o is {}", f.len(), self.i);
        }
        let mut sum = self.seed;
        let mut ret: Vec<F> = Vec::<F>::with_capacity(f.len());
        f.iter().for_each(|v| {
            sum = sum + v.exp();
        });
        f.iter().for_each(|v|{
            ret.push(v.exp()/sum);
        });
        ret
    }

    fn generate_mapping(&self)-> Vec<(usize,usize)> {
        println!("NOT Required");
        vec![]
    }
}
