//Copyright (c) 2025, tree-chutes

use num_traits::Float;
use super::loss_functions::Loss;

pub(in super) struct CrossEntropy<F: Float>{
    pub (in super) seed: F,
}

impl<F: Float> Loss<F> for CrossEntropy<F>{
    fn calculate(&self, p: &[F], q: &[F])-> F{
        let mut ret: F = self.seed;
        if p.len() != q.len(){
            panic!("p shape is {}, q shape is {}", p.len(), q.len());
        }
        for i in 0..(p.len()){
            ret = ret + p[i] * q[i].log2();
        }
        -ret
    }
}
