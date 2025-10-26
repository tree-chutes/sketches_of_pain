//Copyright (c) 2025, tree-chutes

use super::loss_functions::Loss;
use super::register::REGISTER_WIDTH;
use num_traits::Float;

pub(super) struct CrossEntropy<F: Float> {
    pub(super) len: usize,
    pub(super) zero: F,
}

impl<F: Float> Loss<F> for CrossEntropy<F> {
    fn resize(&self, t: Vec<F>) -> Vec<F> {
        assert!(t.len() == self.len);
        let mut ret: Vec<F> = t;
        ret.resize(
            ret.len() + (REGISTER_WIDTH / (size_of::<F>() * 8))
                - ret.len() % (REGISTER_WIDTH / (size_of::<F>() * 8)),
            self.zero,
        );
        ret
    }

    fn forward(&self, t: &[F], p: &[F]) -> Vec<F> {
        let truth_len: usize = self.len + (REGISTER_WIDTH / (size_of::<F>() * 8))
            - self.len % (REGISTER_WIDTH / (size_of::<F>() * 8));

        assert!(t.len() == truth_len);
        assert!(p.len() == truth_len);
        self.execute_forward(t, p)
    }

    fn backward(&self, t: &[F], p: &mut [F]) -> Vec<F> {
        vec![]
    }
}

impl<F:Float> CrossEntropy<F>{
    fn execute_forward(&self, t: &[F], p: &[F]) -> Vec<F>{
        let mut ret: Vec<F> = vec![self.zero; 1];
        let mut idx: usize = 0;
        loop{
            if t[idx] != self.zero{
                ret[0] = -p[idx].ln();
                break;
            }
            idx += 1;
        }
        ret
    }
}
