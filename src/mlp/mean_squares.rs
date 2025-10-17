//Copyright (c) 2025, tree-chutes

use std::ffi::{c_double, c_float, c_uchar, c_ulong};
use std::ptr;

use super::register::REGISTER_WIDTH;
use super::loss_functions::Loss;
use num_traits::Float;

#[link(name = "co5_dl_c", kind = "static")]
#[allow(improper_ctypes)]
unsafe extern "C" {
    unsafe fn squared_loss_float(
        s: c_ulong,
        p_inout: *const c_float,
        t: *const c_float,
        out: *const c_float,
    ) -> c_uchar;

    unsafe fn squared_loss_double(
        s: c_ulong,
        p_inout: *const c_double,
        t: *const c_double,
        out: *const c_double,
    ) -> c_uchar;

    unsafe fn scalar_X_matrix_double(c: c_ulong, m_inout: *const c_double, s: *const c_double, o: *const c_double)-> c_uchar;
    unsafe fn scalar_X_matrix_float(c: c_ulong, m_inout: *const c_float, s: *const c_float, o: *const c_float)-> c_uchar;
}

pub(super) struct MeanSquares<F: Float> {
    pub(super) one: F,
    pub(super) n: usize,
    pub(super) m: usize,
}

impl<F: Float> Loss<F> for MeanSquares<F> {
    fn flatten(&mut self, mut t: Vec<Vec<F>>) -> Vec<F> {
        assert!(t.len() != 0);
        assert!(t[0].len() != 0);
        let mut ret: Vec<F> = vec![];
        let check = t[0].len();

        if self.n == 0 {
            self.n = t.len();
            self.m = check;
        } else {
            assert!(self.n == t.len());
            assert!(self.m == check);
        }
        for idx in 0..t.len() {
            if t[idx].len() != check {
                panic!("Truth out of shape: {}", idx)
            }
            ret.append(&mut t[idx]);
        }
        ret.resize(
            ret.len() + (REGISTER_WIDTH / (size_of::<F>() * 8)) - ret.len() % (REGISTER_WIDTH / (size_of::<F>() * 8)),
            self.one - self.one,
        );
        ret
    }

    fn forward(&self, t: &[F], p: &[F]) -> Vec<F> {
        let truth_len: usize = self.n * self.m + (REGISTER_WIDTH / (size_of::<F>() * 8))
            - self.n * self.m % (REGISTER_WIDTH / (size_of::<F>() * 8));
        assert!(t.len() == truth_len);
        assert!(p.len() == truth_len);
        self.execute_forward(t, p)
    }

    fn backward(&self, loss: F, dl: &mut [F])-> Vec<F>{
        self.execute_backward(loss,dl)
    }
}

impl<F: Float> MeanSquares<F> {
    fn execute_backward(&self, loss: F, dl: &mut [F])-> Vec<F>{
        let width = REGISTER_WIDTH / (size_of::<F>() * 8);  
        let backpropagating_gradients = vec![self.one; dl.len()];      
        unsafe {
            if size_of::<F>() == 8 {
                if scalar_X_matrix_double(
                    dl.len() as c_ulong,
                    dl.as_ptr() as *const c_double,
                    vec![loss ; width].as_ptr() as *const c_double,
                    backpropagating_gradients.as_ptr() as *const c_double
                ) != 0{
                    panic!("can't dl")
                }
            }
            else{
                //let diff = 2.0 / (self.n * self.m) as f32;
                if scalar_X_matrix_float(
                    dl.len() as c_ulong,
                    dl.as_ptr() as *const c_float,
                    vec![loss; width].as_ptr() as *const c_float,
                    backpropagating_gradients.as_ptr() as *const c_float
                ) != 0{
                    panic!("can't dl")
                }
            }
        }
        backpropagating_gradients
    }

    fn execute_forward(&self, t: &[F], p: &[F]) -> Vec<F> {
        let zero = self.one - self.one;
        let total = self.n * self.m;
        let ret: Vec<F> = vec![zero; total];
        unsafe {
            if size_of::<F>() == 8 {
                if squared_loss_double(
                    total as c_ulong,
                    p.as_ptr() as *const c_double,
                    t.as_ptr() as *const c_double,
                    ret.as_ptr() as *const c_double,
                ) != 0
                {
                    panic!("failed mse");
                }
            } else {
                if squared_loss_float(
                    p.len() as c_ulong,
                    p.as_ptr() as *const c_float,
                    t.as_ptr() as *const c_float,
                    ret.as_ptr() as *const c_float,
                ) != 0
                {
                    panic!("failed mse");
                }
            }
        }
        ret
    }
}
