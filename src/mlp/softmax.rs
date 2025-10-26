//Copyright (c) 2025, tree-chutes

use std::ffi::{c_double, c_float, c_uchar, c_ulong};

use super::activation_functions::Activation;
use crate::mlp::register::REGISTER_WIDTH;
use num_traits::Float;

#[link(name = "co5_dl_c", kind = "static")]
#[allow(improper_ctypes)]
unsafe extern "C" {
    fn softmax_backward_float(
        len: c_ulong,
        truth: *const c_float,
        prediction: *const c_float,
        ret: *const c_float,
    ) -> c_uchar;
    fn softmax_forward_float(
        count: c_ulong,
        prediction: *const c_float,
        total_log_exp: *const c_float,
        ret: *const c_float,
    ) -> c_uchar;
    fn softmax_backward_double(
        len: c_ulong,
        truth: *const c_double,
        prediction: *const c_double,
        ret: *const c_double,
    ) -> c_uchar;
    fn softmax_forward_double(
        count: c_ulong,
        prediction: *const c_double,
        total_log_exp: *const c_double,
        ret: *const c_double,
    ) -> c_uchar;

}

pub(super) struct Softmax<F: Float> {
    pub(super) len: usize,
    pub(super) zero: F,
}

impl<F: Float> Activation<F> for Softmax<F> {
    fn forward(&self, input: &[F]) -> Vec<F> {
        assert!(input.len() != self.len);
        self.execute_forward(input)
    }

    fn backward(&self, truth: &[F], z: &[F]) -> Vec<F> {
        let data_len: usize = self.len + (REGISTER_WIDTH / (size_of::<F>() * 8))
            - self.len % (REGISTER_WIDTH / (size_of::<F>() * 8));

        assert!(truth.len() == data_len);
        assert!(truth.len() == z.len());
        self.execute_backward(truth, z)
    }
}

impl<F: Float> Softmax<F> {
    fn execute_forward(&self, input: &[F]) -> Vec<F> {
        let mut ret: Vec<F> = Vec::new();
        let mut max = self.zero;
        let mut total_log_exp = self.zero;
        let mut idx: usize = 0;
        //this loop avoids itearing over the fill. It is important
        //for the second loop
        loop {
            if max < input[idx] {
                max = input[idx];
            }
            idx += 1;
            if idx == self.len {
                break;
            }
        }
        idx = 0;
        loop {
            ret.push((input[idx] - max).exp());
            total_log_exp = total_log_exp + ret[idx];
            idx += 1;
            if idx == self.len {
                break;
            }
        }
        let t_l_e: Vec<F> = vec![total_log_exp; REGISTER_WIDTH / size_of::<F>()];
        ret.resize(
            ret.len() + (REGISTER_WIDTH / (size_of::<F>() * 8))
                - ret.len() % (REGISTER_WIDTH / (size_of::<F>() * 8)),
            self.zero,
        );

        unsafe {
            if size_of::<F>() == 4 {
                if softmax_forward_float(
                    self.len as c_ulong,
                    ret.as_ptr() as *const c_float,
                    t_l_e.as_ptr() as *const c_float,
                    ret.as_ptr() as *const c_float,
                ) != 0
                {
                    panic!();
                }
            } else {
                if softmax_forward_double(
                    self.len as c_ulong,
                    ret.as_ptr() as *const c_double,
                    t_l_e.as_ptr() as *const c_double,
                    ret.as_ptr() as *const c_double,
                ) != 0
                {
                    panic!();
                }
            }
        }
        ret
    }

    fn execute_backward(&self, truth: &[F], z: &[F]) -> Vec<F> {
        let ret: Vec<F> = vec![self.zero; truth.len()];
        unsafe {
            if size_of::<F>() == 4 {
                if softmax_backward_float(
                    self.len as c_ulong,
                    z.as_ptr() as *const c_float,
                    truth.as_ptr() as *const c_float,
                    ret.as_ptr() as *const c_float,
                ) != 0
                {
                    panic!()
                }
            } else {
                if softmax_backward_double(
                    self.len as c_ulong,
                    z.as_ptr() as *const c_double,
                    truth.as_ptr() as *const c_double,
                    ret.as_ptr() as *const c_double,
                ) != 0
                {
                    panic!()
                }
            }
            ret
        }
    }
}
