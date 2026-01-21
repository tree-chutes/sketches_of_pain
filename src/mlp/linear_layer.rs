//Copyright (c) 2025, tree-chutes

use super::layers::Layer;
use super::register::REGISTER_WIDTH;
use super::aggregator_functions::Aggregator;
use num_traits::Float;
use std::{
    ffi::{c_double, c_float, c_uchar, c_ulong},
    ptr,
};

#[link(name = "co5_dl_c", kind = "static")]
#[allow(improper_ctypes)]
unsafe extern "C" {
    fn dot_product_double(
        n: c_ulong,
        d: c_ulong,
        m: c_ulong,
        nXd: *const c_double,
        dXm: *const c_double,
        b: *const c_double,
        out: *const c_double,
    ) -> c_uchar;

    fn dot_product_float(
        n: c_ulong,
        d: c_ulong,
        m: c_ulong,
        nXd: *const c_float,
        dXm: *const c_float,
        b: *const c_float,
        out: *const c_float,
    )-> c_uchar;

    fn linear_sgd_double(
        n: c_ulong,
        d: c_ulong,
        m: c_ulong,
        learning_rate: *const c_double,
        weight_gradients: *const c_double,
        weights: *const c_double,
        loss_gradient: *const c_double,
        backpropagating_gradients: *const c_double
    ) -> c_uchar;

    fn linear_sgd_float(
        n: c_ulong,
        d: c_ulong,
        m: c_ulong,
        learning_rate: *const c_float,
        input: *const c_float,
        weights: *const c_float,
        loss_gradient: *const c_float,
        updated_weights: *const c_float,
        updated_input: *const c_float,
    ) -> c_uchar;
}

pub(super) struct LinearLayer<F: Float> {
    pub(super) zero: F,
    pub(super) d: usize,
    pub(super) m: usize,
    pub(super) n: usize,
    pub(super) is_first_layer: bool
}

impl<F: Float> Layer<F> for LinearLayer<F> {

    fn set_first_layer_flag(&mut self) {
        self.is_first_layer = true;
    }

    fn flatten(
        &self,
        mut x: Vec<Vec<F>>,
        w: Vec<Vec<F>>,
        mut b: Vec<Vec<F>>,
    ) -> (Vec<F>, Vec<F>, Vec<F>) {
        assert!(x.len() != 0);
        assert!(x[0].len() != 0);
        assert!(x[0].len() == w.len()); //D
        // assert!(b.len() == x.len());
        // assert!(b[0].len() == x[0].len());
        let mut ret_x = Vec::<F>::new();
        let mut b1 = Vec::<F>::new();
        let check = x[0].len();

        assert!(self.n == x.len());
        assert!(self.d == x[0].len());
//        assert!(self.m == w[0].len());

        for idx in 0..x.len() {
            if x[idx].len() != check {
                panic!("X out of shape: {}", idx)
            }
            ret_x.append(&mut x[idx]);
        }
        ret_x.resize(
            ret_x.len() + (REGISTER_WIDTH / (size_of::<F>() * 8))
                - ret_x.len() % (REGISTER_WIDTH / (size_of::<F>() * 8)),
            self.zero,
        );
        for idx in 0..b.len() {
            // if b[idx].len() != check {
            //     panic!("W out of shape: {}", idx)
            // }
            b1.append(&mut b[idx]);
        }
        b1.resize(
            b1.len() + (REGISTER_WIDTH / (size_of::<F>() * 8))
                - b1.len() % (REGISTER_WIDTH / (size_of::<F>() * 8)),
            self.zero,
        );

        (ret_x, self.flatten_kernel(w), b1)
    }

    fn flatten_kernel(&self, mut w: Vec<Vec<F>>) -> Vec<F> {
        assert!(w.len() != 0);
        assert!(w[0].len() != 0); //D
        assert!(self.d == w.len());
        assert!(self.m == w[0].len());        
        let check = w[0].len();
        let mut ret_w = vec![];

        for idx in 0..w.len() {
            if w[idx].len() != check {
                panic!("W out of shape: {}", idx)
            }
            ret_w.append(&mut w[idx]);
        }
        ret_w.resize(
            ret_w.len() + (REGISTER_WIDTH / (size_of::<F>() * 8))
                - ret_w.len() % (REGISTER_WIDTH / (size_of::<F>() * 8)),
            self.zero,
        );
        ret_w
    }

    fn forward(&self, mut d: (&[F], &mut [F], &[F]), _a: Option<Box<dyn Aggregator<F>>>) -> Vec<F> {
        let data_len: usize = self.n * self.d + (REGISTER_WIDTH / (size_of::<F>() * 8))
            - self.n * self.d % (REGISTER_WIDTH / (size_of::<F>() * 8));
        let weights_len: usize = self.d * self.m + (REGISTER_WIDTH / (size_of::<F>() * 8))
            - self.d * self.m % (REGISTER_WIDTH / (size_of::<F>() * 8));
        let _biases_len: usize = self.n * self.m + (REGISTER_WIDTH / (size_of::<F>() * 8))
            - self.n * self.m % (REGISTER_WIDTH / (size_of::<F>() * 8));

        if d.0.len() != data_len || d.1.len() != weights_len /*|| d.2.len() != data_len*/ {
            return vec![];
        }
        self.execute_forward(&mut d.0, d.1, &d.2)
    }

    fn backward(
        &self,
        d: (&mut [F], &mut [F], &mut [F]),
        l_r: F,
        l: F,
    ) -> (Vec<F>, Vec<F>) {
        let data_len: usize = self.n * self.d + (REGISTER_WIDTH / (size_of::<F>() * 8))
            - self.n * self.d % (REGISTER_WIDTH / (size_of::<F>() * 8));
        let weights_len: usize = self.d * self.m + (REGISTER_WIDTH / (size_of::<F>() * 8))
            - self.d * self.m % (REGISTER_WIDTH / (size_of::<F>() * 8));
        let _biases_len: usize = self.n * self.m + (REGISTER_WIDTH / (size_of::<F>() * 8))
            - self.n * self.m % (REGISTER_WIDTH / (size_of::<F>() * 8));
        
        if d.0.len() != data_len || d.1.len() != weights_len {
            todo!("properly log this without causing a shutdown");
        }
        self.execute_backward(d.0, d.1, d.2, l_r, l)
    }
}

impl<F: Float> LinearLayer<F> {
    fn transpose(&self, m: &mut [F], n: usize, d: usize) {
        let len = n * d;
        let mut idx0: usize;
        let mut transposed: Vec<F> = vec![self.zero; len];

        for c in 0..len {
            // i = c % d;
            // j = c / d;
            idx0 = (c % d) * n + (c / d);
            transposed[idx0] = m[c];
        }
        unsafe {
            ptr::copy_nonoverlapping(transposed.as_ptr(), m.as_mut_ptr(), len);
        }
    }

    fn execute_forward(&self, x: &[F], w: &mut [F], b: &[F]) -> Vec<F> {
        let mut ret = vec![self.zero; self.n * self.m];

        unsafe {
            if size_of::<F>() == 8 {
                if dot_product_double(
                    self.n as c_ulong,
                    self.d as c_ulong,
                    self.m as c_ulong,
                    x.as_ptr() as *const c_double,
                    w.as_ptr() as *const c_double,
                    b.as_ptr() as *const c_double,
                    ret.as_ptr() as *const c_double,
                ) != 0{
                    panic!("linear double forward pass failed")
                }
            } else {
                if dot_product_float(
                    self.n as c_ulong,
                    self.d as c_ulong,
                    self.m as c_ulong,
                    x.as_ptr() as *const c_float,
                    w.as_ptr() as *const c_float,
                    b.as_ptr() as *const c_float,
                    ret.as_ptr() as *const c_float,
                ) != 0{
                    panic!("linear float forward pass failed")
                }
            }
        }
        ret.resize(
            ret.len() + (REGISTER_WIDTH / (size_of::<F>() * 8))
                - ret.len() % (REGISTER_WIDTH / (size_of::<F>() * 8)),
            self.zero,
        );
        ret
    }

    fn execute_backward(
        &self,
        input: &[F],
        weights: &[F],
        _bias: &[F],
        l_r: F,
        l: F,
    ) -> (Vec<F>, Vec<F>) {
        let out_len = self.d * self.m + (REGISTER_WIDTH / (size_of::<F>() * 8)) - self.d * self.m % (REGISTER_WIDTH / (size_of::<F>() * 8));
        let updated_weights = vec![self.zero; out_len];
        let updated_input = vec![self.zero; out_len];
        let learning_rate = vec![l_r; REGISTER_WIDTH / (size_of::<F>() * 8)];
        //if loss_gradient were a vector, just pass it and avoid this
        let loss_gradient = vec![l; REGISTER_WIDTH / (size_of::<F>() * 8)];

        unsafe {
            if size_of::<F>() == 4 {
                if linear_sgd_float(
                    self.d as c_ulong,
                    self.m as c_ulong,
                    self.n as c_ulong,
                    learning_rate.as_ptr() as *const c_float,
                    input.as_ptr() as *const c_float,
                    weights.as_ptr() as *const c_float,
                    loss_gradient.as_ptr() as *const c_float,
                    updated_weights.as_ptr() as *const c_float,
                    updated_input.as_ptr() as *const c_float
                ) != 0{
                    panic!("linear float backpass failed")
                }
            } else {
                if linear_sgd_double(
                    self.n as c_ulong,
                    self.d as c_ulong,
                    self.m as c_ulong,
                    learning_rate.as_ptr() as *const c_double,
                    input.as_ptr() as *const c_double,
                    weights.as_ptr() as *const c_double,
                    loss_gradient.as_ptr() as *const c_double,
                    updated_weights.as_ptr() as *const c_double,
                ) != 0{
                    panic!("linear double backpass failed")
                }
            }
        }
        (updated_input, updated_weights)
    }
}
