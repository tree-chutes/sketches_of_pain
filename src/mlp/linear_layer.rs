//Copyright (c) 2025, tree-chutes

use super::register::REGISTER_WIDTH;
use super::layers::Layer;
use super::{activation_functions::Activation, aggregator_functions::Aggregator};
use num_traits::Float;
use std::{ptr,ffi::{c_double, c_float, c_uchar, c_ulong}};

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
        out: *const c_double
    );

    fn dot_product_float(
        n: c_ulong,
        d: c_ulong,
        m: c_ulong,
        nXd: *const c_float,
        dXm: *const c_float,
        b: *const c_float,
        out: *const c_float
    );

    fn differentiate_double(
        n: c_ulong,
        d: c_ulong,
        m: c_ulong,
        l_r: *const c_double,
        nXd: *const c_double,
        dXm: *const c_double,
        p: *const c_double,
        out: *const c_double
    ) -> c_uchar;

    fn differentiate_float(
        n: c_ulong,
        d: c_ulong,
        m: c_ulong,
        l_r: *const c_float,
        nXd: *const c_float,
        dXm: *const c_float,
        p: *const c_float,
        out: *const c_float
    ) -> c_uchar;
}

pub(super) struct LinearLayer<F: Float> {
    pub(super) zero: F,
    pub(super) d: usize,
    pub(super) m: usize,
    pub(super) n: usize,
}

impl<F: Float> Layer<F> for LinearLayer<F> {
    fn flatten(
        &mut self,
        mut x: Vec<Vec<F>>,
        mut w: Vec<Vec<F>>,
        mut b: Vec<Vec<F>>,
    ) -> (Vec<F>, Vec<F>, Vec<F>) {
        assert!(x.len() != 0);
        assert!(x[0].len() != 0);
        assert!(w.len() != 0); 
        assert!(w[0].len() != 0); //D
        assert!(x[0].len() == w.len()); //D
        assert!(b.len() == x.len());
        assert!(b[0].len() == w[0].len());
        let mut x1 = Vec::<F>::new();
        let mut w1 = Vec::<F>::new();
        let mut b1 = Vec::<F>::new();
        let mut check = x[0].len();

        if self.d == 0 {
            self.n = x.len();
            self.d = x[0].len();
            self.m = w[0].len();
        } else {
            assert!(self.n == x.len());
            assert!(self.d == x[0].len());
            assert!(self.m == w[0].len());
        }
        for idx in 0..x.len() {
            if x[idx].len() != check {
                panic!("X out of shape: {}", idx)
            }
            x1.append(&mut x[idx]);
        }
        x1.resize(
            x1.len() + (REGISTER_WIDTH / (size_of::<F>() * 8)) - x1.len() % (REGISTER_WIDTH / (size_of::<F>() * 8)),
            self.zero,
        );
        check = w[0].len();
        for idx in 0..w.len() {
            if w[idx].len() != check {
                panic!("W out of shape: {}", idx)
            }
            w1.append(&mut w[idx]);
        }
        w1.resize(
            w1.len() + (REGISTER_WIDTH / (size_of::<F>() * 8)) - w1.len() % (REGISTER_WIDTH / (size_of::<F>() * 8)),
            self.zero,
        );
        for idx in 0..b.len() {
            if b[idx].len() != check {
                panic!("W out of shape: {}", idx)
            }
            b1.append(&mut b[idx]);
        }
        b1.resize(
            b1.len() + (REGISTER_WIDTH / (size_of::<F>() * 8)) - b1.len() % (REGISTER_WIDTH / (size_of::<F>() * 8)),
            self.zero,
        );

        (x1, w1, b1)
    }

    fn forward(&self, mut d: (&[F], &mut [F], &[F]), _a: Option<Box<dyn Aggregator<F>>>) -> Vec<F> {
        let data_len: usize = self.n * self.d + (REGISTER_WIDTH / (size_of::<F>() * 8))
            - self.n * self.d % (REGISTER_WIDTH / (size_of::<F>() * 8));
        let weights_len: usize = self.d * self.m + (REGISTER_WIDTH / (size_of::<F>() * 8))
            - self.d * self.m % (REGISTER_WIDTH / (size_of::<F>() * 8));
        let biases_len: usize = self.n * self.m + (REGISTER_WIDTH / (size_of::<F>() * 8))
            - self.n * self.m % (REGISTER_WIDTH / (size_of::<F>() * 8));

        if  d.0.len() != data_len ||
            d.1.len() != weights_len ||
            d.2.len() != biases_len{
                return vec![];
            }
        self.execute_forward(&mut d.0, d.1, &d.2)
    }

    fn backward(&self, d: (&mut [F], &mut [F], &[F]), z: &mut [F], l_r: F) -> (Vec<F>, Vec<F>, Vec<F>) {
        let data_len: usize = self.n * self.d + (REGISTER_WIDTH / (size_of::<F>() * 8))
            - self.n * self.d % (REGISTER_WIDTH / (size_of::<F>() * 8));
        let weights_len: usize = self.d * self.m + (REGISTER_WIDTH / (size_of::<F>() * 8))
            - self.d * self.m % (REGISTER_WIDTH / (size_of::<F>() * 8));
        let biases_len: usize = self.n * self.m + (REGISTER_WIDTH / (size_of::<F>() * 8))
            - self.n * self.m % (REGISTER_WIDTH / (size_of::<F>() * 8));

        if  d.0.len() != data_len ||
            d.1.len() != weights_len ||
            d.2.len() != biases_len ||
            z.len() != biases_len{
                return (vec![], vec![], vec![]);
        }
        self.execute_backward(d.0, d.1, d.2, z, l_r)
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
            transposed[idx0] =  m[c];
        }
        unsafe{
            ptr::copy_nonoverlapping(transposed.as_ptr(), m.as_mut_ptr(), len);
        }
    }

    fn execute_forward(&self, x: &[F], w: &mut [F], b: &[F]) -> Vec<F> {
        let mut ret = vec![self.zero; self.n * self.m];

        self.transpose(w, self.d, self.m);

        unsafe {
            if size_of::<F>() == 8 {
                dot_product_double(
                    self.n as c_ulong,
                    self.d as c_ulong,
                    self.m as c_ulong,
                    x.as_ptr() as *const c_double,
                    w.as_ptr() as *const c_double,
                    b.as_ptr() as *const c_double,
                    ret.as_ptr() as *const c_double,
                );
            } else {
                dot_product_float(
                    self.n as c_ulong,
                    self.d as c_ulong,
                    self.m as c_ulong,
                    x.as_ptr() as *const c_float,
                    w.as_ptr() as *const c_float,
                    b.as_ptr() as *const c_float,
                    ret.as_ptr() as *const c_float,
                );
            }
        }
        ret.resize(
            ret.len() + (REGISTER_WIDTH / (size_of::<F>() * 8)) - ret.len() % (REGISTER_WIDTH / (size_of::<F>() * 8)),
            self.zero,
        );
        ret
    }

    fn execute_backward(&self, x: &mut [F], w: &mut [F], b: &[F], z: &mut [F], l_r: F) -> (Vec<F>, Vec<F>, Vec<F>) {
        let ret_w = vec![self.zero; self.d * self.m];
        let ret_x = vec![self.zero; self.d * self.n];
        let learning_rate = vec![l_r; REGISTER_WIDTH / (size_of::<F>() * 8)]; 

        self.transpose(x, self.n, self.d);
        self.transpose(w, self.m, self.d);
        self.transpose(z, self.n, self.m);
        unsafe {
            if size_of::<F>() == 4 {
                differentiate_float(
                    self.d as c_ulong,
                    self.m as c_ulong,
                    self.n as c_ulong,
                    learning_rate.as_ptr() as *const c_float,
                    x.as_ptr() as *const c_float,
                    z.as_ptr() as *const c_float,                                         
                    w.as_ptr() as *const c_float,
                    ret_w.as_ptr() as *const c_float,
                );
            }
            else{
                differentiate_double(
                    self.d as c_ulong,
                    self.m as c_ulong,
                    self.n as c_ulong,
                    learning_rate.as_ptr() as *const c_double,
                    x.as_ptr() as *const c_double,
                    z.as_ptr() as *const c_double,                                         
                    w.as_ptr() as *const c_double,
                    ret_w.as_ptr() as *const c_double,
                );                
            }
            self.transpose(x, self.d, self.n);
            self.transpose(w, self.d, self.m);
            if size_of::<F>() == 4 {
                differentiate_float(
                    self.d as c_ulong,
                    self.m as c_ulong,
                    self.n as c_ulong,
                    learning_rate.as_ptr() as *const c_float,
                    z.as_ptr() as *const c_float,
                    w.as_ptr() as *const c_float,                                         
                    x.as_ptr() as *const c_float,
                    ret_x.as_ptr() as *const c_float,
                );        
            }
            else{
                differentiate_double(
                    self.d as c_ulong,
                    self.m as c_ulong,
                    self.n as c_ulong,
                    learning_rate.as_ptr() as *const c_double,
                    z.as_ptr() as *const c_double,
                    w.as_ptr() as *const c_double,                                         
                    x.as_ptr() as *const c_double,
                    ret_x.as_ptr() as *const c_double,
                );        
            }
        }
        (ret_x, ret_w, z.to_vec())
    }
}
