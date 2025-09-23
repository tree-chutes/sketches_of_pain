//Copyright (c) 2025, tree-chutes

use super::layers::Layer;
use super::{activation_functions::Activation, aggregator_functions::Aggregator};
use num_traits::Float;
use std::ffi::{c_double, c_uchar, c_ulong, c_float};

#[link(name = "co5_dl_c", kind = "static")]
#[allow(improper_ctypes)]
unsafe extern "C" {
    fn linear_double(
        n: c_ulong,
        d: c_ulong,
        m: c_ulong,
        rearranged: *const c_double,
        dXm: *const c_double,
        b: *const c_double,
        out: *const c_double,
    );

    fn linear_float(
        n: c_ulong,
        d: c_ulong,
        m: c_ulong,
        rearranged: *const c_float,
        dXm: *const c_float,
        b: *const c_float,
        out: *const c_float,
    );

}

pub(super) struct LinearLayer<F: Float> {
    pub(super) zero: F,
    pub(super) d: usize,
    pub(super) m: usize,
    pub(super) n: usize
}

impl<F: Float> Layer<F> for LinearLayer<F> {
    fn flatten(&mut self, mut x: Vec<Vec<F>>, mut w: Vec<Vec<F>>, mut b: Vec<Vec<F>>) -> (Vec<F>, Vec<F>, Vec<F>) {
        assert!(x.len() != 0);
        assert!(x[0].len() != 0);
        assert!(w.len() != 0);
        assert!(w[0].len() != 0);
        assert!(x[0].len() == w.len());
        assert!(b.len() == x.len());
        assert!(b[0].len() == w[0].len());
        let mut x1 = Vec::<F>::new();
        let mut w1 = Vec::<F>::new();
        let mut b1 = Vec::<F>::new();
        let mut check = x[0].len();

        if self.d == 0{
            self.n = x.len();
            self.d = x[0].len();
            self.m = w[0].len();
        }
        else{
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
        x1.resize(x1.len() + (256 / (size_of::<F>() * 8)) - x1.len() % (256 / (size_of::<F>() * 8)), self.zero);
        check = w[0].len();
        for idx in 0..w.len() {
            if w[idx].len() != check {
                panic!("W out of shape: {}", idx)
            }
            w1.append(&mut w[idx]);
        }
        w1.resize(w1.len()  + (256 / (size_of::<F>() * 8)) - w1.len() % (256 / (size_of::<F>() * 8)), self.zero);
        for idx in 0..b.len() {
            if b[idx].len() != check {
                panic!("W out of shape: {}", idx)
            }
            b1.append(&mut b[idx]);
        }
        //TODO: make 256 a constant
        b1.resize(b1.len() + (256 / (size_of::<F>() * 8)) - b1.len() % (256 / (size_of::<F>() * 8)), self.zero);

        (x1, w1, b1)
    }

    fn forward(&self, mut d: (&mut [F], &[F], &[F]), _a: Option<Box<dyn Aggregator<F>>>) -> Vec<F> {
        let data_len: usize = self.n * self.d  + (256 / (size_of::<F>() * 8)) - self.n * self.d % (256 / (size_of::<F>() * 8));
        let weights_len: usize = self.d * self.m + (256 / (size_of::<F>() * 8)) - self.d * self.m % (256 / (size_of::<F>() * 8));
        let biases_len: usize = self.n * self.m  + (256 / (size_of::<F>() * 8)) - self.n * self.m % (256 / (size_of::<F>() * 8));

        assert!(d.0.len() == data_len);
        assert!(d.1.len() == weights_len);
        assert!(d.2.len() == biases_len);
        self.execute(&mut d.0, &d.1, &d.2)
    }

    fn backward(&self, d: (&mut [F], &[F], &[F]), z: &[F]) -> Vec<F> {
        todo![]
    }
}

impl<F: Float> LinearLayer<F> {

    fn transpose(&self, m: &mut [F], n: usize, d: usize) {
        let mut tmp: F;
        let mut idx0: usize;
        let mut j: usize;
        let mut i: usize;
        let mut idx: usize = 1;
        let mut idx1: usize = 1;

        for c in 1..(n * d) - 1 {
            j = idx / d;
            i = idx % d;
            idx0 = i * n + j % n;
            tmp = m[idx0];
            if c != 1{
                m[idx0] = m[idx1];
                m[idx1] = tmp;                    
            }
            else {
                m[idx0] = m[idx];
                idx1 = idx; 
            }
            m[idx1] = tmp;                    
            idx = idx0;
        }
    }

    fn execute(&self, x: &mut [F], w: &[F], b: &[F]) -> Vec<F> {
        let mut ret = vec![self.zero; self.n * self.m];
        let mut rearranged = Vec::<F>::new();

        self.transpose(x, self.n, self.d);

        for i in 0..(self.n * self.d){
            rearranged.resize((1 + i) * self.m, x[i]);
        }

        unsafe {
            if size_of::<F>() == 8{
                linear_double(
                    self.n as c_ulong,
                    self.d as c_ulong,
                    self.m as c_ulong,
                    rearranged.as_ptr() as *const c_double,
                    w.as_ptr() as *const c_double,
                    b.as_ptr() as *const c_double,
                    ret.as_ptr() as *const c_double,
                );
            }
            else{
                linear_float(
                    self.n as c_ulong,
                    self.d as c_ulong,
                    self.m as c_ulong,
                    rearranged.as_ptr() as *const c_float,
                    w.as_ptr() as *const c_float,
                    b.as_ptr() as *const c_float,
                    ret.as_ptr() as *const c_float,
                );
            }
        }
        ret.resize(
            ret.len() +  (256 / (size_of::<F>() * 8)) - ret.len() % (256 / (size_of::<F>() * 8)),
            self.zero
        );
        ret
    }
}
