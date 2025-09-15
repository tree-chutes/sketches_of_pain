//Copyright (c) 2025, tree-chutes

use super::activation_functions::Activation;
use super::layers::Layer;
use num_traits::Float;
use std::ffi::{c_double, c_uchar, c_ulong};

#[link(name = "co5_dl_c", kind = "static")]
#[allow(improper_ctypes)]
unsafe extern "C" {
    unsafe fn multiply(
        c: c_ulong,
        t: c_ulong,
        j: c_ulong,
        in0: *const c_double,
        in1: *const c_double,
        out: *mut c_double,
    ) -> c_uchar;
}

pub(super) struct Linear<F: Float> {
    pub(super) zero: F,
    pub(super) fill: usize,
    pub(super) d: usize,
    pub(super) m: usize,
    pub(super) n: usize,
    pub(super) a: Box<dyn Activation<F>>,
    pub(super) activations: Vec<F>,
}

impl<F: Float> Layer<F> for Linear<F> {
    fn get_output_shape(&self) -> usize {
        self.m * self.n
    }

    fn forward(&self, w: &[F], f: &[F], b: F) -> (Vec<F>, Vec<F>) {
        self.multiply(w, f)
    }
}

impl<F: Float> Linear<F> {
    pub(super) fn flatten_weights(&mut self, w: Vec<Vec<F>>) -> Vec<F> {
        assert!(w.len() == self.d);
        let mut ret = Vec::<F>::new();
        for j in 0..self.m {
            for i in 0..self.d {
                ret.push(w[i][j]);
            }
        }
        self.fill = ret.len() % 4; //TODO CONSTANt
        if self.fill != 0 {
            self.fill = 4 - self.fill;
            ret.append(&mut vec![self.zero; self.fill]);
        }
        ret
    }

    pub(super) fn generate_input_mapping(&self) -> Vec<(usize, usize)> {
        let fill: usize;
        let mut ret = Vec::<(usize, usize)>::with_capacity(self.n * self.m);

        for i in 0..self.n {
            for _ in 0..self.m {
                for j in 0..self.d {
                    ret.push((j, i));
                }
            }
        }
        fill = ret.len() % 4;
        if fill != 0 {
            ret.append(&mut vec![(0, 0); 4 - fill]);
        }
        ret
    }

    fn multiply(&self, f: &[F], w: &[F]) -> (Vec<F>, Vec<F>) {
        assert!(w.len() != 0 && f.len() != 0);
        let mut reti: usize;
        let mut ret: Vec<F>;
        let mut transposed: Vec<F>;
        let mut sum = self.zero;
        let len = self.n * self.m;
        assert!(f.len() == len * self.d);
        assert!(w.len() == len + self.fill);
        let counter = f.len() / self.d;
        let trigger = w.len() / 4; //TODO CONSTANT

        ret = vec![self.zero; len * self.d];
        unsafe {
            if multiply(
                counter as c_ulong,
                trigger as c_ulong,
                self.fill as c_ulong,
                f.as_ptr() as *const f64,
                w.as_ptr() as *const f64,
                ret.as_mut_ptr() as *mut f64,
            ) != 0
            {
                panic!("failed to multiply");
            }
            transposed = vec![self.zero; self.n * self.m];
            for i in 0..(len * self.d) {
                if i != 0 && i % self.d == 0 {
                    reti = i / self.d - 1;
                    ret[reti] = sum;
                    if reti % self.m == 0{
                        transposed[reti / self.m] = ret[reti] 
                    }
                    else{
                        transposed[reti / self.m + reti % self.m * self.d] = ret[reti];
                    }
                    sum = self.zero;
                }
                sum = sum + ret[i];
            }
            ret.resize(self.n * self.m, self.zero);
            ret[len - 1] = sum;
            transposed[len - 1] = sum;
            (ret, transposed)
        }
    }
}
