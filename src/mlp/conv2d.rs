//Copyright (c) 2025, tree-chutes

use super::layers::Layer;
use super::register::REGISTER_WIDTH;
use super::{activation_functions::Activation, aggregator_functions::Aggregator};
use num_traits::Float;
use std::ffi::{c_double, c_uchar, c_ulong, c_ushort};

#[link(name = "co5_dl_c", kind = "static")]
#[allow(improper_ctypes)]
unsafe extern "C" {
    fn convolve_forward_2d_double(
        w: c_ulong,
        h: c_ulong,
        k: c_ulong,
        c: c_ulong,
        p: c_ushort,
        s: c_ushort,
        hXw: *const c_double,
        kXk: *const c_double,
        out: *mut c_double,
    ) -> c_uchar;
}

pub(super) struct Conv2D<F: Float> {
    pub(super) zero: F,
    pub(super) h: usize,
    pub(super) w: usize,
    pub(super) k: usize,
    pub(super) p: u16,
    pub(super) s: u16,
}

impl<F: Float> Layer<F> for Conv2D<F> {
    fn flatten(
        &mut self,
        mut f: Vec<Vec<F>>,
        mut k: Vec<Vec<F>>,
        b: Vec<Vec<F>>,
    ) -> (Vec<F>, Vec<F>, Vec<F>) {
        assert!(f.len() != 0);
        assert!(f[0].len() != 0);
        assert!(k.len() != 0);
        assert!(k[0].len() != 0);
        assert!(k.len() == k[0].len());
        assert!(f.len() >= k.len());
        assert!(f[0].len() >= k.len());
        let check_f = f[0].len();
        let check_k = k.len();
        let mut f1: Vec<F> = vec![];
        let mut k1: Vec<F> = vec![];

        if self.h == 0 {
            self.h = f.len();
            self.w = f[0].len();
            self.k = k.len();
        } else {
            assert!(self.h == f.len());
            assert!(self.w == f[0].len());
            assert!(self.k == k.len());
        }

        for idx in 0..f.len() {
            if f[idx].len() != check_f {
                panic!("F out of shape: {}", idx)
            }
            f1.append(&mut f[idx]);
        }

        f1.resize(
            f1.len() + (REGISTER_WIDTH / (size_of::<F>() * 8))
                - f1.len() % (REGISTER_WIDTH / (size_of::<F>() * 8)),
            self.zero,
        );
        for idx in 0..k.len() {
            if k[idx].len() != check_k {
                panic!("K out of shape: {}", idx)
            }
            k1.append(&mut k[idx]);
        }
        k1.resize(
            k1.len() + (REGISTER_WIDTH / (size_of::<F>() * 8))
                - k1.len() % (REGISTER_WIDTH / (size_of::<F>() * 8)),
            self.zero,
        );
        (f1, k1, vec![])
    }

    fn forward(
        &self,
        mut data: (&[F], &mut [F], &[F]),
        a: Option<Box<dyn Aggregator<F>>>,
    ) -> Vec<F> {
        let feature_len: usize = self.h * self.w + (REGISTER_WIDTH / (size_of::<F>() * 8))
            - self.h * self.w % (REGISTER_WIDTH / (size_of::<F>() * 8));
        let kernel_len: usize = self.k * self.k + (REGISTER_WIDTH / (size_of::<F>() * 8))
            - self.k * self.k % (REGISTER_WIDTH / (size_of::<F>() * 8));

        assert!(data.0.len() == feature_len);
        assert!(data.1.len() == kernel_len);
        self.execute_forward(&mut data.0, data.1)
    }
}

impl<F: Float> Conv2D<F> {
    fn execute_forward(&self, x: &[F], w: &mut [F]) -> Vec<F> {
        //Assert during creation
        let w_out = (self.w - self.k + 2 * self.p as usize) / self.s as usize + 1;
        let h_out = (self.h - self.k + 2 * self.p as usize) / self.s as usize + 1;
        let count = h_out * w_out;
        let ret: Vec<F> = vec![self.zero; count];
        unsafe {
            convolve_forward_2d_double(
                self.w as c_ulong,
                self.h as c_ulong,
                self.k as c_ulong,
                count as c_ulong,
                self.p as c_ushort,
                self.s as c_ushort,
                x.as_ptr() as *const c_double,
                w.as_ptr() as *const c_double,
                ret.as_ptr() as *mut c_double
            );
        }
        ret
    }
}
