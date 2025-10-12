//Copyright (c) 2025, tree-chutes

use super::layers::Layer;
use super::register::REGISTER_WIDTH;
use super::{activation_functions::Activation, aggregator_functions::Aggregator};
use num_traits::Float;
use std::ffi::{c_double, c_float, c_uchar, c_ulong, c_ushort};
use std::ptr;

#[link(name = "co5_dl_c", kind = "static")]
#[allow(improper_ctypes)]
unsafe extern "C" {
    fn convolve_2d_double(
        w: c_ulong,
        h: c_ulong,
        k: c_ulong,
        o_w: c_ulong,
        c: c_ulong,
        p: c_ushort,
        s: c_ushort,
        hXw: *const c_double,
        kXk: *const c_double,
        prev: *const c_double,
        l_r: *const c_double,
        out: *mut c_double,
    ) -> c_uchar;

    fn convolve_2d_float(
        w: c_ulong,
        h: c_ulong,
        k: c_ulong,
        o_w: c_ulong,
        c: c_ulong,
        p: c_ushort,
        s: c_ushort,
        hXw: *const c_float,
        kXk: *const c_float,
        prev: *const c_float,
        l_r: *const c_float,
        out: *mut c_float,
    ) -> c_uchar;
}

pub(super) struct Conv2D<F: Float> {
    pub(super) zero: F,
    pub(super) h: usize,
    pub(super) w: usize,
    pub(super) k: usize,
    pub(super) p: u16,
    pub(super) s: u16,
    pub(super) h_out: usize,
    pub(super) w_out: usize,
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
            assert!(self.s as usize <= self.w - self.k);
            self.w_out = (self.w - self.k + 2 * self.p as usize) / self.s as usize + 1;
            self.h_out = (self.h - self.k + 2 * self.p as usize) / self.s as usize + 1;
            
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

    fn backward(&self, d: (&mut [F], &mut [F], &[F]), z: &mut [F], l_r: F ) -> (Vec<F>, Vec<F>, Vec<F>) {
        let kernel_len: usize = self.k * self.k + (REGISTER_WIDTH / (size_of::<F>() * 8))
            - self.k * self.k % (REGISTER_WIDTH / (size_of::<F>() * 8));
        let out_len: usize = self.w_out * self.h_out + (REGISTER_WIDTH / (size_of::<F>() * 8))
            - self.w_out * self.h_out % (REGISTER_WIDTH / (size_of::<F>() * 8));
        assert!(d.1.len() == kernel_len);
        assert!(z.len() == out_len);
        let lr = [l_r];
        self.execute_backward(d.0, d.1, z, &lr)
    }

}

impl<F: Float> Conv2D<F> {

    fn flip_180(&self, matrix: &mut [F], n: usize, m: usize){
        let mut tmp: F;
        let mut swap: usize;
        let mut idx_row: usize;
        let mut swap_row: usize;
        let mut offset: usize = 0;
        let mut swap_offset: usize = 0;
        let mut row_counter: usize = 0;
        let mut idx = 0;
        let mut stop_row =  n / 2 ;

        if n % 2 != 0{
            stop_row += 1;
        }

        loop{            
            idx_row = idx / m;
            swap_row = n - 1 - idx_row;            
            offset = idx % m;
            swap_offset = m - 1 - offset; 
            swap = (n * swap_row) + swap_offset;
            idx = (n * idx_row) + offset;
            tmp = matrix[swap];
            matrix[swap] = matrix[idx];
            matrix[idx] = tmp;
            idx += 1;
            if idx % m == 0{
                row_counter += 1;
                if (row_counter == stop_row){
                    break;
                }
            }
        }
    }

    fn execute_backward(&self, x: &[F], w: &mut [F], z: &mut [F], l_r: &[F] ) -> (Vec<F>, Vec<F>, Vec<F>) {
        let padding = self.k - 1;
        let ret_x: Vec<F> = vec![self.zero; self.h * self.w];
        let ret_k: Vec<F> = vec![self.zero; self.k * self.k];
        
        unsafe {
            if size_of::<F>() == 8 {
                convolve_2d_double(
                    self.w as c_ulong,
                    self.h as c_ulong,
                    self.w_out as c_ulong,
                    self.k as c_ulong,
                    (self.k * self.k) as c_ulong,
                    0 as c_ushort,
                    self.s as c_ushort,
                    x.as_ptr() as *const c_double,
                    z.as_ptr() as *const c_double,
                    ptr::null() as *const c_double, 
                    l_r.as_ptr() as *const c_double,
                    ret_k.as_ptr() as *mut c_double
                );
            }
            else{
                convolve_2d_float(
                    self.w as c_ulong,
                    self.h as c_ulong,
                    self.w_out as c_ulong,
                    self.k as c_ulong,
                    (self.k * self.k) as c_ulong,
                    0 as c_ushort,
                    self.s as c_ushort,
                    x.as_ptr() as *const c_float,
                    z.as_ptr() as *const c_float,
                    ptr::null() as *const c_float, 
                    l_r.as_ptr() as *const c_float,
                    ret_k.as_ptr() as *mut c_float
                );
            }
        }
        self.flip_180(w, self.k, self.k);
        unsafe {
            if size_of::<F>() == 8 {
                convolve_2d_double(
                    self.w_out as c_ulong,
                    self.h_out as c_ulong,
                    self.k as c_ulong,
                    self.w as c_ulong,
                    (self.h * self.w) as c_ulong,
                    padding as c_ushort,
                    self.s as c_ushort,
                    z.as_ptr() as *const c_double,
                    w.as_ptr() as *const c_double,
                    ptr::null() as *const c_double,
                    l_r.as_ptr() as *const c_double,
                    ret_x.as_ptr() as *mut c_double
                );
            }
            else{
                convolve_2d_float(
                    self.w_out as c_ulong,
                    self.h_out as c_ulong,
                    self.k as c_ulong,
                    self.w as c_ulong,
                    (self.h * self.w) as c_ulong,
                    padding as c_ushort,
                    self.s as c_ushort,
                    z.as_ptr() as *const c_float,
                    w.as_ptr() as *const c_float,
                    ptr::null() as *const c_float,
                    l_r.as_ptr() as *const c_float,
                    ret_x.as_ptr() as *mut c_float
                );
            }
        }
        (ret_x, ret_k, vec![])
    }

    fn execute_forward(&self, x: &[F], w: &mut [F]) -> Vec<F> {
        let count = self.h_out * self.w_out;
        let mut ret: Vec<F> = vec![self.zero; count];
        unsafe {
            if size_of::<F>() == 8 {
                convolve_2d_double(
                    self.w as c_ulong,
                    self.h as c_ulong,
                    self.k as c_ulong,
                    self.w_out as c_ulong,
                    count as c_ulong,
                    self.p as c_ushort,
                    self.s as c_ushort,
                    x.as_ptr() as *const c_double,
                    w.as_ptr() as *const c_double,
                    ptr::null() as *const c_double,
                    ptr::null() as *const c_double,
                    ret.as_ptr() as *mut c_double
                );
            }
            else{
                convolve_2d_float(
                    self.w as c_ulong,
                    self.h as c_ulong,
                    self.k as c_ulong,
                    self.w_out as c_ulong,
                    count as c_ulong,
                    self.p as c_ushort,
                    self.s as c_ushort,
                    x.as_ptr() as *const c_float,
                    w.as_ptr() as *const c_float,
                    ptr::null() as *const c_float,
                    ptr::null() as *const c_float,
                    ret.as_ptr() as *mut c_float
                );
            }
        }
        ret.resize(
            ret.len() + (REGISTER_WIDTH / (size_of::<F>() * 8)) - ret.len() % (REGISTER_WIDTH / (size_of::<F>() * 8)),
            self.zero,
        );
        ret
    }
}
