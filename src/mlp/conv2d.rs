//Copyright (c) 2025, tree-chutes

use super::layers::Layer;
use super::register::REGISTER_WIDTH;
use super::aggregator_functions::Aggregator;
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
    pub(super) is_first_layer: bool,
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

    fn set_first_layer_flag(&mut self) {
        self.is_first_layer = true;
    }

    fn flatten(
        &self,
        mut f: Vec<Vec<F>>,
        k: Vec<Vec<F>>,
        b: Vec<Vec<F>>,
    ) -> (Vec<F>, Vec<F>, Vec<F>) {
        assert!(f.len() != 0);
        assert!(f[0].len() != 0);
        assert!(f.len() >= k.len());
        assert!(f[0].len() >= k.len());

        let check_f = f[0].len();
        let mut ret_feature: Vec<F> = vec![];

        assert!(self.h == f.len());
        assert!(self.w == f[0].len());

        for idx in 0..f.len() {
            if f[idx].len() != check_f {
                panic!("F out of shape: {}", idx)
            }
            ret_feature.append(&mut f[idx]);
        }

        ret_feature.resize(
            ret_feature.len() + (REGISTER_WIDTH / (size_of::<F>() * 8))
                - ret_feature.len() % (REGISTER_WIDTH / (size_of::<F>() * 8)),
            self.zero,
        );
        (ret_feature, self.flatten_kernel(k), vec![])
    }

    fn flatten_kernel(&self, mut k: Vec<Vec<F>>) -> Vec<F> {
        assert!(k.len() != 0);
        assert!(k[0].len() != 0);
        assert!(k.len() == k[0].len());
        let check_k = k.len();
        let mut ret_kernel: Vec<F> = vec![];

        for idx in 0..k.len() {
            if k[idx].len() != check_k {
                panic!("K out of shape: {}", idx)
            }
            ret_kernel.append(&mut k[idx]);
        }
        ret_kernel.resize(
            ret_kernel.len() + (REGISTER_WIDTH / (size_of::<F>() * 8))
                - ret_kernel.len() % (REGISTER_WIDTH / (size_of::<F>() * 8)),
            self.zero,
        );
        ret_kernel
    }

    fn forward(
        &self,
        mut data: (&[F], &mut [F], &[F]),
        _a: Option<Box<dyn Aggregator<F>>>,
    ) -> Vec<F> {
        let feature_len: usize = self.h * self.w + (REGISTER_WIDTH / (size_of::<F>() * 8))
            - self.h * self.w % (REGISTER_WIDTH / (size_of::<F>() * 8));
        let kernel_len: usize = self.k * self.k + (REGISTER_WIDTH / (size_of::<F>() * 8))
            - self.k * self.k % (REGISTER_WIDTH / (size_of::<F>() * 8));

        assert!(data.0.len() == feature_len);
        assert!(data.1.len() == kernel_len);
        self.execute_forward(&mut data.0, data.1)
    }

    fn backward(&self, d: (&mut [F], &mut [F], &mut [F]), l_r: F, l: F) -> (Vec<F>, Vec<F>) {
        let kernel_len: usize = self.k * self.k + (REGISTER_WIDTH / (size_of::<F>() * 8))
            - self.k * self.k % (REGISTER_WIDTH / (size_of::<F>() * 8));
        let _out_len: usize = self.w_out * self.h_out + (REGISTER_WIDTH / (size_of::<F>() * 8))
            - self.w_out * self.h_out % (REGISTER_WIDTH / (size_of::<F>() * 8));
        assert!(d.1.len() == kernel_len);
        // assert!(z.len() == out_len);
        let lr = [l_r];
        self.execute_backward(d.0, d.1, d.2, &lr)
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

    fn execute_backward(&self, previous_z: &[F], backpropagated_weights: &mut [F], previous_weights: &mut [F], l_r: &[F] ) -> (Vec<F>, Vec<F>) {
        let count = ((self.h_out - self.k + 2 * (self.h_out - 1))) / (self.s as usize) + 1;
        let mut updated_kernel: Vec<F> = vec![self.zero; self.k * self.k];
        let mut backward_gradients = vec![self.zero; self.h * self.w];

        unsafe{
            if size_of::<F>() == 8 {
                if convolve_2d_double(
                    self.w as c_ulong,
                    self.h as c_ulong,
                    self.w_out as c_ulong,
                    self.k as c_ulong,
                    (self.k * self.k) as c_ulong,
                    0 as c_ushort,
                    self.s as c_ushort,
                    previous_z.as_ptr() as *const c_double,                     
                    backpropagated_weights.as_ptr() as *const c_double,
                    previous_weights.as_ptr() as *const c_double,
                    l_r.as_ptr() as *const c_double,
                    updated_kernel.as_ptr() as *mut c_double
                ) != 0{
                    panic!("conv2d double forward pass failed")
                }
            }
            else{
                if convolve_2d_float(
                    self.w as c_ulong,
                    self.h as c_ulong,
                    self.w_out as c_ulong,
                    self.k as c_ulong,
                    (self.k * self.k) as c_ulong,
                    0 as c_ushort,
                    self.s as c_ushort,
                    previous_z.as_ptr() as *const c_float,                     
                    backpropagated_weights.as_ptr() as *const c_float,
                    previous_weights.as_ptr() as *const c_float,
                    l_r.as_ptr() as *const c_float,
                    updated_kernel.as_ptr() as *mut c_float
                ) != 0{
                    panic!("conv2d float forward pass failed")
                }
            }
        }
        if !self.is_first_layer{
            self.flip_180(previous_weights, self.k, self.k);
            unsafe {
                if size_of::<F>() == 8 {
                    if convolve_2d_double(
                        self.w_out as c_ulong, //we are moving from k back to w
                        self.h_out as c_ulong, //we are moving from k back to h
                        self.k as c_ulong,
                        self.w as c_ulong, //we are moving from k back to w
                        (count * count) as c_ulong,
                        (self.k - 1) as c_ushort,
                        self.s as c_ushort,
                        backpropagated_weights.as_ptr() as *const c_double,
                        previous_weights.as_ptr() as *const c_double,                     
                        ptr::null() as *const c_double,
                        ptr::null() as *const c_double,
                        backward_gradients.as_ptr() as *mut c_double
                    ) != 0{
                        panic!("conv2d double backward pass failed")
                    }
                }
                else{
                    if convolve_2d_float(
                        self.w_out as c_ulong, //we are moving from k back to w
                        self.h_out as c_ulong, //we are moving from k back to h
                        self.k as c_ulong,
                        self.w as c_ulong, //we are moving from k back to w
                        (count * count) as c_ulong,
                        (self.k - 1) as c_ushort,
                        self.s as c_ushort,
                        backpropagated_weights.as_ptr() as *const c_float,
                        previous_weights.as_ptr() as *const c_float,                     
                        ptr::null() as *const c_float,
                        ptr::null() as *const c_float,
                        backward_gradients.as_ptr() as *mut c_float
                    ) != 0{
                        panic!("conv2d float backward pass failed")
                    }
                }
            }
        }
        updated_kernel.resize(
            updated_kernel.len() + (REGISTER_WIDTH / (size_of::<F>() * 8)) - updated_kernel.len() % (REGISTER_WIDTH / (size_of::<F>() * 8)),
            self.zero,
        );

        if !self.is_first_layer{
            backward_gradients.resize(
                backward_gradients.len() + (REGISTER_WIDTH / (size_of::<F>() * 8)) - backward_gradients.len() % (REGISTER_WIDTH / (size_of::<F>() * 8)),
                self.zero,
            );
        }
        (updated_kernel, backward_gradients)
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
