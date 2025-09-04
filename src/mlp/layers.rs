//Copyright (c) 2025, tree-chutes

use std::{ffi::{c_double, c_uchar, c_ulong}, mem};
use num_traits::Float;

use super::activation_functions::Activation;

#[link(name="co5_dl_c", kind="static")]
#[allow(improper_ctypes)]
unsafe extern  "C"{ 
    unsafe fn multiply(in1: *const c_double, out: *mut c_double) -> c_uchar;
    unsafe fn init_kernel(o: c_ulong, w: c_ulong, k: *const c_double) -> c_uchar;
    unsafe fn drop() -> c_uchar;
}

pub trait Layer<F: Float>{
    fn forward(&self, f: Vec<F>, b: F)-> Vec<F>;
    fn generate_mapping(&self)-> Vec<(usize,usize)>;
}

pub fn layer_factory<F: Float>(mut k: Vec<Vec<F>>, i: usize, s: F, a: Box<dyn Activation<F>>) -> impl Layer<F>{
    let o = i - k.len() + 1;
    let w = k.len();
    map_kernel::<F>(&mut k, s);
    Default{i: i, w: w, o: o * o, seed: s, a: a}
}

fn map_kernel<F: Float>(k: &mut Vec<Vec<F>>, s: F){
    let w = k.len();
    let mut tmp = flatten(k);
    let fill = tmp.len() % 4;
    if fill != 0{
        tmp.append(&mut vec![s; 4 - fill]);
    }
    unsafe{
        let (p, _, _) = Vec::into_raw_parts(tmp);
    
        if init_kernel((w * w) as c_ulong, w as c_ulong,  p as *const c_double) != 0 {
            panic!("failed to set kernel");
        }
    }
}

pub(in crate::mlp) fn flatten<F: Float>(k: &mut Vec<Vec<F>>)-> Vec<F>{
    let shape = k.len();
    let mut tmp = Vec::<F>::with_capacity(shape * shape);
    for kr in k.iter_mut(){
        if kr.len() != shape{
            panic!("{} is out of shape (shape is {})", kr.len(), shape);
        }
        tmp.append(kr);
    }
    tmp
}

struct Default<F: Float> {
    seed: F,
    o: usize,
    i: usize,
    w: usize,
    a: Box<dyn Activation<F>>
}

impl<F: Float> Drop for Default<F>{
    fn drop(&mut self) {
        unsafe{
            if drop() != 0 as c_uchar{
                println!("Failed to free the kernel");
            }
        }
    }
}

impl<F: Float> Default<F>{
    fn multiply(&self, f: Vec<F>, b: F)-> Vec<F>{
        let mut ret: Vec<F> = vec![self.seed; self.o];
        let mut tmp: Vec<F> = vec![self.seed; f.len()];
        let mut sum: F = self.seed;

        unsafe{
            let len = tmp.len();
            let capacity = tmp.capacity();
            let out = tmp.as_mut_ptr() as *mut c_double;
            mem::forget(tmp);
            let (x_pntr, _x_len, _x_capacity) = f.into_raw_parts();
            if multiply(x_pntr as *const f64, out) != 0{
                panic!("failed to multiply");
            }
            tmp = Vec::from_raw_parts(out as *mut F, len, capacity);
            for i in 0..(self.o * self.o){
                if i > 0 && i % self.o == 0{
                    self.a.calculate(&mut sum, b);
                    ret[i / self.o - 1] = sum;
                    sum = self.seed;
                }
                self.a.calculate(&mut sum, b);
                sum = sum + tmp[i];
            }
            ret[self.o - 1] = sum;
        }
        ret
    }
}

impl<F: Float> Layer<F> for Default<F> {

    fn forward(&self, f: Vec<F>, b: F)-> Vec<F>{
        self.multiply(f, b)
    }

    fn generate_mapping(&self)-> Vec<(usize,usize)> {
        let mut min_x: usize;
        let mut max_x: usize;
        let mut min_y: usize;
        let mut max_y: usize;
        let mut flag: bool;
        let output_shape = self.o;
        let o = output_shape.isqrt();
        let mut horizontal_stride: usize = 0;
        let mut vertical_stride: usize = 0;
        let mut input_mapping = Vec::<(usize,usize)>::new();

        for _ in 0..(output_shape){
            flag = false;
            min_y =  vertical_stride;
            max_y =  vertical_stride + self.w;                            
            min_x = horizontal_stride;
            max_x = horizontal_stride + self.w;                    
    
            for y in 0..self.i{
                for x in 0..self.i{
                    if x >= min_x && x < max_x{
                        if y >= min_y && y < max_y{
                            input_mapping.push((x,y));
                            if input_mapping.len() % (output_shape) == 0{
                                let count = input_mapping.len() / (output_shape);
                                if count % o != 0{
                                    horizontal_stride += 1;
                                }
                                else{
                                    horizontal_stride = 0;
                                    vertical_stride += 1;
                                }
                                flag = true;
                                break;
                            }
                        }
                    }
                }
                if flag{
                    break;
                }
            }        
        }
        input_mapping
    }    
}
