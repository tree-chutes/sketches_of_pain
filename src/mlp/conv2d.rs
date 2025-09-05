//Copyright (c) 2025, tree-chutes

use num_traits::Float;
use super::activation_functions::Activation;
use std::{ffi::{c_double, c_uchar}, mem};

use super::layers::Layer;

#[link(name="co5_dl_c", kind="static")]
#[allow(improper_ctypes)]
unsafe extern  "C"{ 
    unsafe fn multiply(in1: *const c_double, out: *mut c_double) -> c_uchar;
    unsafe fn drop() -> c_uchar;
}

pub (in super) struct Conv2D<F: Float> {
    pub(in super) seed: F,
    pub(in super) o: usize,
    pub(in super) i: usize,
    pub(in super) w: usize,
    pub(in super) a: Box<dyn Activation<F>>
}

impl<F: Float> Drop for Conv2D<F>{
    fn drop(&mut self) {
        unsafe{
            if drop() != 0 as c_uchar{
                println!("Failed to free the kernel");
            }
        }
    }
}

impl<F: Float> Conv2D<F>{
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
                sum = sum + tmp[i];
            }
            self.a.calculate(&mut sum, b);
            ret[self.o - 1] = sum;
        }
        ret
    }
}

impl<F: Float> Layer<F> for Conv2D<F> {

    fn get_output_shape(&self)-> usize {
        self.o
    }

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
