//Copyright (c) 2025, tree-chutes

use std::ffi::{c_double, c_uchar, c_ulong};

use super::conv2d::Conv2D;
use super::softmax::Softmax;
use num_traits::Float;
use super::activation_functions::Activation;

#[link(name="co5_dl_c", kind="static")]
#[allow(improper_ctypes)]
unsafe extern  "C"{ 
    unsafe fn init_kernel(o: c_ulong, w: c_ulong, k: *const c_double) -> c_uchar;
}

pub enum Layers {
    Conv2D,
    Softmax
}

pub trait Layer<F: Float>{
    fn forward(&self, f: &[F], b: F)-> Vec<F>;
    fn generate_mapping(&self)-> Vec<(usize,usize)>;
    fn get_output_shape(&self)-> usize;
}

pub fn layer_factory<F: Float + 'static>(n: Layers, k: Option<Vec<Vec<F>>>, i: usize, s: F, a: Option<Box<dyn Activation<F>>>) -> Box<dyn Layer<F>>{
    match n {
        Layers::Conv2D => Box::new(create_2d::<F>(k.unwrap(), i, s,a.unwrap())),
        Layers::Softmax => Box::new(Softmax{i: i, seed: s})
    }
}

fn create_2d<F: Float>(k: Vec<Vec<F>>, i: usize, s: F, a: Box<dyn Activation<F>>)->Conv2D<F>{
    let o = i - k.len() + 1;
    let w = k.len();
    map_kernel::<F>(k, s);
    Conv2D{i: i, w: w, o: o * o, seed: s, a: a}

}

fn map_kernel<F: Float>(k: Vec<Vec<F>>, s: F){
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

fn flatten<F: Float>(mut k: Vec<Vec<F>>)-> Vec<F>{
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
