//Copyright (c) 2025, tree-chutes

use super::activation_functions::Activation;
use super::conv2d::Conv2D;
use super::linear::Linear;
use super::softmax::Softmax;
use num_traits::Float;

pub enum Layers {
    Linear,
    Conv2D,
    Softmax,
}

pub trait Layer<F: Float> {
    fn forward(&self, w: &[F], f: &[F], b: F)-> Vec<F>;
    fn differential(&self, s: &[F], p: &[F]) -> Vec<F> {
        panic!("Should NOT be called");
    }
    fn get_output_shape(&self) -> usize {
        println!("NOT Required");
        0
    }
}

pub fn layer_factory<F: Float + 'static>(
    l: Layers,
    w: Vec<Vec<F>>,
    n: usize,
    s: F,
    o: F,
    a: Option<Box<dyn Activation<F>>>,
) -> (Vec<F>, Vec<(usize, usize)>, Box<dyn Layer<F>>) {
    let m: usize;
    let d = w.len();
    assert!(d != 0);
    m = w[0].len();
    w.iter().for_each(|r| assert!(r.len() == m));

    match l {
        Layers::Linear => {
            assert!(w.len() == n);
            let mut l1 = Linear {
                fill: 0,
                d: d,
                m: m,
                n: n,
                zero: s,
                a: a.unwrap(),
                activations: vec![s; n * m],
            };        
            (
                l1.flatten_weights(w),
                l1.generate_input_mapping(),
                Box::new(l1)
            )
        }
        Layers::Conv2D =>{
            let o = n - w.len() + 1;
            let w1 = w.len();        
            let mut l1 = Conv2D {
                fill: 0,
                i: n,
                d: w1,
                output_shape: o * o,
                zero: s,
                a: a.unwrap(),
                activations: vec![s; o * o]
            };        
            (
                l1.flatten_weights(w),
                l1.generate_input_mapping(),
                Box::new(l1)
            )
        }
        _ => todo!(),
    }
}
