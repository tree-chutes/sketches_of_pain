//Copyright (c) 2025, tree-chutes

use super::activation_functions::Activation;
use super::aggregator_functions::Aggregator;
use super::conv2d::Conv2D;
use super::linear_layer::LinearLayer;
use num_traits::Float;

pub enum Layers {
    Linear,
    Conv2D,
    Softmax,
}

pub trait Layer<F: Float> {
    fn set_first_layer_flag(&mut self);

    fn forward(&self, d: (&[F], &mut [F], &[F]), a: Option<Box<dyn Aggregator<F>>>)-> Vec<F>;

    fn flatten(&self, x: Vec<Vec<F>>, w: Vec<Vec<F>>, b: Vec<Vec<F>>) -> (Vec<F>, Vec<F>, Vec<F>){
        todo!("Complete me");
    }

    fn flatten_kernel(&self, k: Vec<Vec<F>>) -> Vec<F>{
        todo!("Complete me");
    }

    fn backward(&self, d: (&mut [F], &mut [F], &mut [F]), l_r: F, l: F) -> (Vec<F>, Vec<F>) {
        panic!("Should NOT be called");
    }
}

pub fn layer_factory<F: Float + 'static>(
    l: Layers,
    n: usize,
    d: usize,
    m: usize,
    c: Option<(u16, u16)>,    
    z: F
) -> Box<dyn Layer<F>> {
    match l {
        Layers::Linear => {
            let mut ret = LinearLayer {
                d: d,
                m: m,
                n: n,
                zero: z,
                is_first_layer: false
            };        
            Box::new(ret)
        },
        Layers::Conv2D =>{
            let (padding, step) = c.unwrap();
            assert!(step > 0);
            assert!(step as usize <= n - d);

            let mut ret = Conv2D {
                h: n,
                w: m,
                k: d,
                p: padding,
                s: step,
                h_out: 0,
                w_out: 0,
                zero: z,
                is_first_layer: false
            };   
            ret.w_out = (ret.w - ret.k + 2 * ret.p as usize) / ret.s as usize + 1;
            ret.h_out = (ret.h - ret.k + 2 * ret.p as usize) / ret.s as usize + 1;
     
            Box::new(ret)
        }
        _ => todo!(),
    }
}
