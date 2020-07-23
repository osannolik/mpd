#![allow(dead_code)]

mod common;

use common::ToDecibel;
use common::{RangePulse, ScanProperties, SendPulse};

use ndarray; //::{self, s}; //::{self, prelude::*};
             //use ndarray::Array1;
use num_complex::Complex;
use rustfft::FFTplanner;

fn main() {
    let c: Complex<f64> = Complex::new(1.0, 0.1);

    println!("{:?}", c);

    let mut input = ndarray::Array::from_elem(8, c);
    let mut output = ndarray::Array::zeros(input.shape());

    let mut planner = FFTplanner::new(false);
    let fft = planner.plan_fft(input.len());

    fft.process(
        input.as_slice_mut().unwrap(),
        output.as_slice_mut().unwrap(),
    );

    println!("Input {:?}", input);
    println!("Output {:?}", output);
    /*
       let t = [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];

       let t2 = Array1::linspace(-5.0, 5.0, 11);
       let k = 1.0 / 11.0;
       println!("Chirp {:?}", common::chirp_linear(&t2, 0.0, k));
    */

    let p = ScanProperties {
        carrier_freq: 3.0e9,
        sample_freq: 1.0e6,
        duty_cycle: 0.1,
        nof_pulses: 512,
        nof_range_bins: 115,
    };

    let pulse = SendPulse::new(p.sample_freq, &p);

    println!("Pulse {:?}", pulse);

    let noise_inti = RangePulse::noise((-75.0).db(), &p);

    println!("Noise {:?}", noise_inti);

    let clutter_inti = RangePulse::clutter((-75.0 + 40.0).db(), &p);

    println!("Clutter {:?}", clutter_inti);
}
