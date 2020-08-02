#![allow(dead_code)]
//#![feature(test)]

mod common;
mod iir;

use common::ToDecibel;
use common::{RangePulse, ScanProperties, Target};

use ndarray; //::{self, s}; //::{self, prelude::*};
             //use ndarray::Array1;
use num_complex::Complex;
use rustfft::FFTplanner;

use std::path::Path;

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

    let p = ScanProperties {
        carrier_freq: 3.0e9,
        sample_freq: 1.0e6,
        duty_cycle: 0.1,
        nof_pulses: 512,
        nof_range_bins: 115,
    };

    let targets = [
        Target {
            velocity: 300.0,
            range: 50e3,
            level: -65.0.db(),
        },
        Target {
            velocity: -450.0,
            range: 100e3,
            level: -45.0.db(),
        },
    ];

    let video = RangePulse::noise(-75.0.db(), &p)
        + RangePulse::clutter((-75.0 + 40.0).db(), &p)
        + targets.iter().map(|tgt| RangePulse::target(tgt, &p)).sum();

    println!("Video {:?}", video);

    video
        .to_file(Path::new("video.json"))
        .expect("Could not write to file");
}
