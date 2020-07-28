#![allow(dead_code)]

mod common;

use common::ToDecibel;
use common::{RangePulse, ScanProperties, Target};

use ndarray; //::{self, s}; //::{self, prelude::*};
             //use ndarray::Array1;
use num_complex::Complex;
use rustfft::FFTplanner;

use serde_json;

use std::fs::File;
use std::io::prelude::*;
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
            level: (-65.0).db(),
        },
        Target {
            velocity: -450.0,
            range: 100e3,
            level: (-45.0).db(),
        },
    ];

    let video: RangePulse = RangePulse::noise((-75.0).db(), &p)
        + RangePulse::clutter((-75.0 + 40.0).db(), &p)
        + targets.iter().map(|tgt| RangePulse::target(tgt, &p)).sum();

    println!("Video {:?}", video);

    let path = Path::new("video.json");
    let display = path.display();

    let mut file = match File::create(&path) {
        Err(why) => panic!("couldn't create {}: {}", display, why),
        Ok(file) => file,
    };

    let s = serde_json::to_string(&video).unwrap();
    println!("s {:?}", s);

    match file.write_all(s.as_bytes()) {
        Err(why) => panic!("couldn't write to {}: {}", display, why),
        Ok(_) => println!("successfully wrote to {}", display),
    }
}
