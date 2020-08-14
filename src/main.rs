#![allow(dead_code)]
//#![feature(test)]

mod common;
mod iir;

use crate::common::{RangeDoppler, RangePulse, ScanProperties, Storable, Target, ToDecibel};

use std::path::Path;

fn main() {
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

    let mut received_video = video;

    received_video.pulse_compress(&p);

    println!("Pulse Compress {:?}", received_video);

    received_video
        .to_file(Path::new("pulse_compress.json"))
        .expect("Could not write to file");

    // let range_doppler = received_video.doppler_filtering();
    // let range_doppler: RangeDoppler = received_video.into();
    let range_doppler = RangeDoppler::from(received_video);

    println!("Doppler Filtering {:?}", range_doppler);

    range_doppler
        .to_file(Path::new("range_doppler.json"))
        .expect("Could not write to file");
}
