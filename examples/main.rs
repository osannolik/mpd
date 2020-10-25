#![allow(dead_code)]
//#![feature(test)]

extern crate mpd;

use mpd::{CfarConfig, Resolver, ScanProperties, Target, Units};

use mpd::video::generate;

//use std::path::Path;

fn main() {
    let mut p = ScanProperties {
        carrier_freq: 3.0e9,
        sample_freq: 1.0e6,
        duty_cycle: 0.1,
        nof_pulses: 512,
        nof_range_bins: 115,
    };

    let resolved_scan_prfs: Vec<_> = [115, 106, 97, 93, 88, 80, 71, 62]
        .iter()
        .map(|&nrs| p.to_prf(nrs))
        .collect();

    let targets = vec![
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

    let noise = -75.0.db();
    let clutter = noise + 40.0.db();

    let cfg = CfarConfig {
        mainlobe_clutter_margin: 6,
        min_value: noise - 10.0.db(),
        threshold_offset: 15.db(),
        n_rs_thr: 16,
        n_guard_bins: 1,
    };

    let mut resolver = Resolver::new(200000.0, resolved_scan_prfs.len(), 3, p.range_bin_length());

    for prf in resolved_scan_prfs {
        p.set_prf(prf);

        let video = generate(&targets, noise, clutter, &p);

        let video_pulse_compressed = video.pulse_compress(&p);

        let video_range_doppler = video_pulse_compressed.doppler_filtering();

        let cfar_detections = video_range_doppler.cfar(&cfg, &p);

        resolver.add(&cfar_detections, &p);

        let resolved_detections = resolver.process(30.0, 1000.0);

        println!(
            "Resolved detections using prf={:?}: {:?}",
            prf, resolved_detections
        );
    }

    /*
       let noise_level = -75.0.db();
       let video = RangePulse::noise(noise_level, &p)
           + RangePulse::clutter(noise_level + 40.0.db(), &p)
           + targets.iter().map(|tgt| RangePulse::target(tgt, &p)).sum();

    println!("Video {:?}", video);

    video
        .to_file(Path::new("video.json"))
        .expect("Could not write to file");
    */
}
