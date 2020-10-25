#![allow(dead_code)]
//#![feature(test)]

extern crate mpd;

use mpd::{
    CfarDetection, RangeDoppler, RangePulse, Resolver, ScanProperties, Storable, Target, Units,
};

use mpd::CfarConfig;

use std::path::Path;

fn main() {
    let mut p = ScanProperties {
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

    let noise_level = -75.0.db();
    let video = RangePulse::noise(noise_level, &p)
        + RangePulse::clutter(noise_level + 40.0.db(), &p)
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

    println!("Unambiguous_range {:?}", p.unambiguous_range());
    println!("unambiguous_velocity {:?}", p.unambiguous_velocity());
    println!(
        "Aliased target 1 range velocity {:?} {:?} {:?}",
        p.alias_range(targets[0].range),
        p.alias_velocity(targets[0].velocity),
        targets[0].level.value()
    );
    println!(
        "Aliased target 2 range velocity {:?} {:?} {:?}",
        p.alias_range(targets[1].range),
        p.alias_velocity(targets[1].velocity),
        targets[1].level.value()
    );

    let cfg = CfarConfig {
        mainlobe_clutter_margin: 6,
        min_value: noise_level - 10.0.db(),
        threshold_offset: 15.db(),
        n_rs_thr: 16,
        n_guard_bins: 1,
    };
    let cfar_detections = range_doppler.cfar(&cfg, &p);

    println!("{:?}", cfar_detections);

    let cfar_dets1 = vec![
        CfarDetection {
            range_bin: 93,
            velocity: 16.1184,
            level: 0.0.db(),
        },
        CfarDetection {
            range_bin: 105,
            velocity: 134.8854,
            level: 0.0.db(),
        },
    ];

    let cfar_dets2 = vec![
        CfarDetection {
            range_bin: 17,
            velocity: 172.1081,
            level: 0.0.db(),
        },
        CfarDetection {
            range_bin: 32,
            velocity: 450.9784,
            level: 0.0.db(),
        },
    ];

    let cfar_dets3 = vec![
        CfarDetection {
            range_bin: 44,
            velocity: 216.2381,
            level: 0.0.db(),
        },
        CfarDetection {
            range_bin: 86,
            velocity: 450.5799,
            level: 0.0.db(),
        },
        CfarDetection {
            range_bin: 99,
            velocity: 100.5799,
            level: 0.0.db(),
        },
    ];

    let mut resolver = Resolver::new(200000.0, 8, 3, p.range_bin_length());

    resolver
        .add(&cfar_dets1, &p)
        .add(&cfar_dets2, p.set_prf(9434.0))
        .add(&cfar_dets3, p.set_prf(10309.0));

    let res_dets = resolver.process(30.0, 1000.0);

    println!("res_dets: {:?}", res_dets);
}
