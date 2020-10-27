extern crate mpd;

use mpd::video::generate;
use mpd::{CfarConfig, Resolver, ScanProperties, Target, Units};

use clap::App;
use mpd::Storable;
use std::path::Path;

fn main() {
    let matches = App::new("MPD example")
        .version("0.1.0")
        .author("")
        .about(
            "A very simple example of a pulse doppler radar signal processing chain:\n
                        1. Generation of sampled video\n
                        2. Pulse compression\n
                        3. Doppler filtering (FFT) into the range-frequency domain\n
                        4. Constant false alarm rate processing\n
                        5. Range and velocity resolving",
        )
        .arg("--files... 'Dump data from each processing step into json files'")
        .get_matches();

    let do_it = matches.is_present("files");

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
        to_file(&video, "input.json", do_it);

        let video_pulse_compressed = video.pulse_compress(&p);
        to_file(&video_pulse_compressed, "pulse_compressed.json", do_it);

        let video_range_doppler = video_pulse_compressed.doppler_filtering();
        to_file(&video_range_doppler, "range_doppler.json", do_it);

        let cfar_detections = video_range_doppler.cfar(&cfg, &p);

        resolver.add(&cfar_detections, &p);

        let resolved_detections = resolver.process(30.0, 1000.0);

        println!(
            "Resolved detections using prf={:?}: {:?}",
            prf, resolved_detections
        );
    }
}

fn to_file<T>(inti: &T, s: &str, enable: bool)
where
    T: Storable,
{
    if enable {
        inti.to_file(Path::new(s)).expect("Failed to write file");
    }
}
