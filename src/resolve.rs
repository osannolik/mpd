use crate::common::{CfarDetection, Real, ScanProperties, Storable};
use ndarray::{s, stack, Array2, Axis};
use std::collections::VecDeque;
use std::path::Path;

pub struct ResolverConfig {
    pub nof_detections: usize,
    pub nof_prf: usize,
    pub velocity_window: Real,
    pub range_max: Real,
}

pub struct Resolver {
    v_prf: VecDeque<Real>,
    nof_range_bins: VecDeque<usize>,
    velocity_map: Array2<Real>,
    last_det_bins: Vec<usize>,
}

impl Resolver {
    pub fn new(config: &ResolverConfig, properties: &ScanProperties) -> Resolver {
        let max_range_in_bins = (config.range_max / properties.range_bin_length()).floor();
        Resolver {
            v_prf: VecDeque::from(vec![0.0; config.nof_prf]),
            nof_range_bins: VecDeque::from(vec![0; config.nof_prf]),
            velocity_map: Array2::zeros([max_range_in_bins as usize, config.nof_prf]),
            last_det_bins: vec![],
        }
    }

    fn write_velocity_map(&self, name: &str) {
        self.velocity_map
            .to_file(Path::new(name))
            .expect("could not write vel_map");
    }

    fn binary_integration(&self, nof_detections: usize) -> Vec<usize> {
        self.last_det_bins
            .iter()
            .filter_map(|&rb| {
                let nof_det_in_bin: usize =
                    self.velocity_map
                        .slice(s![rb, ..-1])
                        .fold(
                            1,
                            |nof_det, &vel| {
                                if vel > 0.0 {
                                    nof_det + 1
                                } else {
                                    nof_det
                                }
                            },
                        );
                if nof_det_in_bin >= nof_detections {
                    Some(rb)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn add(&mut self, detections: &Vec<CfarDetection>, nof_range_bins: usize, v_prf: Real) {
        let max_range_in_bins = self.velocity_map.nrows();
        let nof_folds = ((max_range_in_bins + nof_range_bins - 1) as Real / nof_range_bins as Real)
            .floor() as usize;

        self.nof_range_bins.pop_front().unwrap();
        self.nof_range_bins.push_back(nof_range_bins);
        self.v_prf.pop_front().unwrap();
        self.v_prf.push_back(v_prf);

        self.velocity_map = stack(
            Axis(1),
            &[
                self.velocity_map.slice(s![.., 1..]),
                Array2::zeros([self.velocity_map.nrows(), 1]).view(),
            ],
        )
        .unwrap();

        for detection in detections {
            for i in 0..nof_folds {
                let bin = detection.range_bin + i * nof_range_bins;
                if bin < self.velocity_map.nrows() {
                    self.last_det_bins.push(bin);
                    self.velocity_map
                        .slice_mut(s![bin, -1])
                        .fill(detection.velocity);
                }
            }
        }

        self.write_velocity_map("vel_map_add.json");
    }
}
