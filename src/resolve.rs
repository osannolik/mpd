use crate::common::{CfarDetection, Real, ScanProperties, Storable};
use ndarray::{s, stack, Array1, Array2, Axis};
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

    pub fn binary_integration(&self, nof_detections: usize) -> Vec<usize> {
        self.last_det_bins
            .iter()
            .filter_map(|&rb| {
                let nof_det_in_bin: usize = 1 + self
                    .velocity_map
                    .slice(s![rb, ..-1])
                    .iter()
                    .filter(|&vel| *vel > 0.0)
                    .count();

                if nof_det_in_bin >= nof_detections {
                    Some(rb)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn resolve_velocity(
        &self,
        bins: Vec<usize>,
        nof_detections: usize,
        velocity_window: Real,
        max_velocity: Real,
    ) -> Vec<(usize, Real)> {
        bins.iter()
            .filter_map(|&rb_index| {
                let mut unfolded = Vec::<Real>::new();

                self.velocity_map
                    .row(rb_index)
                    .indexed_iter()
                    .filter(|(_, &vel)| vel > 0.0)
                    .for_each(|(n, &vel)| {
                        let prf = *self.v_prf.get(n).unwrap();

                        let mut neg = Array1::range(vel - prf, -max_velocity, -prf).to_vec();
                        let mut pos = Array1::range(vel, max_velocity, prf).to_vec();

                        unfolded.append(&mut neg);
                        unfolded.append(&mut pos);
                    });

                unfolded.sort_by(|a, b| a.partial_cmp(b).unwrap());

                println!("{:?}", unfolded);

                let all_vel = Array1::from(unfolded);

                let diff: Array1<_> = &all_vel.slice(s![nof_detections - 1..])
                    - &all_vel.slice(s![..1 - nof_detections as isize]);

                let min_diff_index: Option<usize> = diff
                    .indexed_iter()
                    .filter(|(_, &v_diff)| v_diff < velocity_window)
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(index, _)| index);

                if let Some(i) = min_diff_index {
                    Some((
                        rb_index,
                        all_vel.slice(s![i..i + nof_detections]).mean().unwrap(),
                    ))
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn add(&mut self, detections: &Vec<CfarDetection>, nof_range_bins: usize, v_prf: Real) {
        let max_range_in_bins = self.velocity_map.nrows();
        let nof_unfolds = ((max_range_in_bins + nof_range_bins - 1) as Real
            / nof_range_bins as Real)
            .floor() as usize;

        self.nof_range_bins.pop_front().unwrap();
        self.nof_range_bins.push_back(nof_range_bins);
        self.v_prf.pop_front().unwrap();
        self.v_prf.push_back(v_prf);
        self.last_det_bins.clear();

        self.velocity_map = stack(
            Axis(1),
            &[
                self.velocity_map.slice(s![.., 1..]),
                Array2::zeros([self.velocity_map.nrows(), 1]).view(),
            ],
        )
        .unwrap();

        for detection in detections {
            for i in 0..nof_unfolds {
                let bin = detection.range_bin + i * nof_range_bins;
                if bin < self.velocity_map.nrows() {
                    self.last_det_bins.push(bin);
                    self.velocity_map
                        .slice_mut(s![bin, -1])
                        .fill(detection.velocity);
                }
            }
        }
    }
}
