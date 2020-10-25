use crate::common::{CfarDetection, Real, ResolverDetection, ScanProperties};
use ndarray::{s, stack, Array1, Array2, Axis};
use std::collections::VecDeque;

pub struct Resolver {
    range_bin_length: Real,
    nof_detections: usize,
    v_prf: VecDeque<Real>,
    nof_range_bins: VecDeque<usize>,
    velocity_map: Array2<Real>,
    last_det_bins: Vec<usize>,
}

fn unfold(start: usize, max: usize, step: usize) -> Vec<usize> {
    let nof_unfolds = ((max - start) as Real / step as Real).floor() as usize;
    (0..=nof_unfolds).map(|i| start + i * step).collect()
}

impl Resolver {
    pub fn new(
        range_max: Real,
        nof_prf: usize,
        nof_detections: usize,
        range_bin_length: Real,
    ) -> Resolver {
        let max_range_in_bins = (range_max / range_bin_length).floor() as usize;
        Resolver {
            range_bin_length,
            nof_detections,
            v_prf: VecDeque::from(vec![0.0; nof_prf]),
            nof_range_bins: VecDeque::from(vec![0; nof_prf]),
            velocity_map: Array2::zeros([max_range_in_bins, nof_prf]),
            last_det_bins: vec![],
        }
    }

    fn binary_integration(&self) -> Vec<usize> {
        self.last_det_bins
            .iter()
            .filter_map(|&rb| {
                let nof_det_in_bin: usize = 1 + self
                    .velocity_map
                    .slice(s![rb, ..-1])
                    .iter()
                    .filter(|&vel| *vel > 0.0)
                    .count();

                if nof_det_in_bin >= self.nof_detections {
                    Some(rb)
                } else {
                    None
                }
            })
            .collect()
    }

    fn resolve_velocity(
        &self,
        bins: &Vec<usize>,
        velocity_window: Real,
        max_velocity: Real,
    ) -> Vec<(usize, Real)> {
        bins.iter()
            .filter_map(|&bin_index| {
                let mut unfolded = Vec::<Real>::new();

                self.velocity_map
                    .row(bin_index)
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

                let all_vel = Array1::from(unfolded);

                let diff: Array1<_> = &all_vel.slice(s![self.nof_detections - 1..])
                    - &all_vel.slice(s![..1 - self.nof_detections as isize]);

                let min_diff_index: Option<usize> = diff
                    .indexed_iter()
                    .filter(|(_, &v_diff)| v_diff < velocity_window)
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(index, _)| index);

                if let Some(i) = min_diff_index {
                    Some((
                        bin_index,
                        all_vel
                            .slice(s![i..i + self.nof_detections])
                            .mean()
                            .unwrap(),
                    ))
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn process(&mut self, velocity_window: Real, max_velocity: Real) -> Vec<ResolverDetection> {
        let range_bin_detections = self.binary_integration();
        let detections =
            self.resolve_velocity(&range_bin_detections, velocity_window, max_velocity);
        let term_bins: Vec<usize> = detections.iter().map(|&(bin, _)| bin).collect();
        self.terminate(&term_bins);
        detections
            .iter()
            .map(|&(bin, vel)| ResolverDetection {
                range: (bin - 1) as Real * self.range_bin_length,
                velocity: -vel,
            })
            .collect()
    }

    fn terminate(&mut self, bins: &Vec<usize>) {
        let max_range_in_bins = self.velocity_map.nrows();

        for &bin_index in bins {
            let vel_dets: Vec<usize> = self
                .velocity_map
                .row(bin_index)
                .indexed_iter()
                .filter(|(_, &vel)| vel > 0.0)
                .map(|(n, _)| n)
                .collect();

            vel_dets.iter().for_each(|&n| {
                let nof_range_bins = *self.nof_range_bins.to_owned().get(n).unwrap();
                let start_bin = bin_index % nof_range_bins;
                unfold(start_bin, max_range_in_bins, nof_range_bins)
                    .iter()
                    .for_each(|&bin| {
                        self.velocity_map.slice_mut(s![bin, n]).fill(0.0);
                    })
            });
        }
    }

    pub fn add(
        &mut self,
        detections: &Vec<CfarDetection>,
        properties: &ScanProperties,
    ) -> &mut Self {
        self.nof_range_bins.pop_front().unwrap();
        self.nof_range_bins.push_back(properties.nof_range_bins);
        self.v_prf.pop_front().unwrap();
        self.v_prf.push_back(properties.unambiguous_velocity());
        self.last_det_bins.clear();

        self.velocity_map = stack(
            Axis(1),
            &[
                self.velocity_map.slice(s![.., 1..]),
                Array2::zeros([self.velocity_map.nrows(), 1]).view(),
            ],
        )
        .unwrap();

        let max_range_in_bins = self.velocity_map.nrows();

        for detection in detections {
            let mut det_bins = unfold(
                detection.range_bin,
                max_range_in_bins,
                properties.nof_range_bins,
            );

            det_bins.iter().for_each(|&bin| {
                self.velocity_map
                    .slice_mut(s![bin, -1])
                    .fill(detection.velocity);
            });

            self.last_det_bins.append(&mut det_bins);
        }

        self
    }
}
