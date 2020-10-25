use crate::common::{
    CfarDetection, DataMatrix, Decibel, RangeDoppler, Real, ScanProperties, Units,
};
use crate::mpd::{BoolMatrix, CpxMatrix, RealMatrix};

use ndarray::{s, Array2, Axis, Zip};
use num::Zero;

#[derive(Debug)]
pub struct CfarConfig {
    pub mainlobe_clutter_margin: usize,
    pub min_value: Decibel,
    pub threshold_offset: Decibel,
    pub n_rs_thr: usize,
    pub n_guard_bins: usize,
}

fn local_max(matrix: &RealMatrix, edge_level: Decibel, cell_radius: usize) -> BoolMatrix {
    let (n_rs, n_ch) = (matrix.nrows(), matrix.ncols());
    let r = cell_radius as isize;
    let mut edged_matrix = Array2::from_elem(
        [n_rs + 2 * cell_radius, n_ch + 2 * cell_radius],
        edge_level.value(),
    );
    edged_matrix
        .slice_mut(s![r..r + n_rs as isize, r..r + n_ch as isize])
        .assign(matrix);

    let mut is_local_max = BoolMatrix::from_elem([n_rs, n_ch], true);
    for dx in -r..=r {
        for dy in -r..=r {
            if dx != 0 || dy != 0 {
                let s = s![
                    r + dx..r + dx + n_rs as isize,
                    r + dy..r + dy + n_ch as isize
                ];
                is_local_max =
                    is_local_max & (matrix - &edged_matrix.slice(s)).map(|&diff| diff > 0.0)
            }
        }
    }
    is_local_max
}

fn local_threshold(matrix: &RealMatrix, n_rs_thr: usize, n_guard_bins: usize) -> RealMatrix {
    let (n_rs, n_ch) = (matrix.nrows(), matrix.ncols());
    let pad = n_rs_thr + n_guard_bins;
    let n_rb_pad = n_rs + 2 * pad + 1;

    let mut padded = RealMatrix::from_elem([n_rb_pad, n_ch], 0.0);

    let first_range_bins = s![1 + n_guard_bins..1 + n_guard_bins + pad, ..];
    let last_range_bins = s![n_rs - n_guard_bins - pad - 1..n_rs - n_guard_bins - 1, ..];

    // Repeat first and last range bins and let first row be 0
    padded
        .slice_mut(s![1..=pad, ..])
        .assign(&matrix.slice(first_range_bins));
    padded
        .slice_mut(s![1 + pad..=pad + n_rs, ..])
        .assign(&matrix);
    padded
        .slice_mut(s![1 + pad + n_rs.., ..])
        .assign(&matrix.slice(last_range_bins));

    // Cumulative sum and then difference with shiftet by n gives average in interval of n
    padded.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);
    let avg = (&padded.slice(s![n_rs_thr.., ..]) - &padded.slice(s![..n_rb_pad - n_rs_thr, ..]))
        / n_rs_thr as Real;

    let delta = 1 + n_rs_thr + 2 * n_guard_bins;
    let n_avg = n_rb_pad - n_rs_thr;

    // Greatest of early and late average
    let (early, late) = (
        avg.slice(s![..n_avg - delta, ..]),
        avg.slice(s![delta.., ..]),
    );

    Zip::from(early)
        .and(late)
        .apply_collect(|&f, &l| Real::max(f, l))
}

impl RangeDoppler<CpxMatrix> {
    pub fn cfar(&self, config: &CfarConfig, properties: &ScanProperties) -> Vec<CfarDetection> {
        let abs_matrix_db: RealMatrix = self.matrix.map(|&x| x.norm().ratio().db().value());

        let no_ml_clutter_pris = s![
            ..,
            config.mainlobe_clutter_margin..abs_matrix_db.ncols() - config.mainlobe_clutter_margin
        ];
        let noise = abs_matrix_db.slice(no_ml_clutter_pris).mean().unwrap();

        let threshold: RealMatrix =
            local_threshold(&abs_matrix_db, config.n_rs_thr, config.n_guard_bins).map(|&lth| {
                config.threshold_offset.db().value() + noise + (lth - noise).max(Real::zero())
            });

        let is_above_threshold: BoolMatrix = Zip::from(&abs_matrix_db)
            .and(&threshold)
            .apply_collect(|signal, thd| signal > thd);

        let is_local_max: BoolMatrix = local_max(&abs_matrix_db, config.min_value, 1);

        let is_allowed_region: BoolMatrix = {
            let mut allowed = BoolMatrix::from_elem(abs_matrix_db.size(), false);
            allowed.slice_mut(no_ml_clutter_pris).fill(true);
            allowed
        };

        let detections: BoolMatrix = is_allowed_region & is_local_max & is_above_threshold;

        let n_rb_min = properties.nof_send_samples() - properties.pulse_compress_rb_start();

        detections
            .indexed_iter()
            .filter(|&(_, &is_det)| is_det)
            .map(|((rb, ch), _)| CfarDetection {
                range_bin: n_rb_min + rb,
                velocity: properties.unambiguous_velocity() * ch as Real
                    / properties.nof_pulses as Real,
                level: abs_matrix_db[[rb, ch]].db(),
            })
            .collect()
    }
}
