use crate::common::{
    DataMatrix, Decibel, RangeDoppler, RangePulse, Real, ScanProperties, Target, Units,
};
use crate::iir;

use std::iter::FromIterator;

use num_complex::Complex64;
use num_traits::{FloatConst, Num, Zero};

use rustfft::FFTplanner;

use ndarray::{s, Array1, Array2, Axis, Zip};
use ndarray_rand::rand_distr::{StandardNormal, Uniform};
use ndarray_rand::RandomExt;
use serde::Serialize;
//use std::fs::File;
//use std::io::Write;
use std::ops::Add;
//use std::path::Path;

pub type CpxMatrix = Array2<Complex64>;
type RealMatrix = Array2<Real>;
type BoolMatrix = Array2<bool>;

impl<T: Num + Clone + Serialize + Add> DataMatrix for Array2<T> {
    fn zero(size: (usize, usize)) -> Self {
        Array2::from_elem(size, T::zero())
    }

    fn size(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }
}

trait Reshape2d<T> {
    fn to_1d(self) -> Array1<T>;
}

trait Reshape1d<T> {
    fn to_2d(self) -> Array2<T>;
}

impl<T> Reshape2d<T> for Array2<T> {
    fn to_1d(self) -> Array1<T> {
        let len = self.len();
        self.into_shape(len).unwrap()
    }
}

impl<T> Reshape1d<T> for Array1<T> {
    fn to_2d(self) -> Array2<T> {
        let len = self.len();
        self.into_shape([1, len]).unwrap()
    }
}

pub fn chirp_linear(t: &Array1<Real>, f0: Real, k: Real) -> Array1<Complex64> {
    t.map(|&t| -2.0 * Real::PI() * (f0 * t + k * t * t / 2.0))
        .map(|&im| Complex64::new(0.0, im).exp())
}

pub fn hanning(length: usize) -> Array1<Real> {
    /*
    let n = (length - 1) as Real;
    Array1::linspace(0.0, n, length)
        .map(|&i| 0.5 * (1.0 - Real::cos(2.0 * Real::PI() * i / n)))
     */
    let l = length as Real;
    Array1::linspace(0.5, l - 0.5, length).map(|&n| Real::sin(Real::PI() * n / l).powf(2.0))
}

impl ScanProperties {
    pub fn send_pulse(&self, sweep_freq: Real) -> Array1<Complex64> {
        let n = self.nof_send_samples();
        let i = (n as Real - 1.0) / 2.0;
        let t = Array1::linspace(-i, i, n);
        let sweep_rate = sweep_freq / self.sample_freq / n as Real;
        chirp_linear(&t, 0.0, sweep_rate)
    }

    pub fn pulse_compress_rb_start(&self) -> usize {
        self.nof_send_samples() / 2
    }
}

impl RangePulse<CpxMatrix> {
    pub fn noise<L>(level: L, properties: &ScanProperties) -> RangePulse<CpxMatrix>
    where
        L: Into<Decibel>,
    {
        // using Array2::random with ComplexDistribution causes
        // "note: perhaps two different versions of crate `rand_core` are being used?"
        // let complex_dist = ComplexDistribution {re: StandardNormal, im: StandardNormal};

        let s: Real = level.into().ratio().value() / Real::SQRT_2();
        let shape = properties.receive_shape();
        // Uniform?
        let im = s * Array2::random(shape, StandardNormal);
        let mut re = s * Array2::random(shape, StandardNormal);

        Self::new(
            Zip::from(&mut re)
                .and(&im)
                .apply_collect(|&mut re, &im| Complex64::new(re, im)),
        )
    }

    pub fn clutter<L>(level: L, properties: &ScanProperties) -> RangePulse<CpxMatrix>
    where
        L: Into<Decibel>,
    {
        let (nof_comps, nof_pulses, nof_samples) =
            (11, properties.nof_pulses, properties.nof_receive_samples());
        let freqs_range = Array1::linspace(-0.005, 0.005, nof_comps).to_2d();
        let phase_range = 2.0 * Real::PI() * &freqs_range;
        let pri_range = Array1::linspace(1.0, nof_pulses as Real, nof_pulses).to_2d();
        let freqs_scale = level.into().ratio().value()
            * freqs_range.map(|&f| Real::powf(10.0, -1.25 / 81.0 * Real::powf(f * 2048.0, 2.0)));

        let to_weight = |n: &Real| -> Complex64 {
            Complex64::from_polar(&(-2.0 * n.ln()).sqrt(), &(2.0 * Real::PI() * n))
        };
        let weights = Array2::random((nof_samples, nof_comps), Uniform::new(Real::EPSILON, 1.0))
            .map(to_weight)
            * &freqs_scale;
        let phase = phase_range
            .t()
            .dot(&pri_range)
            .map(|im| Complex64::from_polar(&1.0, im));

        Self::new(weights.dot(&phase))
    }

    pub fn target(target: &Target, properties: &ScanProperties) -> RangePulse<CpxMatrix> {
        let phase_shift_per_pulse = 2.0 * Real::PI() * properties.doppler_shift(target.velocity)
            / properties.pulse_repetition_freq();
        let nof_pulses = properties.nof_pulses;
        let amp = target.level.ratio().value();
        let reflection = Array1::linspace(0.0, (nof_pulses - 1) as Real, nof_pulses)
            .map(|&x| Complex64::from_polar(&amp, &(phase_shift_per_pulse * x)))
            .to_2d();
        let send_pulse = properties.send_pulse(properties.sample_freq).to_2d();
        let echo = send_pulse.t().dot(&reflection);
        let target_rx_bin = (properties.alias_range(target.range) / properties.range_bin_length())
            .round() as usize
            - send_pulse.len();

        // Limit these and slice echo correctly for targets in pulse or close to Rprf...
        let start = target_rx_bin.max(0);
        let stop = start + echo.dim().0;

        let mut range_pulses = CpxMatrix::zeros(properties.receive_shape());

        range_pulses.slice_mut(s![start..stop, ..]).assign(&echo);

        Self::new(range_pulses)
    }

    pub fn pulse_compress(&mut self, properties: &ScanProperties) {
        let send_pulse = properties.send_pulse(properties.sample_freq);
        let pulse_len = send_pulse.len();
        let window = hanning(pulse_len).map(|&re| Complex64::new(re, 0.0));
        let window = &window / window.dot(&window).sqrt();
        let pulse = window * send_pulse;

        let (n_rs, n_pri) = self.matrix.size();
        let (n_rs_pad, n_pri_pad) = (n_rs + pulse_len - 1, n_pri);
        let mut padded = CpxMatrix::from_elem([n_rs_pad, n_pri_pad], Complex64::new(0.0, 0.0));
        let pad = properties.pulse_compress_rb_start();

        padded
            .slice_mut(s![pad..(n_rs + pad), ..])
            .assign(&self.matrix);

        let coeffs: Array1<_> = pulse.iter().rev().map(|&c| c.conj()).collect();
        let all_range_bins = Array1::from_iter(padded.t().iter().cloned());

        let matched = iir::iir_filter(
            coeffs.as_slice().unwrap(),
            &[Complex64::new(1.0, 0.0)],
            all_range_bins.as_slice().unwrap(),
        );

        let matched = CpxMatrix::from_shape_vec([n_pri_pad, n_rs_pad], matched).unwrap();

        self.matrix = matched.t().slice(s![(pulse_len - 1).., ..]).into_owned();
    }

    pub fn doppler_filtering(self) -> RangeDoppler<CpxMatrix> {
        let (n_rs, n_pri) = self.matrix.size();
        let window = hanning(n_pri).map(|&re| Complex64::new(re, 0.0));
        let window = &window / window.dot(&window).sqrt();
        let window2d = CpxMatrix::ones([n_rs, 1]).dot(&window.to_2d());

        let mut range_pulse: CpxMatrix = window2d * &self.matrix;

        let mut planner: FFTplanner<f64> = FFTplanner::new(false);
        let fft = planner.plan_fft(n_pri);

        let mut range_doppler: CpxMatrix = DataMatrix::zero(self.matrix.size());

        Zip::from(range_pulse.genrows_mut())
            .and(range_doppler.genrows_mut())
            .apply(|mut rp, mut rd| {
                fft.process(rp.as_slice_mut().unwrap(), rd.as_slice_mut().unwrap());
            });

        RangeDoppler::new(range_doppler)
    }
}

pub struct CfarConfig {
    pub mainlobe_clutter_margin: usize,
    pub min_value: Decibel,
    pub threshold_offset: Decibel,
    pub n_rs_thr: usize,
    pub n_guard_bins: usize,
}

fn local_max(matrix: &RealMatrix, edge_level: Decibel, cell_radius: usize) -> BoolMatrix {
    let (n_rs, n_pri) = (matrix.nrows(), matrix.ncols());
    let r = cell_radius as isize;
    let mut edged_matrix = Array2::from_elem(
        [n_rs + 2 * cell_radius, n_pri + 2 * cell_radius],
        edge_level.value(),
    );
    edged_matrix
        .slice_mut(s![r..r + n_rs as isize, r..r + n_pri as isize])
        .assign(matrix);

    let mut is_local_max = BoolMatrix::from_elem([n_rs, n_pri], true);
    for dx in -r..=r {
        for dy in -r..=r {
            if dx != 0 || dy != 0 {
                let s = s![
                    r + dx..r + dx + n_rs as isize,
                    r + dy..r + dy + n_pri as isize
                ];
                is_local_max =
                    is_local_max & (matrix - &edged_matrix.slice(s)).map(|&diff| diff > 0.0)
            }
        }
    }
    is_local_max
}

fn local_threshold(matrix: &RealMatrix, n_rs_thr: usize, n_guard_bins: usize) -> RealMatrix {
    let (n_rs, n_pri) = (matrix.nrows(), matrix.ncols());
    let pad = n_rs_thr + n_guard_bins;
    let n_rb_pad = n_rs + 2 * pad + 1;

    let mut padded = RealMatrix::from_elem([n_rb_pad, n_pri], 0.0);

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
    pub fn cfar(&self, config: &CfarConfig, properties: &ScanProperties) -> Vec<Target> {
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
            .map(|((rb, pri), _)| Target {
                range: properties.to_range(n_rb_min + rb),
                velocity: properties.unambiguous_velocity() * pri as Real
                    / properties.nof_pulses as Real,
                level: 0.0.db(),
            })
            .collect()
    }
}

impl From<RangePulse<CpxMatrix>> for RangeDoppler<CpxMatrix> {
    fn from(range_pulse: RangePulse<CpxMatrix>) -> Self {
        range_pulse.doppler_filtering()
    }
}
