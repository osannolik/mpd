use crate::common::{DataMatrix, Decibel, RangeDoppler, RangePulse, Real, ScanProperties, Target};
use crate::iir;

use std::iter::FromIterator;

use num_complex::Complex64;
use num_traits::{FloatConst, Zero};

use rustfft::FFTplanner;

use ndarray::{s, Array1, Array2, Zip};
use ndarray_rand::rand_distr::{StandardNormal, Uniform};
use ndarray_rand::RandomExt;

pub type CpxMatrix = Array2<Complex64>;

impl DataMatrix for CpxMatrix {
    fn zero(size: (usize, usize)) -> Self {
        CpxMatrix::from_elem(size, Complex64::zero())
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
}

impl RangePulse<CpxMatrix> {
    pub fn noise<L: Into<Decibel>>(level: L, properties: &ScanProperties) -> RangePulse<CpxMatrix> {
        // using Array2::random with ComplexDistribution causes
        // "note: perhaps two different versions of crate `rand_core` are being used?"
        // let complex_dist = ComplexDistribution {re: StandardNormal, im: StandardNormal};

        let s = level.into().unit() / Real::SQRT_2();
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

    pub fn clutter<L: Into<Decibel>>(
        level: L,
        properties: &ScanProperties,
    ) -> RangePulse<CpxMatrix> {
        let (nof_comps, nof_pulses, nof_samples) =
            (11, properties.nof_pulses, properties.nof_receive_samples());
        let freqs_range = Array1::linspace(-0.005, 0.005, nof_comps).to_2d();
        let phase_range = 2.0 * Real::PI() * &freqs_range;
        let pri_range = Array1::linspace(1.0, nof_pulses as Real, nof_pulses).to_2d();
        let freqs_scale = level.into().unit()
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
        let amp = target.level.unit();
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
        let pad = pulse_len / 2;

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

impl From<RangePulse<CpxMatrix>> for RangeDoppler<CpxMatrix> {
    fn from(range_pulse: RangePulse<CpxMatrix>) -> Self {
        range_pulse.doppler_filtering()
    }
}
