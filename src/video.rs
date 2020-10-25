use crate::common::{Decibel, RangePulse, Real, ScanProperties, Target, Units};
use crate::mpd::{CpxMatrix, Reshape1d};

use ndarray::{s, Array1, Array2, Zip};
use ndarray_rand::rand_distr::{StandardNormal, Uniform};
use ndarray_rand::RandomExt;
use num::complex::Complex64;
use num::traits::FloatConst;

pub fn generate<L>(
    targets: &Vec<Target>,
    noise: L,
    clutter: L,
    properties: &ScanProperties,
) -> RangePulse<CpxMatrix>
where
    L: Into<Decibel>,
{
    RangePulse::noise(noise, properties)
        + RangePulse::clutter(clutter, properties)
        + targets
            .iter()
            .map(|tgt| RangePulse::target(tgt, properties))
            .sum()
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
            Complex64::from_polar((-2.0 * n.ln()).sqrt(), 2.0 * Real::PI() * n)
        };
        let weights = Array2::random((nof_samples, nof_comps), Uniform::new(Real::EPSILON, 1.0))
            .map(to_weight)
            * &freqs_scale;
        let phase = phase_range
            .t()
            .dot(&pri_range)
            .map(|&im| Complex64::from_polar(1.0, im));

        Self::new(weights.dot(&phase))
    }

    pub fn target(target: &Target, properties: &ScanProperties) -> RangePulse<CpxMatrix> {
        let phase_shift_per_pulse = 2.0 * Real::PI() * properties.doppler_shift(target.velocity)
            / properties.pulse_repetition_freq();
        let nof_pulses = properties.nof_pulses;
        let amp = target.level.ratio().value();
        let reflection = Array1::linspace(0.0, (nof_pulses - 1) as Real, nof_pulses)
            .map(|&x| Complex64::from_polar(amp, phase_shift_per_pulse * x))
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
}
