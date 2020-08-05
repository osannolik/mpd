use crate::iir;

use ndarray::{s, Array1, Array2, Zip};
use ndarray_rand::rand_distr::{StandardNormal, Uniform};
use ndarray_rand::RandomExt;
//use ndarray_rand::rand_distr::Distribution;
use num_complex::Complex64;
use num_traits::{FloatConst, Num, ToPrimitive};
//use rand::Rng;
use serde::Serialize;
use std::iter::{FromIterator, Sum};
use std::ops::{Add, Neg};

use std::fs::File;
use std::io::Write;

pub type Real = f64;
pub type Natural = u64;

const SPEED_OF_LIGHT: Real = 2.997e8;
/*
/// A generic random value distribution for complex numbers.
#[derive(Clone, Copy, Debug)]
pub struct ComplexDistribution<Re, Im = Re> {
    pub re: Re,
    pub im: Im,
}

impl<T, Re, Im> Distribution<Complex<T>> for ComplexDistribution<Re, Im>
    where
        T: Num + Clone,
        Re: Distribution<T>,
        Im: Distribution<T>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Complex<T> {
        Complex::new(self.re.sample(rng), self.im.sample(rng))
    }
}
*/

#[derive(Copy, Clone)]
pub struct Decibel(Real);

impl Decibel {
    pub fn unit(self) -> Real {
        Real::powf(10.0, self.0 / 20.0)
    }
}

impl Neg for Decibel {
    type Output = Decibel;

    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl<T: Num + ToPrimitive> From<T> for Decibel {
    fn from(value: T) -> Self {
        Self {
            0: value.to_f64().unwrap(),
        }
    }
}

pub trait ToDecibel<T> {
    fn db(self) -> Decibel;
}

impl<T: Num + ToPrimitive> ToDecibel<T> for T {
    fn db(self) -> Decibel {
        Decibel::from(self)
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

pub struct ScanProperties {
    pub carrier_freq: Real,
    pub sample_freq: Real,

    pub duty_cycle: Real,

    pub nof_pulses: usize,
    pub nof_range_bins: usize,
}

impl ScanProperties {
    pub fn nof_send_samples(&self) -> usize {
        ((self.nof_range_bins as Real) * self.duty_cycle) as usize
    }

    pub fn nof_receive_samples(&self) -> usize {
        self.nof_range_bins - self.nof_send_samples()
    }

    pub fn wavelength(&self) -> Real {
        SPEED_OF_LIGHT / self.carrier_freq
    }

    pub fn range_bin_length(&self) -> Real {
        SPEED_OF_LIGHT / self.sample_freq / 2.0
    }

    pub fn pulse_repetition_freq(&self) -> Real {
        self.sample_freq / (self.nof_range_bins as Real)
    }

    pub fn unambiguous_range(&self) -> Real {
        (self.nof_range_bins as Real) * self.range_bin_length()
    }

    pub fn unambiguous_velocity(&self) -> Real {
        self.wavelength() * self.pulse_repetition_freq() / 2.0
    }

    fn alias_velocity(&self, velocity: Real) -> Real {
        velocity % self.unambiguous_velocity()
    }

    fn alias_range(&self, range: Real) -> Real {
        range % self.unambiguous_range()
    }

    pub fn doppler_shift(&self, velocity: Real) -> Real {
        -2.0 * self.alias_velocity(velocity) / self.wavelength()
    }

    pub fn send_pulse(&self, sweep_freq: Real) -> Array1<Complex64> {
        let n = self.nof_send_samples();
        let i = (n as Real - 1.0) / 2.0;
        let t = Array1::linspace(-i, i, n);
        let sweep_rate = sweep_freq / self.sample_freq / n as Real;
        chirp_linear(&t, 0.0, sweep_rate)
    }

    pub fn receive_shape(&self) -> (usize, usize) {
        (self.nof_receive_samples(), self.nof_pulses)
    }
}

pub struct Target {
    pub range: Real,
    pub velocity: Real,
    pub level: Decibel,
}

#[derive(Debug, Serialize)]
pub struct RangePulse {
    matrix: Array2<Complex64>,
}

impl Add for RangePulse {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        RangePulse {
            matrix: self.matrix + rhs.matrix,
        }
    }
}

impl Sum for RangePulse {
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        if let Some(init) = iter.next() {
            iter.fold(
                RangePulse {
                    matrix: init.matrix,
                },
                |acc, item| RangePulse {
                    matrix: acc.matrix + item.matrix,
                },
            )
        } else {
            RangePulse {
                matrix: Array2::from_elem([0, 0], Complex64::new(0.0, 0.0)),
            }
        }
    }
}

impl RangePulse {
    pub fn noise<L: Into<Decibel>>(level: L, properties: &ScanProperties) -> RangePulse {
        // using Array2::random with ComplexDistribution causes
        // "note: perhaps two different versions of crate `rand_core` are being used?"
        // let complex_dist = ComplexDistribution {re: StandardNormal, im: StandardNormal};

        let s = level.into().unit() / Real::SQRT_2();
        let shape = properties.receive_shape();
        // Uniform?
        let im = s * Array2::random(shape, StandardNormal);
        let mut re = s * Array2::random(shape, StandardNormal);

        RangePulse {
            matrix: Zip::from(&mut re)
                .and(&im)
                .apply_collect(|&mut re, &im| Complex64::new(re, im)),
        }
    }

    pub fn clutter<L: Into<Decibel>>(level: L, properties: &ScanProperties) -> RangePulse {
        let (nof_comps, nof_pulses, nof_samples) =
            (11, properties.nof_pulses, properties.nof_receive_samples());
        let freqs_range = Array1::linspace(-0.005, 0.005, nof_comps).to_2d();
        let phase_range = 2.0 * Real::PI() * &freqs_range;
        let pri_range = Array1::linspace(1.0, nof_pulses as Real, nof_pulses).to_2d();
        let freqs_scale = level.into().unit()
            * freqs_range.map(|&f| Real::powf(10.0, -1.25 / 81.0 * Real::powf(f * 2048.0, 2.0)));
        let to_weight = |n: &Real| -> Complex64 {
            Complex64::new(0.0, 2.0 * Real::PI() * n).exp() * (-2.0 * n.ln()).sqrt()
        };
        let weights = Array2::random((nof_samples, nof_comps), Uniform::new(Real::EPSILON, 1.0))
            .map(to_weight)
            * &freqs_scale;
        let phase = phase_range
            .t()
            .dot(&pri_range)
            .map(|&im| Complex64::new(0.0, im).exp());

        RangePulse {
            matrix: weights.dot(&phase),
        }
    }

    pub fn target(target: &Target, properties: &ScanProperties) -> RangePulse {
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

        let mut range_pulses =
            Array2::from_elem(properties.receive_shape(), Complex64::new(0.0, 0.0));

        range_pulses.slice_mut(s![start..stop, ..]).assign(&echo);

        RangePulse {
            matrix: range_pulses,
        }
    }

    pub fn pulse_compress(&mut self, properties: &ScanProperties) -> () {
        let send_pulse = properties.send_pulse(properties.sample_freq);
        let pulse_len = send_pulse.len();
        let window = hanning(pulse_len).map(|&re| Complex64::new(re, 0.0));
        let window = &window / window.dot(&window).sqrt();
        let pulse = window * send_pulse;

        let (n_rs, n_pri) = (self.matrix.shape()[0], self.matrix.shape()[1]);
        let (n_rs_pad, n_pri_pad) = (n_rs + pulse_len - 1, n_pri);
        let mut padded = Array2::from_elem([n_rs_pad, n_pri_pad], Complex64::new(0.0, 0.0));
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

        let matched = Array2::from_shape_vec([n_pri_pad, n_rs_pad], matched).unwrap();

        self.matrix = matched.t().slice(s![(pulse_len - 1).., ..]).into_owned();
    }

    pub fn to_file(&self, path: &std::path::Path) -> Result<(), std::io::Error> {
        let mut file = File::create(&path)?;
        let s = serde_json::to_string(self)?;
        file.write_all(s.as_bytes())?;
        Ok(())
    }
}

pub struct RangeDoppler {
    matrix: Array2<Complex64>,
}
