use crate::common::{DataMatrix, RangeDoppler, RangePulse, Real, ScanProperties};

use ndarray::{Array1, Array2};
use num::complex::Complex64;
use num::traits::{FloatConst, Num};
use serde::Serialize;

use std::ops::Add;

pub type CpxMatrix = Array2<Complex64>;
pub type RealMatrix = Array2<Real>;
pub type BoolMatrix = Array2<bool>;

impl<T: Num + Clone + Serialize + Add> DataMatrix for Array2<T> {
    fn zero(size: (usize, usize)) -> Self {
        Array2::from_elem(size, T::zero())
    }

    fn size(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }
}

pub trait Reshape2d<T> {
    fn to_1d(self) -> Array1<T>;
}

pub trait Reshape1d<T> {
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

fn chirp_linear(t: &Array1<Real>, f0: Real, k: Real) -> Array1<Complex64> {
    t.map(|&t| -2.0 * Real::PI() * (f0 * t + k * t * t / 2.0))
        .map(|&im| Complex64::new(0.0, im).exp())
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

impl From<RangePulse<CpxMatrix>> for RangeDoppler<CpxMatrix> {
    fn from(range_pulse: RangePulse<CpxMatrix>) -> Self {
        range_pulse.doppler_filtering()
    }
}
