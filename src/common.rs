use core::marker::PhantomData;

use std::fs::File;
use std::io::Write;
use std::iter::Sum;
use std::ops::{Add, Neg, Sub};

use num_traits::{Num, ToPrimitive};
use serde::Serialize;

pub type Real = f64;
pub type Natural = u64;

const SPEED_OF_LIGHT: Real = 2.997e8;

#[derive(Copy, Clone)]
pub struct Decibel(Real);

#[derive(Copy, Clone)]
pub struct Ratio(Real);

impl Add for Decibel {
    type Output = Decibel;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl Sub for Decibel {
    type Output = Decibel;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl Neg for Decibel {
    type Output = Decibel;

    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl From<Decibel> for Ratio {
    fn from(db: Decibel) -> Self {
        Self(Real::powf(10.0, db.0 / 20.0))
    }
}

impl From<Ratio> for Decibel {
    fn from(ratio: Ratio) -> Self {
        Self(20.0 * Real::log10(ratio.0))
    }
}
/*
macro_rules! impl_value_for {
    ($DR: ty) => {
        impl $DR {
            #[inline]
            pub fn value(self) -> Real {
                self.0
            }
        }
    };
}

impl_value_for!(Decibel);
impl_value_for!(Ratio);
*/


macro_rules! impl_from_primitive_for {
    ($DR: ty) => {
        impl<T: Num + ToPrimitive> From<T> for $DR {
            #[inline]
            fn from(value: T) -> Self {
                Self(value.to_f64().unwrap())
            }
        }
    };
}

impl_from_primitive_for!(Decibel);
impl_from_primitive_for!(Ratio);

impl From<Decibel> for Real {
    fn from(db: Decibel) -> Self {
        db.0
    }
}

impl From<Ratio> for Real {
    fn from(ratio: Ratio) -> Self {
        ratio.0
    }
}

pub trait Units {
    fn db(self) -> Decibel;

    fn ratio(self) -> Ratio;

    fn value(self) -> Real;
}

impl<T: Into<Decibel> + Into<Ratio> + Into<Real>> Units for T {
    fn db(self) -> Decibel {
        self.into()
    }

    fn ratio(self) -> Ratio {
        self.into()
    }

    fn value(self) -> Real {
        self.into()
    }
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

    pub fn alias_velocity(&self, velocity: Real) -> Real {
        velocity % self.unambiguous_velocity()
    }

    pub fn alias_range(&self, range: Real) -> Real {
        range % self.unambiguous_range()
    }

    pub fn doppler_shift(&self, velocity: Real) -> Real {
        -2.0 * self.alias_velocity(velocity) / self.wavelength()
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

pub trait DataMatrix<T = Self>: Add<T, Output = T> + Serialize {
    fn zero(size: (usize, usize)) -> Self;

    fn size(&self) -> (usize, usize);
}

#[derive(Debug, Serialize)]
pub struct Time {}

#[derive(Debug, Serialize)]
pub struct Freq {}

#[derive(Debug, Serialize)]
pub struct Inti<M, D> {
    pub matrix: M,
    _domain: PhantomData<D>,
}

impl<M: DataMatrix, D> Inti<M, D> {
    pub fn new(matrix: M) -> Self {
        Self {
            matrix: matrix,
            _domain: PhantomData,
        }
    }

    pub fn size(&self) -> (usize, usize) {
        self.matrix.size()
    }
}

impl<M: DataMatrix, D> Add for Inti<M, D> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::Output::new(self.matrix + rhs.matrix)
    }
}

impl<M: DataMatrix, D> Sum for Inti<M, D> {
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        if let Some(init) = iter.next() {
            iter.fold(init, |acc, item| acc + item)
        } else {
            Self {
                matrix: M::zero((0, 0)),
                _domain: PhantomData,
            }
        }
    }
}

pub trait Storable: Serialize {
    fn to_file(&self, path: &std::path::Path) -> Result<(), std::io::Error> {
        let mut file = File::create(&path)?;
        let s = serde_json::to_string(self)?;
        file.write_all(s.as_bytes())?;
        Ok(())
    }
}

impl<M: DataMatrix, D> Storable for Inti<M, D> {}

pub type RangePulse<M> = Inti<M, Time>;
pub type RangeDoppler<M> = Inti<M, Freq>;
