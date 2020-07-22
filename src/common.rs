use ndarray::{Array1, Array2, Zip};
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
//use ndarray_rand::rand_distr::Distribution;
use num_complex::Complex64;
use num_traits::FloatConst;
//use rand::Rng;

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

pub struct Decibel(Real);

impl Decibel {
    pub fn scale(self) -> Real {
        Real::powf(10.0, self.0 / 20.0)
    }
}

pub trait ToDecibel {
    fn db(self) -> Decibel;
}

impl ToDecibel for Real {
    fn db(self) -> Decibel {
        Decibel(self)
    }
}

pub fn chirp_linear(t: &Array1<Real>, f0: Real, k: Real) -> Array1<Complex64> {
    t.map(|&t| -2.0 * Real::PI() * (f0 * t + k * t * t / 2.0))
        .map(|&im| Complex64::new(0.0, im).exp())
}

pub struct ScanProperties {
    pub carrier_freq: Real,
    pub sample_freq: Real,

    pub duty_cycle: Real,

    pub nof_pulses: Natural,
    pub nof_range_bins: Natural,
}

impl ScanProperties {
    pub fn nof_send_samples(&self) -> Natural {
        ((self.nof_range_bins as Real) * self.duty_cycle) as Natural
    }

    pub fn nof_receive_samples(&self) -> Natural {
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

    pub fn receive_shape(&self) -> (usize, usize) {
        (
            self.nof_receive_samples() as usize,
            self.nof_pulses as usize,
        )
    }
}

#[derive(Debug)]
pub struct SendPulse {
    signal: Array1<Complex64>,
}

impl SendPulse {
    pub fn new(sweep_freq: Real, properties: &ScanProperties) -> SendPulse {
        let n = properties.nof_send_samples() as Real;
        let i = (n - 1.0) / 2.0;
        let t = Array1::linspace(-i, i, n as usize);
        let sweep_rate = sweep_freq / properties.sample_freq / n;

        SendPulse {
            signal: chirp_linear(&t, 0.0, sweep_rate),
        }
    }
}

#[derive(Debug)]
pub struct RangePulse {
    matrix: Array2<Complex64>,
}

impl RangePulse {
    pub fn noise(level: Decibel, properties: &ScanProperties) -> RangePulse {
        // using Array2::random with ComplexDistribution causes
        // "note: perhaps two different versions of crate `rand_core` are being used?"
        // let complex_dist = ComplexDistribution {re: StandardNormal, im: StandardNormal};

        let s = level.scale() / Real::SQRT_2();
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
}

pub struct RangeDoppler {
    matrix: Array2<Complex64>,
}
