use ndarray::{Array1, Array2};
use num_complex::Complex64;
use num_traits::FloatConst;

pub type Real = f64;
pub type Natural = u64;

const SPEED_OF_LIGHT: Real = 2.997e8;

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

pub struct RangePulse(Array2<Real>);

impl RangePulse {
    pub fn generate(_p: &SendPulse) -> RangePulse {
        RangePulse {
            0: Array2::zeros((2, 2)),
        }
    }
}

pub struct RangeDoppler(Array2<Real>);
