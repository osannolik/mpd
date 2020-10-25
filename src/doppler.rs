use crate::common::{DataMatrix, RangeDoppler, RangePulse, Real, ScanProperties};
use crate::iir;
use crate::mpd::{CpxMatrix, Reshape1d};

use ndarray::{s, Array1, Zip};
use num::complex::Complex64;
use num::traits::FloatConst;
use rustfft::FFTplanner;

use std::iter::FromIterator;

pub trait DopplerFiltering {
    fn into_range_doppler(self, properties: &ScanProperties) -> RangeDoppler<CpxMatrix>;
}

fn hanning(length: usize) -> Array1<Real> {
    /*
    let n = (length - 1) as Real;
    Array1::linspace(0.0, n, length)
        .map(|&i| 0.5 * (1.0 - Real::cos(2.0 * Real::PI() * i / n)))
     */
    let l = length as Real;
    Array1::linspace(0.5, l - 0.5, length).map(|&n| Real::sin(Real::PI() * n / l).powf(2.0))
}

impl RangePulse<CpxMatrix> {
    pub fn pulse_compress(&self, properties: &ScanProperties) -> Self {
        let send_pulse = properties.send_pulse(properties.sample_freq);
        let pulse_len = send_pulse.len();
        let window = hanning(pulse_len);
        let norm = window.dot(&window).sqrt();
        let window = window.map(|&re| Complex64::new(re / norm, 0.0));
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

        Self::new(matched.t().slice(s![(pulse_len - 1).., ..]).into_owned())
    }

    pub fn doppler_filtering(self) -> RangeDoppler<CpxMatrix> {
        let (n_rs, n_pri) = self.matrix.size();
        let window = hanning(n_pri);
        let norm = window.dot(&window).sqrt();
        let window = window.map(|&re| Complex64::new(re / norm, 0.0));
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

impl DopplerFiltering for RangePulse<CpxMatrix> {
    fn into_range_doppler(self, properties: &ScanProperties) -> RangeDoppler<CpxMatrix> {
        self.pulse_compress(properties).doppler_filtering()
    }
}
