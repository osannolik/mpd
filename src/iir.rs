extern crate test;

use crate::common;

use common::Real;

use ndarray::Array1;
use ndarray::{s};

use num_traits::Zero;
use std::iter::Sum;
use std::ops::{Div, Mul, Sub};

pub fn iir_filter(b: &Array1<Real>, a: &Array1<Real>, input: &Array1<Real>) -> Array1<Real> {
    assert_eq!(b.len(), a.len());
    let mut w = Array1::from_elem(a.len(), 0.0);
    input.map(|&x| {
        let y = (x * b.first().unwrap() + w.first().unwrap()) / a.first().unwrap();
        let update = x * &b.slice(s![1..]) - y * &a.slice(s![1..]) + w.slice(s![1..]);
        w.slice_mut(s![..w.len() - 1]).assign(&update);
        y
    })
}

pub fn iir_filter_2_reals(b: &[Real], a: &[Real], input: &[Real]) -> Vec<Real> {
    let mut output = vec![0.0; input.len()];

    for (i, _) in input.iter().enumerate() {
        let mut sum = 0.0;
        for (j, &c) in b.iter().enumerate() {
            if i >= j {
                sum += c * input[i - j]
            }
        }
        for (j, &c) in a.iter().enumerate() {
            if i >= j {
                sum -= c * output[i - j]
            }
        }
        output[i] = sum / a.first().unwrap_or(&1.0);
    }

    output
}

pub fn iir_filter_3(b: &[Real], a: &[Real], input: &[Real]) -> Vec<Real> {
    let mut output = vec![0.0; input.len()];

    output = input
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let sum: Real = b
                .iter()
                .enumerate()
                .map(|(j, &c)| if i >= j { c * input[i - j] } else { 0.0 })
                .sum::<Real>()
                - a.iter()
                .enumerate()
                .map(|(j, &c)| if i >= j { c * output[i - j] } else { 0.0 })
                .sum::<Real>();

            sum / a.first().unwrap_or(&1.0)
        })
        .collect();

    output
}

pub fn iir_filter_4_reals(b: &[Real], a: &[Real], input: &[Real]) -> Vec<Real> {
    let mut output = vec![0.0; input.len()];

    let sum = |vals: &[Real], coeffs: &[Real], n: usize| -> Real {
        coeffs
            .iter()
            .take(n + 1)
            .enumerate()
            .map(|(j, &c)| c * vals[n - j])
            .sum()
    };

    for (n, _) in input.iter().enumerate() {
        output[n] = (sum(input, b, n) - sum(&output, a, n)) / a.first().unwrap_or(&1.0);
    }

    output
}

pub fn iir_filter_4<T>(b: &[T], a: &[T], input: &[T]) -> Vec<T>
    where
        T: Copy + Clone + Zero + Mul + Div<Output = T> + Sub<Output = T> + Sum<<T as Mul>::Output>,
{
    let mut output = vec![T::zero(); input.len()];

    let sum = |vals: &[T], coeffs: &[T], n: usize| -> T {
        coeffs
            .iter()
            .take(n + 1)
            .enumerate()
            .map(|(j, &c)| c * vals[n - j])
            .sum()
    };

    for (n, _) in input.iter().enumerate() {
        output[n] = (sum(input, b, n) - sum(&output, a, n)) / *a.first().unwrap();
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;
    use num_complex::Complex64;

    const A: [Real; 4] = [1.00000000, -2.77555756e-16, 3.33333333e-01, -1.85037171e-17];
    const B: [Real; 4] = [0.16666667, 0.5, 0.5, 0.16666667];
    const X: [Real; 20] = [
        -0.917843918645,
        0.141984778794,
        1.20536903482,
        0.190286794412,
        -0.662370894973,
        -1.00700480494,
        -0.404707073677,
        0.800482325044,
        0.743500089861,
        1.01090520172,
        0.741527555207,
        0.277841675195,
        0.400833448236,
        -0.2085993586,
        -0.172842103641,
        -0.134316096293,
        0.0259303398477,
        0.490105989562,
        0.549391221511,
        0.9047198589,
    ];

    #[test]
    fn iir_1() {
        let _ = iir_filter(
            &Array1::from(B.to_vec()),
            &Array1::from(A.to_vec()),
            &Array1::from(X.to_vec()),
        );
    }

    #[test]
    fn iir_2() {
        let _ = iir_filter_2_reals(&B, &A, &X);
    }

    #[test]
    fn iir_3() {
        let _ = iir_filter_3(&B, &A, &X);
    }

    fn to_cpx(arr: &[Real]) -> Vec<Complex64> {
        arr.iter().map(|&re| Complex64::new(re, 0.0)).collect()
    }

    #[test]
    fn iir_4() {
        let bc = to_cpx(&B);
        let ac = to_cpx(&A);
        let xc = to_cpx(&X);

        let _ = iir_filter_4(bc.as_slice(), ac.as_slice(), xc.as_slice());
    }

    #[bench]
    fn bench_iir_1(b: &mut Bencher) {
        let b_arr = &Array1::from(B.to_vec());
        let a_arr = &Array1::from(A.to_vec());
        let inp_arr = &Array1::from(X.to_vec());
        b.iter(|| iir_filter(b_arr, a_arr, inp_arr));
    }

    #[bench]
    fn bench_iir_2_reals(b: &mut Bencher) {
        b.iter(|| iir_filter_2_reals(&B, &A, &X));
    }

    #[bench]
    fn bench_iir_3(b: &mut Bencher) {
        b.iter(|| iir_filter_3(&B, &A, &X));
    }

    #[bench]
    fn bench_iir_4_reals(b: &mut Bencher) {
        b.iter(|| iir_filter_4(&B, &A, &X));
    }

    #[bench]
    fn bench_iir_4(b: &mut Bencher) {
        let bc = to_cpx(&B);
        let ac = to_cpx(&A);
        let xc = to_cpx(&X);

        b.iter(|| iir_filter_4(bc.as_slice(), ac.as_slice(), xc.as_slice()));
    }
}