//! INT8 encode/decode used internally by `Int8VectorStore`.
//!
//! Persistent layout, metadata, and mutation are owned by the Python store.

use rayon::{prelude::*, ThreadPoolBuilder};

fn validate(values: &[f32], rows: usize, dimension: usize, name: &str) -> Result<(), String> {
    if rows == 0 || dimension == 0 {
        return Err(format!("{name} must have non-zero axes"));
    }
    if values.len() != rows * dimension {
        return Err(format!("{name} shape does not match its value count"));
    }
    if !crate::core::all_finite(values) {
        return Err(format!("{name} contains a non-finite value"));
    }
    Ok(())
}

fn install<T: Send>(
    threads: Option<usize>,
    execute: impl FnOnce() -> T + Send,
) -> Result<T, String> {
    if threads == Some(0) {
        return Err("threads must be positive".to_string());
    }
    if let Some(threads) = threads {
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|error| format!("could not create vector-store worker pool: {error}"))
            .map(|pool| pool.install(execute))
    } else {
        Ok(execute())
    }
}

#[inline]
fn scalar_normalize(values: &mut [f32]) {
    let norm = values
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
        .max(1.0e-12);
    for value in values {
        *value /= norm;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_normalize(values: &mut [f32]) {
    use std::arch::x86_64::*;

    let mut sum = _mm256_setzero_ps();
    let mut position = 0;
    while position + 8 <= values.len() {
        let value = unsafe { _mm256_loadu_ps(values.as_ptr().add(position)) };
        sum = _mm256_add_ps(sum, _mm256_mul_ps(value, value));
        position += 8;
    }
    let mut lanes = [0.0f32; 8];
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), sum) };
    let mut squared_norm = lanes.iter().sum::<f32>();
    for value in &values[position..] {
        squared_norm += value * value;
    }
    let inverse = 1.0 / squared_norm.sqrt().max(1.0e-12);
    let scale = _mm256_set1_ps(inverse);
    position = 0;
    while position + 8 <= values.len() {
        let value = unsafe { _mm256_loadu_ps(values.as_ptr().add(position)) };
        unsafe {
            _mm256_storeu_ps(
                values.as_mut_ptr().add(position),
                _mm256_mul_ps(value, scale),
            )
        };
        position += 8;
    }
    for value in &mut values[position..] {
        *value *= inverse;
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn neon_normalize(values: &mut [f32]) {
    use std::arch::aarch64::*;

    if values.len() < 4 {
        return scalar_normalize(values);
    }
    let mut sum = vdupq_n_f32(0.0);
    let mut position = 0;
    while position + 4 <= values.len() {
        let value = unsafe { vld1q_f32(values.as_ptr().add(position)) };
        sum = vfmaq_f32(sum, value, value);
        position += 4;
    }
    let mut squared_norm = vaddvq_f32(sum);
    for value in &values[position..] {
        squared_norm += value * value;
    }
    let inverse = 1.0 / squared_norm.sqrt().max(1.0e-12);
    let scale = vdupq_n_f32(inverse);
    position = 0;
    while position + 4 <= values.len() {
        let value = unsafe { vld1q_f32(values.as_ptr().add(position)) };
        unsafe { vst1q_f32(values.as_mut_ptr().add(position), vmulq_f32(value, scale)) };
        position += 4;
    }
    for value in &mut values[position..] {
        *value *= inverse;
    }
}

#[inline]
fn simd_normalize(values: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return unsafe { avx2_normalize(values) };
        }
        scalar_normalize(values)
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_normalize(values) }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        scalar_normalize(values)
    }
}

pub fn int8_encode(
    embeddings: &[f32],
    rows: usize,
    dimension: usize,
    threads: Option<usize>,
) -> Result<(Vec<i8>, Vec<f32>), String> {
    validate(embeddings, rows, dimension, "embeddings")?;
    let mut codes = vec![0i8; embeddings.len()];
    let mut scales = vec![0.0f32; rows];
    install(threads, || {
        codes
            .par_chunks_mut(dimension)
            .zip(scales.par_iter_mut())
            .enumerate()
            .for_each(|(row, (output, scale))| {
                let input = &embeddings[row * dimension..(row + 1) * dimension];
                let maximum = input.iter().map(|value| value.abs()).fold(0.0, f32::max);
                *scale = (maximum / 127.0).max(1.0e-12);
                for (encoded, &value) in output.iter_mut().zip(input) {
                    *encoded = (value / *scale).round().clamp(-127.0, 127.0) as i8;
                }
            });
    })?;
    Ok((codes, scales))
}

pub fn int8_decode(
    codes: &[i8],
    scales: &[f32],
    rows: usize,
    dimension: usize,
    normalize: bool,
    threads: Option<usize>,
) -> Result<Vec<f32>, String> {
    if codes.len() != rows * dimension || scales.len() != rows {
        return Err("int8 rows, scales, and dimension are inconsistent".to_string());
    }
    if scales
        .iter()
        .any(|scale| !scale.is_finite() || *scale <= 0.0)
    {
        return Err("int8 scales must be finite and positive".to_string());
    }
    let mut output = vec![0.0f32; codes.len()];
    install(threads, || {
        output
            .par_chunks_mut(dimension)
            .enumerate()
            .for_each(|(row, decoded)| {
                let input = &codes[row * dimension..(row + 1) * dimension];
                for (value, &code) in decoded.iter_mut().zip(input) {
                    *value = f32::from(code) * scales[row];
                }
                if normalize {
                    simd_normalize(decoded);
                }
            });
    })?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int8_round_trip_preserves_dominant_coordinates() {
        let values = [1.0, 0.0, 0.0, 0.0, 0.1, 0.99, 0.0, 0.0];
        let (codes, scales) = int8_encode(&values, 2, 4, Some(1)).unwrap();
        let decoded = int8_decode(&codes, &scales, 2, 4, true, Some(1)).unwrap();
        assert_eq!(decoded[0], 1.0);
        assert!(decoded[5] > decoded[4]);
    }
}
