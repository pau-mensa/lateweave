//! CPU representation kernels used internally by lateweave vector stores.
//!
//! These functions are not a codec contract. Persistent layout, metadata,
//! mutation, and representation transition are owned by the corresponding
//! Python store implementation.

use rayon::{prelude::*, ThreadPoolBuilder};
use std::mem::size_of;

pub const INT8_FORMAT: &str = "lateweave-int8-rowwise-v1";
pub const TURBOQUANT_FORMAT: &str = "lateweave-turboquant4-mse-v1";
pub const JZIP_FORMAT: &str = "lateweave-jzip-document-zstd-v1";

const JZIP_MAGIC: [u8; 4] = *b"LWJZ";
const JZIP_VERSION: u16 = 1;
const JZIP_FLAGS_UNIT_NORM: u16 = 1;
const JZIP_HEADER_BYTES: usize = 16;

const TQ_CENTROIDS: [f32; 16] = [
    -2.733, -2.069, -1.618, -1.256, -0.9424, -0.6568, -0.3881, -0.1284, 0.1284, 0.3881, 0.6568,
    0.9424, 1.256, 1.618, 2.069, 2.733,
];
const TQ_BOUNDARIES: [f32; 15] = boundaries(TQ_CENTROIDS);
const PERMUTATION_SEEDS: [u64; 3] = [
    654_605_292_835_415_893,
    8_636_605_637_963_351_413,
    1_775_280_196_666_917_949,
];

const fn boundaries(centroids: [f32; 16]) -> [f32; 15] {
    let mut output = [0.0; 15];
    let mut index = 0;
    while index < 15 {
        output[index] = (centroids[index] + centroids[index + 1]) / 2.0;
        index += 1;
    }
    output
}

fn validate(values: &[f32], rows: usize, dimension: usize, name: &str) -> Result<(), String> {
    if rows == 0 || dimension == 0 {
        return Err(format!("{name} must have non-zero axes"));
    }
    if values.len() != rows * dimension {
        return Err(format!("{name} shape does not match its value count"));
    }
    if values.iter().any(|value| !value.is_finite()) {
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

fn checked_product(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("{name} size overflows usize"))
}

fn document_offsets(lengths: &[usize], rows: usize) -> Result<Vec<usize>, String> {
    if lengths.is_empty() || lengths.contains(&0) {
        return Err("document lengths must be non-empty and positive".to_string());
    }
    let mut offsets = Vec::with_capacity(lengths.len() + 1);
    offsets.push(0usize);
    for &length in lengths {
        let next = offsets
            .last()
            .copied()
            .unwrap_or_default()
            .checked_add(length)
            .ok_or_else(|| "document lengths overflow usize".to_string())?;
        offsets.push(next);
    }
    if offsets.last().copied() != Some(rows) {
        return Err("document lengths do not match embedding rows".to_string());
    }
    Ok(offsets)
}

fn cartesian_to_spherical(values: &[f32], rows: usize, dimension: usize) -> Vec<f32> {
    let angle_dimension = dimension - 1;
    let mut angles = vec![0.0f32; rows * angle_dimension];
    let mut squared_tail = vec![0.0f64; dimension];
    for row in 0..rows {
        let input = &values[row * dimension..(row + 1) * dimension];
        let output = &mut angles[row * angle_dimension..(row + 1) * angle_dimension];
        squared_tail[dimension - 1] = f64::from(input[dimension - 1]).powi(2);
        for position in (0..dimension - 1).rev() {
            squared_tail[position] =
                squared_tail[position + 1] + f64::from(input[position]).powi(2);
        }
        for position in 0..dimension.saturating_sub(2) {
            let radius = squared_tail[position].sqrt();
            output[position] = if radius <= f64::EPSILON {
                0.0
            } else {
                (f64::from(input[position]) / radius)
                    .clamp(-1.0, 1.0)
                    .acos() as f32
            };
        }
        output[dimension - 2] = f64::atan2(
            f64::from(input[dimension - 1]),
            f64::from(input[dimension - 2]),
        ) as f32;
    }
    angles
}

fn spherical_to_cartesian(angles: &[f32], rows: usize, dimension: usize) -> Vec<f32> {
    let angle_dimension = dimension - 1;
    let mut values = vec![0.0f32; rows * dimension];
    for row in 0..rows {
        let input = &angles[row * angle_dimension..(row + 1) * angle_dimension];
        let output = &mut values[row * dimension..(row + 1) * dimension];
        let mut sine_product = 1.0f64;
        for position in 0..dimension.saturating_sub(2) {
            let angle = f64::from(input[position]);
            output[position] = (sine_product * angle.cos()) as f32;
            sine_product *= angle.sin();
        }
        let last = f64::from(input[dimension - 2]);
        output[dimension - 2] = (sine_product * last.cos()) as f32;
        output[dimension - 1] = (sine_product * last.sin()) as f32;
    }
    values
}

fn transpose_angles(angles: &[f32], rows: usize, columns: usize) -> Vec<f32> {
    let mut transposed = vec![0.0f32; angles.len()];
    for row in 0..rows {
        for column in 0..columns {
            transposed[column * rows + row] = angles[row * columns + column];
        }
    }
    transposed
}

fn byte_shuffle(values: &[f32]) -> Vec<u8> {
    let mut shuffled = vec![0u8; std::mem::size_of_val(values)];
    for (position, value) in values.iter().enumerate() {
        let bytes = value.to_le_bytes();
        for byte in 0..size_of::<f32>() {
            shuffled[byte * values.len() + position] = bytes[byte];
        }
    }
    shuffled
}

fn byte_unshuffle(values: &[u8], floats: usize) -> Result<Vec<f32>, String> {
    if values.len() != checked_product(floats, size_of::<f32>(), "jzip angle")? {
        return Err("jzip angle payload has the wrong size".to_string());
    }
    let mut output = Vec::with_capacity(floats);
    for position in 0..floats {
        output.push(f32::from_le_bytes([
            values[position],
            values[floats + position],
            values[floats * 2 + position],
            values[floats * 3 + position],
        ]));
    }
    Ok(output)
}

fn encode_jzip_frame(
    embeddings: &[f32],
    rows: usize,
    dimension: usize,
    compression_level: i32,
) -> Result<Vec<u8>, String> {
    if rows > u32::MAX as usize || dimension > u32::MAX as usize {
        return Err("jzip frame shape exceeds its 32-bit header".to_string());
    }
    let angles = cartesian_to_spherical(embeddings, rows, dimension);
    let transposed = transpose_angles(&angles, rows, dimension - 1);
    let shuffled = byte_shuffle(&transposed);
    let mut compressor = zstd::bulk::Compressor::new(compression_level)
        .map_err(|error| format!("could not create jzip zstd compressor: {error}"))?;
    compressor
        .include_checksum(true)
        .map_err(|error| format!("could not enable jzip zstd checksum: {error}"))?;
    let compressed = compressor
        .compress(&shuffled)
        .map_err(|error| format!("jzip zstd compression failed: {error}"))?;
    let mut frame = Vec::with_capacity(JZIP_HEADER_BYTES + compressed.len());
    frame.extend_from_slice(&JZIP_MAGIC);
    frame.extend_from_slice(&JZIP_VERSION.to_le_bytes());
    frame.extend_from_slice(&JZIP_FLAGS_UNIT_NORM.to_le_bytes());
    frame.extend_from_slice(&(rows as u32).to_le_bytes());
    frame.extend_from_slice(&(dimension as u32).to_le_bytes());
    frame.extend_from_slice(&compressed);
    Ok(frame)
}

fn parse_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes.try_into().expect("four-byte field"))
}

fn decode_jzip_frame(
    frame: &[u8],
    expected_rows: usize,
    expected_dimension: usize,
) -> Result<Vec<f32>, String> {
    if frame.len() < JZIP_HEADER_BYTES {
        return Err("jzip frame is shorter than its header".to_string());
    }
    if frame[..4] != JZIP_MAGIC {
        return Err("jzip frame has invalid magic".to_string());
    }
    let version = u16::from_le_bytes(frame[4..6].try_into().expect("two-byte field"));
    let flags = u16::from_le_bytes(frame[6..8].try_into().expect("two-byte field"));
    if version != JZIP_VERSION || flags != JZIP_FLAGS_UNIT_NORM {
        return Err("jzip frame version or flags are unsupported".to_string());
    }
    let rows = parse_u32(&frame[8..12]) as usize;
    let dimension = parse_u32(&frame[12..16]) as usize;
    if rows != expected_rows || dimension != expected_dimension {
        return Err("jzip frame shape differs from the document directory".to_string());
    }
    let floats = checked_product(rows, dimension - 1, "jzip angle")?;
    let uncompressed_bytes = checked_product(floats, size_of::<f32>(), "jzip angle")?;
    let shuffled = zstd::bulk::decompress(&frame[JZIP_HEADER_BYTES..], uncompressed_bytes)
        .map_err(|error| format!("jzip zstd decompression failed: {error}"))?;
    if shuffled.len() != uncompressed_bytes {
        return Err("jzip decompressed payload has the wrong size".to_string());
    }
    let transposed = byte_unshuffle(&shuffled, floats)?;
    let angles = transpose_angles(&transposed, dimension - 1, rows);
    Ok(spherical_to_cartesian(&angles, rows, dimension))
}

pub fn jzip_encode_documents(
    embeddings: &[f32],
    rows: usize,
    dimension: usize,
    document_lengths: &[usize],
    compression_level: i32,
    threads: Option<usize>,
) -> Result<(Vec<u8>, Vec<u64>), String> {
    validate(embeddings, rows, dimension, "embeddings")?;
    if dimension < 2 {
        return Err("jzip requires an embedding dimension of at least two".to_string());
    }
    if !(1..=22).contains(&compression_level) {
        return Err("jzip compression level must be between 1 and 22".to_string());
    }
    for (row, values) in embeddings.chunks_exact(dimension).enumerate() {
        let squared_norm = values
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>();
        if (squared_norm - 1.0).abs() > 2.1e-3 {
            return Err(format!("jzip embedding row {row} is not unit-normalized"));
        }
    }
    let offsets = document_offsets(document_lengths, rows)?;
    let encoded = install(threads, || {
        document_lengths
            .par_iter()
            .enumerate()
            .map(|(document, &length)| {
                let first = offsets[document] * dimension;
                let last = offsets[document + 1] * dimension;
                encode_jzip_frame(
                    &embeddings[first..last],
                    length,
                    dimension,
                    compression_level,
                )
            })
            .collect::<Vec<_>>()
    })?;
    let mut payload = Vec::new();
    let mut frame_lengths = Vec::with_capacity(encoded.len());
    for result in encoded {
        let frame = result?;
        frame_lengths.push(
            frame
                .len()
                .try_into()
                .map_err(|_| "jzip frame length exceeds u64".to_string())?,
        );
        payload.extend_from_slice(&frame);
    }
    Ok((payload, frame_lengths))
}

pub fn jzip_decode_documents(
    payload: &[u8],
    frame_lengths: &[u64],
    document_lengths: &[usize],
    dimension: usize,
    threads: Option<usize>,
) -> Result<Vec<f32>, String> {
    if dimension < 2 || frame_lengths.len() != document_lengths.len() {
        return Err("jzip frame directory is inconsistent".to_string());
    }
    let mut frame_offsets = Vec::with_capacity(frame_lengths.len() + 1);
    frame_offsets.push(0usize);
    for &length in frame_lengths {
        let length: usize = length
            .try_into()
            .map_err(|_| "jzip frame length exceeds usize".to_string())?;
        let next = frame_offsets
            .last()
            .copied()
            .unwrap_or_default()
            .checked_add(length)
            .ok_or_else(|| "jzip frame lengths overflow usize".to_string())?;
        frame_offsets.push(next);
    }
    if frame_offsets.last().copied() != Some(payload.len()) {
        return Err("jzip frame lengths do not match the payload".to_string());
    }
    let decoded = install(threads, || {
        document_lengths
            .par_iter()
            .enumerate()
            .map(|(document, &rows)| {
                decode_jzip_frame(
                    &payload[frame_offsets[document]..frame_offsets[document + 1]],
                    rows,
                    dimension,
                )
            })
            .collect::<Vec<_>>()
    })?;
    let total_rows = document_lengths.iter().try_fold(0usize, |total, &rows| {
        total
            .checked_add(rows)
            .ok_or_else(|| "jzip document lengths overflow usize".to_string())
    })?;
    let mut output = Vec::with_capacity(checked_product(total_rows, dimension, "jzip output")?);
    for result in decoded {
        output.extend(result?);
    }
    Ok(output)
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

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }
}

fn permutation_map(seed: u64, dimension: usize) -> Vec<usize> {
    let mut values: Vec<usize> = (0..dimension).collect();
    let mut random = Lcg(seed);
    for index in (1..dimension).rev() {
        let selected = ((random.next() >> 32) % (index as u64 + 1)) as usize;
        values.swap(index, selected);
    }
    values
}

fn chunk_sizes(mut dimension: usize) -> Vec<usize> {
    let mut output = Vec::new();
    while dimension != 0 {
        let size = 1usize << dimension.ilog2();
        output.push(size);
        dimension ^= size;
    }
    output
}

fn normalized_hadamard(values: &mut [f32]) {
    let mut width = 1;
    while width < values.len() {
        for start in (0..values.len()).step_by(width * 2) {
            for index in start..start + width {
                let left = values[index];
                let right = values[index + width];
                values[index] = left + right;
                values[index + width] = left - right;
            }
        }
        width *= 2;
    }
    let scale = 1.0 / (values.len() as f32).sqrt();
    for value in values {
        *value *= scale;
    }
}

fn hadamard_chunks(values: &mut [f32], sizes: &[usize]) {
    let mut offset = 0;
    for &size in sizes {
        normalized_hadamard(&mut values[offset..offset + size]);
        offset += size;
    }
}

fn rotate(values: &mut [f32], maps: &[Vec<usize>; 3], sizes: &[usize]) {
    hadamard_chunks(values, sizes);
    let mut scratch = vec![0.0f32; values.len()];
    for map in maps {
        for (output, &source) in scratch.iter_mut().zip(map) {
            *output = values[source];
        }
        values.copy_from_slice(&scratch);
        hadamard_chunks(values, sizes);
    }
}

fn rotation(dimension: usize) -> ([Vec<usize>; 3], Vec<usize>) {
    (
        PERMUTATION_SEEDS.map(|seed| permutation_map(seed, dimension)),
        chunk_sizes(dimension),
    )
}

pub fn turboquant4_encode(
    embeddings: &[f32],
    rows: usize,
    dimension: usize,
    threads: Option<usize>,
) -> Result<Vec<u8>, String> {
    validate(embeddings, rows, dimension, "embeddings")?;
    if dimension % 2 != 0 {
        return Err("TurboQuant4 requires an even dimension".to_string());
    }
    let width = dimension / 2;
    let (maps, sizes) = rotation(dimension);
    let mut output = vec![0u8; rows * width];
    install(threads, || {
        output
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(row, packed)| {
                let mut rotated = embeddings[row * dimension..(row + 1) * dimension].to_vec();
                rotate(&mut rotated, &maps, &sizes);
                let norm = rotated
                    .iter()
                    .map(|value| value * value)
                    .sum::<f32>()
                    .sqrt()
                    .max(1.0e-12);
                let scale = (dimension as f32).sqrt() / norm;
                for byte in 0..width {
                    let left = TQ_BOUNDARIES
                        .partition_point(|boundary| *boundary < rotated[byte * 2] * scale)
                        as u8;
                    let right = TQ_BOUNDARIES
                        .partition_point(|boundary| *boundary < rotated[byte * 2 + 1] * scale)
                        as u8;
                    packed[byte] = (left << 4) | right;
                }
            });
    })?;
    Ok(output)
}

pub fn turboquant4_decode_rotated(
    packed: &[u8],
    rows: usize,
    dimension: usize,
    normalize: bool,
    threads: Option<usize>,
) -> Result<Vec<f32>, String> {
    if dimension % 2 != 0 || packed.len() != rows * dimension / 2 {
        return Err("TurboQuant4 packed shape is incompatible with dimension".to_string());
    }
    let width = dimension / 2;
    let mut output = vec![0.0f32; rows * dimension];
    install(threads, || {
        output
            .par_chunks_mut(dimension)
            .enumerate()
            .for_each(|(row, decoded)| {
                let input = &packed[row * width..(row + 1) * width];
                for (byte, &value) in input.iter().enumerate() {
                    decoded[byte * 2] = TQ_CENTROIDS[(value >> 4) as usize];
                    decoded[byte * 2 + 1] = TQ_CENTROIDS[(value & 0x0f) as usize];
                }
                if normalize {
                    simd_normalize(decoded);
                }
            });
    })?;
    Ok(output)
}

pub fn turboquant_rotate(
    embeddings: &[f32],
    rows: usize,
    dimension: usize,
    threads: Option<usize>,
) -> Result<Vec<f32>, String> {
    validate(embeddings, rows, dimension, "embeddings")?;
    let (maps, sizes) = rotation(dimension);
    let mut output = embeddings.to_vec();
    install(threads, || {
        output.par_chunks_mut(dimension).for_each(|row| {
            rotate(row, &maps, &sizes);
            simd_normalize(row);
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

    #[test]
    fn turboquant_is_deterministic_and_uses_half_a_byte_per_coordinate() {
        let values = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let first = turboquant4_encode(&values, 2, 4, Some(1)).unwrap();
        let second = turboquant4_encode(&values, 2, 4, Some(1)).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 4);
        let decoded = turboquant4_decode_rotated(&first, 2, 4, true, Some(1)).unwrap();
        let query = turboquant_rotate(&values, 2, 4, Some(1)).unwrap();
        assert!(
            decoded
                .iter()
                .zip(query)
                .map(|(left, right)| left * right)
                .sum::<f32>()
                > 1.5
        );
    }

    #[test]
    fn jzip_round_trip_is_near_lossless_and_document_framed() {
        let inverse = 1.0 / 2.0f32.sqrt();
        let values = [
            1.0, 0.0, 0.0, 0.0, inverse, inverse, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        ];
        let (payload, frame_lengths) =
            jzip_encode_documents(&values, 3, 4, &[2, 1], 1, Some(2)).unwrap();
        assert_eq!(frame_lengths.len(), 2);
        assert_eq!(frame_lengths.iter().sum::<u64>() as usize, payload.len());

        let decoded = jzip_decode_documents(&payload, &frame_lengths, &[2, 1], 4, Some(2)).unwrap();
        for (original, reconstructed) in values.iter().zip(decoded) {
            assert!((original - reconstructed).abs() < 2.0e-7);
        }
    }

    #[test]
    fn jzip_rejects_a_corrupt_frame() {
        let values = [1.0, 0.0, 0.0, 0.0];
        let (mut payload, lengths) =
            jzip_encode_documents(&values, 1, 4, &[1], 1, Some(1)).unwrap();
        let last = payload.len() - 1;
        payload[last] ^= 0xff;
        let error = jzip_decode_documents(&payload, &lengths, &[1], 4, Some(1)).unwrap_err();
        assert!(error.contains("decompression"));
    }
}
