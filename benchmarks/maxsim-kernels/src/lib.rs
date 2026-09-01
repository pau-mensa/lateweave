//! Historical MaxSim kernels retained solely for comparative benchmarking.
//!
//! SPDX-License-Identifier: Apache-2.0
//! Derived from mixedbread-ai/maxsim-cpu by Benjamin Clavié and Mixedbread.
//! See ../NOTICE.md. This crate is not part of lateweave's production API.

#[cfg(target_os = "macos")]
extern crate blas_src;

use blas::sgemm;
use numpy::{
    PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cell::RefCell;

thread_local! {
    static PACKED_SIMILARITIES: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static FUSED_SIMILARITIES: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static FUSED_BATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

fn scalar_max(values: &[f32]) -> f32 {
    values.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_max(values: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    if values.len() < 8 {
        return scalar_max(values);
    }
    let mut maximum0 = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut maximum1 = maximum0;
    let mut maximum2 = maximum0;
    let mut maximum3 = maximum0;
    let mut position = 0;
    while position + 32 <= values.len() {
        if position + 64 < values.len() {
            _mm_prefetch(values.as_ptr().add(position + 64) as *const i8, _MM_HINT_T0);
        }
        maximum0 = _mm256_max_ps(maximum0, _mm256_loadu_ps(values.as_ptr().add(position)));
        maximum1 = _mm256_max_ps(maximum1, _mm256_loadu_ps(values.as_ptr().add(position + 8)));
        maximum2 = _mm256_max_ps(
            maximum2,
            _mm256_loadu_ps(values.as_ptr().add(position + 16)),
        );
        maximum3 = _mm256_max_ps(
            maximum3,
            _mm256_loadu_ps(values.as_ptr().add(position + 24)),
        );
        position += 32;
    }
    while position + 8 <= values.len() {
        maximum0 = _mm256_max_ps(maximum0, _mm256_loadu_ps(values.as_ptr().add(position)));
        position += 8;
    }
    maximum0 = _mm256_max_ps(maximum0, maximum1);
    maximum2 = _mm256_max_ps(maximum2, maximum3);
    maximum0 = _mm256_max_ps(maximum0, maximum2);
    let high = _mm256_extractf128_ps(maximum0, 1);
    let low = _mm256_castps256_ps128(maximum0);
    let maximum = _mm_max_ps(high, low);
    let maximum = _mm_max_ps(maximum, _mm_movehl_ps(maximum, maximum));
    let maximum = _mm_max_ss(maximum, _mm_shuffle_ps(maximum, maximum, 0b01));
    let mut result = _mm_cvtss_f32(maximum);
    for &value in &values[position..] {
        result = result.max(value);
    }
    result
}

#[cfg(target_arch = "aarch64")]
fn neon_max(values: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    if values.len() < 4 {
        return scalar_max(values);
    }
    unsafe {
        let mut maximum0 = vdupq_n_f32(f32::NEG_INFINITY);
        let mut maximum1 = maximum0;
        let mut maximum2 = maximum0;
        let mut maximum3 = maximum0;
        let mut position = 0;
        while position + 16 <= values.len() {
            maximum0 = vmaxq_f32(maximum0, vld1q_f32(values.as_ptr().add(position)));
            maximum1 = vmaxq_f32(maximum1, vld1q_f32(values.as_ptr().add(position + 4)));
            maximum2 = vmaxq_f32(maximum2, vld1q_f32(values.as_ptr().add(position + 8)));
            maximum3 = vmaxq_f32(maximum3, vld1q_f32(values.as_ptr().add(position + 12)));
            position += 16;
        }
        while position + 4 <= values.len() {
            maximum0 = vmaxq_f32(maximum0, vld1q_f32(values.as_ptr().add(position)));
            position += 4;
        }
        maximum0 = vmaxq_f32(maximum0, maximum1);
        maximum2 = vmaxq_f32(maximum2, maximum3);
        maximum0 = vmaxq_f32(maximum0, maximum2);
        let mut result = vmaxvq_f32(maximum0);
        for &value in &values[position..] {
            result = result.max(value);
        }
        result
    }
}

#[inline]
fn optimized_max(values: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return unsafe { avx2_max(values) };
        }
        scalar_max(values)
    }
    #[cfg(target_arch = "aarch64")]
    {
        neon_max(values)
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        scalar_max(values)
    }
}

unsafe fn multiply(
    query: &[f32],
    documents: &[f32],
    output: &mut [f32],
    query_tokens: usize,
    document_tokens: usize,
    dimension: usize,
) {
    sgemm(
        b'T',
        b'N',
        document_tokens as i32,
        query_tokens as i32,
        dimension as i32,
        1.0,
        documents,
        dimension as i32,
        query,
        dimension as i32,
        0.0,
        output,
        document_tokens as i32,
    );
}

#[cfg(target_arch = "aarch64")]
const PACKED_BATCH_TOKENS: usize = 8 * 1024;

#[cfg(not(target_arch = "aarch64"))]
const PACKED_BATCH_TOKENS: usize = 32 * 1024;

fn packed_kernel(
    query: &[f32],
    documents: &[f32],
    offsets: &[usize],
    query_tokens: usize,
    dimension: usize,
) -> Vec<f32> {
    let document_count = offsets.len() - 1;
    if document_count == 0 {
        return Vec::new();
    }
    let mut batches = Vec::new();
    let mut first_document = 0;
    while first_document < document_count {
        let first_token = offsets[first_document];
        let mut last_document = first_document + 1;
        while last_document < document_count
            && offsets[last_document + 1] - first_token <= PACKED_BATCH_TOKENS
        {
            last_document += 1;
        }
        batches.push((first_document, last_document));
        first_document = last_document;
    }

    let batches: Vec<(usize, Vec<f32>)> = batches
        .into_par_iter()
        .map(|(first_document, last_document)| {
            let first_token = offsets[first_document];
            let last_token = offsets[last_document];
            let token_count = last_token - first_token;
            let documents = &documents[first_token * dimension..last_token * dimension];
            let scores = PACKED_SIMILARITIES.with(|buffer| {
                let mut buffer = buffer.borrow_mut();
                buffer.resize(query_tokens * token_count, 0.0);
                unsafe {
                    multiply(
                        query,
                        documents,
                        buffer.as_mut_slice(),
                        query_tokens,
                        token_count,
                        dimension,
                    );
                }
                (first_document..last_document)
                    .map(|document| {
                        let start = offsets[document] - first_token;
                        let end = offsets[document + 1] - first_token;
                        (0..query_tokens)
                            .map(|query_token| {
                                let row = query_token * token_count;
                                optimized_max(&buffer[row + start..row + end])
                            })
                            .sum()
                    })
                    .collect()
            });
            (first_document, scores)
        })
        .collect();

    let mut scores = vec![0.0; document_count];
    for (first_document, values) in batches {
        scores[first_document..first_document + values.len()].copy_from_slice(&values);
    }
    scores
}

fn single_document(
    query: &[f32],
    document: &[f32],
    query_tokens: usize,
    document_tokens: usize,
    dimension: usize,
) -> f32 {
    FUSED_SIMILARITIES.with(|buffer| {
        let mut buffer = buffer.borrow_mut();
        buffer.resize(query_tokens * document_tokens, 0.0);
        unsafe {
            multiply(
                query,
                document,
                buffer.as_mut_slice(),
                query_tokens,
                document_tokens,
                dimension,
            );
        }
        (0..query_tokens)
            .map(|token| {
                optimized_max(&buffer[token * document_tokens..(token + 1) * document_tokens])
            })
            .sum()
    })
}

fn fused_fixed_kernel(
    query: &[f32],
    documents: &[f32],
    query_tokens: usize,
    document_count: usize,
    document_tokens: usize,
    dimension: usize,
) -> Vec<f32> {
    #[cfg(target_arch = "aarch64")]
    {
        (0..document_count)
            .into_par_iter()
            .map(|document| {
                let start = document * document_tokens * dimension;
                let values = &documents[start..start + document_tokens * dimension];
                let mut maxima = vec![f32::NEG_INFINITY; query_tokens];
                for block_start in (0..document_tokens).step_by(64) {
                    let block_end = (block_start + 64).min(document_tokens);
                    let block_tokens = block_end - block_start;
                    let block = &values[block_start * dimension..block_end * dimension];
                    let mut similarities = vec![0.0; query_tokens * block_tokens];
                    unsafe {
                        multiply(
                            query,
                            block,
                            &mut similarities,
                            query_tokens,
                            block_tokens,
                            dimension,
                        );
                    }
                    for token in 0..query_tokens {
                        let row = token * block_tokens;
                        maxima[token] = maxima[token]
                            .max(optimized_max(&similarities[row..row + block_tokens]));
                    }
                }
                maxima.iter().sum()
            })
            .collect()
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let tile_documents = match document_tokens {
            512 => 128,
            1024 => 64,
            2048 => 32,
            4096 => 16,
            _ => 32,
        };
        let mut scores = vec![0.0; document_count];
        for tile_start in (0..document_count).step_by(tile_documents) {
            let tile_end = (tile_start + tile_documents).min(document_count);
            let tile_count = tile_end - tile_start;
            let tile_tokens = tile_count * document_tokens;
            let values = &documents
                [tile_start * document_tokens * dimension..tile_end * document_tokens * dimension];
            let mut similarities = vec![0.0; query_tokens * tile_tokens];
            unsafe {
                multiply(
                    query,
                    values,
                    &mut similarities,
                    query_tokens,
                    tile_tokens,
                    dimension,
                );
            }
            let tile_scores: Vec<f32> = (0..tile_count)
                .into_par_iter()
                .map(|tile_document| {
                    (0..query_tokens)
                        .map(|token| {
                            let start = token * tile_tokens + tile_document * document_tokens;
                            optimized_max(&similarities[start..start + document_tokens])
                        })
                        .sum()
                })
                .collect();
            scores[tile_start..tile_end].copy_from_slice(&tile_scores);
        }
        scores
    }
}

fn score_padded_batch(
    query: &[f32],
    documents: &[f32],
    offsets: &[usize],
    indices: &[usize],
    query_tokens: usize,
    dimension: usize,
) -> Vec<f32> {
    let maximum_length = indices
        .iter()
        .map(|&index| offsets[index + 1] - offsets[index])
        .max()
        .unwrap();
    FUSED_BATCH.with(|buffer| {
        let mut buffer = buffer.borrow_mut();
        buffer.resize(indices.len() * maximum_length * dimension, 0.0);
        buffer.fill(0.0);
        for (batch_index, &document) in indices.iter().enumerate() {
            let first = offsets[document];
            let last = offsets[document + 1];
            let length = last - first;
            let destination = batch_index * maximum_length * dimension;
            buffer[destination..destination + length * dimension]
                .copy_from_slice(&documents[first * dimension..last * dimension]);
        }
        fused_fixed_kernel(
            query,
            buffer.as_slice(),
            query_tokens,
            indices.len(),
            maximum_length,
            dimension,
        )
    })
}

fn fused_variable_kernel(
    query: &[f32],
    documents: &[f32],
    offsets: &[usize],
    query_tokens: usize,
    dimension: usize,
) -> Vec<f32> {
    let document_count = offsets.len() - 1;
    if document_count == 0 {
        return Vec::new();
    }
    let lengths: Vec<usize> = (0..document_count)
        .map(|index| offsets[index + 1] - offsets[index])
        .collect();
    let minimum = *lengths.iter().min().unwrap();
    let maximum = *lengths.iter().max().unwrap();
    if document_count >= 50 && maximum as f32 / minimum as f32 <= 1.2 {
        let indices: Vec<usize> = (0..document_count).collect();
        return score_padded_batch(query, documents, offsets, &indices, query_tokens, dimension);
    }

    let mut sorted: Vec<usize> = (0..document_count).collect();
    sorted.sort_by_key(|&index| lengths[index]);
    let mut scores = vec![0.0; document_count];
    let mut position = 0;
    while position < document_count {
        let maximum_acceptable = (lengths[sorted[position]] as f32 * 1.2) as usize;
        let mut end = position + 1;
        while end < document_count
            && end < position + 128
            && lengths[sorted[end]] <= maximum_acceptable
        {
            end += 1;
        }
        let indices = &sorted[position..end];
        if indices.len() == 1 {
            let document = indices[0];
            let first = offsets[document];
            let last = offsets[document + 1];
            scores[document] = single_document(
                query,
                &documents[first * dimension..last * dimension],
                query_tokens,
                last - first,
                dimension,
            );
        } else {
            let batch =
                score_padded_batch(query, documents, offsets, indices, query_tokens, dimension);
            for (batch_index, &document) in indices.iter().enumerate() {
                scores[document] = batch[batch_index];
            }
        }
        position = end;
    }
    scores
}

fn offsets(lengths: &[i64], document_rows: usize) -> Result<Vec<usize>, String> {
    let mut offsets = Vec::with_capacity(lengths.len() + 1);
    offsets.push(0);
    let mut total = 0usize;
    for (index, &length) in lengths.iter().enumerate() {
        if length <= 0 {
            return Err(format!(
                "document length at position {index} must be positive"
            ));
        }
        total = total
            .checked_add(length as usize)
            .ok_or_else(|| "document lengths overflow usize".to_string())?;
        offsets.push(total);
    }
    if total != document_rows {
        return Err(format!(
            "document lengths sum to {total}, but documents contain {document_rows} rows"
        ));
    }
    Ok(offsets)
}

fn validate_dimensions(query: &[usize], documents: &[usize]) -> PyResult<()> {
    if query[0] == 0 || query[1] == 0 {
        return Err(PyValueError::new_err("query axes must be non-zero"));
    }
    if query[1] != documents[documents.len() - 1] {
        return Err(PyValueError::new_err(
            "query and document dimensions differ",
        ));
    }
    Ok(())
}

#[pyfunction]
fn packed_scores<'py>(
    py: Python<'py>,
    query: PyReadonlyArray2<'py, f32>,
    documents: PyReadonlyArray2<'py, f32>,
    document_lengths: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    validate_dimensions(query.shape(), documents.shape())?;
    let offsets = offsets(document_lengths.as_slice()?, documents.shape()[0])
        .map_err(PyValueError::new_err)?;
    let query_tokens = query.shape()[0];
    let dimension = query.shape()[1];
    let query = query.as_slice()?;
    let documents = documents.as_slice()?;
    let values =
        py.allow_threads(|| packed_kernel(query, documents, &offsets, query_tokens, dimension));
    Ok(PyArray1::from_vec(py, values))
}

#[pyfunction]
fn fused_scores<'py>(
    py: Python<'py>,
    query: PyReadonlyArray2<'py, f32>,
    documents: PyReadonlyArray3<'py, f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    validate_dimensions(query.shape(), documents.shape())?;
    if documents.shape()[0] == 0 || documents.shape()[1] == 0 {
        return Err(PyValueError::new_err("document axes must be non-zero"));
    }
    let query_tokens = query.shape()[0];
    let dimension = query.shape()[1];
    let document_count = documents.shape()[0];
    let document_tokens = documents.shape()[1];
    let query = query.as_slice()?;
    let documents = documents.as_slice()?;
    let values = py.allow_threads(|| {
        fused_fixed_kernel(
            query,
            documents,
            query_tokens,
            document_count,
            document_tokens,
            dimension,
        )
    });
    Ok(PyArray1::from_vec(py, values))
}

#[pyfunction]
fn fused_scores_variable<'py>(
    py: Python<'py>,
    query: PyReadonlyArray2<'py, f32>,
    documents: PyReadonlyArray2<'py, f32>,
    document_lengths: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    validate_dimensions(query.shape(), documents.shape())?;
    let offsets = offsets(document_lengths.as_slice()?, documents.shape()[0])
        .map_err(PyValueError::new_err)?;
    let query_tokens = query.shape()[0];
    let dimension = query.shape()[1];
    let query = query.as_slice()?;
    let documents = documents.as_slice()?;
    let values = py.allow_threads(|| {
        fused_variable_kernel(query, documents, &offsets, query_tokens, dimension)
    });
    Ok(PyArray1::from_vec(py, values))
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(packed_scores, module)?)?;
    module.add_function(wrap_pyfunction!(fused_scores, module)?)?;
    module.add_function(wrap_pyfunction!(fused_scores_variable, module)?)?;
    Ok(())
}
