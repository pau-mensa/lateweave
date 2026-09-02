//! Packed, allocation-bounded CPU MaxSim.
//!
//! The batching and SIMD reduction are derived from the Apache-2.0 licensed
//! `maxsim-cpu` implementation by Benjamin Clavie and Mixedbread. Keeping the
//! kernel here makes MaxSim the one concrete scoring primitive supplied by
//! lateweave; retrieval backends remain external implementations of the
//! generator/scorer contracts.

#[cfg(any(target_os = "macos", feature = "openblas"))]
use blas::sgemm;
use rayon::{prelude::*, ThreadPoolBuilder};
use std::cell::RefCell;

thread_local! {
    static SIMILARITY_BUFFER: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

#[cfg(target_arch = "aarch64")]
const DEFAULT_BATCH_TOKENS: usize = 8 * 1024;

#[cfg(not(target_arch = "aarch64"))]
const DEFAULT_BATCH_TOKENS: usize = 32 * 1024;

#[derive(Clone, Copy)]
struct Work {
    first_document: usize,
    last_document: usize,
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
fn simd_max(values: &[f32]) -> f32 {
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

fn batches(offsets: &[usize], maximum_tokens: usize) -> Vec<Work> {
    let document_count = offsets.len() - 1;
    let mut output = Vec::new();
    let mut first_document = 0;
    while first_document < document_count {
        let first_token = offsets[first_document];
        let mut last_document = first_document + 1;
        while last_document < document_count
            && offsets[last_document + 1] - first_token <= maximum_tokens
        {
            last_document += 1;
        }
        output.push(Work {
            first_document,
            last_document,
        });
        first_document = last_document;
    }
    output
}

/// `out[query_tokens, token_count]` row-major = `query · documentsᵀ`.
///
/// Both inputs are row-major and share `dimension` as their minor axis: `query`
/// is `[query_tokens, dimension]` and `documents` is `[token_count, dimension]`.
/// One SGEMM is the whole of the kernel's floating-point cost, which is why it
/// is the only thing here with an alternative implementation.
///
/// The implementation is chosen at compile time, never at run time: a symbol
/// resolved from the environment is how a build ends up quietly paired with a
/// slower BLAS than the one it was measured against.
#[inline]
fn similarities(
    query: &[f32],
    documents: &[f32],
    out: &mut [f32],
    query_tokens: usize,
    token_count: usize,
    dimension: usize,
) {
    #[cfg(any(target_os = "macos", feature = "openblas"))]
    {
        // BLAS is column-major and the data is row-major, so each operand is
        // read as its own transpose. Transposing the documents operand is what
        // makes `out` come back row-major, which is the layout the caller's
        // maximum reduction scans.
        unsafe {
            sgemm(
                b'T',
                b'N',
                token_count as i32,
                query_tokens as i32,
                dimension as i32,
                1.0,
                documents,
                dimension as i32,
                query,
                dimension as i32,
                0.0,
                out,
                token_count as i32,
            );
        }
    }
    #[cfg(not(any(target_os = "macos", feature = "openblas")))]
    {
        // Strides say the same thing without the transpose flags: `documents`
        // is passed with its row and column strides swapped, which is its
        // transpose. Single-threaded on purpose -- the caller already runs one
        // of these per rayon worker, and a threaded inner kernel would
        // oversubscribe against that.
        unsafe {
            matrixmultiply::sgemm(
                query_tokens,
                dimension,
                token_count,
                1.0,
                query.as_ptr(),
                dimension as isize,
                1,
                documents.as_ptr(),
                1,
                dimension as isize,
                0.0,
                out.as_mut_ptr(),
                token_count as isize,
                1,
            );
        }
    }
}

fn score_work(
    query: &[f32],
    documents: &[f32],
    offsets: &[usize],
    query_tokens: usize,
    dimension: usize,
    work: Work,
) -> (usize, Vec<f32>) {
    let first_token = offsets[work.first_document];
    let last_token = offsets[work.last_document];
    let token_count = last_token - first_token;
    let documents = &documents[first_token * dimension..last_token * dimension];

    let scores = SIMILARITY_BUFFER.with(|storage| {
        let mut storage = storage.borrow_mut();
        storage.resize(query_tokens * token_count, 0.0);
        similarities(
            query,
            documents,
            storage.as_mut_slice(),
            query_tokens,
            token_count,
            dimension,
        );

        (work.first_document..work.last_document)
            .map(|document| {
                let start = offsets[document] - first_token;
                let end = offsets[document + 1] - first_token;
                (0..query_tokens)
                    .map(|query_token| {
                        let row = query_token * token_count;
                        simd_max(&storage[row + start..row + end])
                    })
                    .sum()
            })
            .collect()
    });
    (work.first_document, scores)
}

pub fn maxsim_scores_packed(
    query: &[f32],
    documents: &[f32],
    offsets: &[usize],
    query_tokens: usize,
    dimension: usize,
    maximum_batch_tokens: Option<usize>,
    threads: Option<usize>,
) -> Result<Vec<f32>, String> {
    let document_count = offsets.len() - 1;
    if document_count == 0 {
        return Ok(Vec::new());
    }
    let work = batches(
        offsets,
        maximum_batch_tokens.unwrap_or(DEFAULT_BATCH_TOKENS),
    );
    let execute = || {
        work.into_par_iter()
            .map(|item| score_work(query, documents, offsets, query_tokens, dimension, item))
            .collect::<Vec<_>>()
    };
    let results = if let Some(threads) = threads {
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|error| format!("could not create MaxSim worker pool: {error}"))?
            .install(execute)
    } else {
        execute()
    };
    let mut scores = vec![0.0; document_count];
    for (first_document, values) in results {
        scores[first_document..first_document + values.len()].copy_from_slice(&values);
    }
    Ok(scores)
}
