//! The subset of the NumPy `.npy` format the vector stores use: little-endian,
//! C-order arrays of `f32`, `i8`, and `u64`, memory-mapped for reading.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::mem::{align_of, size_of};
use std::path::{Path, PathBuf};
use std::slice;

use memmap2::Mmap;

use crate::error::{Error, Result};

const MAGIC: &[u8] = b"\x93NUMPY";
const ALIGNMENT: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Dtype {
    F32,
    I8,
    U64,
}

impl Dtype {
    fn descr(self) -> &'static str {
        match self {
            Self::F32 => "<f4",
            Self::I8 => "|i1",
            Self::U64 => "<u8",
        }
    }

    pub(crate) fn size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::I8 => 1,
            Self::U64 => 8,
        }
    }

    fn parse(descr: &str) -> Option<Self> {
        [Self::F32, Self::I8, Self::U64]
            .into_iter()
            .find(|dtype| dtype.descr() == descr || (*dtype == Self::I8 && descr == "<i1"))
    }
}

/// A plain numeric type every bit pattern of which is a valid value.
pub(crate) trait Element: Copy + 'static {
    const DTYPE: Dtype;
}

impl Element for f32 {
    const DTYPE: Dtype = Dtype::F32;
}

impl Element for i8 {
    const DTYPE: Dtype = Dtype::I8;
}

impl Element for u64 {
    const DTYPE: Dtype = Dtype::U64;
}

pub(crate) fn as_bytes<T: Element>(values: &[T]) -> &[u8] {
    // SAFETY: `Element` types are plain numbers without padding.
    unsafe { slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

fn require_little_endian() -> Result<()> {
    if cfg!(target_endian = "big") {
        return Err(Error::storage(
            "vector stores are little-endian and cannot be used on this host",
        ));
    }
    Ok(())
}

pub(crate) struct NpyArray {
    map: Mmap,
    data_offset: usize,
    dtype: Dtype,
    shape: Vec<usize>,
}

impl NpyArray {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        require_little_endian()?;
        let file = File::open(path)?;
        // SAFETY: the store format writes every file once, before a commit
        // names it, and never modifies it afterwards, so a mapping never
        // observes a concurrent write from a writer that follows it.
        let map = unsafe { Mmap::map(&file)? };
        let invalid = |reason: &str| Error::storage(format!("{}: {reason}", path.display()));
        if map.len() < 10 || &map[..6] != MAGIC {
            return Err(invalid("not a .npy file"));
        }
        let (header_start, header_length) = match map[6] {
            1 => (10, u16::from_le_bytes([map[8], map[9]]) as usize),
            2 | 3 if map.len() >= 12 => (
                12,
                u32::from_le_bytes([map[8], map[9], map[10], map[11]]) as usize,
            ),
            _ => return Err(invalid("unsupported .npy version")),
        };
        let data_offset = header_start + header_length;
        let header = map
            .get(header_start..data_offset)
            .and_then(|bytes| std::str::from_utf8(bytes).ok())
            .ok_or_else(|| invalid("truncated or non-text header"))?;
        let descr = header_value(header, "descr")
            .and_then(|value| value.strip_prefix('\''))
            .and_then(|value| value.split('\'').next())
            .ok_or_else(|| invalid("header has no descr"))?;
        let dtype = Dtype::parse(descr).ok_or_else(|| invalid("unsupported dtype"))?;
        if !header_value(header, "fortran_order").is_some_and(|value| value.starts_with("False")) {
            return Err(invalid("only C-order arrays are supported"));
        }
        let shape = header_value(header, "shape")
            .and_then(|value| value.strip_prefix('('))
            .and_then(|value| value.split(')').next())
            .ok_or_else(|| invalid("header has no shape"))?
            .split(',')
            .map(str::trim)
            .filter(|axis| !axis.is_empty())
            .map(|axis| axis.parse::<usize>())
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|_| invalid("malformed shape"))?;
        let expected = shape
            .iter()
            .try_fold(dtype.size(), |total, &axis| total.checked_mul(axis))
            .ok_or_else(|| invalid("shape overflows usize"))?;
        if map.len() - data_offset.min(map.len()) != expected {
            return Err(invalid("data length does not match the shape"));
        }
        Ok(Self {
            map,
            data_offset,
            dtype,
            shape,
        })
    }

    pub(crate) fn dtype(&self) -> Dtype {
        self.dtype
    }

    pub(crate) fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        &self.map[self.data_offset..]
    }

    pub(crate) fn values<T: Element>(&self) -> Result<&[T]> {
        let bytes = self.bytes();
        if self.dtype != T::DTYPE || bytes.as_ptr() as usize % align_of::<T>() != 0 {
            return Err(Error::storage("array dtype or alignment does not match"));
        }
        // SAFETY: dtype, length, and alignment are checked, and every bit
        // pattern is a valid `Element`.
        Ok(unsafe { slice::from_raw_parts(bytes.as_ptr().cast(), bytes.len() / size_of::<T>()) })
    }
}

fn header_value<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let quoted = format!("'{key}':");
    let start = header.find(&quoted)? + quoted.len();
    Some(header[start..].trim_start())
}

/// Streams one array to disk; `finish` refuses an array of the wrong length.
pub(crate) struct NpyWriter {
    path: PathBuf,
    file: BufWriter<File>,
    remaining: usize,
}

impl NpyWriter {
    pub(crate) fn create(path: PathBuf, dtype: Dtype, shape: &[usize]) -> Result<Self> {
        require_little_endian()?;
        let axes = match shape {
            [axis] => format!("{axis},"),
            _ => shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", "),
        };
        let mut header = format!(
            "{{'descr': '{}', 'fortran_order': False, 'shape': ({axes}), }}",
            dtype.descr()
        );
        let unpadded = MAGIC.len() + 4 + header.len() + 1;
        header.extend(std::iter::repeat(' ').take(unpadded.next_multiple_of(ALIGNMENT) - unpadded));
        header.push('\n');
        let header_length = u16::try_from(header.len())
            .map_err(|_| Error::storage("array shape is too long for a .npy header"))?;

        let mut file = BufWriter::with_capacity(
            1 << 20,
            File::options().write(true).create_new(true).open(&path)?,
        );
        file.write_all(MAGIC)?;
        file.write_all(&[1, 0])?;
        file.write_all(&header_length.to_le_bytes())?;
        file.write_all(header.as_bytes())?;
        Ok(Self {
            path,
            file,
            remaining: shape.iter().product::<usize>() * dtype.size(),
        })
    }

    pub(crate) fn write_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.len() > self.remaining {
            return Err(Error::storage(format!(
                "{}: more data than its shape holds",
                self.path.display()
            )));
        }
        self.remaining -= bytes.len();
        self.file.write_all(bytes)?;
        Ok(())
    }

    pub(crate) fn write<T: Element>(&mut self, values: &[T]) -> Result<()> {
        self.write_bytes(as_bytes(values))
    }

    pub(crate) fn finish(self) -> Result<()> {
        if self.remaining != 0 {
            return Err(Error::storage(format!(
                "{}: less data than its shape holds",
                self.path.display()
            )));
        }
        self.file
            .into_inner()
            .map_err(|error| error.into_error())?
            .sync_all()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arrays_round_trip_with_aligned_data() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("values.npy");
        let mut writer = NpyWriter::create(path.clone(), Dtype::F32, &[2, 3]).unwrap();
        writer.write(&[1.0f32, 2.0, 3.0]).unwrap();
        writer.write(&[4.0f32, 5.0, 6.0]).unwrap();
        writer.finish().unwrap();

        let array = NpyArray::open(&path).unwrap();
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(
            array.values::<f32>().unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
        assert_eq!(array.data_offset % ALIGNMENT, 0);
        assert!(array.values::<u64>().is_err());
    }

    #[test]
    fn a_short_array_is_refused() {
        let directory = tempfile::tempdir().unwrap();
        let mut writer =
            NpyWriter::create(directory.path().join("values.npy"), Dtype::U64, &[2]).unwrap();
        writer.write(&[1u64]).unwrap();
        assert!(writer.finish().is_err());
    }
}
