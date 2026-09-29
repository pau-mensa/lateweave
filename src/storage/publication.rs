use std::fs::{self, File};
use std::path::{Path, PathBuf};

use crate::error::Result;

/// Owns private files and the final paths installed by one publication attempt.
pub(super) struct ImmutableFiles {
    staging: tempfile::TempDir,
    installed: Vec<PathBuf>,
}

impl ImmutableFiles {
    pub(super) fn new(directory: &Path) -> Result<Self> {
        Ok(Self {
            staging: tempfile::Builder::new()
                .prefix(".staging-")
                .tempdir_in(directory)?,
            installed: Vec::new(),
        })
    }

    pub(super) fn path(&self) -> &Path {
        self.staging.path()
    }

    /// Installs finished files without replacing any existing destination.
    pub(super) fn install(&mut self, directory: &Path) -> Result<()> {
        for entry in fs::read_dir(self.path())? {
            let entry = entry?;
            let destination = directory.join(entry.file_name());
            fs::hard_link(entry.path(), &destination)?;
            self.installed.push(destination);
        }
        sync_directory(directory)
    }

    /// Transfers ownership to the writer or a published manifest.
    pub(super) fn retain(&mut self) {
        self.installed.clear();
    }
}

impl Drop for ImmutableFiles {
    fn drop(&mut self) {
        for path in &self.installed {
            let _ = fs::remove_file(path);
        }
    }
}

pub(super) fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}
