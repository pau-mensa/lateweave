# Attribution

The benchmark-only kernels in `src/lib.rs` are derived from the Apache-2.0
licensed [`mixedbread-ai/maxsim-cpu`](https://github.com/mixedbread-ai/maxsim-cpu)
implementation by Benjamin Clavié and the Mixedbread team, including the packed
kernel introduced in commit `d0c9a1e` of the `pau-mensa/maxsim-cpu` fork.

The tiled implementation was preserved by the
[`Novadata-Technologies/maxsim`](https://github.com/Novadata-Technologies/maxsim)
fork. The code is included here solely to make the benchmark reproducible; it
is not part of lateweave's production Python or Rust API.
