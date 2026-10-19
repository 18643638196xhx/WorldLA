# Third-party code

`src/openpi` and `src/openpi_client` are an embedded snapshot of the OpenPI
implementation used by the existing WorldLA checkpoints, not a claim to be
an unmodified upstream release. Upstream: https://github.com/Physical-Intelligence/openpi.
License: Apache-2.0, reproduced in `licenses/OpenPI-APACHE-2.0.txt`.

The exact model/LoRA equations are retained because the published checkpoints
must use the same implementation. In particular, do not replace the attention
output LoRA contraction with a generic headwise merge. `SOURCE_MANIFEST.json`
records SHA256 provenance and identifies modified vendored files.

Portability-only changes in the embedded backend:

- Removed machine-specific training experiment presets and the unused
  multiview preset file. WorldLA constructs its configuration explicitly.
- Changed default runtime caches to the user cache directory and preserved
  caller-provided environment variables/credential-cache locations.
- Removed development-machine paths from comments.

Other libraries are normal installation dependencies. Training pins the
LeRobot revision declared in the tested OpenPI project, rather than silently
using an incompatible current LeRobot release. Fast-WAM and RoboTwin are
external, separately licensed dependencies; their weights/assets are not
redistributed in this code project.
