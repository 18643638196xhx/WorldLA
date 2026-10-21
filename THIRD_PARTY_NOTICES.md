# Third-party code

WorldLA contains code derived from the tested OpenPI implementation, now
reorganized within `src/worldla/`. It is neither wholly original nor an
unmodified upstream release. There are no separate `openpi`/`openpi_client`
Python packages. Upstream: https://github.com/Physical-Intelligence/openpi.
License: Apache-2.0, reproduced in `licenses/OpenPI-APACHE-2.0.txt`.

The exact model/LoRA equations are retained because the published checkpoints
must use the same implementation. In particular, do not replace the attention
output LoRA contraction with a generic headwise merge. `SOURCE_MANIFEST.json`
records current SHA256 provenance. The original source hashes and exact
module/class mapping are in `licenses/provenance/OpenPI-source-snapshot.json`
and `licenses/provenance/namespace_migration.json`. Original copyright headers
are retained; namespace changes do not erase upstream attribution.

Changes in the derived backend:

- Removed machine-specific training experiment presets and the unused
  multiview preset file. WorldLA constructs its configuration explicitly.
- Changed default runtime caches to the user cache directory and preserved
  caller-provided environment variables/credential-cache locations.
- Removed development-machine paths from comments.
- Reorganized modules, filenames, classes and imports under `worldla`; renamed
  the primary model/configuration to `WorldLAModel`/`WorldLAConfig` and the
  underlying backbone to `WorldLABackbone`. Model parameter attribute names,
  checkpoint field names and mathematical operations remain unchanged.
- Unified local cache environment variables as `WORLDLA_CACHE_DIR` and
  `WORLDLA_DATA_HOME`. Remote upstream asset URLs retain their real names.

Other libraries are normal installation dependencies. Training pins the
LeRobot revision declared in the tested OpenPI project, rather than silently
using an incompatible current LeRobot release. Fast-WAM and RoboTwin are
external, separately licensed dependencies; their weights/assets are not
redistributed in this code project.
