# WorldLA 0.2 namespace migration

All Python source is under `src/worldla/`; no OpenPI checkout or Python package
is imported. This is a source/API organization change, not a new backbone
architecture or a claim that upstream code was authored from scratch.

| Component | New module / class |
| --- | --- |
| Complete routed model | `worldla.models.worldla.WorldLAModel` |
| Backbone | `worldla.models.backbone.WorldLABackbone` |
| Model configuration | `worldla.models.configuration.WorldLAConfig` |
| Stage Router | `worldla.models.worldla.ContextAdaptiveStageRouter` |
| Routed visual-message injection | `worldla.models.worldla.StageMessageInjector` |
| Training entry point | `worldla.training.runner` |
| Training presets/settings | `worldla.training.presets`, `worldla.training.settings` |
| Input/output transforms | `worldla.data.transforms` |
| Shared numerical/runtime tools | `worldla.utils` |
| Client and action-chunk handling | `worldla.client` |

Public imports:

```python
from worldla import WorldLAConfig, WorldLAModel, WorldLAPolicy
from worldla.models import WorldLABackbone
```

The `worldla train`, `infer`, `export`, data-preparation and `eval-robotwin` CLI
commands remain available. Python callers must switch to the new imports;
legacy namespace aliases are deliberately not shipped. Optional compatibility
architectures remain under `worldla.models.compat`, and the retained optional
Torch backend is under `worldla.torch_backend`; the tested deployment is JAX.

Existing `original/` and `merged/` checkpoint folders are not rewritten.
Serialized field names, model variants, and parameter paths such as
`base/PaliGemma/...`, `stage_relation_router/...` and
`stage_message_adapter/...` are preserved. Gemma/SigLIP/PaliGemma still identify
the actual upstream architectures. Renaming model *classes* does not require
renaming those saved parameter *attributes*. A fixture records all 89 original
parameter paths/shapes and is checked against the new model.

Local cache settings now use `WORLDLA_CACHE_DIR` and `WORLDLA_DATA_HOME`.
Use the explicit tokenizer/data/checkpoint paths in the training configuration;
upstream download URLs are left intact rather than inventing renamed URLs.

Original copyright notices and Apache-2.0 license text are retained. The exact
old-to-new source/class map is recorded in `licenses/provenance/` separately
from the current source hash manifest. Consult `docs/VALIDATION.md` for which
tests were run before and after this migration; old reports are not presented
as new experiments.
