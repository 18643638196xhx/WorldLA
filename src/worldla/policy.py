"""Load a self-contained WorldLA export without any original checkpoint paths."""
from __future__ import annotations

import json
from pathlib import Path
import sys

def validate_checkpoint_config(cfg):
    """Reject architecture/weight mismatches before allocating the model."""
    schema = cfg.get("schema")
    if schema not in {"worldla_full_folded_v1", "worldla_full_original_v1"}:
        raise ValueError("Not a supported full WorldLA checkpoint")
    folded = schema == "worldla_full_folded_v1"
    if cfg.get("lora_merged") is not folded:
        raise ValueError("Checkpoint schema and lora_merged flag disagree")
    variants = [cfg["base_model_config"][name] for name in
                ("paligemma_variant", "action_expert_variant")]
    if any(name.endswith("_lora") == folded for name in variants):
        raise ValueError("Architecture variants do not match the checkpoint format")


def load_model(directory, *, dtype=None):
    import flax.nnx as nnx
    from flax import traverse_util
    import jax
    from openpi.models import model as model_api
    from openpi.models.pi0_config import Pi0Config
    from worldla.model import StageRelationRouterPi

    root = Path(directory).expanduser().resolve()
    cfg = json.loads((root / "config.json").read_text())
    validate_checkpoint_config(cfg)
    model_cfg = Pi0Config(**cfg["base_model_config"])
    abstract = nnx.eval_shape(lambda: StageRelationRouterPi(
        model_cfg.create(jax.random.key(0)), **cfg["router_config"], rngs=nnx.Rngs(0)))
    graph, state = nnx.split(abstract)
    restored = model_api.restore_params(root / "params", dtype=dtype)
    expected = traverse_util.flatten_dict(state.to_pure_dict())
    actual = traverse_util.flatten_dict(restored)
    from worldla.checkpoints import validate_tree
    validate_tree(expected, actual)
    state.replace_by_pure_dict(restored)
    model = nnx.merge(graph, state)
    model.eval()
    return model


class LocalTokenizer:
    """Pi0.5 PaliGemma prompt/state format using the bundled tokenizer file."""
    def __init__(self, path, max_len):
        import sentencepiece
        self._tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(path))
        self._max_len = int(max_len)

    def tokenize(self, prompt, state=None):
        import numpy as np
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            bins = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
            text = f"Task: {cleaned}, State: {' '.join(map(str, bins))};\nAction: "
            tokens = self._tokenizer.encode(text, add_bos=True)
        else:
            tokens = self._tokenizer.encode(cleaned, add_bos=True) + self._tokenizer.encode("\n")
        count = min(len(tokens), self._max_len)
        return (np.asarray((tokens + [0] * self._max_len)[:self._max_len]),
                np.asarray([True] * count + [False] * (self._max_len - count)))


def make_transforms(directory):
    from openpi import transforms as t
    from openpi.policies.aloha_policy import AlohaInputs, AlohaOutputs
    from openpi.shared import normalize
    root = Path(directory).expanduser().resolve()
    cfg = json.loads((root / "config.json").read_text())
    stats = normalize.load(root / "assets" / cfg["asset_id"])
    mask = t.make_bool_mask(6, -1, 6, -1)
    tokenizer = LocalTokenizer(root / "assets/paligemma_tokenizer.model", cfg["base_model_config"]["max_token_len"])
    inputs = t.compose([
        t.InjectDefaultPrompt(None), AlohaInputs(adapt_to_pi=False), t.DeltaActions(mask),
        t.Normalize(stats, use_quantiles=True), t.InjectDefaultPrompt(None),
        t.ResizeImages(224, 224), t.TokenizePrompt(tokenizer, discrete_state_input=True),
        t.PadStatesAndActions(cfg["base_model_config"]["action_dim"]),
    ])
    outputs = t.compose([t.Unnormalize(stats, use_quantiles=True), t.AbsoluteActions(mask), AlohaOutputs(adapt_to_pi=False)])
    return inputs, outputs


class WorldLAPolicy:
    """Single-observation policy plus optional old-tail/new-head replanning.

    infer() returns a new 50x14 action chunk without cached-plan fusion.
    plan() returns K=10 commands with the released fusion setting. Execute ALL
    K commands before the next plan() call; reset() between episodes.
    """
    def __init__(self, directory, *, seed=20260901):
        import jax
        from openpi.shared import nnx_utils
        self.root = Path(directory).expanduser().resolve()
        self.config = json.loads((self.root / "config.json").read_text())
        self.model = load_model(self.root)
        self.inputs, self.outputs = make_transforms(self.root)
        self.sample = nnx_utils.module_jit(self.model.sample_actions)
        self.initial_seed = int(seed)
        self.reset()

    def reset(self, *, seed=None):
        import jax
        self.rng = jax.random.key(self.initial_seed if seed is None else int(seed))
        self.previous_plan = None

    def infer(self, observation):
        import jax
        import jax.numpy as jnp
        import numpy as np
        from openpi.models.model import Observation
        values = self.inputs(dict(observation))
        obs = Observation.from_dict(jax.tree.map(lambda value: jnp.asarray(value)[None], values))
        self.rng, rng = jax.random.split(self.rng)
        actions = self.sample(rng, obs, num_steps=self.config["deployment"]["denoising_steps"])
        return self.outputs({"state": np.asarray(values["state"]), "actions": np.asarray(actions[0])})

    def plan(self, observation):
        from worldla.replanning import blend_plan
        plan = self.infer(observation)["actions"]
        cfg = self.config["deployment"]
        executed = blend_plan(self.previous_plan, plan, k=cfg["execute_k"], old_weight=cfg["old_plan_weight"])
        self.previous_plan = executed
        return executed[:cfg["execute_k"]]
