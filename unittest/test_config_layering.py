"""Config layering tests (offline, stdlib only).

Tracked ``nebulonmd.cfg`` ships defaults; ``nebulonmd.local.cfg``
(gitignored) overlays per-machine values; explicit environment wins:

    tracked defaults <- local overlay <- env/.env

Covers: default reads, local-wins merge, writes landing in the local file
(never the tracked one), per-key ``source`` reporting, and ``_load_cfg``
export precedence in both gap-fill and override modes.
"""

import os
import tempfile
import unittest
from configparser import ConfigParser
from pathlib import Path

from nmd_host.core import config as cfgmod
from nmd_host.core.config import NMDConfig

TRACKED = """\
[paths]
nmd_home = {home}
[backend]
ndb_api_host = localhost
[server]
nmd_api_port = 9696
[llm]
nmd_llm_provider = nvidia
nmd_llm_model = default-model
"""


class ConfigLayeringTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.tracked = self.home / "nebulonmd.cfg"
        self.tracked.write_text(TRACKED.format(home=self.home), encoding="utf-8")
        self.local = self.home / cfgmod.LOCAL_CFG_FILENAME
        self._saved_env = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._saved_env)
        self.tmp.cleanup()

    def _load(self):
        return NMDConfig(self.tracked)

    def test_defaults_without_local_file(self):
        cfg = self._load()
        self.assertEqual(cfg.NMD_LLM_PROVIDER, "nvidia")
        self.assertEqual(cfg.key_source("llm", "nmd_llm_provider"), "default")
        self.assertFalse(self.local.exists())

    def test_local_overlay_wins(self):
        local = ConfigParser()
        local.add_section("llm")
        local.set("llm", "nmd_llm_provider", "other")
        with self.local.open("w", encoding="utf-8") as f:
            local.write(f)
        cfg = self._load()
        self.assertEqual(cfg.NMD_LLM_PROVIDER, "other")
        # Untouched keys still come from tracked defaults.
        self.assertEqual(cfg.NMD_LLM_MODEL, "default-model")
        self.assertEqual(cfg.key_source("llm", "nmd_llm_provider"), "local")
        self.assertEqual(cfg.key_source("llm", "nmd_llm_model"), "default")

    def test_update_config_writes_local_never_tracked(self):
        before = self.tracked.read_text(encoding="utf-8")
        cfg = self._load()
        updated = cfg.update_config({"llm": {"nmd_llm_provider": "other"}})
        self.assertEqual(updated, ["llm.nmd_llm_provider=other"])
        # Tracked file byte-identical; local file holds the delta.
        self.assertEqual(self.tracked.read_text(encoding="utf-8"), before)
        self.assertTrue(self.local.exists())
        back = ConfigParser()
        back.read(self.local, encoding="utf-8")
        self.assertEqual(back.get("llm", "nmd_llm_provider"), "other")
        # Reload merges the delta.
        self.assertEqual(self._load().NMD_LLM_PROVIDER, "other")

    def test_env_wins_for_source_and_export(self):
        os.environ["NMD_LLM_MODEL"] = "env-model"
        cfg = self._load()
        self.assertEqual(cfg.key_source("llm", "nmd_llm_model"), "env")

    def test_load_cfg_gap_fill_local_beats_tracked_env_wins(self):
        local = ConfigParser()
        local.add_section("llm")
        local.set("llm", "nmd_llm_provider", "other")
        with self.local.open("w", encoding="utf-8") as f:
            local.write(f)
        os.environ.pop("NMD_LLM_PROVIDER", None)
        cfgmod._load_cfg(self.tracked)
        self.assertEqual(os.environ.get("NMD_LLM_PROVIDER"), "other")
        # Explicit shell env is never clobbered in gap-fill mode.
        os.environ["NMD_LLM_PROVIDER"] = "qwen"
        cfgmod._load_cfg(self.tracked)
        self.assertEqual(os.environ.get("NMD_LLM_PROVIDER"), "qwen")

    def test_load_cfg_override_applies_local(self):
        local = ConfigParser()
        local.add_section("llm")
        local.set("llm", "nmd_llm_provider", "other")
        with self.local.open("w", encoding="utf-8") as f:
            local.write(f)
        os.environ["NMD_LLM_PROVIDER"] = "qwen"
        cfgmod._load_cfg(self.tracked, override=True)
        self.assertEqual(os.environ.get("NMD_LLM_PROVIDER"), "other")


if __name__ == "__main__":
    unittest.main()
