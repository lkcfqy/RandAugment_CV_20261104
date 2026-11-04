import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from portable_dev.paths import project_root, project_path, asset_path


class PathConfiguration(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="randaugment-relocated-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve() / "different-checkout"
        (self.root / "portable_dev").mkdir(parents=True)
        (self.root / "portable_dev/paths.py").write_text("# fixture marker\n")
        (self.root / "README.md").write_text("fixture\n")
        self.source = self.root / "nested/topic/module.py"
        self.source.parent.mkdir(parents=True)
        self.source.write_text("# fixture\n")
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_nested_relocated_source_discovers_checkout(self):
        self.assertEqual(project_root(self.source), self.root.resolve())
        self.assertEqual(project_path(self.source, "data/item.json"), self.root / "data/item.json")

    def test_valid_explicit_project_root(self):
        os.environ["RANDAUGMENT_PROJECT_ROOT"] = str(self.root)
        self.assertEqual(project_root(Path(self.temp.name) / "outside.py"), self.root)

    def test_invalid_override_fails_instead_of_silently_falling_back(self):
        os.environ["RANDAUGMENT_PROJECT_ROOT"] = str(Path(self.temp.name))
        with self.assertRaises(ValueError):
            project_root(self.source)

    def test_missing_root_is_an_explicit_error(self):
        with self.assertRaises(ValueError):
            project_root(Path(self.temp.name) / "elsewhere.py")

    def test_external_asset_root_is_separate_from_source(self):
        assets = Path(self.temp.name).resolve() / "local-assets"
        assets.mkdir()
        os.environ["RANDAUGMENT_ASSET_ROOT"] = str(assets)
        self.assertEqual(asset_path(self.source, "models/model.bin"), assets / "models/model.bin")
        self.assertEqual(project_root(self.source), self.root)

    def test_absolute_and_parent_escape_are_rejected(self):
        for relative in ("../outside", "/tmp/outside"):
            with self.subTest(relative=relative):
                with self.assertRaises(ValueError):
                    project_path(self.source, relative)
                with self.assertRaises(ValueError):
                    asset_path(self.source, relative)

    def test_symlink_escape_is_rejected(self):
        (self.root / "escape").symlink_to(Path(self.temp.name), target_is_directory=True)
        with self.assertRaises(ValueError):
            project_path(self.source, "escape/outside")

    def test_missing_asset_is_reportable_without_creating_it(self):
        target = asset_path(self.source, "missing/weights.bin")
        self.assertFalse(target.exists())
        self.assertFalse(target.parent.exists())
