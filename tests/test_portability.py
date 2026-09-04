"""Standard-library checks for portable tutorial paths and source text."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / ("tutorials" if (ROOT / "tutorials").is_dir() else "datasets")


class TutorialPortabilityTests(unittest.TestCase):
    def test_data_root_accepts_home_relative_paths_and_spaces(self):
        checked = 0
        for path in NOTEBOOK_ROOT.rglob("*.ipynb"):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for cell in notebook["cells"]:
                if cell["cell_type"] != "code":
                    continue
                tree = ast.parse("".join(cell["source"]))
                for statement in tree.body:
                    if not isinstance(statement, ast.Assign):
                        continue
                    if not any(isinstance(t, ast.Name) and t.id == "DATA_ROOT" for t in statement.targets):
                        continue
                    if "SPADIVA_DATA_ROOT" not in ast.unparse(statement.value):
                        continue
                    namespace = {
                        "Path": Path,
                        "os": SimpleNamespace(environ={"SPADIVA_DATA_ROOT": "~/spaDIVA data"}),
                        "PROJECT_ROOT": ROOT, "REPO_ROOT": ROOT, "_config": {},
                    }
                    value = eval(compile(ast.Expression(statement.value), str(path), "eval"), namespace)
                    with self.subTest(notebook=str(path.relative_to(ROOT))):
                        self.assertEqual(value, Path.home() / "spaDIVA data")
                    checked += 1
        self.assertGreater(checked, 0)

    def test_source_text_uses_lf_line_endings(self):
        paths = list((ROOT / "spaDIVA").glob("*.py"))
        paths += list(NOTEBOOK_ROOT.rglob("*.ipynb"))
        paths += [ROOT / "README.md", ROOT / ".gitattributes"]
        for path in paths:
            with self.subTest(path=str(path.relative_to(ROOT))):
                self.assertNotIn(b"\r\n", path.read_bytes())

    def test_git_attributes_fix_source_line_endings(self):
        attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
        self.assertIn("* text=auto eol=lf", attributes.splitlines())


if __name__ == "__main__":
    unittest.main()
