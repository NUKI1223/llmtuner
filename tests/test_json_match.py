"""The json_match grader: compare fields of a JSON output with the case's expected object."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmtuner.config import ConfigError, Grader, load_config  # noqa: E402
from llmtuner.dataset import Case, DatasetError  # noqa: E402
from llmtuner.graders import check_case, grade  # noqa: E402

CWD = Path(tempfile.gettempdir())


def run(expected, output, **options):
    grader = Grader("fields", "json_match", options=options)
    text = output if isinstance(output, str) else json.dumps(output)
    return grade(grader, Case("c1", "in", expected), text, CWD)


class JsonMatchTests(unittest.TestCase):
    def test_subset_of_fields_passes_and_extra_output_keys_are_ignored(self):
        result = run({"mode": "surprise", "count": 3}, {"mode": "surprise", "count": 3, "action": "recommend"})
        self.assertTrue(result.passed)
        self.assertEqual(result.detail, "")

    def test_a_wrong_value_fails_and_names_the_field(self):
        result = run({"mode": "surprise", "count": 3}, {"mode": "close", "count": 3})
        self.assertFalse(result.passed)
        self.assertIn("mode", result.detail)
        self.assertIn("surprise", result.detail)
        self.assertIn("close", result.detail)
        self.assertNotIn("count", result.detail)

    def test_a_missing_field_fails(self):
        result = run({"mode": "close"}, {"count": 3})
        self.assertFalse(result.passed)
        self.assertIn("mode: missing", result.detail)

    def test_list_order_is_ignored_by_default_but_content_is_not(self):
        self.assertTrue(run({"ids": ["a", "b"]}, {"ids": ["b", "a"]}).passed)
        self.assertFalse(run({"ids": ["a", "b"]}, {"ids": ["a"]}).passed)
        self.assertFalse(run({"ids": ["a", "b"]}, {"ids": ["a", "b", "c"]}).passed)
        self.assertFalse(run({"ids": ["a", "a"]}, {"ids": ["a"]}).passed)
        self.assertTrue(run({"ids": []}, {"ids": []}).passed)

    def test_ordered_option_makes_list_order_count(self):
        self.assertFalse(run({"ids": ["a", "b"]}, {"ids": ["b", "a"]}, ordered=True).passed)
        self.assertTrue(run({"ids": ["a", "b"]}, {"ids": ["a", "b"]}, ordered=True).passed)

    def test_nested_objects_match_as_subsets(self):
        self.assertTrue(run({"filters": {"mode": "close"}}, {"filters": {"mode": "close", "n": 1}}).passed)
        result = run({"filters": {"mode": "close"}}, {"filters": {"mode": "explore"}})
        self.assertFalse(result.passed)
        self.assertIn("filters.mode", result.detail)
        self.assertFalse(run({"filters": {"mode": "close"}}, {"filters": "close"}).passed)

    def test_json_types_are_not_confused(self):
        self.assertFalse(run({"flag": True}, {"flag": 1}).passed)
        self.assertFalse(run({"flag": False}, {"flag": 0}).passed)
        self.assertFalse(run({"n": 1}, {"n": "1"}).passed)
        self.assertTrue(run({"n": 1}, {"n": 1.0}).passed)
        self.assertTrue(run({"x": None}, {"x": None}).passed)
        self.assertFalse(run({"x": None}, {}).passed)

    def test_output_that_is_not_a_json_object_fails_cleanly(self):
        for output in ("not json", "", "[1, 2]", "42", "[" * 100000):
            result = run({"mode": "close"}, output)
            self.assertFalse(result.passed)
            self.assertFalse(result.error)

    def test_surrounding_whitespace_is_fine(self):
        self.assertTrue(run({"a": 1}, '\n  {"a": 1}\n').passed)

    def test_field_option_reads_a_sub_object_of_expected(self):
        grader = Grader("fields", "json_match", options={"field": "action"})
        case = Case("c1", "in", {"action": {"mode": "close"}, "other": 1})
        self.assertTrue(grade(grader, case, '{"mode": "close"}', CWD).passed)
        self.assertFalse(grade(grader, case, '{"mode": "explore"}', CWD).passed)

    def test_list_order_is_ignored_at_every_depth(self):
        self.assertTrue(run({"a": [[1, 2]]}, {"a": [[2, 1]]}).passed)
        self.assertTrue(run({"a": [{"k": [1, 2]}]}, {"a": [{"k": [2, 1]}]}).passed)
        self.assertFalse(run({"a": [[1, 2]]}, {"a": [[2, 1]]}, ordered=True).passed)

    def test_objects_inside_lists_must_match_exactly(self):
        self.assertFalse(run({"a": [{"x": 1}]}, {"a": [{"x": 1, "y": 2}]}).passed)
        self.assertTrue(run({"a": [{"x": 1, "y": 2}]}, {"a": [{"y": 2, "x": 1}]}).passed)

    def test_nan_and_infinity_are_not_json(self):
        for output in ('{"a": 1, "b": NaN}', '{"a": Infinity}', '{"a": -Infinity}'):
            result = run({"a": 1}, output)
            self.assertFalse(result.passed)
            self.assertEqual(result.detail, "not valid JSON")
            self.assertFalse(result.error)

    def test_deeply_nested_valid_json_fails_without_a_grader_error(self):
        result = run({"a": 1}, '{"a":' + "[" * 3000 + "]" * 3000 + "}")
        self.assertFalse(result.passed)
        self.assertFalse(result.error)

    def test_long_values_are_cut_with_a_visible_mark(self):
        result = run({"a": "x" * 300 + "1"}, {"a": "x" * 300 + "2"})
        self.assertFalse(result.passed)
        self.assertIn("…", result.detail)
        many = run({str(i): "expected-value" for i in range(80)}, {str(i): "other" for i in range(80)})
        self.assertEqual(len(many.detail), 500)
        self.assertTrue(many.detail.endswith("…"))

    def test_an_empty_expected_object_is_refused_before_the_run(self):
        with self.assertRaises(DatasetError):
            check_case(Grader("fields", "json_match"), Case("bad", "in", {}))
        with self.assertRaises(DatasetError):
            check_case(Grader("fields", "json_match", options={"field": "action"}), Case("bad", "in", {"action": {}}))

    def test_non_ascii_values_compare_and_show_as_text(self):
        self.assertTrue(run({"genre": "хип-хоп"}, {"genre": "хип-хоп"}).passed)
        self.assertIn("хип-хоп", run({"genre": "хип-хоп"}, {"genre": "рок"}).detail)


class JsonMatchConfigTests(unittest.TestCase):
    def test_a_case_without_an_expected_object_is_refused_before_the_run(self):
        grader = Grader("fields", "json_match")
        check_case(grader, Case("ok", "in", {"a": 1}))
        for expected in (None, "text", ["a"]):
            with self.assertRaises(DatasetError):
                check_case(grader, Case("bad", "in", expected))
        with self.assertRaises(DatasetError):
            check_case(Grader("fields", "json_match", options={"field": "action"}), Case("bad", "in", {"a": 1}))

    def write_config(self, body: str) -> Path:
        folder = Path(tempfile.mkdtemp())
        (folder / "llmtuner.toml").write_text(
            'project = "p"\n[targets.t]\ncommand = ["true"]\n[datasets.d]\npath = "d.jsonl"\n' + body, encoding="utf-8")
        return folder

    def test_config_accepts_the_grader_and_its_options(self):
        config = load_config(self.write_config('[graders.fields]\ntype = "json_match"\nordered = true\nfield = "action"\n'))
        self.assertEqual(config.graders["fields"].options, {"ordered": True, "field": "action"})

    def test_config_rejects_bad_options(self):
        for body in ('[graders.f]\ntype = "json_match"\nordered = "yes"\n',
                     '[graders.f]\ntype = "json_match"\nfield = 3\n',
                     '[graders.f]\ntype = "json_match"\nvalues = ["a"]\n'):
            with self.assertRaises(ConfigError):
                load_config(self.write_config(body))


if __name__ == "__main__":
    unittest.main()
