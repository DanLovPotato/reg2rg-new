"""CPU structural/component tests; do not load CT encoders or a large checkpoint."""
import ast
from collections import OrderedDict
from pathlib import Path
import unittest
import warnings

import torch
from torch import nn

SOURCE = Path(__file__).resolve().parents[1] / "src/Model/my_embedding_layer.py"
TREE = ast.parse(SOURCE.read_text())
CLASS = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "MyEmbedding")
METHODS = {n.name: n for n in CLASS.body if isinstance(n, ast.FunctionDef)}


def assignments(name):
    return [n for n in ast.walk(METHODS["forward"]) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]


class DirectGraphFusionTests(unittest.TestCase):
    def test_no_whole_image_gke_module_or_call(self):
        self.assertFalse(any(isinstance(n, ast.Attribute) and n.attr == "gke"
                             for n in ast.walk(CLASS)))
        calls = [n for n in ast.walk(METHODS["forward"]) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == "global_gke"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(ast.unparse(calls[0].args[0]), "local_features")

    def test_actual_fusion_expression_and_gradients(self):
        nodes = [n for n in assignments("enhanced_image_embedding") if isinstance(n.value, ast.BinOp)]
        self.assertEqual(len(nodes), 1)
        self.assertEqual(ast.unparse(nodes[0].value), "image_embedding + graph_global.unsqueeze(1)")
        image = torch.randn(2, 32, 8, requires_grad=True)
        graph = torch.randn(2, 8, requires_grad=True)
        namespace = {"image_embedding": image, "graph_global": graph}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
        result = namespace["enhanced_image_embedding"]
        torch.testing.assert_close(result, image + graph[:, None, :])
        result.sum().backward()
        torch.testing.assert_close(image.grad, torch.ones_like(image))
        torch.testing.assert_close(graph.grad, torch.full_like(graph, 32))

    def test_mask_branch_bypasses_graph_and_preserves_supplied_regions(self):
        branch = next(n for n in ast.walk(METHODS["forward"]) if isinstance(n, ast.If)
                      and ast.unparse(n.test) == "precomputed_region_embedding is not None")
        code = ast.unparse(ast.Module(body=branch.body, type_ignores=[]))
        self.assertNotIn("self.global_gke", code)
        self.assertNotIn("self.rwlke(", code)
        self.assertIn("enhanced_image_embedding = image_embedding", code)
        self.assertIn("rwlke_region_embedding = precomputed_region_embedding", code)

    def test_strict_legacy_checkpoint_compatibility(self):
        # Execute the real load hook in a tiny module; preserve nested prefix behavior.
        definition = ast.ClassDef(name="Tiny", bases=[ast.Attribute(
            value=ast.Name(id="nn", ctx=ast.Load()), attr="Module", ctx=ast.Load())],
            keywords=[], body=[METHODS["_load_from_state_dict"]], decorator_list=[])
        tree = ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[]))
        namespace = {"nn": nn, "warnings": warnings}
        exec(compile(tree, str(SOURCE), "exec"), namespace)
        child = namespace["Tiny"]()
        child.rwlke = nn.Linear(2, 2)
        child.global_gke = nn.Linear(2, 2)
        model = nn.Module()
        model.add_module("embedding", child)
        clean = model.state_dict()
        legacy = OrderedDict(clean)
        legacy["embedding.gke.visual_query.weight"] = torch.zeros(2, 2)
        with self.assertWarnsRegex(UserWarning, "Ignoring 1 legacy"):
            result = model.load_state_dict(legacy, strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        self.assertIn("embedding.gke.visual_query.weight", legacy)
        model.load_state_dict(clean, strict=True)
        bad = OrderedDict(clean)
        bad["embedding.unrelated.weight"] = torch.zeros(2, 2)
        with self.assertRaises(RuntimeError):
            model.load_state_dict(bad, strict=True)
        missing = OrderedDict(clean)
        del missing["embedding.global_gke.weight"]
        with self.assertRaises(RuntimeError):
            model.load_state_dict(missing, strict=True)


if __name__ == "__main__":
    unittest.main()
