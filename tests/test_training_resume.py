"""Exercise the actual trainer RNG restore statements without importing Isaac Lab."""
import ast
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def restore_trainer_rng(checkpoint, device):
    # train.py launches Isaac at module scope; execute its actual two RNG statements
    # rather than copying a second implementation into this test.
    source = Path(__file__).resolve().parents[1] / "scripts/train.py"
    tree = ast.parse(source.read_text())
    resume = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "args.resume")
    statements = [node for node in resume.body
                  if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                      and ast.unparse(node.value.func) == "torch.set_rng_state")
                  or (isinstance(node, ast.If)
                      and ast.unparse(node.test) == "args.device.startswith('cuda')")]
    assert len(statements) == 2
    code = compile(ast.Module(body=statements, type_ignores=[]), str(source), "exec")
    exec(code, {"torch": torch, "checkpoint": checkpoint,
                "args": SimpleNamespace(device=device)})


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_resume_rng_after_checkpoint_device_mapping(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is needed to reproduce checkpoint RNG remapping")
    use_cuda = device.startswith("cuda")
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if use_cuda else []):
        torch.manual_seed(937)
        if use_cuda:
            torch.cuda.manual_seed_all(938)
        checkpoint = {"torch_rng": torch.get_rng_state(),
                      "cuda_rng": torch.cuda.get_rng_state_all() if use_cuda else [],
                      "model_tensor": torch.ones(2)}
        expected_cpu = torch.rand(8)
        expected_cuda = [torch.rand(8, device=f"cuda:{i}")
                         for i in range(torch.cuda.device_count())] if use_cuda else []
        stream = io.BytesIO()
        torch.save(checkpoint, stream)
        stream.seek(0)
        loaded = torch.load(stream, map_location=device, weights_only=False)
        assert loaded["model_tensor"].device == torch.device(device)
        assert all(state.device == torch.device(device) for state in loaded["cuda_rng"])
        restore_trainer_rng(loaded, device)
        assert torch.equal(torch.rand(8), expected_cpu)
        for i, expected in enumerate(expected_cuda):
            assert torch.equal(torch.rand(8, device=f"cuda:{i}"), expected)
        assert loaded["model_tensor"].device == torch.device(device)
