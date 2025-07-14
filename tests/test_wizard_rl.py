import importlib.util
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "wizard_rl.py"
    spec = importlib.util.spec_from_file_location("wizard_rl", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SIMPLE_YAML = """
steps:
  - name: A
    fields:
      - key: x
        required: true
  - name: B
    fields:
      - key: y
        required: false
"""


def test_load_schema(tmp_path):
    yaml_file = tmp_path / "schema.yaml"
    yaml_file.write_text(SIMPLE_YAML)
    mod = load_module()
    schema = mod.load_wizard_schema(str(yaml_file))
    assert schema["steps"][0]["name"] == "A"


def test_state_to_vector():
    mod = load_module()
    schema = {
        "steps": [
            {"name": "A", "fields": [{"key": "x"}]},
            {"name": "B", "fields": [{"key": "y"}]},
        ]
    }
    state = {"wizard_step": "A", "x": "1"}
    vec = mod.state_to_vector(state, schema)
    assert vec.tolist() == [0.0, 1.0, 0.0]


def test_policy_and_persistence(tmp_path):
    mod = load_module()
    policy = mod.WizardPolicy(["A", "B"])
    assert policy.decide_next_step({"wizard_step": "A"}) == 1
    file = tmp_path / "p.pkl"
    mod.save_policy(policy, file)
    loaded = mod.load_policy(file)
    assert loaded.step_order == ["A", "B"]


def test_compute_reward():
    mod = load_module()
    reward = mod.compute_reward(
        {"total_steps": 3, "total_time_sec": 10, "completed": True}
    )
    assert reward > 0
