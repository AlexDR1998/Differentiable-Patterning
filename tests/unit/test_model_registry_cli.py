from pathlib import Path

from Experiments import model_registry as registry_cli


def test_cli_defaults_to_local_models(monkeypatch):
    captured = {}

    class FakeRegistry:
        def __init__(self, root):
            captured["root"] = root

        def reindex(self):
            return Path("models/registry.sqlite")

    monkeypatch.delenv("MODEL_STORE_ROOT", raising=False)
    monkeypatch.setattr(registry_cli, "ModelRegistry", FakeRegistry)
    monkeypatch.setattr(
        registry_cli, "parse_args", lambda: type("Args", (), {"root": None, "command": "reindex"})()
    )

    registry_cli.main()

    assert captured["root"] == Path("models")


def test_cli_environment_precedes_local_default(monkeypatch):
    captured = {}

    class FakeRegistry:
        def __init__(self, root):
            captured["root"] = root

        def reindex(self):
            return Path(root := captured["root"]) / "registry.sqlite"

    monkeypatch.setenv("MODEL_STORE_ROOT", "/tmp/model-store")
    monkeypatch.setattr(registry_cli, "ModelRegistry", FakeRegistry)
    monkeypatch.setattr(
        registry_cli, "parse_args", lambda: type("Args", (), {"root": None, "command": "reindex"})()
    )

    registry_cli.main()

    assert captured["root"] == "/tmp/model-store"
