from pathlib import Path

from Experiments.config_workflow import generate_manifest, load_yaml


BASE_DIR = Path(__file__).resolve().parent


def generate(sweep_name="emoji_attractor_switch_example", emit_files=False):
    """Generate an impulse manifest from the package base and named sweep."""

    base_path = BASE_DIR / "conf" / "base_config.yaml"
    sweep_path = BASE_DIR / "conf" / "experiments" / f"{sweep_name}.yaml"
    output_dir = BASE_DIR / "conf" / "generated" / sweep_name
    base_cfg = load_yaml(base_path)
    sweep_cfg = load_yaml(sweep_path)
    sweep_cfg["base_config"] = str(base_path)
    sweep_cfg["sweep_file"] = str(sweep_path)
    return generate_manifest(base_cfg, sweep_cfg, output_dir, emit_files=emit_files)


if __name__ == "__main__":
    manifest = generate()
    print(f"Wrote {manifest['count']} impulse configs")

