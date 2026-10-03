"""Images flagged as low quality, and the file they are kept in.

Images are flagged by hand in ``Experiments/micropatterns/data_cleaning.py``
and saved in a quality flags file that the training loader reads
(``data.micropattern.quality_flags_file``). Each flag is one image file, so
one experiment group at one timestep of one replicate: the loader treats it
as not measured (see ``load_micropattern_260726``).
"""

from pathlib import Path
from typing import Mapping

import yaml

QUALITY_FLAGS_FORMAT = "quality_flags_v1"


def save_quality_flags(path, flags):
    """Write ``{image path relative to the dataset root: reason}`` to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        file.write(
            "# Images flagged as low quality in Experiments/micropatterns/data_cleaning.py.\n"
            "# Training treats each as not measured. Keys are image paths relative to\n"
            "# the dataset root; values are the reasons given.\n"
        )
        yaml.safe_dump(
            {
                "format": QUALITY_FLAGS_FORMAT,
                "flagged": {str(name): str(flags[name]) for name in sorted(flags)},
            },
            file,
            sort_keys=False,
        )


def load_quality_flags(path):
    """Read a file written by ``save_quality_flags``; returns ``{path: reason}``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Quality flags file {path} does not exist. Export it from "
            "Experiments/micropatterns/data_cleaning.py (section Quality flags)."
        )
    with open(path) as file:
        content = yaml.safe_load(file)
    if not isinstance(content, Mapping) or content.get("format") != QUALITY_FLAGS_FORMAT:
        raise ValueError(f"{path} is not a {QUALITY_FLAGS_FORMAT} quality flags file")
    return {str(name): str(reason or "") for name, reason in (content.get("flagged") or {}).items()}
