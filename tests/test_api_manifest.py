"""Machine-readable public API manifest, compared in CI.

The manifest (`api_manifest.json` at the repository root) records every
public symbol per module together with its stability class. A symbol
removed or added without updating the manifest fails here — API changes
must be deliberate.

Stability classes:
- ``stable``: covered by the deprecation policy in README.md
- ``provisional``: functional and tested, but the API may change in a minor
  release while real-world validation is ongoing (currently: the navigation
  stack, fusion pipeline, invariant filter, robustness layer, interop, and
  the extended calibration modules).
"""

import importlib
import json
import pathlib

MANIFEST_PATH = pathlib.Path(__file__).resolve().parent.parent / "api_manifest.json"

#: module -> stability; every module here must define __all__.
PUBLIC_MODULES = {
    "qnav": "stable",
    "qnav.attitude.quaternion": "stable",
    "qnav.attitude.so3": "stable",
    "qnav.attitude.dcm": "stable",
    "qnav.attitude.euler": "stable",
    "qnav.frames.core": "stable",
    "qnav.frames.transforms": "stable",
    "qnav.frames.graph": "stable",
    "qnav.frames.earth": "stable",
    "qnav.determination": "stable",
    "qnav.filters": "provisional",
    "qnav.filters.contracts": "provisional",
    "qnav.filters.robust": "provisional",
    "qnav.filters.pipeline": "provisional",
    "qnav.filters.invariant": "provisional",
    "qnav.nav": "provisional",
    "qnav.nav.measurements": "provisional",
    "qnav.nav.preintegration": "provisional",
    "qnav.interop": "provisional",
    "qnav.calibration": "provisional",
    "qnav.metrics": "stable",
    "qnav.validation": "provisional",
    "qnav.types": "stable",
}


def build_manifest() -> dict:
    manifest = {}
    for module, stability in PUBLIC_MODULES.items():
        mod = importlib.import_module(module)
        assert hasattr(mod, "__all__"), f"{module} must define __all__"
        manifest[module] = {
            "stability": stability,
            "symbols": sorted(str(s) for s in mod.__all__),
        }
    return manifest


def test_manifest_matches_implementation():
    """Regenerate with: python -c "from tests.test_api_manifest import write; write()" """
    assert MANIFEST_PATH.exists(), (
        "api_manifest.json missing — generate it with tests.test_api_manifest.write()"
    )
    recorded = json.loads(MANIFEST_PATH.read_text())
    current = build_manifest()
    assert current == recorded, (
        "public API changed without updating api_manifest.json — review the "
        "diff, then regenerate via tests.test_api_manifest.write()"
    )


def test_all_symbols_resolve():
    for module, entry in build_manifest().items():
        mod = importlib.import_module(module)
        for sym in entry["symbols"]:
            assert hasattr(mod, sym), f"{module}.{sym} in __all__ but missing"


def write() -> None:
    MANIFEST_PATH.write_text(json.dumps(build_manifest(), indent=2) + "\n")
