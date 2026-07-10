"""Executes every ```python block in README.md and docs/getting_started.md.

Documentation examples are part of the public contract: an API change that
breaks an example fails here before it reaches a user.
"""

import pathlib
import re
import warnings

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOC_FILES = [ROOT / "README.md", ROOT / "docs" / "getting_started.md"]

_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.S)


def _blocks():
    out = []
    for path in DOC_FILES:
        for i, m in enumerate(_BLOCK_RE.finditer(path.read_text(encoding="utf-8"))):
            out.append(pytest.param(m.group(1), id=f"{path.name}-block{i}"))
    return out


@pytest.mark.parametrize("code", _blocks())
def test_doc_example_executes(code):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # examples may demonstrate warning paths
        exec(compile(code, "<doc-example>", "exec"), {"__name__": "__doc_example__"})
