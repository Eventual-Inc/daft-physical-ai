from __future__ import annotations

import daft

from daft_ext_template import greet


def test_greet() -> None:
    df = daft.from_pydict({"name": ["John", "Paul"]})
    result = df.select(greet(df["name"]).alias("greet")).collect().to_pydict()
    assert result["greet"] == ["Hello, John!", "Hello, Paul!"]
