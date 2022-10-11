import pathlib

import pytest


@pytest.fixture
def data_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data"
