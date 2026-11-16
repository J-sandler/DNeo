import json
import os
from pathlib import Path

import pytest

from dneo.runtime import dneo_production


@pytest.mark.parametrize("workspace", [None, ":4096:8", ":16:8"])
def test_qualified_workspace_default_and_conflicts(monkeypatch, workspace):
    profile = json.loads(Path(dneo_production.__file__).with_name("profile.json").read_text())
    for key, value in {**profile["environment"]["common"], **profile["environment"]["h100"]}.items():
        monkeypatch.setenv(key, value)
    if workspace is None:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    else:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", workspace)
    if workspace == ":16:8":
        with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG"):
            dneo_production.activate_production_runtime(hardware="h100")
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == workspace
    else:
        dneo_production.activate_production_runtime(hardware="h100")
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
