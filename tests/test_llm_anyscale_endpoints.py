from click.testing import CliRunner
from llm.cli import cli
from llm_anyscale_endpoints import MODELS
import json
import pytest


@pytest.fixture
def user_path(monkeypatch, tmpdir):
    dir = tmpdir / "llm.datasette.io"
    dir.mkdir()
    monkeypatch.setenv("LLM_USER_PATH", str(dir))
    return dir


@pytest.mark.parametrize("set_key", (False, True))
def test_llm_models(set_key, user_path):
    runner = CliRunner()
    if set_key:
        (user_path / "keys.json").write_text(
            json.dumps({"anyscale-endpoints": "x"}), "utf-8"
        )
    result = runner.invoke(cli, ["models", "list"])
    assert result.exit_code == 0, result.output
    fragments = ["AnyscaleEndpoints: {}".format(model_id) for model_id in MODELS]
    for fragment in fragments:
        if set_key:
            assert fragment in result.output
        else:
            assert fragment not in result.output
