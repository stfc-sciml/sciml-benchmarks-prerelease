import pytest
from pathlib import Path
from click.testing import CliRunner
from sciml_bench.core.command import cli

@pytest.fixture()
def data_dir():
    return Path("data/").absolute()

def test_command_em_denoise(data_dir, tmpdir):
    runner = CliRunner()
    with runner.isolated_filesystem():
        model_dir = str(tmpdir)
        data_dir = str(Path(data_dir / 'em_denoise'))
        result = runner.invoke(cli, ['em-denoise', '--data-dir',  data_dir, '--model-dir', model_dir, '--epochs', '1', '--batch-size', '1'])
        assert result.exit_code == 0
