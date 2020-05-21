import os
from pathlib import Path
from click.testing import CliRunner
from sciml_bench.core.command import cli, BENCHMARK_DICT

def test_command_run_single_benchmark(tmpdir, mocker):
    mocker.patch('sciml_bench.core.command._run_benchmark')
    runner = CliRunner()
    with runner.isolated_filesystem():
        model_dir = str(tmpdir)

        # create fake data directory
        data_dir = Path(tmpdir / 'em_denoise')
        data_dir.mkdir(parents=True)
        data_dir = str(data_dir)

        # Run command
        result = runner.invoke(cli, ['run', '--data-dir',  data_dir, '--model-dir', model_dir, 'em-denoise'])
        assert result.exit_code == 0

def test_command_run_single_benchmark_set_verbosity(tmpdir, mocker, caplog):
    mocker.patch('sciml_bench.core.command._run_benchmark')
    runner = CliRunner()
    with runner.isolated_filesystem():
        model_dir = str(tmpdir)

        # create fake data directory
        data_dir = Path(tmpdir )
        (data_dir / 'em_denoise').mkdir(parents=True)
        data_dir = str(data_dir)

        # Run command with verbosity "silence"
        result = runner.invoke(cli, ['run', '--verbosity=0', '--data-dir',  data_dir, '--model-dir', model_dir, 'em-denoise'])
        assert result.exit_code == 0
        assert os.environ['TF_CPP_MIN_LOG_LEVEL'] != '-1'
        assert caplog.text == ''

        caplog.clear()

        # Run command with verbosity at maximum
        result = runner.invoke(cli, ['run', '--verbosity=2', '--data-dir',  data_dir, '--model-dir', model_dir, 'em-denoise'])
        assert result.exit_code == 0
        assert os.environ['TF_CPP_MIN_LOG_LEVEL'] != '-1'
        assert 'INFO' in caplog.text

        caplog.clear()

        # Run command with verbosity at info level (but no TF debug)
        result = runner.invoke(cli, ['run', '--verbosity=2', '--data-dir',  data_dir, '--model-dir', model_dir, 'em-denoise'])
        assert result.exit_code == 0
        assert os.environ['TF_CPP_MIN_LOG_LEVEL'] != '-1'
        assert 'INFO' in caplog.text

        caplog.clear()

        # Run command with verbosity at maximum, but log level at error
        result = runner.invoke(cli, ['run', '--verbosity=3', '--log-level=error', '--data-dir',  data_dir, '--model-dir', model_dir, 'em-denoise'])
        assert result.exit_code == 0
        assert os.environ['TF_CPP_MIN_LOG_LEVEL'] == '-1'
        assert 'INFO' not in caplog.text


def test_command_run_single_benchmark_set_environment(tmpdir, mocker, caplog):
    mocker.patch('sciml_bench.core.command._run_benchmark')
    runner = CliRunner()
    with runner.isolated_filesystem():
        model_dir = str(tmpdir)

        # create fake data directory
        data_dir = Path(tmpdir )
        (data_dir / 'em_denoise').mkdir(parents=True)
        data_dir = str(data_dir)

        # Enable auto mixed precision
        result = runner.invoke(cli, ['run', '--use-amp', '--data-dir',  data_dir, '--model-dir', model_dir, 'em-denoise'])
        assert result.exit_code == 0
        assert os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] == '1'

        # Disable GPU and use CPU only
        result = runner.invoke(cli, ['run', '--cpu-only', '--data-dir',  data_dir, '--model-dir', model_dir, 'em-denoise'])
        assert result.exit_code == 0
        assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1'

def test_command_run_single_benchmark_fails_invalid_data_dir(tmpdir, mocker, caplog):
    mocker.patch('sciml_bench.core.command._run_benchmark')
    runner = CliRunner()
    with runner.isolated_filesystem():
        model_dir = str(tmpdir)

        # create fake data directory that does not exist!
        data_dir = Path(tmpdir / 'em_denoise')
        data_dir = str(data_dir)

        # Run command
        result = runner.invoke(cli, ['run', '--data-dir',  data_dir, '--model-dir', model_dir, 'em-denoise'])

        assert result.exit_code == 1
        assert 'Data directory {} does not exist!'.format(data_dir) in caplog.text


def test_command_run_single_benchmark_fails_invalid_benchmark_name(tmpdir, mocker):
    mocker.patch('sciml_bench.core.command._run_benchmark')
    runner = CliRunner()
    with runner.isolated_filesystem():
        model_dir = str(tmpdir)

        # create fake data directory that does not exist!
        data_dir = Path(tmpdir / 'em_denoise')
        data_dir.mkdir(parents=True)
        data_dir = str(data_dir)

        # Run command
        result = runner.invoke(cli, ['run', '--data-dir',  data_dir, '--model-dir', model_dir, 'random-benchmark'])

        assert result.exit_code == 2
        assert 'invalid choice: random-benchmark' in result.output


def test_command_run_single_benchmark_skips_invalid_data_dir_for_specific_benchmark(tmpdir, mocker, caplog):
    mocker.patch('sciml_bench.core.command._run_benchmark')
    runner = CliRunner()
    with runner.isolated_filesystem():
        model_dir = str(tmpdir)

        # create fake data directory that does not exist!
        data_dir = Path(tmpdir)
        data_dir.mkdir(parents=True, exist_ok=True)
        data_dir = str(data_dir)

        # Run command
        result = runner.invoke(cli, ['run', '--data-dir',  data_dir, '--model-dir', model_dir, 'em-denoise'])

        assert result.exit_code == 0
        assert 'Data directory {} does not exist!'.format(data_dir + '/em_denoise') in caplog.text
        assert 'Skipping benchmark em-denoise'.format(data_dir) in caplog.text


def test_command_run_single_benchmark_fails_invalid_data_dir_for_specific_benchmark(tmpdir, mocker, caplog):
    mocker.patch('sciml_bench.core.command._run_benchmark')
    runner = CliRunner()
    with runner.isolated_filesystem():
        model_dir = str(tmpdir)

        # create fake data directory that does not exist!
        data_dir = Path(tmpdir)
        data_dir.mkdir(parents=True, exist_ok=True)
        data_dir = str(data_dir)

        # Run command
        result = runner.invoke(cli, ['run', '--no-skip', '--data-dir',  data_dir, '--model-dir', model_dir, 'em-denoise'])

        assert result.exit_code == 1
        assert 'Data directory {} does not exist!'.format(data_dir + '/em_denoise') in caplog.text
        assert 'Skipping benchmark em-denoise'.format(data_dir) not in caplog.text


def test_command_run_all_benchmarks(tmpdir, mocker, caplog):
    mocker.patch('sciml_bench.core.command._run_benchmark')
    runner = CliRunner()
    with runner.isolated_filesystem():
        model_dir = str(tmpdir / 'models')

        # create fake data directory that does not exist!
        data_dir = Path(tmpdir)
        data_dir.mkdir(parents=True, exist_ok=True)

        for name in BENCHMARK_DICT.keys():
            (data_dir / name.replace('-', '_')).mkdir()

        data_dir = str(data_dir)

        # Run command
        result = runner.invoke(cli, ['run', '--data-dir',  data_dir, '--model-dir', model_dir])

        assert result.exit_code == 0
        assert 'ERROR' not in caplog.text

def test_command_sysinfo():
    runner = CliRunner()
    result = runner.invoke(cli, ['sysinfo'])
    assert result.exit_code == 0

def test_command_list_default():
    runner = CliRunner()
    result = runner.invoke(cli, ['list'])
    print(result.output)
    assert result.exit_code == 0
