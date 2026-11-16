import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts/explanations'))
import run_budgets  # noqa: E402


def fake_runner(tmp_path, body):
    script = tmp_path / 'fake_runner.py'
    script.write_text('import json, sys\nfrom pathlib import Path\nargs = sys.argv[1:]\n' + body)
    return script


def test_budgets_run_in_order_and_done_ones_are_skipped(tmp_path):
    log = tmp_path / 'calls.txt'
    script = fake_runner(tmp_path, f"open({str(log)!r}, 'a').write(args[args.index('--budget') + 1] + '\\n')\n")
    argv = ['--budgets', '3000', '5000', '--state-dir', str(tmp_path / 'state'), '--', str(script), '--config', 'X']
    assert run_budgets.main(argv) == 0
    assert log.read_text().split() == ['3000', '5000']
    assert run_budgets.main(argv) == 0  # A repeat finds both done.
    assert log.read_text().split() == ['3000', '5000']
    folder = next((tmp_path / 'state').iterdir())
    assert json.loads((folder / 'B5000.done').read_text())['attempts'] == 1


def test_a_failed_budget_stops_before_the_next_and_resumes_there(tmp_path):
    flag = tmp_path / 'fail_once'
    flag.write_text('')
    log = tmp_path / 'calls.txt'
    script = fake_runner(tmp_path, f"""budget = args[args.index('--budget') + 1]
open({str(log)!r}, 'a').write(budget + '\\n')
flag = Path({str(flag)!r})
if budget == '5000' and flag.exists():
    flag.unlink()
    sys.exit(3)
""")
    argv = ['--budgets', '3000', '5000', '--state-dir', str(tmp_path / 'state'), '--', str(script)]
    assert run_budgets.main(argv) == 1
    assert run_budgets.main(argv) == 0
    assert log.read_text().split() == ['3000', '5000', '5000']
    folder = next((tmp_path / 'state').iterdir())
    outcomes = [item['outcome'] for item in json.loads((folder / 'B5000_attempts.json').read_text())]
    assert outcomes == ['failed', 'done']


def test_budget_and_no_resume_options_are_rejected(tmp_path):
    for extra in (['--budget', '1000'], ['--max-evals=1000'], ['--no-resume']):
        with pytest.raises(SystemExit):
            run_budgets.parse(['--budgets', '3000', '--', 'OD_yolo.py', *extra])


def test_different_commands_keep_separate_state(tmp_path):
    first = run_budgets.parse(['--budgets', '3000', '--', 'OD_yolo.py', '--config', 'A'])
    second = run_budgets.parse(['--budgets', '3000', '--', 'OD_yolo.py', '--config', 'B'])
    assert run_budgets.state_folder(first) != run_budgets.state_folder(second)
