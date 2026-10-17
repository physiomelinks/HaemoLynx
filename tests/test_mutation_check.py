"""scripts/mutation_check.py: the "each rewritten test fails once against broken code" proof."""
import contextlib
import importlib.util
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
from conftest import TOY_ENV, realdata_test, toy_git

pytest.importorskip("coverage")  # the script's clean runs; [dev]'s pytest-cov installs it

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "mutation_check.py"
HELPER = REPO / "scripts" / "_worktree.py"
_spec = importlib.util.spec_from_file_location("mutation_check", SCRIPT)
mc = importlib.util.module_from_spec(_spec)
sys.modules["mutation_check"] = mc  # dataclasses look their module up here
_spec.loader.exec_module(mc)

CALC = '''\
def add(a, b):
    return a + b


def sign(x):
    if x < 0:
        return -1
    return 1
'''


@pytest.fixture
def toy(tmp_path):
    """A one-module git repo with the script in scripts/, so the script's REPO is this repo."""
    repo = tmp_path / "toy"
    (repo / "src" / "toy").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "scripts").mkdir()
    shutil.copy(SCRIPT, repo / "scripts" / "mutation_check.py")
    shutil.copy(HELPER, repo / "scripts" / "_worktree.py")
    (repo / "pyproject.toml").write_text('[tool.pytest.ini_options]\npythonpath = ["src"]\n')
    (repo / ".gitignore").write_text("__pycache__/\n")
    (repo / "src" / "toy" / "__init__.py").write_text("")
    (repo / "src" / "toy" / "calc.py").write_text(CALC)
    (repo / "tests" / "test_calc.py").write_text(
        "from toy.calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n")
    toy_git(repo, "init", "-q")
    toy_git(repo, "add", ".")
    toy_git(repo, "commit", "-q", "-m", "toy")
    return repo


def _run(repo, spec_text, *args):
    """Run the script on `spec_text`; its temp folders go in a fresh TMPDIR that must end empty."""
    spec = repo.parent / "spec.toml"
    spec.write_text(textwrap.dedent(spec_text))
    tmp = repo.parent / "tmp"
    tmp.mkdir()
    proc = subprocess.run([sys.executable, str(repo / "scripts" / "mutation_check.py"), str(spec), *args],
                          cwd=repo, env={**TOY_ENV, "TMPDIR": str(tmp)}, capture_output=True, text=True,
                          timeout=120)
    assert list(tmp.iterdir()) == [], "the script left temp files behind"
    return proc


def _row(stdout, name):
    rows = [line for line in stdout.splitlines() if line.startswith(f"| {name} |")]
    assert len(rows) == 1, stdout
    return [cell.strip() for cell in rows[0].strip("|").split("|")]


def test_a_mutation_its_uncommitted_test_catches_exits_0(toy):
    # The test exists only as an untracked file, so it is found only if uncommitted work is copied in.
    (toy / "tests" / "test_sign.py").write_text(
        "from toy.calc import sign\n\n\ndef test_negative():\n    assert sign(-2) == -1\n")
    proc = _run(toy, '''
        [[mutation]]
        name = "negative read as positive"
        file = "src/toy/calc.py"
        old = "        return -1\\n"
        new = "        return 1\\n"
        test = "tests/test_sign.py::test_negative"
    ''')
    assert proc.returncode == 0, proc.stdout + proc.stderr
    name, test, expected, result, ran, failure = _row(proc.stdout, "negative read as positive")
    assert (test, expected, result, ran) == ("tests/test_sign.py::test_negative", "caught", "caught", "yes")
    assert failure == "tests/test_sign.py:5: assert 1 == -1"  # the test's file:line, then the E line
    head = subprocess.run(["git", "rev-parse", "--short=7", "HEAD"], cwd=toy, env=TOY_ENV,
                          capture_output=True, text=True).stdout.strip()
    lines = proc.stdout.splitlines()
    assert lines[0] == f"Worktree: {head} + 1 uncommitted file(s) copied in, 0 data path(s) linked in."
    assert lines[1] == "Copied in: tests/test_sign.py"  # named, so a stray note is not taken for a test input
    assert lines[-1] == "1 mutation(s): 1 as expected, 0 not; 0 never ran the mutated line; 0 error(s)."


def test_a_survivor_the_spec_expected_caught_exits_1(toy):
    proc = _run(toy, '''
        [[mutation]]
        name = "operands swapped"
        file = "src/toy/calc.py"
        old = "return a + b"
        new = "return b + a"
        test = "tests/test_calc.py::test_add"
    ''')
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert _row(proc.stdout, "operands swapped")[2:5] == ["caught", "survived", "yes"]


def test_a_survivor_the_spec_expected_to_survive_exits_0(toy):
    proc = _run(toy, '''
        [[mutation]]
        name = "operands swapped"
        file = "src/toy/calc.py"
        old = "return a + b"
        new = "return b + a"
        test = "tests/test_calc.py::test_add"
        expect = "survived"
    ''')
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _row(proc.stdout, "operands swapped")[2:5] == ["survived", "survived", "yes"]


def test_a_mutated_line_the_test_never_ran_exits_1(toy):
    proc = _run(toy, '''
        [[mutation]]
        name = "positive sign doubled"
        file = "src/toy/calc.py"
        old = "    return 1\\n"
        new = "    return 2\\n"
        test = "tests/test_calc.py::test_add"
        expect = "survived"
    ''')
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert _row(proc.stdout, "positive sign doubled")[2:5] == ["survived", "survived", "no"]


def test_a_mutation_pytest_errors_on_is_an_error_not_a_catch_and_exits_3(toy):
    proc = _run(toy, '''
        [[mutation]]
        name = "syntax broken"
        file = "src/toy/calc.py"
        old = "return a + b"
        new = "return a +"
        test = "tests/test_calc.py::test_add"
    ''')
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert _row(proc.stdout, "syntax broken")[3] == "error"


def test_a_mutation_the_same_size_as_the_line_it_replaces_is_still_caught(toy):
    # Python reuses cached bytecode when a file's mtime (to the second) and size are unchanged.
    # This conftest pins calc.py's mtime before import, so a same-size edit would always test
    # the old code if the clean run left bytecode behind.
    (toy / "tests" / "conftest.py").write_text(textwrap.dedent('''\
        import os
        from pathlib import Path

        os.utime(Path(__file__).parents[1] / "src" / "toy" / "calc.py", (1e9, 1e9))
    '''))
    proc = _run(toy, '''
        [[mutation]]
        name = "minus"
        file = "src/toy/calc.py"
        old = "return a + b"
        new = "return a - b"
        test = "tests/test_calc.py::test_add"
    ''')
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_only_given_twice_runs_the_names_from_both(toy):
    spec = "".join(f'''
        [[mutation]]
        name = "{name}"
        file = "src/toy/calc.py"
        old = "return a + b"
        new = "{new}"
        test = "tests/test_calc.py::test_add"
    ''' for name, new in [("minus", "return a - b"), ("times", "return a * b"), ("first", "return a")])
    proc = _run(toy, spec, "--only", "minus", "--only", "first")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert [line.split("|")[1].strip() for line in proc.stdout.splitlines()[4:6]] == ["minus", "first"]
    assert "| times |" not in proc.stdout


def test_a_test_that_fails_before_any_mutation_exits_3_with_no_table(toy):
    (toy / "tests" / "test_calc.py").write_text(
        "from toy.calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 6\n")
    proc = _run(toy, '''
        [[mutation]]
        name = "operands swapped"
        file = "src/toy/calc.py"
        old = "return a + b"
        new = "return b + a"
        test = "tests/test_calc.py::test_add"
    ''')
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "| operands swapped |" not in proc.stdout
    assert "tests/test_calc.py::test_add fails before any mutation (pytest exit 1)" in proc.stderr


def test_a_test_that_skips_before_any_mutation_exits_3_naming_the_skip_reason(toy):
    # A skip also exits 0: read as a pass, the mutated run's skip would then read as "survived".
    (toy / "tests" / "test_calc.py").write_text(textwrap.dedent('''\
        import pytest


        @pytest.mark.skip(reason="the real data is not on disk")
        def test_add():
            pass
    '''))
    proc = _run(toy, '''
        [[mutation]]
        name = "operands swapped"
        file = "src/toy/calc.py"
        old = "return a + b"
        new = "return b + a"
        test = "tests/test_calc.py::test_add"
        expect = "survived"
    ''')
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "| operands swapped |" not in proc.stdout
    assert "tests/test_calc.py::test_add skips before any mutation" in proc.stderr
    assert "the real data is not on disk" in proc.stderr


def test_a_mutation_its_test_skips_on_is_skipped_not_survived_and_exits_3(toy):
    (toy / "tests" / "test_calc.py").write_text(textwrap.dedent('''\
        import pytest

        from toy.calc import add


        def test_add():
            if add(0, 0) != 0:
                pytest.skip("add looks broken, so the data is missing")
            assert add(2, 3) == 5
    '''))
    proc = _run(toy, '''
        [[mutation]]
        name = "off by one"
        file = "src/toy/calc.py"
        old = "return a + b"
        new = "return a + b + 1"
        test = "tests/test_calc.py::test_add"
        expect = "survived"
    ''')
    assert proc.returncode == 3, proc.stdout + proc.stderr
    name, test, expected, result, ran, failure = _row(proc.stdout, "off by one")
    assert (expected, result, ran) == ("survived", "skipped", "yes")
    assert "add looks broken, so the data is missing" in failure
    assert proc.stdout.splitlines()[-1] == (
        "1 mutation(s): 0 as expected, 1 not; 0 never ran the mutated line; 1 error(s).")


def test_a_worktree_git_will_not_remove_still_reports_the_first_error_and_leaves_nothing_behind(toy):
    # A single --force cannot remove a locked worktree; that must not hide why the run stopped.
    (toy / "tests" / "test_calc.py").write_text(textwrap.dedent('''\
        import subprocess


        def test_add():
            subprocess.run(["git", "worktree", "lock", "."], check=True)
            assert False
    '''))
    proc = _run(toy, '''
        [[mutation]]
        name = "operands swapped"
        file = "src/toy/calc.py"
        old = "return a + b"
        new = "return b + a"
        test = "tests/test_calc.py::test_add"
    ''')
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "tests/test_calc.py::test_add fails before any mutation" in proc.stderr
    worktrees = subprocess.run(["git", "worktree", "list"], cwd=toy, env=TOY_ENV,
                               capture_output=True, text=True).stdout
    assert len(worktrees.splitlines()) == 1, worktrees


def test_a_mutation_that_hangs_is_timed_out_and_exits_1(toy):
    proc = _run(toy, '''
        [[mutation]]
        name = "add never returns"
        file = "src/toy/calc.py"
        old = "return a + b"
        new = "while True: pass"
        test = "tests/test_calc.py::test_add"
    ''', "--timeout", "3")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert _row(proc.stdout, "add never returns")[3] == "timed out"


def test_ctrl_c_kills_the_running_test_and_leaves_no_worktree_or_temp_files(toy):
    # pytest runs in its own session, so Ctrl-C reaches only the script: the script itself must
    # kill the hung test, then still remove the worktree.
    started = toy.parent / "pytest.pid"
    (toy / "tests" / "test_calc.py").write_text(textwrap.dedent(f'''\
        import os
        import time


        def test_add():
            with open({str(started)!r} + ".part", "w") as f:
                f.write(str(os.getpid()))
            os.replace({str(started)!r} + ".part", {str(started)!r})
            time.sleep(60)
    '''))
    spec = toy.parent / "spec.toml"
    spec.write_text(textwrap.dedent('''
        [[mutation]]
        name = "operands swapped"
        file = "src/toy/calc.py"
        old = "return a + b"
        new = "return b + a"
        test = "tests/test_calc.py::test_add"
    '''))
    tmp = toy.parent / "tmp"
    tmp.mkdir()
    # SIGINT back to its default, in case this run was started with it ignored (a background job).
    proc = subprocess.Popen([sys.executable, str(toy / "scripts" / "mutation_check.py"), str(spec)],
                            cwd=toy, env={**TOY_ENV, "TMPDIR": str(tmp)}, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
    pid = None
    try:
        deadline = time.monotonic() + 60
        while not started.exists():
            if proc.poll() is not None or time.monotonic() > deadline:
                proc.kill()
                pytest.fail("the hung test never started:\n" + proc.communicate()[0])
            time.sleep(0.1)
        pid = int(started.read_text())
        proc.send_signal(signal.SIGINT)
        out = proc.communicate(timeout=30)[0]
        assert proc.returncode == -signal.SIGINT, out
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)  # the hung pytest is gone
    finally:
        proc.kill()
        if pid is not None:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
    worktrees = subprocess.run(["git", "worktree", "list"], cwd=toy, env=TOY_ENV,
                               capture_output=True, text=True).stdout
    assert len(worktrees.splitlines()) == 1, worktrees
    assert list(tmp.iterdir()) == [], "the script left temp files behind"


def test_a_test_that_imports_the_main_checkout_instead_of_the_worktree_exits_3(toy):
    # A committed conftest that puts the main checkout's src/ first: the worktree's copy of
    # calc.py is never imported, so any mutation of it would "survive" for the wrong reason.
    (toy / "tests" / "conftest.py").write_text(f"import sys\nsys.path.insert(0, {str(toy / 'src')!r})\n")
    toy_git(toy, "add", ".")
    toy_git(toy, "commit", "-q", "-m", "conftest")
    proc = _run(toy, '''
        [[mutation]]
        name = "operands swapped"
        file = "src/toy/calc.py"
        old = "return a + b"
        new = "return b + a"
        test = "tests/test_calc.py::test_add"
        expect = "survived"
    ''')
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "imported the main checkout's src/toy/calc.py, not the worktree's" in proc.stderr


def test_a_line_run_only_in_a_subprocess_the_test_starts_counts_as_ran(toy):
    (toy / "tests" / "test_cli.py").write_text(textwrap.dedent('''\
        import subprocess, sys


        def test_sign_in_a_subprocess():
            code = "import sys; sys.path.insert(0, 'src'); from toy.calc import sign; sys.exit(sign(-1) != -1)"
            assert subprocess.run([sys.executable, "-c", code]).returncode == 0
    '''))
    proc = _run(toy, '''
        [[mutation]]
        name = "negative read as positive"
        file = "src/toy/calc.py"
        old = "        return -1\\n"
        new = "        return 1\\n"
        test = "tests/test_cli.py::test_sign_in_a_subprocess"
    ''')
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _row(proc.stdout, "negative read as positive")[3:5] == ["caught", "yes"]


def test_uncommitted_edits_and_deletions_reach_the_worktree_and_the_checkout_is_left_as_it_was(toy):
    calc = toy / "src" / "toy" / "calc.py"
    calc.write_text(CALC + "\n\ndef neg(x):\n    return -x\n")  # tracked, edited, not committed
    (toy / "tests" / "test_calc.py").unlink()  # tracked, deleted, not committed
    (toy / "tests" / "test_neg.py").write_text(textwrap.dedent('''\
        from pathlib import Path

        from toy.calc import neg


        def test_neg_and_the_deleted_file_is_gone():
            assert not Path("tests/test_calc.py").exists()
            assert neg(3) == -3
    '''))
    status = subprocess.run(["git", "status", "--porcelain"], cwd=toy, env=TOY_ENV,
                            capture_output=True, text=True).stdout
    before = {p: p.read_bytes() for p in toy.rglob("*") if p.is_file() and ".git" not in p.parts}
    proc = _run(toy, '''
        [[mutation]]
        name = "neg returns x"
        file = "src/toy/calc.py"
        old = "return -x"
        new = "return x"
        test = "tests/test_neg.py::test_neg_and_the_deleted_file_is_gone"
    ''')
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _row(proc.stdout, "neg returns x")[3:5] == ["caught", "yes"]
    after = {p: p.read_bytes() for p in toy.rglob("*") if p.is_file() and ".git" not in p.parts}
    assert after == before
    assert subprocess.run(["git", "status", "--porcelain"], cwd=toy, env=TOY_ENV,
                          capture_output=True, text=True).stdout == status
    worktrees = subprocess.run(["git", "worktree", "list"], cwd=toy, env=TOY_ENV,
                               capture_output=True, text=True).stdout
    assert len(worktrees.splitlines()) == 1, worktrees


def test_a_realdata_test_reads_the_gitignored_data_linked_in_and_its_break_is_caught(toy):
    # Ticket 10's realdata break, in miniature: the test reads a gitignored file under
    # examples/images and skips where it is missing, as it did in the script's worktree.
    (toy / ".gitignore").write_text("__pycache__/\nexamples/images/sum.txt\nexamples/outputs/\n")
    (toy / "examples" / "images").mkdir(parents=True)
    (toy / "examples" / "images" / "README").write_text("tracked\n")
    (toy / "pyproject.toml").write_text('[tool.pytest.ini_options]\npythonpath = ["src"]\n'
                                        'markers = ["realdata: reads gitignored data"]\n')
    (toy / "tests" / "test_data.py").write_text(realdata_test("toy"))
    toy_git(toy, "add", ".")
    toy_git(toy, "commit", "-q", "-m", "data test")
    data = toy / "examples" / "images" / "sum.txt"
    data.write_text("5\n")
    stamp = data.stat().st_mtime_ns
    proc = _run(toy, '''
        [[mutation]]
        name = "add subtracts"
        file = "src/toy/calc.py"
        old = "return a + b"
        new = "return a - b"
        test = "tests/test_data.py::test_add_matches_the_data"
    ''')
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _row(proc.stdout, "add subtracts")[3:5] == ["caught", "yes"]
    assert (data.read_text(), data.stat().st_mtime_ns) == ("5\n", stamp)


def test_data_written_through_a_link_stops_the_run(toy):
    # A symlink cannot be read-only, so a write through it must at least be caught and named.
    (toy / ".gitignore").write_text("__pycache__/\nexamples/outputs/\n")
    (toy / "examples" / "outputs").mkdir(parents=True)
    (toy / "examples" / "outputs" / "old.txt").write_text("old\n")
    (toy / "tests" / "test_calc.py").write_text(textwrap.dedent('''\
        from pathlib import Path

        from toy.calc import add


        def test_add():
            Path("examples/outputs/new.txt").write_text("written by a test\\n")
            assert add(2, 3) == 5
    '''))
    toy_git(toy, "add", ".")
    toy_git(toy, "commit", "-q", "-m", "a test that writes data")
    proc = _run(toy, '''
        [[mutation]]
        name = "operands swapped"
        file = "src/toy/calc.py"
        old = "return a + b"
        new = "return a - b"
        test = "tests/test_calc.py::test_add"
    ''')
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "changed the linked data in the main checkout:" in proc.stderr
    assert "examples/outputs/new.txt" in proc.stderr
    assert (toy / "examples" / "outputs" / "new.txt").exists()  # caught, not prevented


GOOD = {"name": "n", "file": "src/m.py", "old": "x = 1", "new": "x = 2", "test": "tests/test_m.py::test_x"}


def _spec_file(tmp_path, *entries, extra=""):
    def toml(value):
        return '"""' + value + '"""' if isinstance(value, str) else str(value)
    body = "".join("[[mutation]]\n" + "".join(f"{k} = {toml(v)}\n" for k, v in e.items()) + "\n" for e in entries)
    path = tmp_path / "spec.toml"
    path.write_text(extra + body)
    return path


@pytest.fixture
def plain(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "src" / "m.py").write_text("x = 1\ny = 1\ny = 1\n")
    (repo / "src" / "notes.txt").write_text("x = 1\n")
    (repo / "tests" / "test_m.py").write_text("x = 1\n")
    return repo


def test_load_spec_reads_each_mutation_with_caught_as_the_default_expect(tmp_path, plain):
    spec = _spec_file(tmp_path, GOOD, {**GOOD, "name": "m", "expect": "survived"})
    assert mc.load_spec(spec, plain) == [mc.Mutation(**GOOD), mc.Mutation(**{**GOOD, "name": "m"}, expect="survived")]


def test_load_spec_writes_each_file_the_plain_way_coverage_reports_it(tmp_path, plain):
    # Coverage reports src/m.py, so ./src/m.py kept as written would read as "never ran".
    spec = _spec_file(tmp_path, {**GOOD, "file": "./src//m.py"})
    assert [m.file for m in mc.load_spec(spec, plain)] == ["src/m.py"]


def test_load_spec_keeps_only_the_named_mutations_in_spec_order(tmp_path, plain):
    spec = _spec_file(tmp_path, GOOD, {**GOOD, "name": "m"}, {**GOOD, "name": "k"})
    assert [m.name for m in mc.load_spec(spec, plain, only=["k", "n"])] == ["n", "k"]


@pytest.mark.parametrize("entries, extra, message", [
    ([], "", "no [[mutation]] tables"),
    ([], "mutation = ['x']\n", "mutation 1 is not a table"),
    ([GOOD], "title = 'x'\n", "unknown top-level key(s): title"),
    ([{k: v for k, v in GOOD.items() if k != "test"}], "", "mutation 1 (n): missing test"),
    ([{**GOOD, "tset": "t"}], "", "mutation 1 (n): unknown key(s): tset"),
    ([{**GOOD, "old": 3}], "", "mutation 1 (n): old must be a string"),
    ([{**GOOD, "old": ""}], "", "mutation 1 (n): old is empty"),
    ([{**GOOD, "new": "x = 1"}], "", "mutation 1 (n): new is the same as old"),
    ([{**GOOD, "expect": "fail"}], "", "mutation 1 (n): expect must be 'caught' or 'survived', not 'fail'"),
    ([GOOD, GOOD], "", "mutation 2 (n): name already used"),
    ([{**GOOD, "file": "src/gone.py"}], "", "mutation 1 (n): src/gone.py is not a file in the repo"),
    ([{**GOOD, "file": "/abs/m.py"}], "", "mutation 1 (n): file must be a path inside the repo"),
    ([{**GOOD, "file": "src/../../m.py"}], "", "mutation 1 (n): file must be a path inside the repo"),
    ([{**GOOD, "file": ""}], "", "mutation 1 (n): file must be a path inside the repo"),
    ([{**GOOD, "file": "."}], "", "mutation 1 (n): file must be a path inside the repo"),
    ([{**GOOD, "file": "tests/test_m.py"}], "", "mutation 1 (n): tests/test_m.py is under tests/"),
    ([{**GOOD, "file": "src/notes.txt"}], "", "mutation 1 (n): src/notes.txt is not a .py file"),
    ([{**GOOD, "old": "z = 9"}], "", "mutation 1 (n): old appears 0 times in src/m.py, not once"),
    ([{**GOOD, "old": "y = 1"}], "", "mutation 1 (n): old appears 2 times in src/m.py (lines 2, 3), not once"),
    ([{**GOOD, "old": "x = 1 "}], "",
     "mutation 1 (n): old appears 0 times in src/m.py, not once; it ends with whitespace ' ', "
     "and without that whitespace it appears 1 time(s)"),
    ([{**GOOD, "old": "\t x = 1"}], "",
     "mutation 1 (n): old appears 0 times in src/m.py, not once; it starts with whitespace "
     "'\\t ', and without that whitespace it appears 1 time(s)"),
])
def test_load_spec_refuses_a_spec_it_cannot_use_as_written(tmp_path, plain, entries, extra, message):
    with pytest.raises(mc.SpecError) as error:
        mc.load_spec(_spec_file(tmp_path, *entries, extra=extra), plain)
    assert str(error.value) == message


def test_load_spec_refuses_an_only_name_not_in_the_spec(tmp_path, plain):
    with pytest.raises(mc.SpecError, match=r"^--only name\(s\) not in the spec: nope$"):
        mc.load_spec(_spec_file(tmp_path, GOOD), plain, only=["n", "nope"])


def test_load_spec_refuses_a_file_that_is_not_toml(tmp_path, plain):
    path = tmp_path / "spec.toml"
    path.write_text("[[mutation]\n")
    with pytest.raises(mc.SpecError, match=r"^cannot read .*spec\.toml: "):
        mc.load_spec(path, plain)


def test_a_spec_refusal_exits_2_before_any_worktree_is_made(toy):
    proc = _run(toy, '''
        [[mutation]]
        name = "not there"
        file = "src/toy/calc.py"
        old = "return a - b"
        new = "return b + a"
        test = "tests/test_calc.py::test_add"
    ''')
    assert proc.returncode == 2
    assert proc.stdout == ""
    assert "mutation 1 (not there): old appears 0 times in src/toy/calc.py, not once" in proc.stderr


MULTI = "x = f(\n    1,\n    2,\n)\n"
BRANCH = "if a:\n    b = 1\n# note\nc = 2\n"
DECORATED = "@dec\ndef f():\n    pass\n"
HANDLER = "try:\n    a()\nexcept E:\n    b()\n"


@pytest.mark.parametrize("source, old, executed, ran", [
    (BRANCH, "c = 2", {4}, True),
    (BRANCH, "c = 2", {1, 2}, False),
    (MULTI, "    2,\n", {1}, True),  # mid-statement span: the statement's line ran
    (MULTI, "    2,\n", set(), False),
    (BRANCH, "    b = 1", {1, 4}, False),  # the if line ran, its body did not
    (BRANCH, "if a:", {1}, True),
    (BRANCH, "# note", {1, 2, 4}, False),  # a comment is no statement, so nothing ran it
    (DECORATED, "@dec", {1, 2}, True),
    (HANDLER, "except E:", {1, 2, 3, 4}, True),
    (HANDLER, "except E:", {1, 2}, False),
])
def test_ran_line_counts_a_statement_overlapping_the_old_text_whose_line_ran(source, old, executed, ran):
    assert mc.ran_line(source, old, old.replace("1", "9").replace("2", "8") + "  # m", executed) is ran


@pytest.mark.parametrize("old, new, executed, ran", [
    # Ticket 12's miss: the unchanged `if` line ran, the changed body line did not.
    ("if a:\n    b = 1\n", "if a:\n    b = 9\n", {1, 4}, False),
    ("if a:\n    b = 1\n", "if a:\n    b = 9\n", {1, 2}, True),
    ("if a:\n    b = 1\n", "", {1}, True),  # deleted outright: every old line is changed
    ("    b = 1\n", "    b = 1\n    b = 9\n", {2}, True),  # pure insertion: judged by the line before it
    ("    b = 1\n", "    b = 1\n    b = 9\n", {1, 4}, False),
])
def test_ran_line_judges_only_the_lines_new_changes_not_the_unchanged_context(old, new, executed, ran):
    assert mc.ran_line(BRANCH, old, new, executed) is ran


def test_a_timeout_that_is_not_a_positive_number_is_a_usage_error():
    with pytest.raises(SystemExit) as error:
        mc.main(["spec.toml", "--timeout", "0"])
    assert error.value.code == 2


def test_a_repo_with_no_commit_to_build_a_worktree_from_exits_3(tmp_path):
    repo = tmp_path / "bare"
    (repo / "scripts").mkdir(parents=True)
    (repo / "src").mkdir()
    shutil.copy(SCRIPT, repo / "scripts" / "mutation_check.py")
    shutil.copy(HELPER, repo / "scripts" / "_worktree.py")
    (repo / "src" / "m.py").write_text("x = 1\n")
    toy_git(repo, "init", "-q")
    proc = _run(repo, '''
        [[mutation]]
        name = "n"
        file = "src/m.py"
        old = "x = 1"
        new = "x = 2"
        test = "tests/test_m.py::test_x"
    ''')
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "cannot run: git worktree add" in proc.stderr


def test_load_spec_reports_every_error_in_the_spec_at_once(tmp_path, plain):
    spec = _spec_file(tmp_path, {**GOOD, "old": "z = 9", "expect": "fail"},
                      {k: v for k, v in GOOD.items() if k != "test"} | {"name": "m"},
                      {**GOOD, "name": "k", "file": "tests/test_m.py"}, extra="title = 'x'\n")
    with pytest.raises(mc.SpecError) as error:
        mc.load_spec(spec, plain, only=["nope"])
    assert str(error.value).splitlines() == [
        "unknown top-level key(s): title",
        "mutation 1 (n): expect must be 'caught' or 'survived', not 'fail'",
        "mutation 1 (n): old appears 0 times in src/m.py, not once",
        "mutation 2 (m): missing test",
        "mutation 3 (k): tests/test_m.py is under tests/",
        "--only name(s) not in the spec: nope",
    ]


def test_every_spec_error_reaches_stderr_in_one_run(toy):
    proc = _run(toy, '''
        [[mutation]]
        name = "a"
        file = "src/toy/calc.py"
        old = "return a - b"
        new = "return b + a"
        test = "tests/test_calc.py::test_add"

        [[mutation]]
        name = "a"
        file = "src/toy/gone.py"
        old = "x"
        new = "y"
        test = "tests/test_calc.py::test_add"
    ''')
    assert proc.returncode == 2
    assert proc.stderr.splitlines() == [
        "3 spec error(s):",
        "mutation 1 (a): old appears 0 times in src/toy/calc.py, not once",
        "mutation 2 (a): name already used",
        "mutation 2 (a): src/toy/gone.py is not a file in the repo",
    ]


def test_a_bare_assertion_error_shows_its_detail_and_the_test_line_it_failed_on(toy):
    # numpy's assert_allclose raises like this: the message starts with blank lines.
    (toy / "tests" / "test_close.py").write_text(textwrap.dedent('''\
        from toy.calc import add


        def check(got):
            if got != 5:
                raise AssertionError("\\n\\nNot equal to tolerance rtol=1e-07\\nmore")


        def test_add_close():
            check(add(2, 3))
    '''))
    proc = _run(toy, '''
        [[mutation]]
        name = "minus"
        file = "src/toy/calc.py"
        old = "return a + b"
        new = "return a - b"
        test = "tests/test_close.py::test_add_close"
    ''')
    assert proc.returncode == 0, proc.stdout + proc.stderr
    name, test, expected, result, ran, failure = _row(proc.stdout, "minus")
    assert result == "caught"
    # The last frame in the test's own file: the helper's raise, not the call in the test.
    assert failure == "tests/test_close.py:6: AssertionError: Not equal to tolerance rtol=1e-07"


def test_a_catch_by_an_error_other_than_an_assert_is_marked(toy):
    proc = _run(toy, '''
        [[mutation]]
        name = "add returns None"
        file = "src/toy/calc.py"
        old = "return a + b"
        new = "return None + b"
        test = "tests/test_calc.py::test_add"
    ''')
    assert proc.returncode == 0, proc.stdout + proc.stderr  # still a catch, as the spec expected
    name, test, expected, result, ran, failure = _row(proc.stdout, "add returns None")
    assert result == "caught (TypeError, not an assert)"
    assert failure.startswith("tests/test_calc.py:5: TypeError: unsupported operand")
    assert proc.stdout.splitlines()[-1] == (
        "1 mutation(s): 1 as expected, 0 not; 0 never ran the mutated line; 0 error(s); "
        "1 caught by an error other than an assert.")


@pytest.mark.parametrize("out, kind", [
    ("E   assert 1 == -1", "AssertionError"),
    ("E   AssertionError: x", "AssertionError"),
    ("E   Failed: DID NOT RAISE <class 'ValueError'>", "Failed"),
    ("E   TypeError: x", "TypeError"),
    ("E   numpy.linalg.LinAlgError", "numpy.linalg.LinAlgError"),
    ("1 failed in 0.1s", None),
])
def test_the_error_that_failed_a_test_is_read_from_its_first_E_line(out, kind):
    assert mc.failure(out, "tests/test_m.py::test_x")[1] == kind


def test_help_says_the_main_checkout_is_never_edited():
    proc = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, env=TOY_ENV)
    assert proc.returncode == 0
    assert "never edits the main checkout" in " ".join(proc.stdout.split())
