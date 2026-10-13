"""scripts/wayfinder_progress.py: a local wayfinder map's frontier and progress page."""
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("wayfinder_progress", REPO / "scripts" / "wayfinder_progress.py")
wp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wp)


def _map(tmp_path):
    (tmp_path / "issues").mkdir()
    (tmp_path / "map.md").write_text("# Map: demo\n\n## Destination\n\nA spec.\n\n## Decisions so far\n\n## Not yet specified\n")
    tickets = {
        "01-first.md": ("First", "resolved", "none"),
        "02-second.md": ("Second", "open", "01"),
        "03-third.md": ("Third", "open", "02"),
        "04-fourth.md": ("Fourth", "claimed", "01"),
    }
    for name, (title, status, blocked) in tickets.items():
        (tmp_path / "issues" / name).write_text(
            f"# {name[:2]}: {title}\n\nType: grilling (HITL)\nStatus: {status}\nBlocked by: {blocked}\n\n## Question\n\nWhy?\n")
    return tmp_path


def test_frontier_lists_open_tickets_whose_blockers_are_resolved(tmp_path):
    tickets, _ = wp.load_tickets(_map(tmp_path))
    assert wp.frontier_lines(tickets) == [
        "02 Second (grilling (HITL)) issues/02-second.md",
        "claimed: 04 Fourth issues/04-fourth.md",
    ]


def test_frontier_says_so_when_empty(tmp_path):
    here = _map(tmp_path)
    (here / "issues" / "02-second.md").unlink()
    tickets, _ = wp.load_tickets(here)
    assert wp.frontier_lines(tickets)[0] == "no ticket on the frontier"


def test_the_page_is_written_without_sessions_or_inventory(tmp_path, capsys):
    here = _map(tmp_path)
    assert wp.main([str(here)]) == 0
    page = (here / "progress.html").read_text()
    assert "1 / 4" in page and "Inventory headline" not in page
    assert "progress.html" in capsys.readouterr().out
