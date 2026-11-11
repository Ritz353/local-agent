import agent
import notes_search
import homelab


def make_vault(tmp_path):
    """Build a tiny fake vault so tests never touch your real notes."""
    (tmp_path / "Class Notes").mkdir()
    (tmp_path / "Class Notes" / "Lecture 05.md").write_text(
        "Two's complement represents negative numbers by inverting bits and adding one.",
        encoding="utf-8",
    )
    (tmp_path / "Reading List.md").write_text("ReAct paper, Toolformer paper", encoding="utf-8")
    hidden = tmp_path / ".venv"
    hidden.mkdir()
    (hidden / "junk.md").write_text("complement complement complement", encoding="utf-8")
    return tmp_path


def test_multiply():
    assert agent.multiply(12, 34) == 408
    assert agent.multiply("2.5", "4") == 10


def test_search_finds_right_note(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "VAULT", make_vault(tmp_path))
    result = agent.keyword_search("two's complement")
    assert "Lecture 05" in result
    assert "junk" not in result  # notes inside .venv must be skipped


def test_search_no_match(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "VAULT", make_vault(tmp_path))
    assert agent.keyword_search("quantum chromodynamics") == "No notes matched those keywords."


def test_read_note(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "VAULT", make_vault(tmp_path))
    assert "inverting bits" in agent.read_note("lecture 05")
    assert "No note named" in agent.read_note("Does Not Exist")

def test_search_notes_keyword_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "VAULT", make_vault(tmp_path))
    monkeypatch.setattr(agent, "SEARCH_MODE", "keyword")
    assert "Lecture 05" in agent.search_notes("two's complement")


def test_chunk_text_overlaps_and_covers_everything():
    text = "\n".join(f"line {i} " + "x" * 40 for i in range(100))
    chunks = notes_search.chunk_text(text, size=500, overlap=100)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)
    assert "line 0" in chunks[0] and "line 99" in chunks[-1]


def test_cosine():
    assert abs(notes_search.cosine([1, 0], [1, 0]) - 1.0) < 1e-9
    assert abs(notes_search.cosine([1, 0], [0, 1])) < 1e-9
    assert notes_search.cosine([0, 0], [1, 1]) == 0.0


def test_check_homelab_reports_up_and_down(monkeypatch):
    monkeypatch.setattr(homelab, "load_config", lambda: {"ssh_host": "x", "services": {"A": "http://a", "B": "http://b"}})
    monkeypatch.setattr(homelab, "probe", lambda url, timeout=5: (True, "HTTP 200", 12) if url == "http://a"
                        else (False, "unreachable: timeout", None))
    result = homelab.check_homelab()
    assert result.startswith("1/2 services up")
    assert "UP: A (HTTP 200, 12 ms)" in result
    assert "DOWN: B (unreachable: timeout)" in result


def test_backup_status_handles_ssh_failure(monkeypatch):
    class Failed:
        returncode, stdout, stderr = 255, "", "Permission denied (publickey)."
    monkeypatch.setattr(homelab, "load_config", lambda: {"ssh_host": "x", "services": {}})
    monkeypatch.setattr(homelab.subprocess, "run", lambda *a, **k: Failed())
    assert "Permission denied" in homelab.get_backup_status()
