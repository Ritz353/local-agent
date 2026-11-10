import agent


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
    result = agent.search_notes("two's complement")
    assert "Lecture 05" in result
    assert "junk" not in result  # notes inside .venv must be skipped


def test_search_no_match(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "VAULT", make_vault(tmp_path))
    assert agent.search_notes("quantum chromodynamics") == "No notes matched those keywords."


def test_read_note(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "VAULT", make_vault(tmp_path))
    assert "inverting bits" in agent.read_note("lecture 05")
    assert "No note named" in agent.read_note("Does Not Exist")