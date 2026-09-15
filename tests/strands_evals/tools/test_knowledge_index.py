import pytest

from strands_evals.tools.knowledge_index import KnowledgeEntry, KnowledgeIndex


@pytest.fixture
def catalog():
    return {
        "issue_refund": "Refund a payment. args: order_id, amount. Refuses amounts over $500.",
        "lookup_order": "Look up an order by id. args: order_id. Returns status and total.",
        "escalate": "Escalate a case to a human agent. args: case_id, reason.",
    }


def test_overview_is_compact_and_sorted(catalog):
    index = KnowledgeIndex(catalog)
    overview = index.overview()
    lines = overview.splitlines()

    assert "3 entries" in lines[0]
    # Keys appear in sorted order, one line each, bracketed.
    entry_lines = [ln for ln in lines if ln.startswith("[")]
    assert [ln.split("]")[0][1:] for ln in entry_lines] == ["escalate", "issue_refund", "lookup_order"]
    # Overview is far smaller than the full corpus.
    assert len(overview) < sum(len(v) for v in catalog.values()) + 500


def test_description_shown_in_overview():
    index = KnowledgeIndex([KnowledgeEntry(key="k1", content="body", description="a short summary")])
    assert "a short summary" in index.overview()


def test_get_entry_returns_full_content(catalog):
    index = KnowledgeIndex(catalog)
    _, tools = index.for_judge()
    get_entry = _tool(tools, "get_entry")
    assert catalog["issue_refund"] in get_entry(key="issue_refund")


def test_get_entry_unknown_key_is_error(catalog):
    index = KnowledgeIndex(catalog)
    get_entry = _tool(index.tools, "get_entry")
    out = get_entry(key="nope")
    assert "ERROR" in out and "nope" in out


def test_get_entry_windows_oversized_content():
    big = "A" * 25_000
    index = KnowledgeIndex({"big": big}, max_read_chars=8_000)
    get_entry = _tool(index.tools, "get_entry")
    page1 = get_entry(key="big")
    assert "TRUNCATED" in page1
    assert len(page1) < 8_200
    # Paging via the reported offset continues the content.
    page2 = get_entry(key="big", offset=8_000)
    assert page2.startswith("A")


def test_search_literal_finds_entry(catalog):
    index = KnowledgeIndex(catalog)
    search = _tool(index.tools, "search_entries")
    out = search(pattern="$500")
    assert "issue_refund" in out
    assert "lookup_order" not in out


def test_search_is_case_insensitive(catalog):
    index = KnowledgeIndex(catalog)
    search = _tool(index.tools, "search_entries")
    assert "escalate" in search(pattern="ESCALATE")


def test_search_regex_anchors_match_line_boundaries():
    index = KnowledgeIndex({"doc": "first line\nSTATUS: ok\nlast line"})
    search = _tool(index.tools, "search_entries")
    assert "doc" in search(pattern=r"^STATUS:", is_regex=True)


def test_search_invalid_regex_is_reported():
    index = KnowledgeIndex({"doc": "text"})
    search = _tool(index.tools, "search_entries")
    out = search(pattern="[", is_regex=True)
    assert "ERROR" in out and "regex" in out


def test_search_no_match(catalog):
    index = KnowledgeIndex(catalog)
    search = _tool(index.tools, "search_entries")
    assert "No matches" in search(pattern="zzz-not-present")


def test_search_bounded_by_max_read_chars():
    entries = {f"k{i:04d}": f"needle body {i}" for i in range(800)}
    index = KnowledgeIndex(entries, max_read_chars=2_000)
    search = _tool(index.tools, "search_entries")
    out = search(pattern="needle", max_matches=800)
    assert len(out) < 2_300
    assert "budget reached" in out


def test_search_signals_max_matches():
    entries = {f"k{i}": "needle" for i in range(10)}
    index = KnowledgeIndex(entries, max_read_chars=100_000)
    search = _tool(index.tools, "search_entries")
    out = search(pattern="needle", max_matches=3)
    assert "stopped at 3 entries" in out


def test_render_injects_selected_entries(catalog):
    index = KnowledgeIndex(catalog)
    block = index.render(keys=["issue_refund", "escalate"])
    assert block.startswith("<Knowledge>") and block.endswith("</Knowledge>")
    assert catalog["issue_refund"] in block
    assert catalog["escalate"] in block
    # Unselected entry is not injected.
    assert catalog["lookup_order"] not in block


def test_render_reports_unknown_key(catalog):
    index = KnowledgeIndex(catalog)
    block = index.render(keys=["issue_refund", "ghost"])
    assert "ghost" in block and "no such knowledge entry" in block
    assert catalog["issue_refund"] in block


def test_render_dedupes_keys(catalog):
    index = KnowledgeIndex(catalog)
    block = index.render(keys=["issue_refund", "issue_refund"])
    assert block.count("[issue_refund]") == 1


def test_render_bounds_by_max_chars():
    entries = {f"k{i}": "B" * 5_000 for i in range(10)}
    index = KnowledgeIndex(entries, max_read_chars=8_000)
    block = index.render(keys=list(entries))
    assert "TRUNCATED" in block
    assert len(block) < 8_300


def test_for_judge_returns_overview_and_tools(catalog):
    index = KnowledgeIndex(catalog)
    section, tools = index.for_judge()
    assert section.startswith("<KnowledgeOverview>")
    assert "3 entries" in section
    assert {t.tool_name for t in tools} == {"list_entries", "get_entry", "search_entries"}


def test_accepts_reference_entry_iterable():
    index = KnowledgeIndex(
        [
            KnowledgeEntry(key="a", content="alpha"),
            KnowledgeEntry(key="b", content="beta"),
        ]
    )
    assert index.keys == ["a", "b"]


def test_accepts_pairs():
    index = KnowledgeIndex([("a", "alpha"), ("b", "beta")])
    assert index.keys == ["a", "b"]


def test_duplicate_key_raises():
    with pytest.raises(ValueError, match="duplicate knowledge key"):
        KnowledgeIndex([("a", "1"), ("a", "2")])


def test_invalid_max_read_chars_raises(catalog):
    with pytest.raises(ValueError, match="max_read_chars"):
        KnowledgeIndex(catalog, max_read_chars=0)


def test_overview_paging():
    entries = {f"k{i:03d}": "x" * 500 for i in range(50)}
    index = KnowledgeIndex(entries, max_read_chars=2_000)
    page1 = index.overview()
    assert "MORE" in page1
    # The MORE marker reports the next offset; paging from it makes progress.
    next_offset = int(page1.split("offset=")[1].split("]")[0])
    assert next_offset > 0
    page2 = index.overview(offset=next_offset)
    assert "Showing entries" in page2


def test_overview_offset_out_of_range(catalog):
    index = KnowledgeIndex(catalog)
    assert "ERROR" in index.overview(offset=99)


def _tool(tools, name):
    """Return the @tool-decorated callable named `name` (the tool objects are callable)."""
    for t in tools:
        if t.tool_name == name:
            return t
    raise KeyError(name)
