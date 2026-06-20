"""Cover report_utils lines 57-58 — third-level _safe_write_line fallback."""
from unittest.mock import MagicMock
from services import report_utils

def test_safe_write_line_triple_failure():
    """All 3 levels fail: multi_cell → multi_cell(safe) → cell([line skipped]) → pass."""
    pdf = MagicMock()
    # First multi_cell fails, second multi_cell fails, then cell also fails
    pdf.multi_cell.side_effect = [RuntimeError("level1"), RuntimeError("level2")]
    pdf.cell.side_effect = RuntimeError("level3")
    # Should not raise
    report_utils._safe_write_line(pdf, "x" * 500)
    assert pdf.multi_cell.call_count == 2
    assert pdf.cell.call_count == 1
