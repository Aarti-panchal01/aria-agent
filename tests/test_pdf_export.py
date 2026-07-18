"""Tests for PDF export of research reports."""

from report.pdf_exporter import export_to_pdf

_SAMPLE = """# Research Report

## Executive Summary

Redis and Memcached are both **in-memory** stores with different trade-offs.

## Comparison Table

| Dimension | Redis | Memcached |
| --- | --- | --- |
| Data types | Rich | Strings only |
| Persistence | Yes | No |

## Key Conclusions

- Redis is more feature-rich
- Memcached is simpler and very fast

## Key Takeaway

Pick based on whether you need persistence and data structures.
"""


def test_pdf_export_produces_valid_pdf():
    out = export_to_pdf(_SAMPLE, "test-session-id", "Compare Redis vs Memcached")
    assert isinstance(out, bytes)
    assert out[:4] == b"%PDF"
    assert len(out) > 1000


def test_pdf_export_handles_empty_report():
    out = export_to_pdf("", "sid", "some goal")
    assert isinstance(out, bytes)
    assert out[:4] == b"%PDF"
