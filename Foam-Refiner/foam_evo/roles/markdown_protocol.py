"""The deliberately tiny control protocol used by Foam-Refiner roles.

Role prose is unconstrained Markdown.  Only a small section is interpreted by
the engine, using a small line-oriented control protocol.  Missing or
ambiguous control text is never guessed as approval.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


class MarkdownProtocolError(ValueError):
    """A role document cannot be used to make a deterministic routing choice."""


AMBIGUOUS = "AMBIGUOUS"


def section_text(markdown: str, heading: str) -> str | None:
    """Return text below an H2 control heading, or ``None`` if it is absent.

    Foam-Refiner accepts harmless heading annotations such as ``## Verdict (final)``.
    The control heading is still restricted to H2 so ordinary prose headings
    cannot become routing input.
    """

    in_section = False
    lines: list[str] = []
    title = heading.removeprefix("##").strip().lower()
    for raw in markdown.splitlines():
        line = raw.rstrip()
        if in_section:
            if line.strip().startswith("## "):
                break
            lines.append(line)
        else:
            candidate = line.strip()
            if candidate.startswith("## "):
                candidate_title = candidate[3:].strip()
                # Match the requested H2 title as a complete title, while
                # retaining parenthesized/colon annotations.
                # Prefix matching would incorrectly treat headings such as
                # ``## Verdict Justification`` as the control section.
                if candidate_title.lower() == title or re.match(
                    rf"^{re.escape(title)}\s*[(:-]", candidate_title, re.IGNORECASE
                ):
                    in_section = True
    return "\n".join(lines) if in_section else None


def scan_token(markdown: str, allowed: Iterable[str], *, heading: str = "## Verdict") -> str | None:
    """Scan a closed token under *heading*; with a bounded tail fallback.

    The explicit section wins.  When no section exists, only the final 2 KiB is
    considered.  A token must be present as a word (not a substring of a prose
    sentence), and any line containing two different candidates is ambiguous.
    """

    candidates = tuple(str(item).upper() for item in allowed)
    section = section_text(markdown, heading)
    if section is not None:
        # A role may put a label before the token (``Decision: APPROVE``), or
        # add one explanatory line before the final bare token.  Search all
        # non-empty lines in the bounded control section, but require a line to
        # contain exactly one candidate.  This preserves deterministic routing
        # without parsing arbitrary Markdown prose.
        lines = [line.strip() for line in section.splitlines() if line.strip()]
    else:
        # Search each line in the bounded tail when no control section exists;
        # never inspect unbounded role prose.
        lines = [line.strip() for line in markdown[-2000:].splitlines() if line.strip()]
    exact: list[str] = []
    loose: list[str] = []
    for line in lines:
        if not line:
            continue
        upper = line.upper()
        if "APPROVE" in upper and ("REJECT" in upper or "REVISE" in upper):
            return None
        found = [token for token in candidates if re.search(rf"(?<![A-Z0-9_]){re.escape(token)}(?![A-Z0-9_])", upper)]
        if len(found) == 1:
            (exact if _is_control_line(line, found[0]) else loose).append(found[0])
        if len(found) > 1:
            return None
        # Accept a bare REJECT fallback for reviewer prose.  Keep this
        # deterministic but never infer an approval from unrelated text.
        if "REJECT" in upper and "APPROVE" not in upper and "REVISE" not in upper:
            for token in candidates:
                if token.startswith("REJECT"):
                    loose.append(token)
        if "APPROVE" in upper and "REJECT" not in upper and "REVISE" not in upper:
            for token in candidates:
                if token == "APPROVE":
                    loose.append(token)
        if "REVISE" in upper and "APPROVE" not in upper and "REJECT" not in upper:
            for token in candidates:
                if token == "REVISE":
                    loose.append(token)
    values = exact or loose
    unique = set(values)
    return next(iter(unique)) if len(unique) == 1 else None


def _is_control_line(line: str, token: str) -> bool:
    """Recognize a bare token or a conventional label around the token."""

    escaped = re.escape(token)
    return bool(
        re.fullmatch(rf"[`*_#>\-\s]*(?:verdict|decision|status|outcome)?\s*[:\-]?\s*{escaped}[.!`*_\s]*", line, re.IGNORECASE)
    )


def parse_plan_exit(markdown: str) -> str | None:
    if section_text(markdown, "## Exit") is None:
        return None
    return scan_token(
        markdown,
        ("CODE_ONLY", "KNOWLEDGE_ONLY", "CODE_AND_KNOWLEDGE", "NO_EVALUATION"),
        heading="## Exit",
    )


def parse_plan_reviewer_verdict(markdown: str) -> str:
    return scan_token(
        markdown,
        ("REJECT_ARCHITECTURAL", "REJECT_COMPLETENESS", "APPROVE"),
    ) or AMBIGUOUS


def parse_code_reviewer_verdict(markdown: str) -> str:
    return scan_token(markdown, ("REJECT_IMPL", "APPROVE")) or AMBIGUOUS


def parse_iteration_verdict(markdown: str) -> str:
    return scan_token(
        markdown,
        (
            "FUNDAMENTAL_LIMIT_ARCHITECTURE",
            "FUNDAMENTAL_LIMIT_MODEL",
            "NEED_MORE_WORK",
            "CONVERGED",
        ),
    ) or AMBIGUOUS


def parse_cluster_cases(markdown: str) -> list[tuple[str, list[str]]]:
    """Read the minimal cluster protocol from a complete Markdown document.

    The canonical form is ``Cases: a, b``.  A short list form (``Cases:``
    followed by ``- a`` lines) is accepted as well because it is common in
    otherwise valid Markdown, while prose and tables remain uninterpreted.
    """

    clusters: list[tuple[str, list[str]]] = []
    current: str | None = None
    cases: list[str] = []
    collecting_list = False
    for raw in markdown.splitlines():
        line = raw.strip()
        match = re.fullmatch(r"##\s+Cluster\s+([^\s]+)", line, flags=re.IGNORECASE)
        if match:
            if current is not None:
                clusters.append((current, cases))
            current = match.group(1)
            cases = []
            collecting_list = False
            continue
        if current is not None:
            # Accept the harmless emphasis commonly emitted by Markdown
            # writers (``**Cases:**`` / ``_Case:_``), but keep the field name
            # and delimiter deterministic.
            match = re.match(r"[*_`\s]*Cases?[*_`\s]*\s*:\s*(.*)$", line, flags=re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                collecting_list = not value
                if value:
                    cases.extend(_case_tokens(value))
                continue
            if collecting_list:
                bullet = re.fullmatch(r"[-*+]\s+(.+?)\s*", line)
                if bullet:
                    cases.extend(_case_tokens(bullet.group(1)))
                    continue
                if line.startswith("## "):
                    collecting_list = False
    if current is not None:
        clusters.append((current, cases))
    return clusters


def _case_tokens(value: str) -> list[str]:
    return [
        token
        for token in (item.strip().strip("`* ") for item in re.split(r",", value))
        if token
    ]


def require_token(markdown: str, allowed: Iterable[str], *, heading: str = "## Verdict") -> str:
    token = scan_token(markdown, allowed, heading=heading)
    if token is None:
        raise MarkdownProtocolError(f"missing or ambiguous {heading} control token")
    return token


def read_markdown(path: str | Path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise MarkdownProtocolError(f"cannot read role artifact: {path}") from exc
