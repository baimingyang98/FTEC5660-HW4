"""FTEC5660 Homework 4: the retrieval half of a RAG assistant over company reports.

Fill in the five functions marked ### YOUR CODE HERE. You may add imports, constants and helper
functions above them, but do not change the provided code below them (the models, the generator,
the loaders, the runner and the scoring).

    python3 hw4.py --data-folder public_test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
import traceback
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek

load_dotenv()


# =============================================================================
# Your code: add imports, constants and helper functions here.
# =============================================================================

from bs4 import BeautifulSoup, NavigableString, Tag
from rank_bm25 import BM25Okapi

# Switches for the ablation study (ablation.py turns them off one at a time). The defaults are the
# submitted pipeline; with every switch off, the pipeline is the naive baseline B0.
SETTINGS: dict[str, Any] = {
    "table_text": True,       # tables through table_to_text (off: BeautifulSoup's get_text, a stream of cells)
    "section_chunks": True,   # chunks follow the headings (off: RecursiveCharacterTextSplitter over the text)
    "chunk_header": True,     # each chunk starts with "company | document | section"
    "tokenizer": "stemmed",   # "stemmed": words without punctuation, light stemming; "whitespace": text.lower().split()
    "query_words": True,      # BM25 query: content words, synonyms, abbreviations (off: the whole question)
    "metadata_filter": True,  # parse_query, then only chunks of the companies, years and document type asked about
    "per_company": True,      # a comparison takes the best chunks of each company in turn
    "table_boost": 1.3,       # BM25 factor for chunks with a table when the question asks for a figure (1.0: off)
    "abstain": True,          # return [] when the corpus cannot answer
}
CHUNK_OVERLAP = 200       # RecursiveCharacterTextSplitter overlap, as in Tutorial 7 (baseline only)

LINE_LIMIT = 380          # Task 1.1 counts a line only up to 400 characters; keep a margin
MIN_CHUNK_CHARS = 250     # a section shorter than this joins the next section of the same part
HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
TEXT_TAGS = {"p", "caption", "blockquote", "pre", "dt", "dd", "figcaption", "address"}
SKIP_TAGS = {"nav", "script", "style", "head", "noscript", "button", "form"}
SKIP_SECTIONS = {"contents", "table of contents"}
SUFFIX_WORDS = {"inc", "corp", "corporation", "ltd", "limited", "plc", "llc", "co"}  # as in company_key()
DOC_LABELS = {"annual_report": "Annual report", "earnings_call": "Earnings call",
              "research_report": "Research report"}


def _clean(text: str) -> str:
    """Collapse whitespace (including non-breaking spaces) to single spaces."""
    return " ".join(str(text).split())


def _expand_row(tr: Tag, repeat: bool = False) -> list[str]:
    """The text of each cell of a row, spread over its colspan so that it lines up with the header.

    A header cell is repeated in every column it spans; a body cell fills only its first column.
    """
    texts = []
    for cell in tr.find_all(["td", "th"], recursive=False):
        try:
            span = max(1, int(cell.get("colspan", 1)))
        except (TypeError, ValueError):
            span = 1
        text = _clean(cell.get_text(" ", strip=True))
        texts.extend([text] + [text if repeat else ""] * (span - 1))
    return texts


def _row_lines(label: str, pairs: list[str]) -> list[str]:
    """One line per row; a row too long for one line is split, and every part repeats the row label."""
    line = " | ".join([label, *pairs]) if label else " | ".join(pairs)
    if len(line) <= LINE_LIMIT or len(pairs) <= 1:
        return [line]
    lines, current = [], []
    for pair in pairs:
        if current and len(" | ".join([label, *current, pair])) > LINE_LIMIT:
            lines.append(" | ".join([label, *current]))
            current = []
        current.append(pair)
    lines.append(" | ".join([label, *current]))
    return lines


def _table_lines(table: Tag) -> list[str]:
    """table_to_text for a parsed <table>: caption, then one line per row with column headers."""
    lines = []
    caption = table.find("caption")
    if caption is not None and _clean(caption.get_text(" ", strip=True)):
        lines.append(_clean(caption.get_text(" ", strip=True)))

    rows = [tr for tr in table.find_all("tr") if tr.find_parent("table") is table]
    header_rows = [tr for tr in rows if tr.find_parent("thead") is not None]
    if not header_rows and rows and rows[0].find("td") is None and len(rows[0].find_all("th")) > 1:
        header_rows = [rows[0]]  # no <thead>: a first row of <th> cells is the header
    body_rows = [tr for tr in rows if tr not in header_rows]

    # Column headers; several header rows are joined per column ("2025 Q4").
    headers: list[str] = []
    for tr in header_rows:
        for j, text in enumerate(_expand_row(tr, repeat=True)):
            if j >= len(headers):
                headers.append(text)
            elif text and text not in headers[j]:
                headers[j] = f"{headers[j]} {text}".strip()

    for tr in body_rows:
        cells = _expand_row(tr)
        filled = [c for c in cells if c]
        if not filled:
            continue
        if len(filled) == 1:  # a section row such as "Current assets", or a one-cell row
            lines.append(filled[0])
            continue
        label, pairs = cells[0], []
        for j, value in enumerate(cells[1:], start=1):
            if not value:
                continue
            header = headers[j] if j < len(headers) and headers[j] else f"Column {j + 1}"
            pairs.append(f"{header}: {value}")
        lines.extend(_row_lines(label, pairs))
    return lines


def _strip_suffix(name: str) -> str:
    """'Northgate Robotics Inc.' -> 'Northgate Robotics'; also drops a trailing '(TSX: BWE)'."""
    name = _clean(re.sub(r"\s*\([^)]*\)\s*$", "", name))
    words = name.split()
    while len(words) > 1 and re.sub(r"[^a-z]", "", words[-1].lower()) in SUFFIX_WORDS:
        words.pop()
    return " ".join(words).rstrip(",")


def _document_meta(soup: BeautifulSoup) -> tuple[str, int | None, str]:
    """(company, fiscal_year, doc_type) from the first lines of a document."""
    body = soup.body or soup
    for tag in body.find_all(SKIP_TAGS):
        tag.decompose()
    head = _clean(body.get_text(" ", strip=True))[:400]
    low = head.lower()

    if re.search(r"earnings\s+(conference\s+)?call", low):
        doc_type = "earnings_call"
    elif re.search(r"results\s+review|equity\s+research|target\s+price|\brating\b", low):
        doc_type = "research_report"
    else:
        doc_type = "annual_report"

    patterns = {
        "annual_report": [r"annual\s+report\s+(?:fy\s*)?(\d{4})", r"fiscal\s+year\s+ended[^0-9]*?\d{1,2}\s+\w+\s+(\d{4})"],
        "earnings_call": [r"full\s+year\s+(?:fy\s*)?(\d{4})", r"\bfy\s*(\d{4})", r"(?:q4|fourth\s+quarter)\D{0,30}(\d{4})"],
        "research_report": [r"\bfy\s*(\d{4})\s*results", r"results\s+review\D{0,20}(\d{4})", r"\bfy\s*(\d{4})"],
    }[doc_type]
    fiscal_year = None
    for pattern in patterns:
        found = re.search(pattern, low)
        if found:
            fiscal_year = int(found.group(1))
            break
    if fiscal_year is None:
        years = re.findall(r"\b(20\d{2})\b", low)
        fiscal_year = int(years[0]) if years else None

    h1 = body.find("h1")
    if h1 is not None:
        company = h1.get_text(" ", strip=True)
    elif soup.title is not None:
        company = soup.title.get_text(" ", strip=True).split("|")[0]
    else:
        company = head[:60]
    return _strip_suffix(company), fiscal_year, doc_type


def _blocks(node: Tag):
    """The document in reading order as ("heading", level, text), ("text", text) or ("table", lines)."""
    for child in node.children:
        if isinstance(child, NavigableString):
            if type(child) is NavigableString and _clean(child):
                yield ("text", _clean(child))
            continue
        if not isinstance(child, Tag) or child.name in SKIP_TAGS:
            continue
        name = child.name
        if name == "table":
            if SETTINGS["table_text"]:
                yield ("table", _table_lines(child))
            else:  # naive extraction: one cell per line, the link between value, row and column is lost
                yield ("table", [_clean(t) for t in child.get_text("\n", strip=True).split("\n") if _clean(t)])
        elif name in HEADING_TAGS:
            yield ("heading", HEADING_TAGS[name], _clean(child.get_text(" ", strip=True)))
        elif name in ("ul", "ol"):
            for li in child.find_all("li", recursive=False):
                text = _clean(li.get_text(" ", strip=True))
                if text:
                    yield ("text", f"- {text}")
        elif name in TEXT_TAGS or (name in ("div", "span", "li", "section") and not child.find(
                ["p", "div", "table", "ul", "ol", "section"] + list(HEADING_TAGS))):
            text = _clean(child.get_text(" ", strip=True))
            if text:
                yield ("speaker" if "speaker" in (child.get("class") or []) else "text", text)
        else:
            yield from _blocks(child)


def _split_text(text: str, limit: int) -> list[str]:
    """Split a long paragraph at sentence ends (or, failing that, at spaces) into pieces <= limit."""
    if len(text) <= limit:
        return [text]
    pieces, current = [], ""
    for sentence in re.split(r"(?<=[.!?;])\s+", text):
        while len(sentence) > limit:  # a sentence longer than the limit: cut at a space
            cut = sentence.rfind(" ", 0, limit) if " " in sentence[:limit] else limit
            cut = cut if cut > 0 else limit
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:cut])
            sentence = sentence[cut:].strip()
        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= limit:
            current = candidate
        else:
            pieces.append(current)
            current = sentence
    if current:
        pieces.append(current)
    return pieces


def _split_table(lines: list[str], limit: int) -> list[str]:
    """Split table text into pieces <= limit at row boundaries; every piece repeats the caption.

    A piece preferably ends before a section row ("Liabilities and equity"), so that the section
    row starts the next piece together with its rows.
    """
    text = "\n".join(lines)
    if len(text) <= limit:
        return [text]
    caption = lines[0] if lines else ""
    parts = [part for line in lines[1:] for part in _split_text(line, limit - len(caption) - 1)]
    pieces, current = [], [caption]
    for part in parts:
        if len("\n".join(current + [part])) > limit and len(current) > 1:
            # back up to the last section row (a line without "|") if it is not the first row
            cut = next((i for i in range(len(current) - 1, 1, -1) if " | " not in current[i]), None)
            if cut and len("\n".join([caption, *current[cut:], part])) <= limit:
                pieces.append("\n".join(current[:cut]))
                current = [caption, *current[cut:]]
            else:
                pieces.append("\n".join(current))
                current = [caption]
        current.append(part)
    pieces.append("\n".join(current))
    return pieces


def _recursive_chunks(soup: BeautifulSoup, base: dict, head: str) -> list[Document]:
    """Baseline chunking (ablation only): the whole text, split by RecursiveCharacterTextSplitter."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if SETTINGS["table_text"]:
        for table in soup.find_all("table"):
            table.replace_with(NavigableString("\n" + "\n".join(_table_lines(table)) + "\n"))
    text = soup.get_text("\n", strip=True)
    head = head if SETTINGS["chunk_header"] else ""
    size = MAX_CHUNK_CHARS - (len(head) + 1 if head else 0)
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=CHUNK_OVERLAP)
    return [Document(page_content=(head + "\n" if head else "") + piece,
                     metadata={**base, "section": "", "has_table": False, "chunk": i})
            for i, piece in enumerate(splitter.split_text(text))]


# ----------------------------------------------------------------------------- what chunk_document found

# Filled by chunk_document (the runner calls it on every document before parse_query) and again by
# build_index from the chunk metadata: the companies of the corpus, and names that are not companies.
KNOWN_COMPANIES: dict[str, str] = {}   # company key -> name as in the metadata
KNOWN_BROKERS: set[str] = set()        # research report authors and analysts' employers (keys)
KNOWN_ORGS: set[str] = set()           # brokers, audit firms and other names that are not companies (keys)
KNOWN_PEOPLE: set[str] = set()         # directors, officers, call participants (keys)
KNOWN_TEXTS: set[str] = set()          # every document, normalised, to look names up


def _register(company: str, doc_type: str, lines: list[str]) -> None:
    """Remember the company of a document and the brokers, audit firms and people it names."""
    if company:
        KNOWN_COMPANIES.setdefault(company_key(company), company)
    KNOWN_TEXTS.add(normalize("\n".join(lines)))
    for line in lines:
        if doc_type == "research_report":
            found = re.match(r"^(.+?)\s*\|\s*Equity Research\b", line)
            if found:
                KNOWN_BROKERS.add(company_key(found.group(1)))
        # audit firms: the capitalised name right before "audited", or any name ending in LLP or LLC
        for found in re.finditer(r"((?:[A-Z][\w&'.-]*\s+){0,4}[A-Z][\w&'.-]*)\s+(?:has\s+)?audited\b", line):
            KNOWN_ORGS.add(company_key(found.group(1)))
        for found in re.finditer(r"((?:[A-Z][\w&'.-]*\s+){1,4}(?:LLP|LLC))\b", line):
            KNOWN_ORGS.add(company_key(found.group(1)))
        found = re.match(r"^- ([A-Z][\w.'-]+(?: [A-Z][\w.'-]+){1,3}) - (.+)$", line)  # "- Ivan Xu - Harbourline Securities"
        if found:
            KNOWN_PEOPLE.add(company_key(found.group(1)))
            if not any(w in TITLE_WORDS for w in company_key(found.group(2)).split()):
                KNOWN_BROKERS.add(company_key(found.group(2)))  # an analyst's employer, not a job title
    KNOWN_ORGS.update(KNOWN_BROKERS)
    for line in lines:
        found = re.match(r"^([A-Z][\w.'-]+(?: [A-Z][\w.'-]+){1,3}) \| (?:Role|Title): ", line)  # board, executives
        if found:
            KNOWN_PEOPLE.add(company_key(found.group(1)))


def table_to_text(table_html: str) -> str:
    """Task 1.1: turn one HTML <table> into text that keeps every value with its row and its column.

    Put each value on a line (at most 400 characters) together with its row label and its column
    header, e.g. "Revenue | 2025: 2,710.4 | 2024: 2,629.1", one table row per line, keep the text of
    every cell unchanged, and include the caption (it holds the unit) in the output. Do not call the
    LLM here.
    """
    ### YOUR CODE HERE
    soup = BeautifulSoup(table_html, "lxml")
    table = soup.find("table")
    if table is None:
        return _clean(soup.get_text(" ", strip=True))
    return "\n".join(_table_lines(table))


def chunk_document(file_name: str, html: str) -> list[Document]:
    """Task 1.2: split one document into chunks of at most MAX_CHUNK_CHARS characters.

    Every chunk needs the metadata "source" (= file_name), "company", "fiscal_year" (an int) and
    "doc_type" (one of DOC_TYPES). Do not call the LLM here.
    """
    ### YOUR CODE HERE
    soup = BeautifulSoup(html, "lxml")
    company, fiscal_year, doc_type = _document_meta(soup)
    label = f"{DOC_LABELS[doc_type]} FY{fiscal_year}" if fiscal_year else DOC_LABELS[doc_type]
    base = {"source": file_name, "company": company, "fiscal_year": fiscal_year, "doc_type": doc_type}

    if not SETTINGS["section_chunks"]:  # the baseline: fixed-size pieces of the whole text
        chunks = _recursive_chunks(BeautifulSoup(html, "lxml"), base, f"{company} | {label}")
        _register(company, doc_type, [line for c in chunks for line in c.page_content.split("\n")])
        return chunks

    chunks: list[Document] = []
    titles: dict[int, str] = {}  # heading level -> current title (h2 and below)
    buffer: list[str] = []
    buffer_titles: list[str] = []  # the section path of the chunk being filled
    has_table = False
    skipping = False
    speaker = ""

    def path() -> list[str]:
        return [titles[level] for level in sorted(titles)]

    def header(parts: list[str]) -> str:
        if not SETTINGS["chunk_header"]:
            return ""
        return " | ".join(part for part in (company, label, " > ".join(parts)) if part)

    def flush() -> None:
        nonlocal buffer, has_table
        if buffer:
            head = header(buffer_titles)
            chunks.append(Document(page_content=(head + "\n" if head else "") + "\n".join(buffer),
                                   metadata={**base, "section": " > ".join(buffer_titles),
                                             "has_table": has_table, "chunk": len(chunks)}))
        buffer, has_table = [], False

    def add(text: str, is_table: bool) -> None:
        nonlocal has_table, buffer_titles
        if not buffer:
            buffer_titles = path()
        room = MAX_CHUNK_CHARS - max(len(header(buffer_titles)), len(header(path()))) - 1
        pieces = _split_table(text.split("\n"), room) if is_table else _split_text(text, room)
        for piece in pieces:
            if buffer and len("\n".join(buffer + [piece])) > room:
                flush()
                buffer_titles = path()
            buffer.append(piece)
            has_table = has_table or is_table

    for kind, *rest in _blocks(soup.body or soup):
        if kind == "heading":
            level, title = rest
            if level == 1:  # the document title belongs to the opening chunk
                if not skipping:
                    add(title, False)
                continue
            speaker = ""
            new_titles = {k: v for k, v in titles.items() if k < level}
            new_titles[level] = title
            new_skip = title.lower() in SKIP_SECTIONS or (level > 2 and skipping)
            opening = not buffer_titles
            same_part = level > 2 and buffer_titles[:1] == [titles.get(2)]
            if buffer and not new_skip and len("\n".join(buffer)) < MIN_CHUNK_CHARS and (opening or same_part):
                # a short section joins the next one; the heading stays in the text as a line
                titles, skipping = new_titles, new_skip
                common = 0
                while common < min(len(buffer_titles), len(path())) and buffer_titles[common] == path()[common]:
                    common += 1
                if buffer_titles[common:]:  # the title dropped from the header moves into the text
                    buffer.insert(0, " > ".join(buffer_titles[common:]))
                buffer_titles = buffer_titles[:common]
                buffer.append(title)
                continue
            flush()
            titles, skipping = new_titles, new_skip
            continue
        if skipping:
            continue
        if kind == "table":
            if speaker:
                add(speaker, False)
                speaker = ""
            add("\n".join(rest[0]), True)
        elif kind == "speaker":
            if speaker:
                add(speaker, False)
            speaker = rest[0]
        else:
            text = f"{speaker}: {rest[0]}" if speaker else rest[0]
            speaker = ""
            add(text, False)
    if speaker:
        add(speaker, False)
    flush()
    _register(company, doc_type, [line for c in chunks for line in c.page_content.split("\n")])
    return chunks


# ----------------------------------------------------------------------------- query parsing

MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august", "september",
          "october", "november", "december"}
# Capitalised words that start or end a name but are not part of a company name.
NOT_NAME_WORDS = MONTHS | {
    "what", "which", "who", "whom", "whose", "how", "when", "where", "why", "in", "on", "at", "for", "of",
    "did", "does", "do", "was", "were", "is", "are", "the", "a", "an", "its", "their", "by", "from",
    "according", "during", "as", "and", "or", "to", "with", "between", "usd", "hkd", "eur", "gbp", "cny",
    "rmb", "jpy", "ifrs", "gaap", "eps", "ceo", "cfo", "coo", "cto", "q1", "q2", "q3", "q4", "h1", "h2"}
TITLE_WORDS = {"chief", "officer", "executive", "financial", "operating", "technology", "chair", "chairman",
               "chairwoman", "director", "directors", "president", "counsel", "secretary", "head", "board",
               "committee", "investor", "relations", "treasurer", "controller"}
ORG_WORDS = {"securities", "capital", "research", "partners", "llp", "bank", "advisors", "advisers",
             "brokers", "markets", "investments", "equity"}
CONNECTORS = {"of", "and", "&"}
UNIT_WORDS = {"segment", "segments", "division", "business", "unit", "region", "brand", "subsidiary"}


def _name_spans(question: str) -> list[tuple[str, bool, int]]:
    """Runs of capitalised words, cut at words that cannot be part of a company name.

    "of", "and" and "&" between capitalised words stay in the run. Returns (name, possessive, end
    position) for every run.
    """
    text = question.replace("’", "'")
    spans = []
    for run in re.finditer(r"[A-Z][\w&.\-]*(?:\s+(?:(?:of|and|&)\s+)?[A-Z][\w&.\-]*)*", text):
        words, start = [], run.start()
        for word in re.finditer(r"\S+", run.group(0)):
            token = word.group(0).strip(".,")
            low = token.lower()
            if low in CONNECTORS:
                words.append(token)
            elif low in NOT_NAME_WORDS or re.fullmatch(r"fy\d*|\d+[a-z]*", low):
                if words:
                    spans.append((words, start + word.start()))
                words = []
            else:
                words.append(token)
        if words:
            spans.append((words, run.end()))
    result = []
    for words, end in spans:
        while words and words[-1].lower() in CONNECTORS:
            words = words[:-1]
        while words and words[0].lower() in CONNECTORS:
            words = words[1:]
        if not words:
            continue
        name = " ".join(words)
        possessive = text[end:end + 2] in ("'s", "' ") or text[end:end + 1] == "'" or name.endswith("'")
        result.append((name.rstrip("'"), possessive, end))
    return result


def _in_corpus(name: str) -> bool:
    """The name occurs, as whole words, in some document of the corpus."""
    phrase = normalize(name)
    return any(phrase in text for text in KNOWN_TEXTS)


def _as_company(name: str, possessive: bool, after: str, compared: set[str]) -> str | None:
    """The company a name in the question refers to, or None if it is not a company."""
    key = company_key(name)
    words = key.split()
    if not words or key in KNOWN_ORGS or key in KNOWN_PEOPLE:
        return None
    if key not in KNOWN_COMPANIES and re.fullmatch(r"[A-Z0-9&/.\-]{1,6}", name):
        return None  # an abbreviation or a ticker: R&D, EPS, SG&A, NGR
    if words[-1] in ORG_WORDS or any(w in TITLE_WORDS for w in words):
        return None
    if key in KNOWN_COMPANIES or key in compared:
        return name
    # a known company followed by a segment name: "Brightwater Energy Midstream segment"
    prefix = next((k for k in KNOWN_COMPANIES if k and key.startswith(k + " ")), None)
    if prefix and re.match(r"(s\s+)?(" + "|".join(UNIT_WORDS) + r")\b", after):
        return KNOWN_COMPANIES[prefix]
    # a company that is not in the corpus: it looks like one that is ("Halcyon Air Cargo"), it is
    # in the possessive, or the documents never mention it (segments, regions and people they do)
    known_first = {k.split()[0] for k in KNOWN_COMPANIES if k}
    if len(words) >= 2 and (words[0] in known_first or possessive or not _in_corpus(name)):
        return name
    if len(words) == 1 and possessive and not _in_corpus(name):
        return name
    return None


def _parse_companies(question: str) -> list[str]:
    compared: set[str] = set()
    for pattern in (r":\s*(.+?)\s+or\s+(.+?)\s*\?*\s*$", r"\b[Bb]etween\s+(.+?)\s+and\s+(.+?)\s*[,?]"):
        found = re.search(pattern, question)
        if found:
            compared |= {company_key(n) for n in found.groups()}
    companies: list[str] = []
    for name, possessive, end in _name_spans(question):
        after = question[end:].lstrip(" '’").lower()
        parts = [(name, possessive)]
        if re.search(r"\s(?:and|&)\s", name) and company_key(name) not in KNOWN_COMPANIES:
            # "Northgate Robotics and Northgate Payments": two companies if both parts are companies
            pieces = re.split(r"\s+(?:and|&)\s+", name)
            flags = [False] * (len(pieces) - 1) + [possessive]
            if all(_as_company(p, f, "", compared) for p, f in zip(pieces, flags)):
                parts = list(zip(pieces, flags))
        for part, flag in parts:
            company = _as_company(part, flag, after, compared)
            if company:
                companies.append(company)
    unique: dict[str, str] = {}
    for name in companies:
        unique.setdefault(company_key(name), name)
    return list(unique.values())


def _parse_year(question: str) -> int | None:
    found = re.search(r"\bFY\s?'?(\d{4}|\d{2})(?:[AE])?\b", question, re.I)
    if found:
        year = int(found.group(1))
        return year + 2000 if year < 100 else year
    found = re.search(r"\b(19\d{2}|20\d{2})\b", question)
    return int(found.group(1)) if found else None


def _parse_doc_type(question: str) -> str:
    low = question.lower()
    if re.search(r"earnings[\s-]+(?:conference\s+)?call|conference\s+call|\bon\s+(?:its|the)\s+(?:q4\s+|fourth[\s-]quarter\s+)?call\b", low):
        return "earnings_call"
    if re.search(r"\brating\b|target\s+price|price\s+target|forecast|\bestimates?\b|results\s+review|broker|"
                 r"\banalysts?\b|equity\s+research|recommendation|upgrade|downgrade|\bprojects?\b", low):
        return "research_report"
    if any(broker and f" {broker} " in normalize(question) for broker in KNOWN_BROKERS):
        return "research_report"  # a question that names a broker is about its research report
    return "annual_report"


# ----------------------------------------------------------------------------- lexical retrieval


STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "for", "to", "by", "from", "with", "and", "or", "as", "is",
    "are", "was", "were", "be", "been", "it", "its", "their", "this", "that", "these", "those", "what",
    "which", "who", "whom", "how", "much", "many", "did", "does", "do", "had", "has", "have", "end",
    "usd", "million", "millions", "percent", "company", "companies", "value", "amount", "figure", "year",
    "fiscal", "fy", "report", "reported", "hold", "held", "generate", "generated", "earn", "earned", "give",
    "gives", "set", "sets", "announce", "announced", "spend", "spent", "more", "higher", "lower", "less",
    "most", "highest", "lowest", "larger", "smaller", "december", "its", "s", "according", "during"} | MONTHS
# Question words -> document words (the documents' wording comes first).
SYNONYMS = {
    "cash": ["cash", "equivalents"], "headcount": ["employees"], "staff": ["employees"],
    "workforce": ["employees"], "people": ["employees"], "employ": ["employees"], "sales": ["revenue"],
    "turnover": ["revenue"], "debt": ["borrowings"], "loans": ["borrowings"], "borrowed": ["borrowings"],
    "cfo": ["chief", "financial", "officer"], "ceo": ["chief", "executive", "officer"],
    "coo": ["chief", "operating", "officer"], "cto": ["chief", "technology", "officer"],
    "auditor": ["audited", "independent", "auditor"], "audit": ["audited", "auditor"],
    "audited": ["audited", "auditor"], "headquartered": ["headquarters"], "based": ["headquarters"],
    "buyback": ["repurchase"], "buybacks": ["repurchase"], "repurchase": ["repurchase", "programme"],
    "guide": ["expect", "guidance", "outlook", "guided"], "guidance": ["expect", "outlook", "guided"],
    "outlook": ["expect", "guidance"], "forecast": ["estimate", "estimates"], "forecasts": ["estimate"],
    "estimate": ["estimates"], "rating": ["rating"], "recommendation": ["rating"],
    "profit": ["income", "profit"], "earnings": ["income", "eps"], "receivables": ["receivable"],
    "inventory": ["inventories"], "stock": ["inventories"], "equity": ["equity"],
}
# Words that describe the answer rather than the evidence ("In which city ...", "Which audit firm ...").
ANSWER_WORDS = {"city", "town", "country", "place", "location", "firm", "person", "name", "named", "who",
                "large", "big", "small", "size", "figure", "amount", "number", "level", "date", "day",
                "month", "time", "percentage", "pay", "paid", "record", "recorded", "symbol", "code",
                "post", "posted", "achieve", "achieved", "reach", "reached", "disclose", "disclosed",
                "show", "shown", "state", "stated", "say", "said", "mention", "mentioned", "make", "made",
                "assign", "assigned", "publish", "published", "issue", "issued", "raise", "raised",
                "lower", "lowered", "cut", "increase", "increased", "decrease", "decreased",
                # when, how and from whom: these describe the question, not the figure
                "previous", "previously", "prior", "earlier", "old", "new", "latest", "current", "recent",
                "initial", "original", "revised", "updated", "project", "projected", "projection",
                "expect", "expected", "anticipate", "anticipated", "predict", "predicted", "plan", "planned",
                "broker", "brokers", "analyst", "analysts", "approximately", "roughly", "about", "around",
                "exactly", "overall", "combined", "annual", "annually", "full", "whole", "start", "beginning",
                "close", "closing", "ending", "ended", "compared", "versus", "vs", "change", "changed"}
# Abbreviations in questions -> the words the documents use.
ABBREVIATIONS = [
    (r"\bR&D\b", "research and development"), (r"\bSG&A\b", "selling general and administrative"),
    (r"\bPP&E\b", "property plant and equipment"), (r"\bD&A\b", "depreciation and amortisation"),
    (r"\bM&A\b", "acquisitions"), (r"\bEPS\b", "earnings per share"), (r"\bP/E\b", "P/E price earnings"),
    (r"\bcapex\b|\bcapital\s+expenditures?\b", "purchases of property plant and equipment"),
    (r"\bQ4\b", "fourth quarter"), (r"\bYoY\b|\by/y\b", "year on year"), (r"\bIR\b", "investor relations"),
]
# American spellings -> the British spellings of the documents (applied to documents and questions).
SPELLINGS = {"program": "programme", "programs": "programmes", "labor": "labour", "center": "centre",
             "centers": "centres", "favorable": "favourable", "behavior": "behaviour"}
NUMBER_TOKEN_RE = re.compile(r"\d")
OTHER_DOC_RATIO = 1.3  # how much better a chunk of another kind of document must match to be added
MAX_OTHER_DOCS = 2     # at most this many such chunks
# A question that asks for a figure ("in USD millions", "how many") or compares figures ("higher net income").
FIGURE_QUESTION = (r"\b(?:usd|hkd|eur|gbp|millions?|billions?|percent|how\s+much|how\s+many|"
                   r"higher|lower|more|less|larger|smaller|greater|fewer|bigger)\b")


def _stem(word: str) -> str:
    """A light stemmer: plural, -ing, -ed and a final e ('employees' and 'employee' -> 'employe')."""
    word = SPELLINGS.get(word, word)
    if "iz" in word:  # amortization -> amortisation, organized -> organised
        word = re.sub(r"iz(?=e|ed|es|ing|ation|ations)", "is", word)
    if len(word) <= 3 or NUMBER_TOKEN_RE.search(word):
        return word
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word = word[:-1]
    for suffix in ("ing", "ed"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            word = word[:-len(suffix)]
            break
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    return word


def _tokens(text: str) -> list[str]:
    """Lower-cased word and number tokens, stemmed. 'FY2025E' -> 'fy2025e', '2,710.4' -> '2,710.4'."""
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    return [_stem(t) for t in re.findall(r"\d[\d,.]*\d|\d|[a-z][a-z0-9]*", text)]


def _tokenize(text: str) -> list[str]:
    """The BM25 tokens of a chunk or a query, by SETTINGS["tokenizer"]."""
    if SETTINGS["tokenizer"] == "whitespace":
        return text.lower().split()
    return _tokens(text)


def _query_words(question: str, companies: list[str]) -> list[str]:
    """The content words of a question: no company names, years, units or question words."""
    text = question.replace("’", "'")
    for name in companies:
        text = re.sub(re.escape(name) + r"(?:'s|')?", " ", text, flags=re.I)
    for pattern, replacement in ABBREVIATIONS:
        text = re.sub(pattern, replacement, text, flags=re.I)
    words = []
    for word in re.findall(r"[A-Za-z][A-Za-z\-]*", text.replace("’", "'")):
        for part in word.lower().split("-"):
            if part and part not in STOPWORDS and not re.fullmatch(r"fy\d*|q\d|\d+", part) and len(part) > 1:
                words.append(part)
    return words


def _query_tokens(question: str, companies: list[str]) -> list[str]:
    """The BM25 query: the content words of the question with their synonyms, or the whole question."""
    if not SETTINGS["query_words"]:
        return _tokenize(question)
    words = _query_words(question, companies)
    expanded = [w for word in words for w in (word, *SYNONYMS.get(word, []))]
    return [t for w in expanded for t in _tokenize(w)] or _tokenize(question)


class LexicalIndex:
    """BM25 over all chunks, plus the chunks grouped by company, and what each company's documents cover."""

    def __init__(self, chunks: list[Document]) -> None:
        self.chunks = chunks
        self.tokens = [_tokenize(c.page_content) for c in chunks]
        self.bm25 = BM25Okapi(self.tokens, k1=1.2, b=0.75) if chunks else None
        self.by_company: dict[str, list[int]] = {}
        self.annual_years: dict[str, set[int]] = {}
        self.all_vocab: set[str] = set()  # stemmed words of the whole corpus, for the abstention check
        for i, chunk in enumerate(chunks):
            meta = chunk.metadata or {}
            key = company_key(meta.get("company", ""))
            self.by_company.setdefault(key, []).append(i)
            self.all_vocab.update(_tokens(chunk.page_content))
            year = as_year(meta.get("fiscal_year"))
            if meta.get("doc_type") == "annual_report" and year:
                # an annual report has the columns of its year and of the year before
                self.annual_years.setdefault(key, set()).update({year, year - 1})
            KNOWN_COMPANIES.setdefault(key, meta.get("company", ""))


def _rank_company(index: LexicalIndex, key: str, scores, year: int | None, doc_type: str) -> list[int]:
    """The company's chunks, best first, from the documents that can hold the answer."""
    ids = index.by_company.get(key, [])

    def allowed(i: int) -> bool:
        meta = index.chunks[i].metadata
        if meta.get("doc_type") != doc_type:
            return False
        if doc_type == "annual_report" and year is not None:
            return as_year(meta.get("fiscal_year")) in (year, year + 1)
        return True

    def score(i: int) -> float:
        bonus = 0.0
        if year is not None and as_year(index.chunks[i].metadata.get("fiscal_year")) == year:
            bonus += 0.5  # the report of the year asked about comes first
        return scores[i] + bonus

    candidates = [i for i in ids if allowed(i)]
    if not candidates:
        return sorted(ids, key=score, reverse=True)
    ranked = sorted(candidates, key=score, reverse=True)
    # The question may name the wrong kind of document (a repurchase programme that was only announced
    # on the call): the best chunk of another kind of document goes second if it matches the question
    # clearly better than every chunk of the expected documents.
    others = sorted((i for i in ids if index.chunks[i].metadata.get("doc_type") != doc_type),
                    key=lambda i: scores[i], reverse=True)
    better = [i for i in others[:MAX_OTHER_DOCS] if scores[i] > OTHER_DOC_RATIO * max(scores[ranked[0]], 1e-9)]
    return ranked[:1] + better + ranked[1:]


def _unanswerable(index: LexicalIndex, keys: list[str], words: list[str], year: int | None,
                  doc_type: str) -> str | None:
    """Why the corpus cannot answer, or None: "company" (not in the corpus), "year" (a year the
    documents do not cover) or "word" (a question word that occurs nowhere in the corpus, i.e. a
    figure that is not disclosed, such as lobbying expenses)."""
    if any(k not in index.by_company for k in keys):
        return "company"
    for key in keys:
        if year is None:
            continue
        if doc_type == "annual_report":
            if year not in index.annual_years.get(key, set()):
                return "year"
        else:  # a call or a research report: the year (e.g. a forecast for FY2027) must be in it
            ids = [i for i in index.by_company.get(key, []) if index.chunks[i].metadata.get("doc_type") == doc_type]
            pattern = re.compile(rf"(?<!\d){year}(?!\d)")
            if ids and not any(pattern.search(index.chunks[i].page_content) for i in ids):
                return "year"
    for word in words:
        if word in ANSWER_WORDS:
            continue
        options = {_stem(word), *(_stem(s) for s in SYNONYMS.get(word, []))}
        if not options & index.all_vocab:
            return "word"
    return None


ANSWER_CHECK_PROMPT = """Below are passages from company reports, and a question about a company.

Passages:
{context}

Question: {question}

Do the passages state the answer to this question: the figure, name or rating it asks for, for the company and the period it asks about? Reply with one word: YES or NO."""
_checker: ChatDeepSeek | None = None


def _llm_says_answerable(question: str, chunks: list[Document]) -> bool | None:
    """Ask the LLM whether the chunks answer the question; None if the LLM cannot be reached."""
    global _checker
    if not chunks or not os.environ.get("DEEPSEEK_API_KEY"):
        return None
    try:
        if _checker is None:
            _checker = make_llm(temperature=0.0, max_tokens=5)
        reply = str(_checker.invoke(ANSWER_CHECK_PROMPT.format(
            context=format_context(chunks), question=question)).content).strip().upper()
    except Exception:
        return None
    if reply.startswith("YES"):
        return True
    if reply.startswith("NO"):
        return False
    return None


def parse_query(question: str) -> dict:
    """Task 2.2: {"companies": [names], "fiscal_year": int or None, "doc_type": one of DOC_TYPES}.

    The rules for each field are under Task 2.2 in the handout. You may use make_llm() here.
    """
    ### YOUR CODE HERE
    return {"companies": _parse_companies(question), "fiscal_year": _parse_year(question),
            "doc_type": _parse_doc_type(question)}


def build_index(chunks: list[Document]) -> Any:
    """Task 2: index the chunks of the whole corpus (the output of chunk_document), once per run.

    Return anything retrieve() needs: a BM25 or TF-IDF index, the chunks grouped by metadata, ...
    """
    ### YOUR CODE HERE
    return LexicalIndex(chunks)


def retrieve(index: Any, question: str) -> list[Document]:
    """Tasks 2 and 3: at most MAX_RETRIEVED of the chunks given to build_index, best first.

    Return [] when the corpus cannot answer the question: the generator then answers NOT_FOUND.
    """
    ### YOUR CODE HERE
    if not isinstance(index, LexicalIndex) or not index.chunks:
        return []

    def best(scores) -> list[Document]:  # BM25 over the whole corpus
        order = sorted(range(len(index.chunks)), key=lambda i: -scores[i])
        return [index.chunks[i] for i in order[:MAX_RETRIEVED]]

    if not SETTINGS["metadata_filter"]:  # the baseline: no query parsing, never []
        return best(index.bm25.get_scores(_query_tokens(question, [])))

    parsed = parse_query(question)
    names, year, doc_type = parsed["companies"], parsed["fiscal_year"], parsed["doc_type"]
    keys = [company_key(n) for n in names]
    words = _query_words(question, names)
    scores = index.bm25.get_scores(_query_tokens(question, names))  # company names: the filter does that

    reason = None
    if SETTINGS["abstain"]:
        reason = _unanswerable(index, keys, words, year, doc_type) if keys else None
        if reason in ("company", "year"):
            return []
    else:
        keys = [k for k in keys if k in index.by_company]
    if not keys:  # no company of the corpus named: rank the whole corpus
        return best(scores)

    boost = SETTINGS["table_boost"]
    if boost != 1.0 and re.search(FIGURE_QUESTION, question, re.I):
        # the question asks for a figure, or compares figures, and figures are in tables
        scores = [s * boost if index.chunks[i].metadata.get("has_table") else s for i, s in enumerate(scores)]
    ranked = [_rank_company(index, k, scores, year, doc_type) for k in keys]
    if len(ranked) > 1 and not SETTINGS["per_company"]:  # one ranking of all the companies' chunks
        picked = sorted({i for ids in ranked for i in ids}, key=lambda i: -scores[i])
    else:
        # the companies take turns (the best chunk of each first), up to an equal share each,
        # then the best of the rest fill up
        share = max(1, MAX_RETRIEVED // len(keys))
        picked = []
        for rank in range(share):
            for ids in ranked:
                if rank < len(ids) and ids[rank] not in picked:
                    picked.append(ids[rank])
        rest = sorted({i for ids in ranked for i in ids} - set(picked), key=lambda i: -scores[i])
        picked.extend(rest[:MAX_RETRIEVED - len(picked)])
    chunks = [index.chunks[i] for i in picked[:MAX_RETRIEVED]]

    if reason == "word":
        # A question word the corpus never uses usually means the figure is not disclosed, but it may
        # also be unusual wording: the LLM gets the last word when it can be reached.
        return chunks if _llm_says_answerable(question, chunks) else []
    return chunks


# =============================================================================
# Provided code below. Do not change it.
# =============================================================================

LLM_MODEL = "deepseek-v4-flash"
DOC_TYPES = ("annual_report", "earnings_call", "research_report")

MAX_CHUNK_CHARS = 1500  # characters per chunk
MAX_RETRIEVED = 5       # chunks per question
TIME_LIMIT_SECONDS = 30 * 60

# USD per 1M tokens, only for the cost estimate that the runner prints.
PRICE_LLM_INPUT, PRICE_LLM_OUTPUT = 0.15, 0.60

# Points (Section 2 of the handout).
POINTS = {"1.1": 8, "1.2": 7, "2.1": 5, "2.2": 10, "2.3": 15, "3": 10, "4": 20}
RECALL_ZERO, RECALL_FULL = 0.50, 0.90  # Task 2.3: 0 points at or below RECALL_ZERO, full points from RECALL_FULL
MAX_LINE_CHARS = 400                   # Task 1.1
MIN_PROBES_FOUND = 0.9                 # Task 1.2


class UsageTracker(BaseCallbackHandler):
    """Counts LLM tokens, for the cost estimate. There is no budget."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.llm_calls = self.llm_input_tokens = self.llm_output_tokens = 0

    def on_llm_end(self, response, **kwargs) -> None:
        tokens_in = tokens_out = 0
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                tokens_in += usage.get("input_tokens", 0)
                tokens_out += usage.get("output_tokens", 0)
        with self._lock:
            self.llm_calls += 1
            self.llm_input_tokens += tokens_in
            self.llm_output_tokens += tokens_out

    def cost(self) -> float:
        return (self.llm_input_tokens * PRICE_LLM_INPUT + self.llm_output_tokens * PRICE_LLM_OUTPUT) / 1e6

    def summary(self) -> dict:
        return {"llm_calls": self.llm_calls, "llm_input_tokens": self.llm_input_tokens,
                "llm_output_tokens": self.llm_output_tokens, "cost_usd": round(self.cost(), 4)}


USAGE = UsageTracker()            # your calls
GENERATOR_USAGE = UsageTracker()  # the generator's calls


def _chat(usage: UsageTracker, temperature: float, thinking: bool, max_tokens: int | None) -> ChatDeepSeek:
    return ChatDeepSeek(
        model=LLM_MODEL,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=3,
        # Thinking mode ignores `temperature` and its tokens count as output.
        extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
        callbacks=[usage],
    )


def make_llm(temperature: float = 0.0, thinking: bool = False, max_tokens: int | None = None) -> ChatDeepSeek:
    """deepseek-v4-flash. Use it for every LLM call you make (only in parse_query, build_index and retrieve)."""
    return _chat(USAGE, temperature, thinking, max_tokens)


# ----------------------------------------------------------------------------- the generator

GENERATOR_PROMPT = """You answer questions about company reports, using only the context below.

Context:
{context}

Question: {question}

Rules:
- Use only the context. The header of each passage says which document it comes from.
- Check that the passage you use is about the company and the fiscal year that the question asks about.
- If the context does not contain the answer, reply exactly NOT_FOUND.
- For a number, reply with that one number only, in the unit the question asks for, without units, words or other numbers (for example: 1913.4).
- For a name (a person, a company, a city, an audit firm or a rating), reply with the name only.

Answer:"""

_generator: ChatDeepSeek | None = None


def format_context(chunks: list[Document]) -> str:
    """The retrieved chunks as the generator sees them: a header from the metadata, then the text."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata or {}
        header = " | ".join(f"{key}: {str(meta[key])[:100]}"
                            for key in ("source", "company", "fiscal_year", "doc_type") if meta.get(key) not in (None, ""))
        parts.append(f"[{i}] {header}\n{chunk.page_content}")
    return "\n\n".join(parts)


def generate(question: str, chunks: list[Document]) -> str:
    """The fixed generator: deepseek-v4-flash answers from the retrieved chunks only."""
    global _generator
    if not chunks:
        return "NOT_FOUND"
    if _generator is None:
        _generator = _chat(GENERATOR_USAGE, 0.0, False, 64)
    prompt = GENERATOR_PROMPT.format(context=format_context(chunks), question=question)
    return str(_generator.invoke(prompt).content).strip()


# ----------------------------------------------------------------------------- loaders

def load_corpus(folder: str | Path) -> dict[str, str]:
    """{file name: raw HTML} for every .html file in the folder."""
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(Path(folder).glob("*.html"))}


def load_json(path: str | Path) -> Any:
    """The parsed file, or None if it does not exist (the private set has no answers)."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- scoring

# One number: optional minus sign, optional thousands separators, optional decimals.
NUMBER_RE = re.compile(r"(?<![\d.,])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
TITLES = {"mr", "mrs", "ms", "dr", "prof", "professor", "sir"}
LEGAL_SUFFIXES = {"inc", "corp", "corporation", "ltd", "limited", "plc", "llc", "co"}


def parse_number(text: str) -> float | None:
    """The single number in the text, or None if there is not exactly one."""
    found = NUMBER_RE.findall(text.replace("\u2212", "-"))
    if len(found) != 1:
        return None
    return float(found[0].replace(",", ""))


def normalize(text: str) -> str:
    """Lower case, no punctuation, no titles, padded with spaces for whole-word matching."""
    text = unicodedata.normalize("NFKC", str(text)).lower().replace("_", " ")
    words = [w for w in re.sub(r"[^a-z0-9]+", " ", text).split() if w not in TITLES]
    return " " + " ".join(words) + " "


def company_key(name: Any) -> str:
    """A company name without case, punctuation or legal suffix: 'Northgate Robotics Inc.' -> 'northgate robotics'."""
    return " ".join(w for w in normalize(name).split() if w not in LEGAL_SUFFIXES)


def squash(text: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip().casefold()


def as_year(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def answer_correct(answer: str, gold: dict) -> bool:
    answer = str(answer or "")
    said_not_found = " not found " in normalize(answer)
    if gold["type"] == "not_found":
        return said_not_found
    if said_not_found:
        return False
    if gold["type"] == "number":
        value = parse_number(answer)
        if value is None:
            return False
        return abs(value - gold["answer"]) <= max(0.01 * abs(gold["answer"]), 0.05)
    names = [gold["answer"], *gold.get("aliases", [])]
    if not any(normalize(name) in normalize(answer) for name in names):
        return False
    return not any(normalize(other) in normalize(answer) for other in gold.get("reject", []))


def table_score(output: Any, table: dict) -> float:
    """Task 1.1, one table: the share of cells found on a line with their row and column labels and
    without the label of another row, halved if the caption is missing."""
    if not isinstance(output, str) or not table["cells"]:
        return 0.0
    lines = [squash(line) for line in output.splitlines()]
    lines = [line for line in lines if line and len(line) <= MAX_LINE_CHARS]
    labels = {squash(row) for row, _, _ in table["cells"]}

    def kept(row: str, col: str, value: str) -> bool:
        row, col, value = squash(row), squash(col), squash(value)
        others = [o for o in labels if o not in row]  # e.g. "total liabilities" may be in "total liabilities and equity"
        return any(row in line and col in line and value in line and not any(o in line for o in others)
                   for line in lines)

    share = sum(kept(*cell) for cell in table["cells"]) / len(table["cells"])
    return share if squash(table["caption"]) in squash(output) else share / 2


def document_problems(chunks: Any, error: str | None, meta: dict, file_name: str) -> list[str]:
    """Task 1.2, one document: why its chunks fail the checks (an empty list means they pass)."""
    if error:
        return [f"error: {error}"]
    if not isinstance(chunks, list) or not chunks:
        return ["no chunks"]
    if not all(isinstance(c, Document) for c in chunks):
        return ["not a list of Document"]
    problems = []
    longest = max(len(c.page_content) for c in chunks)
    if longest > MAX_CHUNK_CHARS:
        problems.append(f"a chunk has {longest} characters")
    for key, ok in (("source", lambda v: v == file_name),
                    ("company", lambda v: company_key(v) == company_key(meta["company"])),
                    ("fiscal_year", lambda v: as_year(v) == meta["fiscal_year"]),
                    ("doc_type", lambda v: v == meta["doc_type"])):
        wrong = [c.metadata.get(key) for c in chunks if not ok(c.metadata.get(key))]
        if wrong:
            problems.append(f"{key} {wrong[0]!r} (expected {meta.get(key, file_name)!r}) on {len(wrong)} chunk(s)")
    text = squash("\n".join(c.page_content for c in chunks))
    missing = [p for p in meta["probes"] if squash(p) not in text]
    if len(missing) > (1 - MIN_PROBES_FOUND) * len(meta["probes"]):
        problems.append(f"text missing from the chunks: {missing[:3]}")
    return problems


def parse_score(parsed: Any, gold: dict) -> float:
    """Task 2.2, one question: the share of the three fields that are right."""
    if not isinstance(parsed, dict):
        return 0.0
    expected = gold["query"]
    companies = parsed.get("companies")
    if isinstance(companies, str):
        companies = [companies]
    right = 0
    if isinstance(companies, (list, tuple)) and {company_key(c) for c in companies} == {company_key(c) for c in expected["companies"]}:
        right += 1
    if "fiscal_year" in parsed and as_year(parsed["fiscal_year"]) == expected["fiscal_year"] and \
            (parsed["fiscal_year"] is None) == (expected["fiscal_year"] is None):
        right += 1
    if parsed.get("doc_type") == expected["doc_type"]:
        right += 1
    return right / 3


def contains(chunk_text: str, printed: str) -> bool:
    """The fact as printed in the document is in the chunk (a number only as a whole number)."""
    text, printed = squash(chunk_text), squash(printed)
    if re.fullmatch(r"[\d,.%-]+", printed):
        return re.search(rf"(?<![\d.,]){re.escape(printed)}(?![\d]|[.,]\d)", text) is not None
    return printed in text


def retrieval_hit(chunks: list[Document], gold: dict) -> bool:
    """For every fact the question needs, a retrieved chunk of a document that states it contains it."""
    return bool(gold["evidence"]) and all(
        any(c.metadata.get("source") == file and contains(c.page_content, printed)
            for c in chunks for file, printed in group)
        for group in gold["evidence"])


def clip(x: float) -> float:
    return max(0.0, min(1.0, x))


# ----------------------------------------------------------------------------- runner

def main() -> None:
    parser = argparse.ArgumentParser(description="FTEC5660 HW4: retrieval for RAG over company reports")
    parser.add_argument("--data-folder", default="public_test",
                        help="folder with corpus/, questions.json and (optionally) the answer files")
    parser.add_argument("--output", default="results.csv")
    args = parser.parse_args()

    folder = Path(args.data_folder)
    corpus = load_corpus(folder / "corpus")
    questions = load_json(folder / "questions.json")
    ground_truth = load_json(folder / "ground_truth.json")
    docs_meta = load_json(folder / "docs.json")
    tables = load_json(folder / "tables.json")
    print(f"{len(corpus)} documents, {len(questions)} questions")
    start = time.perf_counter()
    points: dict[str, float] = {}

    # Task 1.1: tables (no model calls allowed)
    if tables:
        before = USAGE.llm_calls
        scores = []
        for table in tables:
            try:
                scores.append(table_score(table_to_text(table["html"]), table))
            except Exception as e:
                print(f"table_to_text failed on a table of {table['source']}: {e!r}")
                scores.append(0.0)
        if USAGE.llm_calls != before:
            print("WARNING: table_to_text called the LLM, which is not allowed (Task 1.1 = 0).")
            scores = [0.0]
        points["1.1"] = POINTS["1.1"] * sum(scores) / len(scores)
        print(f"\nTask 1.1 table_to_text: {sum(scores) / len(scores):.1%} of the table cells kept with their labels")

    # Task 1.2: chunks (no model calls allowed)
    before = USAGE.llm_calls
    chunks_by_file: dict[str, list] = {}
    errors: dict[str, str] = {}
    for file_name, html in corpus.items():
        try:
            chunks_by_file[file_name] = list(chunk_document(file_name, html) or [])
        except Exception as e:
            traceback.print_exc()
            chunks_by_file[file_name], errors[file_name] = [], repr(e)
    called_models = USAGE.llm_calls != before
    if called_models:
        print("WARNING: chunk_document called the LLM, which is not allowed (Task 1.2 = 0).")
    if docs_meta:
        problems = {f: document_problems(chunks_by_file.get(f), errors.get(f), m, f) for f, m in docs_meta.items()}
        passed = sum(not p for p in problems.values()) / len(problems)
        points["1.2"] = 0.0 if called_models else POINTS["1.2"] * passed
        print(f"Task 1.2 chunk_document: {passed:.1%} of the documents pass")
        for f, p in [(f, p) for f, p in problems.items() if p][:5]:
            print(f"  {f}: {'; '.join(p)}")

    # the chunks that retrieve() may return
    all_chunks = [c for f in sorted(chunks_by_file) for c in chunks_by_file[f]
                  if isinstance(c, Document) and len(c.page_content) <= MAX_CHUNK_CHARS]
    by_text: dict[str, list[Document]] = {}
    for c in all_chunks:
        by_text.setdefault(c.page_content, []).append(c)
    print(f"{len(all_chunks)} chunks to index")

    # Task 2.2: query parsing
    parsed: dict[str, Any] = {}
    for qid, question in questions.items():
        try:
            parsed[qid] = parse_query(question)
        except Exception as e:
            parsed[qid] = f"error: {e!r}"

    # Tasks 2.1, 2.3 and 3: retrieval
    index = build_index(all_chunks)
    index_seconds = time.perf_counter() - start
    print(f"Index built in {index_seconds:.1f}s | {USAGE.summary()}")
    retrieved: dict[str, list[Document]] = {}
    valid: dict[str, bool] = {}
    for qid, question in questions.items():
        try:
            result = retrieve(index, question)
            ok = isinstance(result, (list, tuple)) and len(result) <= MAX_RETRIEVED
            kept = []
            for chunk in list(result or []) if isinstance(result, (list, tuple)) else []:
                matches = by_text.get(getattr(chunk, "page_content", None), [])
                if not matches:
                    ok = False
                    continue
                source = (getattr(chunk, "metadata", None) or {}).get("source")
                kept.append(next((m for m in matches if m.metadata.get("source") == source), matches[0]))
            retrieved[qid], valid[qid] = kept[:MAX_RETRIEVED], ok
        except Exception:
            traceback.print_exc()
            retrieved[qid], valid[qid] = [], False
    retrieval_seconds = time.perf_counter() - start - index_seconds

    # Task 4: the fixed generator
    def answer(qid: str) -> str:
        try:
            return generate(questions[qid], retrieved[qid])
        except Exception as e:
            print(f"Generator failed on {qid}: {e!r}")
            return ""

    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = dict(zip(questions, pool.map(answer, questions)))
    total_seconds = time.perf_counter() - start

    rows = []
    for qid, question in questions.items():
        sources = sorted({c.metadata.get("source", "") for c in retrieved[qid]})
        row = {"id": qid, "category": "", "question": question, "parsed": json.dumps(parsed[qid], default=str),
               "parse_score": "", "retrieved": ";".join(sources), "valid": valid[qid], "retrieval_hit": "",
               "answer": answers[qid], "correct": "", "expected": ""}
        if ground_truth and qid in ground_truth:
            gold = ground_truth[qid]
            row["category"] = gold["category"]
            row["parse_score"] = round(parse_score(parsed[qid], gold), 3)
            row["retrieval_hit"] = retrieval_hit(retrieved[qid], gold) if gold["evidence"] else ""
            row["correct"] = answer_correct(answers[qid], gold)
            if not row["correct"]:
                row["expected"] = gold["answer"]
        rows.append(row)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")

    if ground_truth:
        graded = [r for r in rows if r["id"] in ground_truth]
        answerable = [r for r in graded if ground_truth[r["id"]]["evidence"]]
        unanswerable = [r for r in graded if not ground_truth[r["id"]]["evidence"]]
        parse_mean = sum(r["parse_score"] for r in graded) / len(graded)
        non_empty = sum(r["valid"] and bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        recall = sum(r["retrieval_hit"] for r in answerable) / max(len(answerable), 1)
        kept = sum(bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        rejected = sum(not r["retrieved"] for r in unanswerable) / max(len(unanswerable), 1)
        accuracy = sum(r["correct"] for r in graded) / len(graded)
        points["2.1"] = POINTS["2.1"] * non_empty
        points["2.2"] = POINTS["2.2"] * parse_mean
        points["2.3"] = POINTS["2.3"] * clip((recall - RECALL_ZERO) / (RECALL_FULL - RECALL_ZERO))
        points["3"] = POINTS["3"] * clip(kept + rejected - 1)
        points["4"] = POINTS["4"] * accuracy
        print(f"Task 2.1 valid, non-empty retrieval on answerable questions: {non_empty:.1%}")
        print(f"Task 2.2 parse_query fields right: {parse_mean:.1%}")
        print(f"Task 2.3 retrieval recall@{MAX_RETRIEVED} on answerable questions: {recall:.1%}")
        print(f"Task 3   answerable questions with chunks: {kept:.1%} | unanswerable questions with []: {rejected:.1%}")
        print(f"Task 4   answer accuracy: {accuracy:.1%}")
        by_category: dict[str, list[tuple]] = {}
        for r in graded:
            by_category.setdefault(r["category"], []).append((r["correct"], r["retrieval_hit"]))
        print("\nBy category       accuracy   recall")
        for category, results in sorted(by_category.items()):
            hits = [h for _, h in results if h != ""]
            recall_text = f"{sum(hits) / len(hits):.1%}" if hits else "-"
            print(f"  {category:<14}{sum(c for c, _ in results) / len(results):>9.1%}{recall_text:>9}")

    if points:
        print("\nEstimated points (the grade uses the private set):")
        for task in POINTS:
            if task in points:
                print(f"  Task {task:<4}{points[task]:6.2f} / {POINTS[task]}")
        print(f"  Total    {sum(points.values()):6.2f} / {sum(POINTS[t] for t in points)}")

    print(f"\nYour usage: {USAGE.summary()}")
    print(f"Generator usage: {GENERATOR_USAGE.summary()}")
    print(f"Time: {total_seconds:.1f}s (index {index_seconds:.1f}s, retrieval {retrieval_seconds:.1f}s)")
    if total_seconds > TIME_LIMIT_SECONDS:
        print(f"WARNING: the run took longer than the {TIME_LIMIT_SECONDS // 60}-minute limit.")


if __name__ == "__main__":
    main()
