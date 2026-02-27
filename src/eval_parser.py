"""Parses block quotes and record IDs from LLM output."""

import re
from dataclasses import dataclass


# Matches:  > "some text" — rec_id
# Also handles unicode em-dash (—) and ASCII alternatives (-- or -)
QUOTE_PATTERN = re.compile(
    r'>\s*"([^"]+)"\s*[—\-–]+\s*(\S+)',
    re.MULTILINE,
)


@dataclass
class ParsedQuote:
    text: str           # verbatim text as written by LLM
    record_id: str      # feedback_record_id as written by LLM


def parse_quotes(answer: str) -> list[ParsedQuote]:
    """Extract all block quotes from an LLM answer string."""
    quotes = []
    for match in QUOTE_PATTERN.finditer(answer):
        quotes.append(ParsedQuote(
            text=match.group(1).strip(),
            record_id=match.group(2).strip()
        ))
    return quotes