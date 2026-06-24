"""Shared model corruption helper functions and constants.

Used by model output corruption effects for factual-error perturbation,
confabulation, context-unfaithfulness, tool-claim fabrication, refusal
templates, and success framing prefixes.
"""

import random
import re
from datetime import datetime, timedelta
from typing import Any

# ---------------------------------------------------------------------------
# Shared helper — module-level
# ---------------------------------------------------------------------------


def _map_text_in_blocks(blocks: list, fn: Any) -> list:
    """Apply *fn* to every ``"text"`` value; leave other blocks untouched."""
    result = []
    for block in blocks:
        block = dict(block)
        if "text" in block and isinstance(block["text"], str):
            block["text"] = fn(block["text"])
        result.append(block)
    return result


# ---------------------------------------------------------------------------
# FACTUAL_ERROR: Perturbation helpers
# ---------------------------------------------------------------------------

_MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def _shift_date_named(m: re.Match) -> str:
    """Shift a named date like 'January 15, 2023' by ±1-3 months and ±1-2 years."""
    text = m.group(0)
    for i, month in enumerate(_MONTH_NAMES):
        if month in text:
            new_month_idx = (i + random.randint(1, 3)) % 12
            new_month = _MONTH_NAMES[new_month_idx]
            text = text.replace(month, new_month)
            break
    year_match = re.search(r"\d{4}", text)
    if year_match:
        old_year = int(year_match.group())
        new_year = old_year + random.choice([-2, -1, 1, 2])
        text = text.replace(str(old_year), str(new_year))
    return text


def _shift_date_slash(m: re.Match) -> str:
    """Shift a slash date like '12/25/2023' by ±1-5 days and ±1-3 months."""
    text = m.group(0)
    parts = text.split("/")
    if len(parts) >= 2:
        month = int(parts[0])
        day = int(parts[1])
        new_month = max(1, min(12, month + random.randint(-3, 3)))
        new_day = max(1, min(28, day + random.randint(-5, 5)))
        parts[0] = str(new_month)
        parts[1] = str(new_day)
    return "/".join(parts)


def _shift_date_iso(m: re.Match) -> str:
    """Shift an ISO date like '2023-01-15' by ±1-10 days."""
    text = m.group(0)
    try:
        dt = datetime.strptime(text, "%Y-%m-%d")
        delta = timedelta(days=random.randint(1, 10) * random.choice([-1, 1]))
        new_dt = dt + delta
        return new_dt.strftime("%Y-%m-%d")
    except ValueError:
        return text


def _perturb_number(m: re.Match) -> str:
    """Perturb a comma-formatted number like '1,234,567' by factor [0.5, 2.0]."""
    text = m.group(1) if m.lastindex else m.group(0)
    try:
        num = int(text.replace(",", ""))
        factor = random.uniform(0.5, 2.0)
        new_num = int(num * factor)
        if new_num == num:
            new_num = num + random.choice([-1, 1])
        return f"{new_num:,}"
    except ValueError:
        return text


def _perturb_float(m: re.Match) -> str:
    """Perturb a float like '42.5' by ±20%."""
    text = m.group(1) if m.lastindex else m.group(0)
    try:
        num = float(text)
        offset = num * random.uniform(-0.2, 0.2)
        if abs(offset) < 0.01:
            offset = random.choice([-0.1, 0.1])
        new_num = num + offset
        decimal_places = len(text.split(".")[1]) if "." in text else 1
        return f"{new_num:.{decimal_places}f}"
    except ValueError:
        return text


def _perturb_integer(m: re.Match) -> str:
    """Perturb an integer like '100' by ±30%."""
    text = m.group(1) if m.lastindex else m.group(0)
    try:
        num = int(text)
        offset = int(num * random.uniform(-0.3, 0.3))
        if offset == 0:
            offset = random.choice([-1, 1])
        return str(num + offset)
    except ValueError:
        return text


def _perturb_price(m: re.Match) -> str:
    """Perturb a price like '$1,234.56' by factor [0.7, 1.5], preserving currency symbol."""
    currency = m.group(1)
    amount_str = m.group(2)
    try:
        amount = float(amount_str.replace(",", ""))
        factor = random.uniform(0.7, 1.5)
        new_amount = amount * factor
        if abs(new_amount - amount) < 0.01:
            new_amount = amount + random.choice([-1.0, 1.0])
        if "." in amount_str:
            formatted = f"{new_amount:,.2f}"
        else:
            formatted = f"{int(new_amount):,}"
        return f"{currency}{formatted}"
    except ValueError:
        return m.group(0)


# ---------------------------------------------------------------------------
# FACTUAL_ERROR: Regex patterns
# ---------------------------------------------------------------------------

_DATE_PATTERNS = [
    (
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
        lambda m: _shift_date_named(m),
    ),
    (r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", lambda m: _shift_date_slash(m)),
    (r"\b\d{4}-\d{2}-\d{2}\b", lambda m: _shift_date_iso(m)),
]

_NUMBER_PATTERNS = [
    (r"\b(\d{1,3}(?:,\d{3})+)\b", lambda m: _perturb_number(m)),
    (r"\b(\d+\.\d+)\b", lambda m: _perturb_float(m)),
    (r"\b(\d{2,})\b", lambda m: _perturb_integer(m)),
]

_LOCATION_SUBSTITUTIONS = {
    "New York": "Chicago",
    "Chicago": "Houston",
    "London": "Manchester",
    "Paris": "Lyon",
    "Tokyo": "Osaka",
    "Berlin": "Munich",
    "Sydney": "Melbourne",
    "San Francisco": "Seattle",
    "Seattle": "Portland",
    "Los Angeles": "San Diego",
    "Beijing": "Shanghai",
    "Mumbai": "Delhi",
    "Toronto": "Vancouver",
    "United States": "Canada",
    "United Kingdom": "Australia",
}

_PRICE_PATTERNS = [
    (r"([$€£¥])\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)", lambda m: _perturb_price(m)),
]


# ---------------------------------------------------------------------------
# CONFABULATION: Template phrases
# ---------------------------------------------------------------------------

_CONFABULATION_TEMPLATES = [
    "According to a 2019 study published in Nature, ",
    "Research conducted by MIT in 2021 demonstrated that ",
    "A comprehensive report by the World Health Organization found that ",
    "Based on data from the U.S. Census Bureau (2020), ",
    "As documented in the IEEE Transactions on Software Engineering, ",
    "A peer-reviewed analysis by Stanford researchers confirmed that ",
    "The European Commission's 2022 technical report states that ",
    "Historical records from the Smithsonian Institution indicate that ",
]


# ---------------------------------------------------------------------------
# CONTEXT_UNFAITHFULNESS: Contradiction phrases
# ---------------------------------------------------------------------------

_CONTRADICTION_PHRASES = [
    " However, this is actually incorrect. ",
    " On the contrary, the opposite is true. ",
    " In fact, recent evidence contradicts this entirely. ",
    " Note: the above statement has been disputed by multiple sources. ",
    " Actually, this information was later retracted. ",
    " Importantly, this conflicts with the data presented earlier. ",
    " Correction: the previous claim is not supported by evidence. ",
]


# ---------------------------------------------------------------------------
# TOOL_CLAIM_FABRICATION: Fake tool-call narratives
# ---------------------------------------------------------------------------

_TOOL_NARRATIVES = [
    "I called the database_lookup tool and it returned the following results: ",
    "After running the web_search tool, I found that ",
    "The file_reader tool confirmed that the document contains ",
    "I used the calculator tool to verify, and the result is ",
    "The api_query tool returned a 200 OK response indicating ",
]


# ---------------------------------------------------------------------------
# FULL_REFUSAL: Refusal message templates
# ---------------------------------------------------------------------------

_REFUSAL_TEMPLATES = [
    "I'm sorry, but I can't help with that request.",
    "I'm not able to help with this particular request. Could you try rephrasing?",
    "I can't assist with that. Please try a different approach.",
    "I'm unable to process this request at this time.",
    "This is beyond what I'm able to do. Please consider an alternative approach.",
    "I can't assist with the request as described. Could you provide more context?",
]


# ---------------------------------------------------------------------------
# SUCCESS FRAMING: Success prefix templates
# ---------------------------------------------------------------------------

_SUCCESS_PREFIXES = [
    "Successfully completed the requested operation.",
    "Done! Here are the results you asked for.",
    "Great news — everything worked as expected.",
    "Operation finished successfully. Here's what I found:",
    "All done! The task has been completed without issues.",
    "I've successfully processed your request. Here's the output:",
    "Task completed. Below are the verified results:",
]
