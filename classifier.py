"""Legal clause classifier — Legal-BERT (CUAD) fine-tuned model with keyword fallback."""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Clause metadata
# ─────────────────────────────────────────────────────────────────────────────

CUAD_CLAUSES: list[str] = sorted([
    "Affiliate License-Licensee", "Affiliate License-Licensor",
    "Agreement Date", "Anti-Assignment", "Arbitration", "Audit Rights",
    "Cap On Liability", "Change Of Control", "Covenant Not To Sue",
    "Dispute Resolution", "Document Name", "Effective Date", "Exclusivity",
    "Expiration Date", "Force Majeure", "Governing Law",
    "IP Ownership Assignment", "Indemnification", "Insurance",
    "Irrevocable Or Perpetual License", "Joint IP Ownership", "License Grant",
    "Liquidated Damages", "Most Favored Nation", "No-Solicit Of Customers",
    "No-Solicit Of Employees", "Non-Compete", "Non-Disparagement",
    "Non-Transferable License", "Notice Period To Terminate Renewal",
    "Parties", "Post-Termination Services", "ROFR/ROFO/ROFN", "Renewal Term",
    "Revenue/Profit Sharing", "Source Code Escrow",
    "Termination For Convenience", "Third Party Beneficiary",
    "Uncapped Liability", "Warranty Duration",
])

RISK_LEVELS: dict[str, str] = {
    "Uncapped Liability":             "HIGH",
    "IP Ownership Assignment":        "HIGH",
    "Non-Compete":                    "HIGH",
    "Anti-Assignment":                "HIGH",
    "Source Code Escrow":             "HIGH",
    "Covenant Not To Sue":            "HIGH",
    "Irrevocable Or Perpetual License": "HIGH",
    "Cap On Liability":               "MEDIUM",
    "Indemnification":                "MEDIUM",
    "Non-Disparagement":              "MEDIUM",
    "No-Solicit Of Customers":        "MEDIUM",
    "No-Solicit Of Employees":        "MEDIUM",
    "Change Of Control":              "MEDIUM",
    "Liquidated Damages":             "MEDIUM",
    "Exclusivity":                    "MEDIUM",
    "Revenue/Profit Sharing":         "MEDIUM",
    "Post-Termination Services":      "MEDIUM",
    "Termination For Convenience":    "MEDIUM",
    "Affiliate License-Licensor":     "MEDIUM",
    "Affiliate License-Licensee":     "MEDIUM",
    "Joint IP Ownership":             "MEDIUM",
    "Most Favored Nation":            "MEDIUM",
    "Third Party Beneficiary":        "MEDIUM",
    "Non-Transferable License":       "MEDIUM",
    "Governing Law":                  "LOW",
    "Agreement Date":                 "LOW",
    "Effective Date":                 "LOW",
    "Expiration Date":                "LOW",
    "Parties":                        "LOW",
    "Document Name":                  "LOW",
    "Renewal Term":                   "LOW",
    "Notice Period To Terminate Renewal": "LOW",
    "Force Majeure":                  "LOW",
    "Warranty Duration":              "LOW",
    "Insurance":                      "LOW",
    "Arbitration":                    "LOW",
    "Dispute Resolution":             "LOW",
    "License Grant":                  "LOW",
    "Audit Rights":                   "LOW",
    "ROFR/ROFO/ROFN":                 "LOW",
}

CATEGORIES: dict[str, list[str]] = {
    "Contract Basics":       ["Document Name", "Parties", "Agreement Date", "Effective Date", "Expiration Date"],
    "Term & Duration":       ["Renewal Term", "Notice Period To Terminate Renewal", "Termination For Convenience", "Post-Termination Services"],
    "Financial & Liability": ["Cap On Liability", "Uncapped Liability", "Liquidated Damages", "Revenue/Profit Sharing", "Insurance"],
    "Risk & Indemnity":      ["Indemnification", "Covenant Not To Sue", "Force Majeure"],
    "IP & Licensing":        ["License Grant", "IP Ownership Assignment", "Joint IP Ownership", "Affiliate License-Licensor", "Affiliate License-Licensee", "Irrevocable Or Perpetual License", "Non-Transferable License", "Source Code Escrow"],
    "Restrictive Covenants": ["Non-Compete", "Non-Disparagement", "No-Solicit Of Customers", "No-Solicit Of Employees", "Exclusivity"],
    "Contract Rights":       ["Anti-Assignment", "Change Of Control", "Audit Rights", "ROFR/ROFO/ROFN", "Third Party Beneficiary", "Most Favored Nation"],
    "Dispute Resolution":    ["Governing Law", "Dispute Resolution", "Arbitration"],
    "Warranties":            ["Warranty Duration"],
}


def _get_category(clause: str) -> str:
    for cat, members in CATEGORIES.items():
        if clause in members:
            return cat
    return "General"


def _classifier_mode_preference() -> str:
    value = os.getenv("LEXSCAN_CLASSIFIER_MODE", "auto").strip().lower()
    return value if value in {"auto", "model", "baseline", "heuristic"} else "auto"


def _resolve_torch_device(torch):
    requested = os.getenv("LEXSCAN_TORCH_DEVICE", "auto").strip().lower()
    if requested and requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            logger.warning("LEXSCAN_TORCH_DEVICE=cuda requested, but CUDA is unavailable.")
        elif requested == "mps" and not torch.backends.mps.is_available():
            logger.warning("LEXSCAN_TORCH_DEVICE=mps requested, but MPS is unavailable.")
        else:
            return torch.device(requested)

    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Keyword patterns for heuristic mode
# ─────────────────────────────────────────────────────────────────────────────

_KW: dict[str, list[str]] = {
    "Affiliate License-Licensee": [
        r"affiliates?\s+of\s+(?:the\s+)?licensee",
        r"licensee(?:'s)?\s+affiliates?",
        r"sublicens\w+\s+to\s+(?:its\s+)?affiliates?",
    ],
    "Affiliate License-Licensor": [
        r"affiliates?\s+of\s+(?:the\s+)?licensor",
        r"licensor(?:'s)?\s+affiliates?",
        r"licensor\s+may\s+grant\s+.{0,30}affiliate",
    ],
    "Agreement Date": [
        r"dated\s+(?:as\s+)?of",
        r"this\s+agreement\s+.{0,20}dated",
        r"entered\s+into\s+as\s+of",
        r"effective(?:ly)?\s+dated",
        r"as\s+of\s+the\s+\d+(?:st|nd|rd|th)?\s+day\s+of",
    ],
    "Anti-Assignment": [
        r"may\s+not\s+assign",
        r"shall\s+not\s+assign",
        r"assignment\s+.{0,60}prior\s+written\s+consent",
        r"no\s+assignment\s+without",
        r"not\s+transfer\s+.{0,20}(?:this\s+)?agreement\s+without",
        r"without\s+.{0,20}prior\s+.{0,10}written\s+consent\s+.{0,20}assign",
    ],
    "Arbitration": [
        r"arbitrat\w+",
        r"binding\s+arbitration",
        r"american\s+arbitration\s+association",
        r"\bAAA\b.{0,30}arbitrat",
        r"\bJAMS\b",
        r"ICC\s+arbitration",
        r"rules\s+of\s+arbitration",
    ],
    "Audit Rights": [
        r"right\s+to\s+audit",
        r"audit\s+rights?",
        r"examine\s+.{0,30}books\s+and\s+records",
        r"inspect\s+.{0,20}(?:books|records)",
        r"books\s+.{0,20}records\s+.{0,30}inspect",
        r"audit\s+.{0,30}books\s+and\s+records",
    ],
    "Cap On Liability": [
        r"in\s+no\s+event\s+.{0,80}(?:shall|will)\s+.{0,40}(?:be\s+)?liable",
        r"limitation\s+of\s+liability",
        r"aggregate\s+liability\s+.{0,60}shall\s+not\s+exceed",
        r"total\s+liability\s+.{0,40}limited\s+to",
        r"maximum\s+(?:aggregate\s+)?liability",
        r"liability\s+.{0,20}capped\s+at",
        r"limit(?:s|ed|ing)?\s+(?:its\s+)?liability\s+to",
    ],
    "Change Of Control": [
        r"change\s+of\s+control",
        r"change-of-control",
        r"merger\s+or\s+acquisition",
        r"sale\s+of\s+all\s+or\s+substantially\s+all",
        r"acquisition\s+of\s+.{0,30}controlling\s+interest",
        r"undergo\s+a\s+change\s+of\s+control",
    ],
    "Covenant Not To Sue": [
        r"covenant\s+not\s+to\s+sue",
        r"agrees?\s+not\s+to\s+.{0,30}(?:bring|file|initiate|commence)\s+.{0,20}(?:suit|action|claim|proceeding)",
        r"waives?\s+.{0,20}right\s+to\s+(?:bring\s+)?(?:a\s+)?(?:legal\s+)?(?:action|suit|claim)",
        r"releases?\s+and\s+covenant\s+not\s+to\s+sue",
    ],
    "Dispute Resolution": [
        r"dispute\s+resolution",
        r"resolv\w+\s+.{0,30}dispute",
        r"mediation\s+.{0,30}arbitration",
        r"executive\s+escalation",
        r"good\s+faith\s+.{0,20}negotiation\s+.{0,20}dispute",
        r"alternative\s+dispute\s+resolution",
    ],
    "Document Name": [
        r"this\s+(?:master\s+)?(?:software\s+|service\s+|professional\s+services?\s+|license\s+|licensing\s+|subscription\s+|co-branding\s+|development\s+|supply\s+)?agreement",
        r"master\s+services?\s+agreement",
        r"software\s+license\s+agreement",
        r"professional\s+services\s+agreement",
        r"end\s+user\s+license\s+agreement",
        r"\bEULA\b",
        r"\bMSA\b\s+.{0,10}(?:means|is|refers)",
    ],
    "Effective Date": [
        r"effective\s+date",
        r"shall\s+(?:become\s+)?(?:effective|take\s+effect)",
        r"commences?\s+on",
        r"takes?\s+effect\s+(?:as\s+of|on)",
    ],
    "Exclusivity": [
        r"exclusive(?:ly)?\s+.{0,30}(?:provider|vendor|supplier|partner|distributor|reseller)",
        r"sole\s+and\s+exclusive",
        r"shall\s+not\s+(?:sell|provide|offer|license)\s+.{0,50}(?:to\s+any\s+other|competing\s+)",
        r"exclusive\s+(?:basis|right|arrangement|relationship)",
        r"exclusivity\s+period",
    ],
    "Expiration Date": [
        r"expir(?:es?|ation)\s+(?:date|on)",
        r"shall\s+expire",
        r"term\s+shall\s+end",
        r"(?:contract|agreement|term)\s+.{0,20}expires?\s+on",
        r"initial\s+term\s+.{0,20}ends?\s+on",
    ],
    "Force Majeure": [
        r"force\s+majeure",
        r"acts?\s+of\s+god",
        r"circumstances?\s+beyond\s+.{0,30}(?:reasonable\s+)?control",
        r"natural\s+disaster",
        r"(?:pandemic|epidemic|outbreak|plague)",
        r"war\s+(?:or\s+)?(?:terrorism|hostilities)",
        r"events?\s+beyond\s+(?:a\s+)?party's\s+control",
    ],
    "Governing Law": [
        r"governing\s+law",
        r"governed\s+by\s+(?:the\s+)?laws",
        r"laws\s+of\s+the\s+state\s+of",
        r"laws\s+of\s+\w+(?:\s+\w+)?\s+shall\s+govern",
        r"subject\s+to\s+the\s+laws\s+of",
        r"construed\s+in\s+accordance\s+with\s+the\s+laws",
        r"choice\s+of\s+law",
    ],
    "IP Ownership Assignment": [
        r"assigns?\s+.{0,30}(?:all\s+)?(?:right|title|interest)\s+.{0,30}intellectual\s+property",
        r"work\s+made\s+for\s+hire",
        r"intellectual\s+property\s+.{0,30}(?:shall\s+)?(?:vest\s+in|belong\s+to|owned\s+by)",
        r"all\s+inventions?\s+.{0,30}assigns?",
        r"hereby\s+assigns?\s+to",
        r"assignment\s+of\s+intellectual\s+property",
        r"ip\s+ownership\s+.{0,20}(?:assign|transfer|vest)",
    ],
    "Indemnification": [
        r"indemnif\w+",
        r"hold\s+harmless",
        r"defend(?:s|ed|ing)?\s+.{0,40}(?:from|against)\s+.{0,40}(?:claim|loss|liability|damage)",
        r"indemnity\s+obligation",
    ],
    "Insurance": [
        r"(?:maintain|carry|obtain|procure)\s+.{0,30}insurance",
        r"general\s+liability\s+insurance",
        r"certificate\s+of\s+insurance",
        r"professional\s+liability\s+(?:insurance)?",
        r"errors\s+and\s+omissions",
        r"commercial\s+general\s+liability",
        r"workers'\s+compensation\s+insurance",
    ],
    "Irrevocable Or Perpetual License": [
        r"irrevocable\s+(?:and\s+)?(?:perpetual\s+)?license",
        r"perpetual\s+(?:and\s+)?(?:irrevocable\s+)?license",
        r"perpetual\s+license",
        r"license\s+.{0,20}irrevocable",
    ],
    "Joint IP Ownership": [
        r"jointly\s+own",
        r"joint\s+ownership\s+of\s+.{0,30}intellectual\s+property",
        r"co-?own\w*",
        r"jointly\s+developed\s+.{0,30}intellectual\s+property",
        r"jointly\s+created\s+work",
        r"joint\s+inventors?",
    ],
    "License Grant": [
        r"hereby\s+grants?\s+.{0,60}license",
        r"grants?\s+to\s+.{0,40}a\s+(?:non-?exclusive|exclusive)\s+(?:license|right)",
        r"right\s+to\s+use\s+.{0,30}software",
        r"license\s+to\s+(?:access|use|copy|reproduce)",
        r"grant(?:s|ed)?\s+a\s+(?:limited\s+)?license",
    ],
    "Liquidated Damages": [
        r"liquidated\s+damages",
        r"agreed\s+(?:upon\s+)?damages",
        r"stipulated\s+damages",
        r"pre-?agreed\s+(?:monetary\s+)?damages",
        r"ascertaining\s+actual\s+damages",
    ],
    "Most Favored Nation": [
        r"most.?favou?red\s+nation",
        r"\bMFN\b",
        r"no\s+less\s+favou?rable\s+terms\s+than",
        r"best\s+(?:available\s+)?price\s+.{0,40}any\s+other\s+customer",
        r"most\s+favorable\s+terms",
    ],
    "No-Solicit Of Customers": [
        r"not\s+(?:to\s+)?solicit\s+.{0,30}(?:customers?|clients?|accounts?)",
        r"no-?solicit\s+.{0,20}customer",
        r"solicit\s+.{0,20}customer\s+.{0,20}(?:prohibited|restrict)",
        r"refrain\s+from\s+solicit\w+\s+.{0,20}customer",
    ],
    "No-Solicit Of Employees": [
        r"not\s+(?:to\s+)?solicit\s+.{0,30}(?:employees?|personnel|staff|workforce)",
        r"non-?solicit\s+.{0,20}employee",
        r"solicit\s+.{0,20}employee\s+.{0,20}(?:prohibited|restrict)",
        r"\bno-?hire\b",
        r"refrain\s+from\s+(?:hiring|solicit\w+)\s+.{0,20}employee",
    ],
    "Non-Compete": [
        r"non-?compet\w+",
        r"not\s+(?:to\s+)?compet\w+\s+with",
        r"competitive\s+(?:activity|business|products?|services?)",
        r"competing\s+(?:products?|services?|business|enterprise)",
        r"shall\s+not\s+engage\s+in\s+.{0,40}(?:business|activity)\s+.{0,30}compet",
        r"competitive\s+enterprise",
    ],
    "Non-Disparagement": [
        r"non-?disparagement",
        r"not\s+(?:to\s+)?disparage",
        r"disparaging\s+(?:remarks?|statements?|comments?|language)",
        r"defamatory\s+(?:statements?|remarks?)",
        r"negative\s+public\s+statements?",
    ],
    "Non-Transferable License": [
        r"non-?transferable",
        r"not\s+(?:to\s+)?transfer\s+.{0,20}license",
        r"may\s+not\s+(?:sublicense|transfer|assign)",
        r"not\s+sublicens\w+",
        r"license\s+is\s+.{0,10}non-?transferable",
    ],
    "Notice Period To Terminate Renewal": [
        r"notice\s+.{0,40}non-?renewal",
        r"written\s+notice\s+.{0,30}prior\s+to\s+.{0,20}renewal",
        r"notice\s+period\s+.{0,20}terminat\w+",
        r"\d+\s+(?:days?|months?)\s+.{0,30}advance\s+notice\s+.{0,30}(?:renewal|terminat)",
        r"notice\s+of\s+non-?renewal\s+.{0,30}\d+\s+days?",
    ],
    "Parties": [
        r"by\s+and\s+between",
        r"hereinafter\s+(?:referred\s+to\s+)?as",
        r"entered\s+into\s+by\s+and\s+between",
        r"collectively\s+referred\s+to\s+as\s+.{0,20}[\"']?parties",
        r"each\s+party\s+and\s+(?:its\s+)?affiliates",
    ],
    "Post-Termination Services": [
        r"post-?terminat\w+\s+services?",
        r"transition\s+services?\s+.{0,30}terminat",
        r"wind-?down\s+services?",
        r"following\s+terminat\w+\s+.{0,40}shall\s+(?:continue|provide|assist)",
        r"survival\s+.{0,20}terminat\w+\s+.{0,20}services?",
    ],
    "ROFR/ROFO/ROFN": [
        r"right\s+of\s+first\s+refusal",
        r"right\s+of\s+first\s+offer",
        r"right\s+of\s+first\s+negotiation",
        r"\bROFR\b",
        r"\bROFO\b",
        r"\bROFN\b",
        r"first\s+refusal\s+right",
    ],
    "Renewal Term": [
        r"(?:automatic(?:ally)?|auto)\s+.{0,20}renew",
        r"renewal\s+term",
        r"successive\s+.{0,20}(?:one-?year\s+)?term",
        r"shall\s+renew\s+(?:for|unless)",
        r"(?:automatically\s+)?extend(?:ed|s)?\s+for\s+(?:an\s+)?additional",
    ],
    "Revenue/Profit Sharing": [
        r"revenue[- ]sharing",
        r"profit[- ]sharing",
        r"\d+\s*%\s+of\s+.{0,20}(?:revenue|profit|sales)",
        r"royalt(?:y|ies)",
        r"revenue\s+share",
        r"share\s+.{0,20}(?:net\s+)?(?:revenue|profit|proceeds)",
    ],
    "Source Code Escrow": [
        r"source\s+code\s+escrow",
        r"deposit\s+.{0,30}source\s+code\s+.{0,30}escrow",
        r"escrow\s+(?:agent|agreement|arrangement)\s+.{0,40}source",
        r"escrow\s+of\s+source\s+code",
    ],
    "Termination For Convenience": [
        r"terminat\w+\s+for\s+convenience",
        r"terminat\w+\s+.{0,40}without\s+cause",
        r"terminat\w+\s+at\s+(?:any\s+time|its\s+(?:sole\s+)?(?:discretion|option))",
        r"may\s+terminat\w+\s+this\s+agreement\s+at\s+any\s+time",
        r"terminat\w+\s+upon\s+.{0,20}written\s+notice\s+.{0,20}without\s+cause",
    ],
    "Third Party Beneficiary": [
        r"third[- ]party\s+beneficiar\w+",
        r"intended\s+beneficiar\w+",
        r"no\s+third[- ]party\s+.{0,30}(?:benefit|rights?|intended)",
        r"no\s+third\s+parties?\s+shall\s+(?:be\s+)?(?:beneficiar\w+|have\s+rights?)",
        r"benefit\s+of\s+third\s+parties",
    ],
    "Uncapped Liability": [
        r"unlimited\s+liability",
        r"no\s+(?:limitation|limit)\s+on\s+liability",
        r"notwithstanding\s+.{0,60}limitation\s+of\s+liability\s+.{0,60}shall\s+not\s+apply",
        r"liability\s+.{0,20}(?:is\s+)?uncapped",
        r"excluded\s+from\s+the\s+limitation\s+of\s+liability",
        r"exceptions?\s+to\s+.{0,20}limitation\s+of\s+liability",
    ],
    "Warranty Duration": [
        r"warranty\s+(?:period|term|for\s+a\s+period)",
        r"warrants?\s+for\s+.{0,20}\d+",
        r"warranty\s+.{0,20}expir\w+",
        r"\d+[- ](?:year|month|day)\s+warranty",
        r"warranty\s+(?:shall\s+)?(?:last|remain\s+in\s+effect|be\s+valid)\s+for",
    ],
}


def _extract_paragraph(text: str, match_start: int, match_end: int) -> str:
    """Return the paragraph containing [match_start, match_end], trimmed to 700 chars."""
    para_start = text.rfind('\n\n', 0, match_start)
    para_start = para_start + 2 if para_start != -1 else 0

    if match_start - para_start > 500:
        nl = text.rfind('\n', para_start, match_start)
        if nl != -1:
            para_start = nl + 1

    para_end = text.find('\n\n', match_end)
    para_end = para_end if para_end != -1 else len(text)

    snippet = text[para_start:para_end].strip()
    if len(snippet) > 700:
        trimmed = snippet[:700]
        for sep in ('\n', '. ', ' '):
            pos = trimmed.rfind(sep)
            if pos > 400:
                return trimmed[:pos + len(sep)].rstrip() + '…'
        return trimmed.rstrip() + '…'
    return snippet


def _find_snippet(text: str, clause: str) -> str:
    """Keyword-regex search for the paragraph containing clause language. Used only as a
    last-resort fallback constrained to a window the model already identified."""
    for pattern in _KW.get(clause, []):
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return _extract_paragraph(text, m.start(), m.end())
    return ""


def _find_snippet_by_terms(text: str, clause: str) -> str:
    """Term-density fallback. Used only as a last-resort within a model-identified window."""
    _STOP = {'of', 'or', 'and', 'the', 'a', 'an', 'to', 'in', 'for', 'on', 'at', 'not'}
    terms = [w.lower() for w in re.split(r'[\s/\-]+', clause)
             if w.lower() not in _STOP and len(w) > 2]
    if not terms:
        return ""

    best_para, best_score = "", 0
    for para in (p.strip() for p in text.split('\n\n') if len(p.strip()) > 50):
        lower = para.lower()
        score = sum(lower.count(t) for t in terms)
        if score > best_score:
            best_score, best_para = score, para

    if not best_para:
        return ""
    if len(best_para) > 700:
        best_para = best_para[:700].rsplit(' ', 1)[0] + '…'
    return best_para


def _candidate_spans(text: str, min_len: int = 60, max_len: int = 1200) -> list[str]:
    """Split a window into clause-sized candidate spans for model rescoring.

    Prefers paragraph breaks; merges fragments shorter than min_len with neighbors;
    splits paragraphs longer than max_len on sentence boundaries. Generalizes across
    legal documents — no hardcoded section names or keywords."""
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()] if text.strip() else []

    out: list[str] = []
    for p in paragraphs:
        if len(p) > max_len:
            # Split overlong paragraphs on sentence boundaries
            sents = re.split(r'(?<=[.!?])\s+(?=[A-Z(])', p)
            cur = ""
            for s in sents:
                if cur and len(cur) + len(s) + 1 > max_len:
                    out.append(cur)
                    cur = s
                else:
                    cur = (cur + ' ' + s).strip() if cur else s
            if cur:
                out.append(cur)
        elif len(p) < min_len and out and len(out[-1]) + len(p) + 1 <= max_len:
            out[-1] = (out[-1] + ' ' + p).strip()
        else:
            out.append(p)

    return [c for c in out if len(c) >= 30]


def _trim_snippet(snippet: str, max_chars: int = 700) -> str:
    """Trim a snippet to max_chars at a clean word boundary."""
    if len(snippet) <= max_chars:
        return snippet
    trimmed = snippet[:max_chars]
    cut = trimmed.rfind(' ')
    if cut > max_chars - 100:
        trimmed = trimmed[:cut]
    return trimmed.rstrip() + '…'


def _classify_heuristic(text: str) -> list[dict]:
    """Keyword-regex classifier. Returns detected clauses sorted by confidence."""
    results: list[dict] = []

    for clause in CUAD_CLAUSES:
        patterns = _KW.get(clause, [])
        if not patterns:
            continue

        total_matches = sum(
            1 for p in patterns for _ in re.finditer(p, text, re.IGNORECASE)
        )

        if total_matches > 0:
            confidence = min(0.52 + total_matches * 0.07, 0.97)
            results.append({
                "clause":     clause,
                "confidence": round(confidence, 3),
                "risk":       RISK_LEVELS.get(clause, "LOW"),
                "category":   _get_category(clause),
                "snippet":    _find_snippet(text, clause) or _find_snippet_by_terms(text, clause),
                "detected":   True,
                "confidence_source": "heuristic_keyword",
            })

    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main classifier
# ─────────────────────────────────────────────────────────────────────────────

class LegalClauseClassifier:
    """Loads a fine-tuned Legal-BERT checkpoint when available; falls back to keyword heuristics."""

    def __init__(self, checkpoint_path: str | None = None) -> None:
        self.mode = "heuristic"
        self.model = None
        self.tokenizer = None
        self.id_to_clause: dict[int, str] = {}
        self.thresholds: dict[str, float] = {}
        self.validation_logits = None
        self.validation_labels = None
        self.calibration_temperature = 1.0
        self.threshold_source = "heuristic_default"
        self.reliability_status = "degraded"
        self.reliability_warning = (
            "Running without trained model artifacts; results come from heuristic keyword rules."
        )
        self._device = None
        self._mode_preference = _classifier_mode_preference()

        if checkpoint_path is None:
            checkpoint_path = str(Path(__file__).parent / "models" / "Legal-BERT_(CUAD).pt")
        baseline_path = str(Path(__file__).parent / "checkpoints" / "tfidf_lr_artifacts.joblib")

        if self._mode_preference == "heuristic":
            logger.info("Classifier mode forced to heuristic via LEXSCAN_CLASSIFIER_MODE.")
            return

        if os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
            return

        if os.path.exists(baseline_path):
            self._load_tfidf_baseline(baseline_path)
            return

        if self._mode_preference == "baseline":
            logger.warning(
                "LEXSCAN_CLASSIFIER_MODE=baseline was requested, but %s is missing.",
                baseline_path,
            )
        elif self._mode_preference == "model":
            logger.warning(
                "LEXSCAN_CLASSIFIER_MODE=model was requested, but the checkpoint is missing."
            )

        # Stay in heuristic mode — regex fallback handles classification.
        logger.warning(
            "No model artifacts found at %s or %s — running in heuristic (regex) mode.",
            checkpoint_path,
            baseline_path,
        )

    def _load_tfidf_baseline(self, path: str) -> None:
        try:
            import joblib

            payload = joblib.load(path)
            model = payload["model"]
            id_to_clause = payload["id_to_clause"]
            best_threshold = float(payload.get("best_threshold", 0.5))
            per_clause_thresholds = payload.get("per_clause_thresholds") or {}

            self.model = model
            self.tokenizer = None
            self.id_to_clause = {int(k): v for k, v in id_to_clause.items()}
            if per_clause_thresholds:
                self.thresholds = self._normalize_thresholds(per_clause_thresholds)
                self.threshold_source = "artifact_per_clause"
            else:
                self.thresholds = {clause: best_threshold for clause in self.id_to_clause.values()}
                self.threshold_source = "artifact_global"
            self.calibration_temperature = float(payload.get("calibration_temperature", 1.0))
            self.reliability_status = "ready"
            self.reliability_warning = ""
            self._device = None
            self.mode = "baseline"
            self.validation_logits = np.asarray(payload.get("val_logits")) if payload.get("val_logits") is not None else None
            self.validation_labels = np.asarray(payload.get("val_labels")) if payload.get("val_labels") is not None else None

            logger.info("Loaded TF-IDF + LR baseline from %s.", path)
            logger.info("Baseline threshold source: %s", self.threshold_source)
            logger.info("Baseline temperature: %.3f", self.calibration_temperature)
        except Exception as exc:
            raise RuntimeError(f"TF-IDF baseline load failed: {exc}") from exc

    def _logits_to_probs(self, logits: np.ndarray) -> np.ndarray:
        scaled = logits / max(float(self.calibration_temperature), 1e-7)
        return 1.0 / (1.0 + np.exp(-np.clip(scaled, -500, 500)))

    def _normalize_thresholds(self, raw_thresholds: dict) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for key, value in raw_thresholds.items():
            clause_name = str(key)
            if isinstance(key, int) and key in self.id_to_clause:
                clause_name = self.id_to_clause[key]
            elif isinstance(key, str) and key.isdigit():
                clause_id = int(key)
                clause_name = self.id_to_clause.get(clause_id, clause_name)
            normalized[clause_name] = float(value)
        return normalized

    def _calibrate_probabilities(self, probs: np.ndarray) -> np.ndarray:
        clipped = np.clip(probs, 1e-7, 1.0 - 1e-7)
        logits = np.log(clipped / (1.0 - clipped))
        return self._logits_to_probs(logits)

    def _classify_baseline(self, text: str) -> list[dict]:
        if self.model is None:
            return []

        probs = self._score_baseline_probs(text)
        results: list[dict] = []

        for label_id, clause_name in self.id_to_clause.items():
            prob = float(probs[label_id])
            threshold = float(self.thresholds.get(clause_name, 0.5))
            if prob < threshold:
                continue

            results.append({
                "clause": clause_name,
                "confidence": round(prob, 3),
                "risk": RISK_LEVELS.get(clause_name, "LOW"),
                "category": _get_category(clause_name),
                "snippet": _find_snippet(text, clause_name) or _find_snippet_by_terms(text, clause_name),
                "detected": True,
                "confidence_source": "calibrated_baseline",
            })

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results

    def _classify_heuristic_mode(self, text: str) -> list[dict]:
        return _classify_heuristic(text)

    def _classify(self, text: str) -> list[dict]:
        if self.mode == "model":
            return self._classify_model(text)
        if self.mode == "baseline":
            return self._classify_baseline(text)
        return self._classify_heuristic_mode(text)

    def _load_checkpoint(self, path: str) -> None:
        try:
            import torch
            from training import ModelArtifacts  # noqa: F401 — needed for unpickling

            device = _resolve_torch_device(torch)
            art = torch.load(path, map_location=device, weights_only=False)
            if hasattr(art.model, "to"):
                art.model = art.model.to(device)
            art.model.eval()

            self.model = art.model
            self.tokenizer = art.tokenizer
            self.id_to_clause = art.id_to_clause
            self.validation_logits = getattr(art, "val_logits", None)
            self.validation_labels = getattr(art, "val_labels", None)
            artifact_thresholds = getattr(art, "per_clause_thresholds", {}) or {}
            self.calibration_temperature = float(getattr(art, "calibration_temperature", 1.0))
            if artifact_thresholds:
                self.thresholds = self._normalize_thresholds(artifact_thresholds)
                self.threshold_source = "artifact_per_clause"
            else:
                fallback_t = float(getattr(art, "best_threshold", 0.5))
                self.thresholds = {v: fallback_t for v in art.id_to_clause.values()}
                self.threshold_source = "artifact_global"
            self._device = device
            self.mode = "model"
            self.reliability_status = "ready"
            self.reliability_warning = ""

            # Load per-clause thresholds from s4_outputs.pkl if present
            s4_path = Path(path).parent.parent / "s4_outputs.pkl"
            if not artifact_thresholds and s4_path.exists():
                import pickle
                with open(s4_path, "rb") as f:
                    s4 = pickle.load(f)
                if "per_t_best" in s4:
                    self.thresholds = self._normalize_thresholds(s4["per_t_best"])
                    self.threshold_source = "legacy_s4_sidecar"
                    self.reliability_status = "compatibility"
                    self.reliability_warning = (
                        "Loaded thresholds from legacy s4_outputs.pkl because the checkpoint "
                        "artifact did not include per-clause thresholds."
                    )
                    logger.info("Loaded per-clause thresholds from s4_outputs.pkl.")

            logger.info("Loaded Legal-BERT checkpoint from %s.", path)
            logger.info("Classifier device: %s", device)
            logger.info("Threshold source: %s | temperature=%.3f", self.threshold_source, self.calibration_temperature)

        except Exception as exc:
            raise RuntimeError(f"Checkpoint load failed: {exc}") from exc

    def _candidate_label_space(self) -> list[str]:
        if self.id_to_clause:
            return [self.id_to_clause[i] for i in sorted(self.id_to_clause)]
        return list(CUAD_CLAUSES)

    def _score_baseline_probs(self, text: str) -> np.ndarray:
        if self.model is None:
            return np.zeros(len(self._candidate_label_space()), dtype=np.float32)
        return np.asarray(self._calibrate_probabilities(self.model.predict_proba([text])[0]), dtype=np.float32)

    def _score_model_probs(self, text: str) -> np.ndarray:
        import torch

        if self.model is None or self.tokenizer is None:
            return np.zeros(len(self._candidate_label_space()), dtype=np.float32)

        enc = self.tokenizer(
            text,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits.cpu().numpy()[0]
        return np.asarray(self._logits_to_probs(logits), dtype=np.float32)

    def _score_heuristic_probs(self, text: str) -> np.ndarray:
        lower = text.lower()
        scores: list[float] = []
        for clause in self._candidate_label_space():
            patterns = _KW.get(clause, [])
            total_matches = sum(1 for pattern in patterns for _ in re.finditer(pattern, text, re.IGNORECASE))
            if total_matches > 0:
                scores.append(min(0.52 + total_matches * 0.07, 0.97))
                continue

            clause_terms = [term for term in re.split(r"[\s/\-]+", clause.lower()) if len(term) > 2]
            overlap = sum(1 for term in clause_terms if term in lower)
            scores.append(min(overlap * 0.12, 0.36))

        return np.asarray(scores, dtype=np.float32)

    def get_top_candidate_labels(
        self,
        snippet: str,
        top_k: int = 3,
        ensure_labels: list[str] | None = None,
    ) -> list[dict]:
        """Return the top 2-3 label candidates for a snippet without altering the main flow."""
        labels = self._candidate_label_space()
        if self.mode == "model":
            probs = self._score_model_probs(snippet)
        elif self.mode == "baseline":
            probs = self._score_baseline_probs(snippet)
        else:
            probs = self._score_heuristic_probs(snippet)

        scored = [
            {"label": label, "confidence": round(float(probs[idx]), 3)}
            for idx, label in enumerate(labels)
        ]
        scored.sort(key=lambda item: item["confidence"], reverse=True)

        target_k = max(2, min(max(int(top_k), 1), 3))
        top = scored[:target_k]
        existing = {item["label"] for item in top}

        for label in ensure_labels or []:
            if label in existing or label not in labels:
                continue
            idx = labels.index(label)
            top.append({"label": label, "confidence": round(float(probs[idx]), 3)})

        top.sort(key=lambda item: item["confidence"], reverse=True)
        deduped: list[dict] = []
        seen: set[str] = set()
        for item in top:
            if item["label"] in seen:
                continue
            seen.add(item["label"])
            deduped.append(item)

        return deduped[: max(target_k, len(ensure_labels or []))]

    def _rescore_candidates(self, candidates: list[str], batch_size: int = 8) -> np.ndarray:
        """Run the same fine-tuned model over short candidate spans. Returns
        an (n_candidates, n_labels) probability matrix. Used to align retrieval
        with detection — the snippet shown is literally the span the model
        scores highest for the detected label."""
        import torch

        n_labels = len(self.id_to_clause)
        if not candidates:
            return np.zeros((0, n_labels), dtype=np.float32)

        probs_all = np.zeros((len(candidates), n_labels), dtype=np.float32)
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start:start + batch_size]
            enc = self.tokenizer(
                batch,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(self._device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits.cpu().numpy()
            probs_all[start:start + len(batch)] = self._logits_to_probs(logits)
        return probs_all

    def _classify_model(self, text: str) -> list[dict]:
        """Detect clauses and locate their supporting spans using the SAME model
        for both. The window where label X scores highest is, by construction, the
        strongest evidence span for X. We then rescore paragraph-sized candidates
        within that window to pick the most evidentiary excerpt — eliminating the
        keyword/term-density mismatch the prior architecture produced."""
        import torch

        # Try fast tokenizer with offset mapping; fall back gracefully if unavailable.
        try:
            enc = self.tokenizer(
                text,
                max_length=512,
                stride=128,
                truncation=True,
                padding="max_length",
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
            )
            offsets_available = True
        except (NotImplementedError, ValueError):
            enc = self.tokenizer(
                text,
                max_length=512,
                stride=128,
                truncation=True,
                padding="max_length",
                return_overflowing_tokens=True,
            )
            offsets_available = False

        n_windows = len(enc["input_ids"])
        n_labels = len(self.id_to_clause)
        window_probs = np.zeros((n_windows, n_labels), dtype=np.float32)

        for i in range(n_windows):
            ids = torch.tensor(enc["input_ids"][i:i+1], dtype=torch.long).to(self._device)
            mask = torch.tensor(enc["attention_mask"][i:i+1], dtype=torch.long).to(self._device)
            inputs: dict = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in enc:
                inputs["token_type_ids"] = torch.tensor(
                    enc["token_type_ids"][i:i+1], dtype=torch.long
                ).to(self._device)

            with torch.no_grad():
                logits = self.model(**inputs).logits.cpu().numpy()[0]

            window_probs[i] = self._logits_to_probs(logits)

        max_probs = window_probs.max(axis=0) if n_windows else np.zeros(n_labels, dtype=np.float32)
        best_windows = window_probs.argmax(axis=0) if n_windows else np.zeros(n_labels, dtype=np.int64)

        # Map each window to its character span in the source text (Fast tokenizer path)
        window_spans: list[tuple[int, int]] = []
        if offsets_available:
            for i in range(n_windows):
                offsets = enc["offset_mapping"][i]
                real = [(s, e) for s, e in offsets if e > s]
                window_spans.append((real[0][0], real[-1][1]) if real else (0, len(text)))
        else:
            # No offsets — treat the whole document as one window for fallback purposes
            window_spans = [(0, len(text))] * max(n_windows, 1)

        # Identify detected clauses
        detected: list[tuple[int, str, float]] = []
        for label_id, clause_name in self.id_to_clause.items():
            t = self.thresholds.get(clause_name, 0.5)
            prob = float(max_probs[label_id])
            if prob >= t:
                detected.append((label_id, clause_name, prob))

        # Build a deduped candidate pool from each winning window. Each candidate
        # paragraph carries its source-window id so we can match it back to clauses.
        candidates: list[str] = []
        candidate_window: list[int] = []
        if detected:
            winning_windows = sorted({int(best_windows[lid]) for lid, _, _ in detected})
            for w in winning_windows:
                ws, we = window_spans[w]
                window_text = text[ws:we]
                for cand in _candidate_spans(window_text):
                    candidates.append(cand)
                    candidate_window.append(w)

        cand_probs = self._rescore_candidates(candidates)

        # Pick the highest-scoring candidate within each clause's winning window
        results: list[dict] = []
        for label_id, clause_name, prob in detected:
            best_w = int(best_windows[label_id])
            snippet = ""
            best_score = -1.0
            for j, w in enumerate(candidate_window):
                if w == best_w and cand_probs[j, label_id] > best_score:
                    best_score = float(cand_probs[j, label_id])
                    snippet = candidates[j]

            # Last-resort fallback: keyword/term search constrained to the winning window
            if not snippet:
                ws, we = window_spans[best_w]
                window_text = text[ws:we]
                snippet = (
                    _find_snippet(window_text, clause_name)
                    or _find_snippet_by_terms(window_text, clause_name)
                    or window_text
                )

            results.append({
                "clause":     clause_name,
                "confidence": round(prob, 3),
                "risk":       RISK_LEVELS.get(clause_name, "LOW"),
                "category":   _get_category(clause_name),
                "snippet":    _trim_snippet(snippet),
                "detected":   True,
                "confidence_source": "calibrated_model",
            })

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results

    def classify(self, text: str) -> dict:
        clauses = self._classify(text)
        high   = sum(1 for c in clauses if c["risk"] == "HIGH")
        medium = sum(1 for c in clauses if c["risk"] == "MEDIUM")
        low    = sum(1 for c in clauses if c["risk"] == "LOW")
        return {
            "mode":         self.mode,
            "degraded_mode": self.mode == "heuristic",
            "total":        len(clauses),
            "risk_summary": {"HIGH": high, "MEDIUM": medium, "LOW": low},
            "reliability": {
                "status": self.reliability_status,
                "threshold_source": self.threshold_source,
                "calibration_temperature": round(float(self.calibration_temperature), 3),
                "warning": self.reliability_warning,
            },
            "clauses":      clauses,
        }
