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
    """Primary: keyword-regex search for the paragraph containing clause language."""
    for pattern in _KW.get(clause, []):
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return _extract_paragraph(text, m.start(), m.end())
    return ""


def _find_snippet_by_terms(text: str, clause: str) -> str:
    """Fallback: score paragraphs by clause-name term density when no keyword fires."""
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
        self._device = None

        if checkpoint_path is None:
            checkpoint_path = str(Path(__file__).parent / "models" / "Legal-BERT_(CUAD).pt")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

        self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, path: str) -> None:
        try:
            import torch
            from training import ModelArtifacts  # noqa: F401 — needed for unpickling

            device = torch.device(
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
            art = torch.load(path, map_location=device, weights_only=False)
            if hasattr(art.model, "to"):
                art.model = art.model.to(device)
            art.model.eval()

            self.model = art.model
            self.tokenizer = art.tokenizer
            self.id_to_clause = art.id_to_clause
            self.thresholds = {v: 0.5 for v in art.id_to_clause.values()}
            self._device = device
            self.mode = "model"

            # Load per-clause thresholds from s4_outputs.pkl if present
            s4_path = Path(path).parent.parent / "s4_outputs.pkl"
            if s4_path.exists():
                import pickle
                with open(s4_path, "rb") as f:
                    s4 = pickle.load(f)
                if "per_t_best" in s4:
                    self.thresholds = s4["per_t_best"]
                    logger.info("Loaded per-clause thresholds from s4_outputs.pkl.")

            logger.info("Loaded Legal-BERT checkpoint from %s.", path)

        except Exception as exc:
            raise RuntimeError(f"Checkpoint load failed: {exc}") from exc

    def _classify_model(self, text: str) -> list[dict]:
        import torch

        enc = self.tokenizer(
            text,
            max_length=512,
            stride=128,
            truncation=True,
            padding="max_length",
            return_overflowing_tokens=True,
        )

        n_labels = len(self.id_to_clause)
        max_probs = np.zeros(n_labels, dtype=np.float32)

        for i in range(len(enc["input_ids"])):
            ids = torch.tensor(enc["input_ids"][i:i+1], dtype=torch.long).to(self._device)
            mask = torch.tensor(enc["attention_mask"][i:i+1], dtype=torch.long).to(self._device)
            inputs: dict = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in enc:
                inputs["token_type_ids"] = torch.tensor(
                    enc["token_type_ids"][i:i+1], dtype=torch.long
                ).to(self._device)

            with torch.no_grad():
                logits = self.model(**inputs).logits.cpu().numpy()[0]

            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
            for j in range(n_labels):
                if probs[j] > max_probs[j]:
                    max_probs[j] = probs[j]

        results: list[dict] = []
        for label_id, clause_name in self.id_to_clause.items():
            t = self.thresholds.get(clause_name, 0.5)
            prob = float(max_probs[label_id])
            if prob >= t:
                results.append({
                    "clause":     clause_name,
                    "confidence": round(prob, 3),
                    "risk":       RISK_LEVELS.get(clause_name, "LOW"),
                    "category":   _get_category(clause_name),
                    "snippet":    _find_snippet(text, clause_name) or _find_snippet_by_terms(text, clause_name),
                    "detected":   True,
                })

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results

    def classify(self, text: str) -> dict:
        clauses = self._classify_model(text)
        high   = sum(1 for c in clauses if c["risk"] == "HIGH")
        medium = sum(1 for c in clauses if c["risk"] == "MEDIUM")
        low    = sum(1 for c in clauses if c["risk"] == "LOW")
        return {
            "mode":         self.mode,
            "total":        len(clauses),
            "risk_summary": {"HIGH": high, "MEDIUM": medium, "LOW": low},
            "clauses":      clauses,
        }
