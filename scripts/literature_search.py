"""Literature search for prior work on hiding weighted patterns (collection and screening only).

Goal: find methods that hide (sanitize) patterns whose disclosure is defined by the weighted
utility support (wus), i.e. frequent weighted utility patterns / itemsets (FWUP / FWUI).

Queries A x B, publication years 2008-2026:
  A = "weighted utility support" | "frequent weighted utility" | FWUI | FWUP |
      "weighted frequent itemset" | "weighted itemset"
  B = hiding | hide | sanitization | sanitize | "privacy preserving" | "knowledge hiding"

Sources queried through their APIs (`fetch`): OpenAlex, Crossref, Semantic Scholar (36 queries
each), DBLP (6 title queries "<A> hid|sanitiz|privacy"). Google Scholar through SerpAPI
(`serpapi`, needs the environment variable SERPAPI_KEY; the key is never written) was an
experiment with incomplete results and is not counted as a systematic source. Two inputs were
made by hand in a browser and are described in literature_search/README.md:
  - DBLP: literature_search/dblp_manual/dblp_Q1.json ... dblp_Q6.json (used by `fetch` if present);
  - Google Scholar: literature_search/scholar_browser_screening.csv (records + author decisions).

Screening: a record is a candidate when its title + abstract (title only if no abstract) match
>= 1 phrase of A and >= 1 of B. Candidates and records added on request are assessed by the
authors in literature_search/candidates_review.csv (input of `build`).

  python scripts/literature_search.py fetch     -> literature_search/raw/ (not distributed)
  python scripts/literature_search.py serpapi   -> literature_search/raw/ (experimental)
  python scripts/literature_search.py build     -> literature_search/{queries,records}.csv,
                                                   prisma.json
Set LITSEARCH_MAILTO to a contact address for the OpenAlex / Crossref polite pools. Results
depend on the date of the search (each query records retrieved_at).
"""

import collections
import csv
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "literature_search"
)
RAW = os.environ.get("LITSEARCH_RAW", os.path.join(HERE, "raw"))  # API responses
MAILTO = os.environ.get("LITSEARCH_MAILTO", "")
UA = "hiding-sfwup-litsearch/1.0" + (f" (mailto:{MAILTO})" if MAILTO else "")
YEAR0, YEAR1 = 2008, 2026
A = [
    '"weighted utility support"',
    '"frequent weighted utility"',
    "FWUI",
    "FWUP",
    '"weighted frequent itemset"',
    '"weighted itemset"',
]
B = ["hiding", "hide", "sanitization", "sanitize", '"privacy preserving"', '"knowledge hiding"']
A_RE = [
    r"weighted utility support",
    r"frequent weighted utility",
    r"\bfwuis?\b",
    r"\bfwups?\b",
    r"weighted frequent itemsets?",
    r"weighted itemsets?",
]
B_RE = [
    r"\bhid(e|es|ing|den)\b",
    r"saniti[sz](e|es|ed|ing|ation)",
    r"privacy[- ]preserving",
    r"knowledge hiding",
]
DBLP_A = [
    "weighted utility support",
    "frequent weighted utility",
    "FWUI",
    "FWUP",
    "weighted frequent itemset",
    "weighted itemset",
]
DBLP_B = "hid|sanitiz|privacy"  # DBLP: space = AND, | = OR, prefix match
# Google Scholar in a browser: "About N results" shown for each query
BROWSER_TOTALS = {
    '"weighted itemset" "privacy preserving"': 40,
    '"weighted frequent itemset" hiding': 123,
    '"frequent weighted utility" hiding': 32,
    '"weighted utility support" "privacy preserving"': 2,
}
SERPAPI_MAX_CALLS = 80
DELAY = {
    "openalex": 1.0,
    "s2": float(os.environ.get("LITSEARCH_DELAY_S2", 60.0)),
    "dblp": float(os.environ.get("LITSEARCH_DELAY_DBLP", 10.0)),
    "crossref": 1.0,
    "serpapi": 1.0,
}
# (source, batch directory under raw/); the order fixes which duplicate is kept
BATCHES = [
    ("openalex", "openalex"),
    ("s2", "s2"),
    ("dblp", "dblp_ab"),
    ("dblp", "dblp_title"),
    ("crossref", "crossref"),
    ("scholar_serpapi", "serpapi_or"),
    ("scholar_serpapi", "serpapi_pairs"),
]
NAMES = {
    "openalex": "OpenAlex",
    "crossref": "Crossref",
    "s2": "Semantic Scholar",
    "dblp": "DBLP",
    "scholar_browser": "Google Scholar (manual browser search)",
    "scholar_serpapi": "Google Scholar via SerpAPI (experimental, incomplete)",
}


def qid(a, b=None):
    return re.sub(r"[^a-z0-9]+", "_", (a + ("_" + b if b else "")).lower()).strip("_")


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())


# ------------------------------------------------------------------ fetch
def get(url, src, tries=None):
    tries = tries or int(os.environ.get("LITSEARCH_TRIES_" + src.upper(), 8))
    err = "HTTP 429/5xx"
    for k in range(tries):
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode("utf-8"))
            time.sleep(DELAY[src])
            return data, None
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(DELAY[src] * (2**k) + 5)
                continue
            return None, f"HTTP {e.code}"
        except Exception as e:  # network hiccup
            time.sleep(5 * (k + 1))
            err = type(e).__name__
    return None, f"failed after {tries} tries: {err}"


def save(batch, name, payload):
    d = os.path.join(RAW, batch)
    os.makedirs(d, exist_ok=True)
    json.dump(payload, open(os.path.join(d, f"{name}.json"), "w"), indent=1)


def with_mailto(params):
    return dict(params, mailto=MAILTO) if MAILTO else params


def fetch_openalex(a, b):
    q = f"{a} {b}"
    flt = f"from_publication_date:{YEAR0}-01-01,to_publication_date:{YEAR1}-12-31"
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(
        with_mailto({"search": q, "filter": flt, "per-page": 200})
    )
    data, err = get(url, "openalex")
    return {
        "query": q,
        "error": err,
        "total_results": (data or {}).get("meta", {}).get("count"),
        "results": (data or {}).get("results", []),
    }


def fetch_s2(a, b):
    q = f"{a} {b}"
    url = "https://api.semanticscholar.org/graph/v1/paper/search?" + urllib.parse.urlencode(
        {
            "query": q,
            "year": f"{YEAR0}-{YEAR1}",
            "limit": 100,
            "fields": "title,abstract,year,externalIds,venue,authors,url,publicationTypes",
        }
    )
    data, err = get(url, "s2")
    return {
        "query": q,
        "error": err,
        "total_results": (data or {}).get("total"),
        "results": (data or {}).get("data", []),
    }


def fetch_crossref(a, b):
    q = f"{a} {b}"
    flt = f"from-pub-date:{YEAR0},until-pub-date:{YEAR1}"
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(
        with_mailto({"query.bibliographic": q, "filter": flt, "rows": 100})
    )
    data, err = get(url, "crossref")
    msg = (data or {}).get("message", {})
    return {
        "query": q,
        "error": err,
        "total_results": msg.get("total-results"),
        "results": msg.get("items", []),
    }


def dblp_record(i, raw, note, at, err=None):
    hits = ((raw or {}).get("result") or {}).get("hits") or {}
    return {
        "query": f"{DBLP_A[i]} {DBLP_B}",
        "source_note": note,
        "error": err,
        "total_results": int(hits["@total"]) if "@total" in hits else None,
        "results": hits.get("hit", []),
        "retrieved_at": at,
    }


def fetch_dblp_title():
    """6 DBLP title queries; a file in literature_search/dblp_manual/ (saved from a browser) is
    used instead of the API, which may answer with a bot check (HTTP 202, HTML)."""
    for i in range(len(DBLP_A)):
        mp = os.path.join(HERE, "dblp_manual", f"dblp_Q{i + 1}.json")
        if os.path.exists(mp):
            r = dblp_record(i, json.load(open(mp)), "manual browser download", "manual")
        else:
            url = "https://dblp.org/search/publ/api?" + urllib.parse.urlencode(
                {"q": f"{DBLP_A[i]} {DBLP_B}", "h": 1000, "format": "json"}
            )
            data, err = get(url, "dblp", tries=1)
            r = dblp_record(i, data, "API", now(), err)
        r["n_retrieved"] = len(r["results"])
        save("dblp_title", f"dblp_Q{i + 1}", r)
        print(f"dblp Q{i + 1}: total={r['total_results']} retrieved={r['n_retrieved']}")


def fetch():
    for src, fn in (("openalex", fetch_openalex), ("s2", fetch_s2), ("crossref", fetch_crossref)):
        for a in A:
            for b in B:
                q = qid(a, b)
                p = os.path.join(RAW, src, f"{q}.json")
                if os.path.exists(p) and not json.load(open(p)).get("error"):
                    continue
                r = fn(a, b)
                r["retrieved_at"] = now()
                r["n_retrieved"] = len(r["results"])
                save(src, q, r)
                print(f"{src} {q}: total={r['total_results']} retrieved={r['n_retrieved']}")
    fetch_dblp_title()


def scrub(o, key):
    """Drop every field named api_key or whose string value contains the key."""
    if isinstance(o, dict):
        return {
            k: scrub(v, key)
            for k, v in o.items()
            if k != "api_key" and not (isinstance(v, str) and key in v)
        }
    if isinstance(o, list):
        return [scrub(v, key) for v in o if not (isinstance(v, str) and key in v)]
    return o


def serpapi():
    """Experimental: Google Scholar through SerpAPI, (1) one query per A with B joined by OR,
    first 100 results; (2) the 36 single pairs "<A>" <B>, first 40 results. No request URL is
    stored (SerpAPI puts the key in the query string)."""
    key = os.environ.get("SERPAPI_KEY")
    if not key:
        sys.exit("SERPAPI_KEY not set")
    rounds = [("serpapi_or", qid(a), f"{a} ({' OR '.join(B)})", 100) for a in A]
    rounds += [("serpapi_pairs", qid(a, b), f'"{a.strip(chr(34))}" {b}', 40) for a in A for b in B]
    calls = 0
    for batch, name, q, n_max in rounds:
        res, total, err, note = [], None, None, None
        for start in range(0, n_max, 20):
            if calls >= SERPAPI_MAX_CALLS:
                err = "call budget reached"
                break
            url = "https://serpapi.com/search.json?" + urllib.parse.urlencode(
                {
                    "engine": "google_scholar",
                    "q": q,
                    "as_ylo": YEAR0,
                    "as_yhi": YEAR1,
                    "hl": "en",
                    "num": 20,
                    "start": start,
                    "api_key": key,
                }
            )
            data, e = get(url, "serpapi", tries=1)
            del url
            calls += 1
            if e:
                err = e
                break
            data = scrub(data, key)
            if data.get("error"):
                if "hasn't returned any results" in data["error"]:
                    note, total = f"no results at start={start}", total or 0
                else:
                    err = data["error"]
                break
            total = (data.get("search_information") or {}).get("total_results", total)
            org = data.get("organic_results", [])
            res += org
            if len(org) < 20 or not (data.get("serpapi_pagination") or {}).get("next"):
                break
        r = {
            "query": q,
            "error": err,
            "source_note": note,
            "total_results": total,
            "results": res[:n_max],
            "retrieved_at": now(),
        }
        r["n_retrieved"] = len(r["results"])
        save(batch, name, r)
        print(f"{batch} {name}: total={total} retrieved={r['n_retrieved']} err={err}")


# ------------------------------------------------------------------ build
def oa_abstract(inv):
    if not inv:
        return ""
    pos = {i: w for w, idx in inv.items() for i in idx}
    return " ".join(pos[i] for i in sorted(pos))


def strip_tags(s):
    return html.unescape(re.sub(r"<[^>]+>", " ", s or "")).strip()


def norm_title(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def to_record(src, it):
    """One API result -> title, year, doi, venue, abstract (abstract used for screening only)."""
    if src == "openalex":
        loc = ((it.get("primary_location") or {}).get("source") or {}).get("display_name", "")
        return {
            "title": it.get("title") or "",
            "year": it.get("publication_year"),
            "doi": (it.get("doi") or "").replace("https://doi.org/", ""),
            "venue": loc,
            "abstract": oa_abstract(it.get("abstract_inverted_index")),
        }
    if src == "s2":
        return {
            "title": it.get("title") or "",
            "year": it.get("year"),
            "doi": (it.get("externalIds") or {}).get("DOI") or "",
            "venue": it.get("venue") or "",
            "abstract": it.get("abstract") or "",
        }
    if src == "dblp":
        info = it.get("info", {})
        v = info.get("venue", "")
        return {
            "title": strip_tags(info.get("title", "")),
            "year": int(info["year"]) if info.get("year") else None,
            "doi": info.get("doi", ""),
            "venue": v if isinstance(v, str) else ",".join(v),
            "abstract": "",
        }
    if src == "crossref":
        yr = None
        for k in ("published-print", "published-online", "issued"):
            dp = (it.get(k) or {}).get("date-parts")
            if dp and dp[0] and dp[0][0]:
                yr = dp[0][0]
                break
        return {
            "title": strip_tags(" ".join(it.get("title") or [])),
            "year": yr,
            "doi": it.get("DOI", ""),
            "venue": " ".join(it.get("container-title") or []),
            "abstract": strip_tags(it.get("abstract", "")),
        }
    # Google Scholar (SerpAPI): the snippet stands in for the abstract
    pub = (it.get("publication_info") or {}).get("summary", "")
    m = re.search(r"(19|20)\d\d", pub)
    return {
        "title": it.get("title", ""),
        "year": int(m.group(0)) if m else None,
        "doi": "",
        "venue": pub,
        "abstract": it.get("snippet", ""),
    }


def browser_query(row):
    return re.sub(r" \(filter=0\)$", "", row["query"].split(": ", 1)[-1])


def load():
    """All records and per-query metadata, in BATCHES order, then the browser Scholar file."""
    recs, meta = [], []
    for src, batch in BATCHES:
        d = os.path.join(RAW, batch)
        if not os.path.isdir(d):
            continue
        for fn in sorted(f for f in os.listdir(d) if f.endswith(".json")):
            r = json.load(open(os.path.join(d, fn)))
            meta.append(
                {
                    "source": src,
                    "batch": batch,
                    "query": r["query"],
                    "retrieved_at": r.get("retrieved_at"),
                    "total_results": r["total_results"],
                    "n_retrieved": r.get("n_retrieved", len(r["results"])),
                    "error": r.get("error"),
                    "note": r.get("source_note") or r.get("note"),
                }
            )
            for rank, it in enumerate(r["results"], 1):
                recs.append(dict(to_record(src, it), source=src, query=r["query"], rank=rank))
    rows = list(csv.DictReader(open(os.path.join(HERE, "scholar_browser_screening.csv"))))
    for q in sorted(BROWSER_TOTALS, key=qid):
        res = [r for r in rows if browser_query(r) == q]
        nf0 = sum(1 for r in res if r["query"].endswith("(filter=0)"))
        note = "'About N results'; Scholar shows fewer (versions / similar results grouped)"
        if nf0:
            note += f"; {len(res) - nf0} shown + {nf0} more with filter=0"
        meta.append(
            {
                "source": "scholar_browser",
                "batch": "scholar_browser",
                "query": q,
                "retrieved_at": "2026-10-16 (manual)",
                "total_results": BROWSER_TOTALS[q],
                "n_retrieved": len(res),
                "error": None,
                "note": note,
            }
        )
        for rank, it in enumerate(res, 1):
            recs.append(
                {
                    "title": it["title"],
                    "year": int(it["year"]) if it.get("year") else None,
                    "doi": "",
                    "venue": it.get("authors_venue", ""),
                    "abstract": "",
                    "source": "scholar_browser",
                    "query": q,
                    "rank": rank,
                    "browser_code": it["reason_code"],
                }
            )
    return recs, meta


def merge(o, d):
    o["n_hits"] += d["n_hits"]
    for s in d["sources"].split(";"):
        if s not in o["sources"].split(";"):
            o["sources"] += ";" + s
    for k in ("abstract", "doi", "browser_code"):
        if not o.get(k) and d.get(k):
            o[k] = d[k]


def dedup(recs):
    """Key = DOI, else normalised title + year; then same normalised title with years within
    +-1 (online vs issue year) are merged, a record with a DOI being kept first."""
    out = {}
    for r in recs:
        k = ("doi:" + r["doi"].lower()) if r["doi"] else f"t:{norm_title(r['title'])}|{r['year']}"
        if k not in out:
            out[k] = dict(r, key=k, sources=r["source"], n_hits=1)
        else:
            merge(out[k], dict(r, sources=r["source"], n_hits=1))
    by_t = {}
    order = sorted(out.items(), key=lambda kv: (kv[0][:4] != "doi:", kv[1]["year"] or 0))
    for k, d in order:
        t = norm_title(d["title"])
        near = [k2 for k2 in by_t.get(t, []) if abs((out[k2]["year"] or 0) - (d["year"] or 0)) <= 1]
        if t and near:
            merge(out[near[0]], d)
            del out[k]
        else:
            by_t.setdefault(t, []).append(k)
    return sorted(out.values(), key=lambda d: (-(d["year"] or 0), norm_title(d["title"])))


def match(text, pats):
    t = (text or "").lower()
    return [p for p in pats if re.search(p, t)]


def write_csv(name, rows, cols):
    with open(os.path.join(HERE, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def build():
    recs, meta = load()
    # browser Scholar records without a year are kept: the authors screened all of them
    inyr = [
        r
        for r in recs
        if (r["year"] and YEAR0 <= int(r["year"]) <= YEAR1)
        or (not r["year"] and r["source"] == "scholar_browser")
    ]
    dd = dedup(inyr)
    review = {
        norm_title(r["title"]): r
        for r in csv.DictReader(open(os.path.join(HERE, "candidates_review.csv")))
    }
    n_cand, stages = 0, collections.Counter()
    for d in dd:
        ta = (d["title"] or "") + " " + (d["abstract"] or "")
        ma, mb = match(ta, A_RE), match(ta, B_RE)
        d["A_match"], d["B_match"] = "|".join(ma), "|".join(mb)
        cand = bool(ma and mb)
        n_cand += cand
        rv = review.get(norm_title(d["title"]))
        if cand or (rv and rv["added_by"].startswith("author request")):
            d["screening_stage"] = "eligibility (authors)"
            d["decision"] = rv["final_decision"] if rv else "pending review"
            d["reason_code"] = rv["reason_code"] if rv else ""
        elif "scholar_browser" in d["sources"].split(";"):
            d["screening_stage"] = "title (authors, Google Scholar browser search)"
            d["decision"], d["reason_code"] = "exclude", d.get("browser_code", "")
        else:
            d["screening_stage"] = "title/abstract (automated)"
            d["decision"] = "exclude"
            d["reason_code"] = "no_A_and_B_in_title" + ("_abstract" if d["abstract"] else "_only")
        stages[d["screening_stage"]] += 1
    write_csv(
        "records.csv",
        dd,
        ["key", "title", "year", "venue", "doi", "sources", "A_match", "B_match"]
        + ["screening_stage", "decision", "reason_code"],
    )
    write_csv(
        "queries.csv",
        meta,
        ["source", "batch", "query", "retrieved_at", "total_results", "n_retrieved", "error"]
        + ["note"],
    )
    sought = [d for d in dd if d["screening_stage"] == "eligibility (authors)"]
    not_retr = [d for d in sought if d["decision"].startswith("not retrieved")]
    assessed = [d for d in sought if d not in not_retr]
    ident = collections.Counter(r["source"] for r in recs)
    prisma = {
        "date": "2026-10-17",
        "final_decisions_by": "authors",
        "identified_per_source": {NAMES[s]: ident[s] for s in NAMES if s in ident},
        "identified_total": len(recs),
        "excluded_year_or_missing_year": len(recs) - len(inyr),
        "duplicates_removed": len(inyr) - len(dd),
        "after_dedup": len(dd),
        "candidates_A_and_B": n_cand,
        "screened": len(dd),
        "excluded_screening": len(dd) - len(sought),
        "excluded_screening_automated": stages["title/abstract (automated)"],
        "excluded_screening_authors_scholar_browser": stages[
            "title (authors, Google Scholar browser search)"
        ],
        "excluded_screening_authors_reasons": dict(
            sorted(
                collections.Counter(
                    d["reason_code"]
                    for d in dd
                    if d["screening_stage"].startswith("title (authors")
                ).items()
            )
        ),
        "reports_sought": len(sought),
        "reports_not_retrieved": len(not_retr),
        "not_retrieved": [d["title"] for d in not_retr],
        "reports_assessed": len(assessed),
        "reports_excluded": dict(
            sorted(
                collections.Counter(
                    d["reason_code"] for d in assessed if d["decision"].startswith("exclude")
                ).items()
            )
        ),
        "pending_review": sum(1 for d in sought if d["decision"] == "pending review"),
        "included": sum(1 for d in assessed if d["decision"].startswith("include")),
        "per_source_after_dedup": {
            NAMES[s]: sum(1 for d in dd if s in d["sources"].split(";")) for s in NAMES
        },
        "per_source_candidates": {
            NAMES[s]: sum(
                1 for d in dd if s in d["sources"].split(";") and d["A_match"] and d["B_match"]
            )
            for s in NAMES
        },
    }
    json.dump(prisma, open(os.path.join(HERE, "prisma.json"), "w"), indent=1)
    print(json.dumps(prisma, indent=1))


if __name__ == "__main__":
    cmds = {"fetch": fetch, "serpapi": serpapi, "build": build}
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        sys.exit(f"usage: {sys.argv[0]} {{{'|'.join(cmds)}}}")
    cmds[sys.argv[1]]()
