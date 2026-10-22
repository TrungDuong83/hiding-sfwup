# Literature search

**Goal.** Find prior methods that hide (sanitize) patterns whose disclosure is defined by the
weighted utility support (wus), i.e. frequent weighted utility patterns or itemsets (FWUP /
FWUI). The search collects and screens records; it draws no conclusion beyond the screening
decisions recorded here.

**Queries.** A × B, publication years 2008–2026.

- A = "weighted utility support", "frequent weighted utility", FWUI, FWUP, "weighted frequent
  itemset", "weighted itemset".
- B = hiding, hide, sanitization, sanitize, "privacy preserving", "knowledge hiding".

## Sources and dates

| Source | How | Queries | Date |
|---|---|---|---|
| OpenAlex | API, 36 A × B queries, up to 200 results each | 36 of 36 succeeded | 2026-10-11 |
| Crossref | API, 36 A × B queries, first 100 results each | 36 of 36 succeeded | 2026-10-11 |
| Semantic Scholar | API, 36 A × B queries, first 100 results each | 19 of 36 succeeded, 17 failed with HTTP 429 | 2026-10-11 to 2026-10-17 |
| DBLP | 6 title queries `<A> hid\|sanitiz\|privacy`, saved from a browser (the API answered the server with a bot check) | 6 of 6 | 2026-10-16 |
| Google Scholar | manual search in a browser, 4 queries (below) | 4 of 4 | 2026-10-16 |
| Google Scholar via SerpAPI | API experiment; results incomplete, **not counted as a systematic source** | 6 + 36 | 2026-10-16 |

The Google Scholar browser queries were `"weighted itemset" "privacy preserving"` (About 40
results, 29 shown), `"weighted frequent itemset" hiding` (About 123, 114 shown),
`"frequent weighted utility" hiding` (About 32; 12 shown, plus 12 more with `filter=0`) and
`"weighted utility support" "privacy preserving"` (2, both shown). Scholar shows fewer results
than "About N" because it groups versions and similar results.

SerpAPI returned far fewer results than the browser: for example
`"weighted itemset" "privacy preserving"` gave 0 results through SerpAPI (also without the year
filter) and about 40 in the browser. Its records are kept in the files and de-duplicated with
the others, but the counts below do not rely on it.

The first DBLP attempt (2026-10-11, 36 A × B queries through the API) failed for every query and
is listed in `queries.csv` (batch `dblp_ab`).

## Screening

1. **De-duplication:** DOI, or normalised title + year; records with the same normalised title
   and years within ±1 (online vs. issue year) are merged.
2. **Automated title/abstract screening:** a record is a candidate when its title plus abstract
   contain at least one phrase of A and at least one of B (case-insensitive regular expressions;
   see `A_match`, `B_match`). Records without an abstract are screened on their title. For
   SerpAPI records the Scholar snippet stands in for the abstract.
3. **Google Scholar browser records** were screened by the authors on title and result page
   (`scholar_browser_screening.csv`); all were excluded, with a reason code.
4. **Eligibility:** every candidate, plus one record added on request (a DBLP hit), was assessed
   by the authors on abstract or full text (`candidates_review.csv`).

**Eligibility criterion (include):** the paper proposes a hiding or sanitization method for
weighted patterns whose disclosure is defined by wus / FWUP / FWUI.

**Reason codes.**

| Code | Meaning |
|---|---|
| E1 | not a scientific publication |
| E2 | pattern-mining method, no hiding or sanitization (B matched generic wording such as "hidden knowledge") |
| E3 | A or B matched only in cited references |
| E4 | application study, or a privacy method that does not hide patterns |
| E5 | hiding method whose disclosure condition is not wus (e.g. multi-level utility) |

## PRISMA flow

| Step | Records |
|---|---:|
| Identified: OpenAlex / Crossref / Semantic Scholar / DBLP / Google Scholar (browser) | 61 / 3,515 / 1,862 / 1 / 169 |
| Identified: Google Scholar via SerpAPI (experimental) | 110 |
| Identified, total | 5,718 |
| Removed: publication year outside 2008–2026 or missing | 9 |
| Removed: duplicates | 3,292 |
| Records screened | 2,417 |
| Excluded at screening, automated rule | 2,258 |
| Excluded at screening by the authors, Google Scholar browser records (E2 126, E3 1, E4 11, E5 1) | 139 |
| Reports sought for retrieval (19 candidates + 1 added on request) | 20 |
| Reports not retrieved (Toshniwal and Verma, DMIN 2010: full text unavailable; cited as related work based on its title) | 1 |
| Reports assessed for eligibility | 19 |
| Reports excluded (E1 1, E2 11, E2+E3 5, E4 2) | 19 |
| **Included** | **0** |

The final decisions were made by the authors on 2026-10-17.

## Limitations

- Semantic Scholar rate-limited the queries: 17 of 36 failed with HTTP 429 after 8 attempts
  (60 s delay, exponential back-off).
- Google Scholar was searched by hand with 4 queries only; SerpAPI results were incomplete.
- DBLP searches titles only (prefix match).
- Crossref ranks by relevance over all metadata; only the first 100 results per query were kept.
- Records without an abstract were screened on their title.

## Files

| File | Content |
|---|---|
| `queries.csv` | every query: source, batch, query, date, total results reported, records retrieved, error, note |
| `records.csv` | de-duplicated records: key, title, year, venue, DOI, sources, A/B phrases matched, screening stage, decision, reason code (automated screening: `no_A_and_B_in_title_abstract`, or `no_A_and_B_in_title_only` when there is no abstract) |
| `candidates_review.csv` | eligibility assessment of the 20 reports sought: decision, reason code, reason, evidence used, final decision |
| `scholar_browser_screening.csv` | Google Scholar browser records (query, title, authors/venue, year) with the authors' decision and reason code |
| `prisma.json` | the counts above |

No abstract, Scholar snippet or raw API response is distributed. For Google Scholar records the
`venue` field is the authors/venue line shown by Scholar.
The records are verbatim metadata from the sources; some titles and venues are not in English.

## Re-running

`scripts/literature_search.py fetch` queries OpenAlex, Crossref and Semantic Scholar and tries
the DBLP API; the responses go to `literature_search/raw/` (not distributed). If the DBLP API
answers with a bot check, save the 6 JSON responses of
`https://dblp.org/search/publ/api?q=<query>&h=1000&format=json` from a browser as
`literature_search/dblp_manual/dblp_Q1.json` … `dblp_Q6.json` (queries in the order of A) and run
`fetch` again. `scripts/literature_search.py serpapi` repeats the SerpAPI experiment (needs the
environment variable `SERPAPI_KEY`). `scripts/literature_search.py build` writes `queries.csv`,
`records.csv` and `prisma.json` from `raw/`, `scholar_browser_screening.csv` and
`candidates_review.csv` (the authors' decisions are inputs). The Google Scholar browser search
is manual. Results depend on the date of the search and on the state of each service.

## Terms of use

The bibliographic metadata comes from OpenAlex, Crossref, Semantic Scholar, DBLP and Google
Scholar; each record is subject to the terms of the service it came from (column `sources`).
Only factual fields are kept: title, year, venue, DOI, source, query and the A/B match flags. The
screening decisions, reason codes and counts were made by the authors and are licensed under
CC BY 4.0 (see `LICENSE-DATA`).
