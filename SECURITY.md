# Security

## Reporting

Found a vulnerability? Please open a private security advisory on GitHub or
contact the maintainer rather than filing a public issue.

## Dependency policy

Dependencies are pinned (`requirements.txt`) with lower bounds in
`pyproject.toml`, and Dependabot is enabled. Known advisories are addressed as
follows.

### Addressed advisories

| Advisory | Package | Severity | Status |
| --- | --- | --- | --- |
| [GHSA-fjqc-hq36-qh5p](https://github.com/advisories/GHSA-fjqc-hq36-qh5p) | `langgraph-checkpoint` | Medium | **Fixed** — upgraded to `4.1.1` |
| [GHSA-gr75-jv2w-4656](https://github.com/advisories/GHSA-gr75-jv2w-4656) | `langchain` | Medium | **Fixed** — upgraded to `1.3.9` |
| [GHSA-f4j7-r4q5-qw2c](https://github.com/advisories/GHSA-f4j7-r4q5-qw2c) | `chromadb` | Critical | **Not exposed** (see below) |

### ChromaDB pre-auth code injection (GHSA-f4j7-r4q5-qw2c)

This advisory describes a **pre-authentication remote code execution** against
the **ChromaDB HTTP server**: an unauthenticated attacker can reach the
`/api/v2/tenants/{tenant}/databases/{db}/collections` endpoint and, with
`trust_remote_code=true`, load a malicious model repository and run arbitrary
code on the server. At the time of writing there is **no patched release** (the
latest published version, `1.5.9`, is still in the vulnerable range).

**ARIA is not exposed to this vulnerability:**

- ARIA uses only the **embedded** `chromadb.PersistentClient(path=...)` — an
  in-process, on-disk store (`memory/chroma_store.py`).
- ARIA never runs the ChromaDB HTTP server, never uses `chromadb.HttpClient`,
  and exposes no ChromaDB network endpoint.
- ARIA never sets `trust_remote_code`.

Because the vulnerable code path (the server's REST endpoint) is never reached,
this critical advisory is not applicable to ARIA's usage. It will be upgraded as
soon as a patched release is available. Do **not** run ARIA's ChromaDB store as
a network-exposed server.

## API keys

`GROQ_API_KEY` and `TAVILY_API_KEY` are read from the environment / `.env`.
`.env` is gitignored — never commit real keys. Research goals and queries are
sent to Groq and Tavily; treat sensitive topics accordingly.
