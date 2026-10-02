# Security design

Scope: this project fetches URLs submitted by anyone who can reach the API, and
renders them in a browser. That makes the server a potential proxy into whatever
network it sits in, and a potential victim of a malicious page. The design below
exists for those two reasons and nothing else.

## 1. Threat model

| Adversary | Capability | Mitigation |
| --- | --- | --- |
| Anonymous user | Reach `/analyze` | Rate limit; SSRF guard on every URL |
| Attacker submitting a crafted URL | Reach internal services, cloud metadata, admin panels | `app/security/url_guard.py` |
| Malicious page served to our renderer | Script execution, resource abuse, local file read | Hardened Chromium, scheme filtering, isolated context |
| Attacker sending huge/slow responses | Memory exhaustion, worker starvation | Streaming byte cap, total read deadline, total body cap |

Explicitly **out of scope**: authentication and authorisation (there is none, by
requirement), multi-tenant isolation, and defence against a Chromium
vulnerability (see §4).

## 2. SSRF defence

`app.security.url_guard.validate_target` is the single gate. It is called from
three places, and all three matter:

1. at the top of `/analyze`, **before** any network work;
2. inside `SecureFetcher.fetch` before each request;
3. again at **every redirect hop**, because a permitted public URL can redirect
   to `169.254.169.254`.

Checks, all deny-by-default:

- scheme in `{http, https}` — no `file:`, `ftp:`, `gopher:`, `javascript:`, `data:`
- port in the permit list — so an attacker's port scan never leaves the host
- no embedded credentials — both a phishing tell and a way to hide the real host
- no control characters — they can smuggle a second request line
- length bounded
- hostname **resolved**, and *every* returned address checked as globally
  routable. Checking the hostname string alone is not enough: `evil.example`
  resolving to `127.0.0.1` is the standard bypass, and a list of bad *names* does
  not catch it.
- known internal names (`localhost`, `metadata.google.internal`,
  `instance-data`, …) refused **by name**, so a resolver override or hosts-file
  entry cannot wave them through

Anything that cannot be classified is refused. An address in a private,
loopback, link-local, multicast, reserved, carrier-grade-NAT, or unspecified
range never leaves the process.

### Known limitation

DNS rebinding between validation and connection is mitigated by validating every
address the name resolves to and re-checking at each hop, but the connection is
still made by hostname. A production deployment should pin the connection to the
validated IP with a matching `Host` header. Recorded here rather than glossed.

### Resource-exhaustion controls

Validation resolves DNS, which is a blocking syscall. Run inline it would stall
every concurrent request for the length of the resolver timeout — a trivially
cheap DoS against a no-auth endpoint. It is therefore dispatched to a thread
(`asyncio.to_thread`) at every call site. Two hard deadlines bound a fetch:
`total_timeout` on the body read, and an outer per-URL deadline in the collector.
The renderer is launched once and shared rather than per URL, and teardown is
bounded so a wedged Chromium process cannot stall a collection pass.

## 3. Rate limiting

A fixed window per client IP on `/analyze` only (default 10/minute,
`RATE_LIMIT_PER_MINUTE`). Without authentication this is the only brake on
abuse, so it deliberately does not apply to `/health`, which is cheap and
non-mutating. A shared store (Redis) would be needed for more than one worker
process; the in-process counter does not coordinate across workers.

## 4. Page rendering

Deployment target: a container per capture. Docker is **not installed on this
host**, so the running system hardens Chromium in-process instead:

Provided:

- fresh browser context per page — no cookie or storage carry-over
- every non-`http(s)` scheme aborted at the request layer, which is what stops
  `file://` reads and `ftp://` / `ws://` exfiltration
- all permissions denied (camera, microphone, geolocation, notifications)
- JS dialogs auto-dismissed, so a page cannot stall the collector
- downloads refused
- hard navigation timeout, and a total byte cap on the response
- service workers blocked, HTTPS errors ignored (phishing hosts have bad certs)

Not provided, and this is the real cost: **no filesystem, PID, or network
namespace.** A Chromium vulnerability runs with the collector's privileges. This
is why `/health` reports `screenshot_service: "in_process"` rather than
`"container"` — the UI shows the weaker posture instead of implying isolation
that does not exist.

TLS verification is disabled for fetches, deliberately: phishing pages routinely
serve expired or self-signed certificates. The artifact is treated as hostile
untrusted input and never trusted for anything.

## 5. Data handling

- Collected HTML and screenshots are written under `backend/data/`, which is
  git-ignored. No fetched page content is ever committed.
- `/analyze` returns acquisition *metadata* only. Raw page HTML and raw PNG bytes
  are carried on internal `_`-prefixed keys that the response builder strips, so
  neither the page source nor megabytes of binary can leak into a JSON response.
  `tests/test_collection_regressions.py` pins both halves of that contract.
- Submitted URLs are not persisted unless artifact storage is explicitly enabled
  (`ALLOW_STORE_ARTIFACTS`, off by default).
- Logs are structured JSON. They record URLs, because that is needed to debug
  collection, but never credentials or page bodies.

## 6. What the tests actually cover

`tests/test_url_guard.py` (50 tests) attacks the guard with the real bypasses:
loopback and link-local literals, IPv6 loopback, the cloud metadata address,
private ranges, non-HTTP schemes, non-default ports, embedded credentials,
control characters, over-length input, and names that resolve inward.
`tests/test_api.py` verifies the API refuses the same targets before touching the
network, and that the rate limiter actually engages.

## 7. Deployment checklist

Before exposing this service to any untrusted network:

- [ ] put a reverse proxy in front of it (it has no auth)
- [ ] move the rate limiter to a shared store
- [ ] build and verify the containerised screenshot service
- [ ] pin connections to validated IPs to close the rebinding window
- [ ] terminate TLS at the proxy