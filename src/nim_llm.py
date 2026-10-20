"""NVIDIA NIM (Nemotron) adapter for RiskLabs — real LLM risk-signal extraction.

Sends PUBLIC annual-report excerpts to NVIDIA's OpenAI-compatible endpoint
(https://integrate.api.nvidia.com/v1/chat/completions) and returns structured
risk signals whose evidence quotes are verified, character-for-character after
whitespace normalisation, against the source excerpt. Quotes that fail
verification are dropped and counted — never reworded, never kept silently.

Configuration (environment):
  NVIDIA_API_KEY   API key (portable mode). In the Muse environment the key
                   is NOT placed in env: if the variable is absent, the
                   adapter falls back to the registered `custom.nvidia-nim`
                   connector surrogate, so the real credential never appears
                   in env, code, logs, or this repo.
  NVIDIA_MODEL     Model id. Default: nvidia/nemotron-3-ultra-550b-a55b
                   (verified live against this account 2026-10-20; several
                   listed Nemotron ids return 404 for the account and some
                   return 503 at capacity — retries + fallbacks handle that).
  NVIDIA_BASE_URL  Default: https://integrate.api.nvidia.com/v1
  NIM_FALLBACK_MODELS  Comma-separated fallback model ids on 404/503.
  NIM_CACHE_DIR    Disk cache dir. Default: data/real/llm_cache.

Every result is labelled with its origin: "live" (a real API response in this
run), "cache" (a previously recorded live response), or "mock" (test double).
A cached or mocked result is never presented as a fresh live call.

Only public report excerpts may be passed in. No company-private data, no
credentials, and no user data are ever sent.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_FALLBACKS = [
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "nvidia/nemotron-nano-3-30b-a3b",
]
PROMPT_VERSION = "risklabs-extract-v2"

CATEGORIES = {
    "liquidity", "leverage", "profitability", "credit",
    "legal", "going_concern", "operational", "positive",
}

SYSTEM_PROMPT = (
    "You are a financial-risk analyst extracting structured risk signals from "
    "an excerpt of a published Indian company annual report. Respond with a "
    "single JSON object and nothing else, of this exact shape:\n"
    '{"signals": [{"category": one of liquidity|leverage|profitability|credit|'
    "legal|going_concern|operational|positive, "
    '"severity": integer 0-3 (0=informational, 3=severe), '
    '"evidence_quote": a verbatim quote copied exactly from the excerpt, '
    '"explanation": one short sentence}], '
    '"overall_assessment": one or two sentences, '
    '"going_concern_flag": true or false}\n'
    "Reason silently. Output the JSON object only — no reasoning, no "
    "commentary, no code fences. Rules: every evidence_quote must be copied "
    "character-for-character from "
    "the excerpt; do not paraphrase quotes; if the excerpt shows no risk "
    "signal, return an empty signals list; use category 'positive' for "
    "evidence of financial strength. Set going_concern_flag true only if the "
    "excerpt explicitly raises going-concern doubt or material uncertainty."
)


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def verify_quote(quote: str, source: str) -> bool:
    """True iff `quote` appears verbatim (modulo whitespace) in `source`."""
    q = _normalize_ws(quote)
    if len(q) < 12:  # too short to be evidence
        return False
    return q in _normalize_ws(source)


def chunk_text(text: str, max_chars: int = 6000, overlap: int = 400) -> list[str]:
    """Split long text into overlapping chunks on paragraph boundaries."""
    text = text or ""
    if len(text) <= max_chars:
        return [text] if text.strip() else []
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            cut = text.rfind("\n", start, end)
            if cut > start + max_chars // 2:
                end = cut
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c.strip()]


@dataclass
class NimConfig:
    api_key: str | None = None
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    fallbacks: list[str] = field(default_factory=lambda: list(DEFAULT_FALLBACKS))
    timeout_s: float = 120.0
    max_retries: int = 3  # per model, on 429/5xx/timeout only
    min_interval_s: float = 0.0  # min spacing between live calls (free tier)

    @classmethod
    def from_env(cls) -> "NimConfig":
        fb = os.environ.get("NIM_FALLBACK_MODELS")
        return cls(
            api_key=os.environ.get("NVIDIA_API_KEY") or None,
            model=os.environ.get("NVIDIA_MODEL", DEFAULT_MODEL),
            base_url=os.environ.get("NVIDIA_BASE_URL", DEFAULT_BASE_URL),
            fallbacks=[m.strip() for m in fb.split(",") if m.strip()] if fb
            else list(DEFAULT_FALLBACKS),
            min_interval_s=float(os.environ.get("NIM_MIN_INTERVAL_S", "0")),
        )


@dataclass
class ExtractionResult:
    signals: list[dict]
    overall_assessment: str
    going_concern_flag: bool
    model: str
    origin: str  # "live" | "cache" | "mock"
    dropped_quotes: int = 0
    raw_signals: int = 0
    # True when the model offered an overall_assessment but no verified
    # signal survived to support it: the assessment is then withheld
    # (overall_assessment == "") and must never be shown as evidence.
    assessment_unsupported: bool = False


class NimError(RuntimeError):
    pass


class NimClient:
    """Chat-completions client with JSON validation, quote verification,
    disk cache keyed by (report, model, prompt), and bounded retries."""

    def __init__(self, config: NimConfig | None = None, cache_dir=None,
                 transport=None):
        self.config = config or NimConfig.from_env()
        self.cache_dir = Path(cache_dir or os.environ.get(
            "NIM_CACHE_DIR", "data/real/llm_cache"))
        # transport(request_dict, model) -> response_dict; injectable for tests
        self._transport = transport
        self._last_live_call = 0.0

    def _respect_min_interval(self):
        """Space out live calls: the free tier rate-limits aggressively
        (HTTP 429 after a handful of rapid calls — observed 2026-10-20)."""
        if self._transport is not None or self.config.min_interval_s <= 0:
            return
        wait = self.config.min_interval_s - (time.time()
                                              - self._last_live_call)
        if wait > 0:
            time.sleep(wait)
        self._last_live_call = time.time()

    # -- HTTP ------------------------------------------------------------
    def _default_transport(self, payload: dict) -> dict:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if self.config.api_key:
            req.add_header("Authorization", f"Bearer {self.config.api_key}")
        else:
            # Muse environment: attach the connector surrogate; the real
            # credential is swapped in by the runtime on this registered host.
            try:
                import sys
                sys.path.insert(
                    0, "/opt/muse/skills/custom-connector-setup/bin")
                from dynamic_credentials import add_surrogate_to_request
                add_surrogate_to_request(
                    req, "custom.nvidia-nim",
                    allowed_hosts=["integrate.api.nvidia.com"])
            except Exception as exc:  # no env key and no connector -> offline
                raise NimError(
                    "No NVIDIA_API_KEY in env and connector credential "
                    f"unavailable ({exc}). Set NVIDIA_API_KEY or run inside "
                    "the Muse environment with the NIM connector connected."
                ) from exc

        class _NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args):
                raise NimError("redirect refused; verify the endpoint")

        with urllib.request.build_opener(_NoRedirect()).open(
                req, timeout=self.config.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # -- Cache -------------------------------------------------------------
    def _cache_key(self, report_id: str, model: str, chunk: str) -> str:
        h = hashlib.sha256()
        h.update(f"{report_id}|{model}|{PROMPT_VERSION}|".encode())
        h.update(chunk.encode("utf-8"))
        return h.hexdigest()

    # -- Core --------------------------------------------------------------
    def _call_with_retries(self, payload: dict, model: str) -> dict:
        attempt = 0
        while True:
            attempt += 1
            try:
                if self._transport is not None:
                    return self._transport(payload, model)
                return self._default_transport(payload)
            except urllib.error.HTTPError as exc:
                retryable = exc.code in (429, 500, 502, 503, 504)
                if not retryable or attempt > self.config.max_retries:
                    raise NimError(f"HTTP {exc.code} from NIM: "
                                   f"{exc.read().decode(errors='replace')[:300]}"
                                   ) from exc
                # 429 = free-tier rate limit: back off much longer.
                delay = min(2 ** attempt, 20)
                if exc.code == 429:
                    delay = max(delay, 30 * attempt)
                time.sleep(delay)
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt > self.config.max_retries:
                    raise NimError(f"network error calling NIM: {exc}") from exc
                time.sleep(min(2 ** attempt, 20))

    @staticmethod
    def _validate(parsed: dict, chunk: str) -> ExtractionResult:
        """Strict schema validation of the model's JSON object.

        Container-level problems (not a dict, missing/non-list `signals`)
        raise NimError. Item-level problems drop that signal and count it.
        Types are checked strictly: severity must be a real int (bool is a
        subclass of int in Python and is rejected, as are floats/strings);
        going_concern_flag must be a real bool (the string "false" is NOT
        truthy-coerced — anything that is not literally true counts as
        false); overall_assessment must be a string, and it is withheld
        entirely when no verified signal survives to support it.
        """
        if not isinstance(parsed, dict):
            raise NimError("model response failed schema validation: "
                           "top level is not a JSON object")
        raw = parsed.get("signals")
        if not isinstance(raw, list):
            raise NimError("model response failed schema validation: "
                           "'signals' is missing or is not a list")
        signals, dropped = [], 0
        for sig in raw:
            if (not isinstance(sig, dict)
                    or sig.get("category") not in CATEGORIES
                    or type(sig.get("severity")) is not int
                    or not 0 <= sig["severity"] <= 3
                    or not isinstance(sig.get("evidence_quote"), str)
                    or not isinstance(sig.get("explanation"), str)):
                dropped += 1
                continue
            if not verify_quote(sig["evidence_quote"], chunk):
                dropped += 1  # unverifiable quote: drop, never reword
                continue
            signals.append({
                "category": sig["category"],
                "severity": sig["severity"],
                "evidence_quote": _normalize_ws(sig["evidence_quote"]),
                "explanation": sig["explanation"].strip(),
            })
        flag = parsed.get("going_concern_flag")
        assessment = parsed.get("overall_assessment")
        assessment = assessment.strip() if isinstance(assessment, str) else ""
        unsupported = bool(assessment) and not signals
        if unsupported:
            assessment = ""  # no verified evidence -> no conclusion shown
        return ExtractionResult(
            signals=signals,
            overall_assessment=assessment,
            going_concern_flag=flag if type(flag) is bool else False,
            model="", origin="live",
            dropped_quotes=dropped,
            raw_signals=len(raw),
            assessment_unsupported=unsupported,
        )

    def extract_signals(self, chunk: str, report_id: str,
                        use_cache: bool = True) -> ExtractionResult:
        """Extract verified risk signals from one chunk of report text."""
        models = [self.config.model] + [m for m in self.config.fallbacks
                                        if m != self.config.model]
        last_error = None
        for model in models:
            key = self._cache_key(report_id, model, chunk)
            cache_file = self.cache_dir / f"{key}.json"
            if use_cache and cache_file.exists():
                cached = json.loads(cache_file.read_text())
                result = self._validate(cached["parsed"], chunk)
                result.model, result.origin = model, "cache"
                return result
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content":
                     f"Report excerpt ({report_id}):\n\n{chunk}"},
                ],
                "temperature": 0.0,
                # Reasoning-style Nemotron models spend tokens thinking
                # before the JSON; too small a budget truncates the reply
                # mid-reasoning and no JSON is emitted at all.
                "max_tokens": 6000,
            }
            try:
                self._respect_min_interval()
                data = self._call_with_retries(payload, model)
            except NimError as exc:
                last_error = exc
                if any(code in str(exc) for code in
                       ("HTTP 404", "HTTP 429", "HTTP 503")):
                    continue  # try the next model
                raise
            content = (data.get("choices", [{}])[0]
                       .get("message", {}).get("content", ""))
            parsed = _parse_json_object(content)
            result = self._validate(parsed, chunk)
            result.model = model
            result.origin = "mock" if self._transport is not None else "live"
            if self._transport is None:  # only real responses enter the cache
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps({
                    "report_id": report_id, "model": model,
                    "prompt_version": PROMPT_VERSION, "parsed": parsed,
                }, indent=2))
            return result
        raise NimError(f"all NIM models failed; last error: {last_error}")


def _parse_json_object(content: str) -> dict:
    """Parse the model's JSON object, tolerating code fences / preamble."""
    content = (content or "").strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    # Prefer the last anchor of our schema (a reasoning preamble may quote
    # braces or schema fragments before the real answer).
    anchor = content.rfind('{"signals"')
    if anchor >= 0:
        try:
            return json.loads(content[anchor:])
        except json.JSONDecodeError:
            pass
    start, end = content.find("{"), content.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(content[start:end + 1])
        except json.JSONDecodeError:
            pass
    raise NimError(f"model did not return JSON: {content[:200]!r}")
