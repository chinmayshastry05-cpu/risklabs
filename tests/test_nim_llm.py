"""Tests for the NIM adapter. ALL API interaction here uses a fake transport
— these are MOCKED tests and prove nothing about the live service. Live-call
evidence lives in data/real/llm_features.csv (origin column) instead."""

import json

import pytest

from src.nim_llm import (NimClient, NimConfig, NimError, chunk_text,
                        verify_quote)

CHUNK = (
    "The auditors have expressed material uncertainty related to going "
    "concern. Total borrowings rose sharply and the company breached a "
    "covenant on one term loan during the year."
)


def fake_transport_factory(parsed):
    def _transport(payload, model):
        return {"choices": [{"message": {
            "content": json.dumps(parsed)}}]}
    return _transport


def test_verify_quote():
    assert verify_quote(
        "material uncertainty related to going concern", CHUNK)
    assert verify_quote(  # whitespace differences are normalised
        "Total borrowings  rose\nsharply", CHUNK)
    assert not verify_quote("this sentence is not in the source", CHUNK)
    assert not verify_quote("short", CHUNK)  # too short to be evidence


def test_chunk_text():
    assert chunk_text("hello") == ["hello"]
    assert chunk_text("   ") == []
    long = "\n\n".join(f"paragraph {i} " + "x" * 200 for i in range(40))
    chunks = chunk_text(long, max_chars=1000, overlap=100)
    assert len(chunks) > 1
    assert all(len(c) <= 1100 for c in chunks)


def test_extract_drops_fabricated_quotes():
    parsed = {
        "signals": [
            {"category": "going_concern", "severity": 3,
             "evidence_quote": "material uncertainty related to going concern",
             "explanation": "Auditor raised going-concern uncertainty."},
            {"category": "leverage", "severity": 2,
             "evidence_quote": "the company defaulted on all its loans",
             "explanation": "Fabricated quote — must be dropped."},
            {"category": "not_a_category", "severity": 1,
             "evidence_quote": "breached a covenant on one term loan",
             "explanation": "Bad category — must be dropped."},
        ],
        "overall_assessment": "High risk.",
        "going_concern_flag": True,
    }
    client = NimClient(config=NimConfig(api_key="test"),
                       transport=fake_transport_factory(parsed))
    result = client.extract_signals(CHUNK, report_id="TEST-FY24")
    assert result.origin == "mock"
    assert len(result.signals) == 1
    assert result.signals[0]["category"] == "going_concern"
    assert result.dropped_quotes == 2
    assert result.going_concern_flag is True


def test_extract_rejects_non_json():
    client = NimClient(
        config=NimConfig(api_key="test"),
        transport=lambda payload, model: {
            "choices": [{"message": {"content": "no json here"}}]})
    with pytest.raises(NimError):
        client.extract_signals(CHUNK, report_id="TEST-FY24")


def test_cache_hit_is_labelled_cache(tmp_path):
    parsed = {"signals": [], "overall_assessment": "ok",
              "going_concern_flag": False}
    client = NimClient(config=NimConfig(api_key="test"),
                       cache_dir=tmp_path,
                       transport=fake_transport_factory(parsed))
    key = client._cache_key("TEST-FY24", client.config.model, CHUNK)
    (tmp_path / f"{key}.json").write_text(json.dumps(
        {"report_id": "TEST-FY24", "model": client.config.model,
         "prompt_version": "risklabs-extract-v2", "parsed": parsed}))

    def boom(payload, model):
        raise AssertionError("transport must not be called on a cache hit")

    client._transport = boom
    result = client.extract_signals(CHUNK, report_id="TEST-FY24")
    assert result.origin == "cache"
