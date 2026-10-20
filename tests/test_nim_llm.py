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


# --- regression: strict validation (independent review, 2026-10-20) ---

_validate = NimClient._validate


def test_signals_null_is_schema_error_not_typeerror():
    with pytest.raises(NimError):
        _validate({"signals": None}, CHUNK)


def test_signals_wrong_container_types():
    for bad in ("text", 42, {"category": "legal"}):
        with pytest.raises(NimError):
            _validate({"signals": bad}, CHUNK)


def test_signals_missing_and_toplevel_not_dict():
    with pytest.raises(NimError):
        _validate({"overall_assessment": "x"}, CHUNK)
    with pytest.raises(NimError):
        _validate(["not", "a", "dict"], CHUNK)


def test_going_concern_flag_string_is_not_truthy():
    for bad in ("false", "true", 1, 0, "yes"):
        r = _validate({"signals": [], "going_concern_flag": bad}, CHUNK)
        assert r.going_concern_flag is False
    r = _validate({"signals": [], "going_concern_flag": True}, CHUNK)
    assert r.going_concern_flag is True


def test_bool_and_float_severity_rejected():
    def one(sev):
        return _validate({"signals": [
            {"category": "going_concern", "severity": sev,
             "evidence_quote": "material uncertainty related to going concern",
             "explanation": "e"}]}, CHUNK)
    assert len(one(True).signals) == 0 and one(True).dropped_quotes == 1
    assert len(one(2.0).signals) == 0
    assert len(one("2").signals) == 0
    assert len(one(2).signals) == 1


def test_non_dict_signal_items_dropped_not_crash():
    r = _validate({"signals": [None, "x", 7]}, CHUNK)
    assert r.signals == [] and r.dropped_quotes == 3


def test_assessment_must_be_string():
    r = _validate({"signals": [], "overall_assessment": 123}, CHUNK)
    assert r.overall_assessment == ""


def test_unsupported_assessment_withheld():
    r = _validate({"signals": [
        {"category": "legal", "severity": 3,
         "evidence_quote": "this quote is fabricated",
         "explanation": "e"}],
        "overall_assessment": "Severe distress."}, CHUNK)
    assert r.signals == []
    assert r.overall_assessment == ""
    assert r.assessment_unsupported is True
    good = [{"category": "going_concern", "severity": 3,
             "evidence_quote": "material uncertainty related to going concern",
             "explanation": "e"}]
    r2 = _validate({"signals": good,
                    "overall_assessment": "Contingencies noted."}, CHUNK)
    assert r2.overall_assessment == "Contingencies noted."
    assert r2.assessment_unsupported is False
    r3 = _validate({"signals": [], "overall_assessment": "Fine."}, CHUNK)
    assert r3.overall_assessment == "" and r3.assessment_unsupported is True


def test_signals_null_end_to_end_raises_nimerror():
    client = NimClient(
        config=NimConfig(api_key="test"),
        transport=lambda payload, model: {"choices": [{"message": {
            "content": '{"signals": null}'}}]})
    with pytest.raises(NimError):
        client.extract_signals(CHUNK, report_id="TEST-FY24")
