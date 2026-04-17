"""Unit tests for NL-to-IR translation control flow."""

from nl2ocel.nl_to_ir import NLtoIRTranslator
from nl2ocel.query_verifier import VerifyResult


class DummyRetriever:
    def retrieve(self, question, k=12):
        return {"tables": ["events"], "event_types": [], "relations": []}

    def to_prompt_text(self, question, k=12):
        return "## Relevant Schema\nTable: events"


def test_zero_repair_attempts_makes_single_llm_call(monkeypatch):
    calls = []

    def fake_call_llm(prompt, backend, model, api_key=None):
        calls.append(prompt)
        return """
        {
          "intent": "path_relation",
          "tables": ["events"],
          "select": [{"col": "*", "agg": "COUNT", "alias": "n"}],
          "filters": [],
          "joins": [{"relation_type": "bad_join"}],
          "group_by": [],
          "order_by": [],
          "limit": null,
          "temporal": null,
          "ctes": []
        }
        """

    def always_repair(ir):
        return VerifyResult(
            status="repair",
            errors=["Join 'bad_join' not in whitelist"],
            repair_hint={"allowed_joins": ["billing_to_ar"]},
        )

    monkeypatch.setattr("nl2ocel.nl_to_ir._call_llm", fake_call_llm)

    translator = NLtoIRTranslator(
        retriever=DummyRetriever(),
        verify_fn=always_repair,
        max_repair_attempts=0,
    )
    result = translator.translate("How many linked objects?")

    assert result["attempts"] == 1
    assert result["status"] == "reject"  # repair budget exhausted → normalised to reject
    assert len(calls) == 1
