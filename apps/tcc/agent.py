"""TCC response agent: turns a verified traffic state into an action plan.

This is the paper's future work ("event-specific response generation at the TCC").
Uses the OpenAI Responses API with a JSON schema. Falls back to rules if the API
is unavailable, so the demo never dies on stage.
"""
from __future__ import annotations

import json
import os

PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "severity": {"type": "string", "enum": ["info", "low", "medium", "high", "critical"]},
        "summary": {"type": "string"},
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"type": "string", "enum": [
                        "dispatch_rescue", "notify_police", "adjust_signal_timing",
                        "update_message_sign", "public_alert", "request_human_review", "inspect_camera"]},
                    "detail": {"type": "string"},
                },
                "required": ["type", "detail"],
            },
        },
        "public_alert_th": {"type": "string"},
        "public_alert_en": {"type": "string"},
        "needs_human_approval": {"type": "boolean"},
    },
    "required": ["severity", "summary", "actions", "public_alert_th", "public_alert_en", "needs_human_approval"],
}

SYSTEM = (
    "You are a traffic control center operations assistant in Thailand. You receive a traffic state "
    "inferred at a roadside unit as a 4-class confidence vector [normal_flow, congestion, accident, "
    "invalid_scene] plus recent history. Propose a short, concrete response plan. Never claim certainty "
    "the scores don't support: if confidence is low or CPV changed the label, set needs_human_approval=true "
    "and include request_human_review. Dispatching rescue always needs human approval. Keep alerts under "
    "200 characters."
)


def _fallback(event: dict) -> dict:
    s = event["state"]
    rules = {
        "accident": ("high", [{"type": "request_human_review", "detail": "Confirm accident on CCTV"},
                              {"type": "dispatch_rescue", "detail": "Prepare rescue unit pending confirmation"},
                              {"type": "update_message_sign", "detail": "Accident ahead, use caution"}],
                     "พบอุบัติเหตุข้างหน้า โปรดใช้ความระมัดระวัง", "Accident ahead, please use caution"),
        "congestion": ("medium", [{"type": "adjust_signal_timing", "detail": "Extend green on congested approach"},
                                  {"type": "update_message_sign", "detail": "Heavy traffic ahead"}],
                       "การจราจรหนาแน่นข้างหน้า", "Heavy traffic ahead"),
        "invalid_scene": ("low", [{"type": "inspect_camera", "detail": "Camera view obstructed or degraded"}],
                          "", ""),
    }
    sev, actions, th, en = rules.get(s, ("info", [], "", ""))
    return {"severity": sev, "summary": f"Rule-based plan for {s} at {event['rsu_id']}", "actions": actions,
            "public_alert_th": th, "public_alert_en": en,
            "needs_human_approval": s == "accident" or event["confidence"] < 0.6, "source": "fallback"}


def plan(event: dict, history: list[dict]) -> dict:
    if not os.getenv("OPENAI_API_KEY"):
        return _fallback(event)
    try:
        from openai import OpenAI

        client = OpenAI(timeout=15)
        resp = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-5.5"),
            instructions=SYSTEM,
            input=json.dumps({"event": event, "recent_history": history[-10:]}),
            text={"format": {"type": "json_schema", "name": "response_plan", "schema": PLAN_SCHEMA, "strict": True}},
        )
        out = json.loads(resp.output_text)
        out["source"] = "openai"
        return out
    except Exception as e:  # noqa: BLE001 — never break the pipeline on the agent
        out = _fallback(event)
        out["summary"] += f" (agent error: {type(e).__name__})"
        return out
