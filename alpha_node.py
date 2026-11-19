"""Node 2: Alpha Macro Circuit Breaker powered by the Gemini Decisions API.

Uses ``google.genai.Client().decisions.create(model="gemini-decision-1")`` with a
single ``noul`` (yes/no) question (``circuit_breaker_active``) evaluated against
``state["macro_context"]``. If the probability exceeds ``0.85``, emits
``CIRCUIT_BREAKER`` to suspend trading and cancel resting orders; otherwise emits
``HOLD``.
"""

import os
from typing import Any
from dotenv import load_dotenv
from google import genai

from state import AgentState

load_dotenv()

DECISION_MODEL: str = "gemini-decision-1"
GEMINI_MODEL: str = DECISION_MODEL
CIRCUIT_BREAKER_THRESHOLD: float = 0.85
CIRCUIT_BREAKER_QUESTION_ID: str = "circuit_breaker_active"
CIRCUIT_BREAKER_INSTRUCTION: str = (
    "Is a high-impact macroeconomic event scheduled within the next 60 minutes, "
    "or is there a severe black-swan news event?"
)


def _extract_noul_probability(answer: Any) -> float:
    """Extract the scalar ``noul`` probability from a Decisions API answer."""
    noul_val = getattr(answer, "noul", None)
    if noul_val is None and isinstance(answer, dict):
        noul_val = answer.get("noul")

    if isinstance(noul_val, (int, float)):
        return float(noul_val)

    for attr in ("probability", "score", "yes", "value"):
        nested = getattr(noul_val, attr, None)
        if nested is not None:
            return float(nested)
        if isinstance(noul_val, dict) and noul_val.get(attr) is not None:
            return float(noul_val[attr])

    raise ValueError(f"Unable to extract noul probability from Decisions answer: {answer!r}")


def alpha_logic_gate_node(state: AgentState) -> AgentState:
    """Evaluate macro hazard probability via ``gemini-decision-1`` Decisions API."""
    symbol = state["asset_symbol"]
    current_price = float(state.get("current_price", 0.0))
    macro_context = str(state.get("macro_context") or "").strip()

    if not macro_context:
        return {
            **state,
            "action": "HOLD",
            "target_price": None,
            "rationale": "No macroeconomic context available; defaulting to HOLD.",
            "sentiment_score": 0.0,
            "last_alpha_eval_price": current_price,
            "error_message": None,
        }

    try:
        client = genai.Client(
            api_key=os.environ.get("GOOGLE_API_KEY"),
            http_options={"api_version": "v1beta"},
        )
        result = client.decisions.create(
            model=DECISION_MODEL,
            input=macro_context,
            questions={
                CIRCUIT_BREAKER_QUESTION_ID: {
                    "type": "noul",
                    "instructions": CIRCUIT_BREAKER_INSTRUCTION,
                }
            },
        )
        answer = result.answers[CIRCUIT_BREAKER_QUESTION_ID]
        confidence = _extract_noul_probability(answer)

        if confidence > CIRCUIT_BREAKER_THRESHOLD:
            action = "CIRCUIT_BREAKER"
            sentiment_score = max(-1.0, min(1.0, -confidence))
            rationale = (
                f"Macro circuit breaker triggered (noul={confidence:.3f} > "
                f"{CIRCUIT_BREAKER_THRESHOLD:.2f}). Suspending trading and cancelling orders."
            )
        else:
            action = "HOLD"
            sentiment_score = 0.0
            rationale = (
                f"No imminent high-impact macro hazard (noul={confidence:.3f} <= "
                f"{CIRCUIT_BREAKER_THRESHOLD:.2f}). Holding."
            )

        print(
            f"[Alpha] Decision: {action} | Circuit Breaker Prob: {confidence:.3f} "
            f"| Rationale: {rationale}"
        )

        return {
            **state,
            "action": action,
            "target_price": None,
            "rationale": rationale,
            "sentiment_score": sentiment_score,
            "last_alpha_eval_price": current_price,
            "error_message": None,
        }
    except Exception as exc:
        return {
            **state,
            "action": "HOLD",
            "target_price": None,
            "rationale": "",
            "sentiment_score": 0.0,
            "last_alpha_eval_price": current_price,
            "error_message": f"Alpha decision node error for {symbol}: {exc}",
        }
