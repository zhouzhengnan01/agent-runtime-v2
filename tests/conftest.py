from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def disable_llm_spec_planner_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_SPEC_PLANNER", "0")
    monkeypatch.delenv("JETLINKS_RUNTIME_BOOTSTRAP", raising=False)
