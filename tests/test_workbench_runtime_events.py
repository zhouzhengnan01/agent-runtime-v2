from pathlib import Path


WORKBENCH = Path("static/workbench.html")


def test_workbench_surfaces_agent_loop_runtime_events() -> None:
    html = WORKBENCH.read_text()

    for event_type in [
        "context.compacted",
        "llm.request.started",
        "llm.request.completed",
        "tool.calls.started",
        "tool.started",
        "tool.completed",
        "tool.failed",
    ]:
        assert f"case '{event_type}':" in html

    for helper_name in [
        "runtimeRoundText",
        "llmRequestText",
        "llmCompletedText",
        "toolCallsText",
        "toolProgressText",
        "toolCompletedText",
        "toolFailedText",
    ]:
        assert f"function {helper_name}" in html


def test_workbench_has_specific_progress_for_session_files_and_artifacts() -> None:
    html = WORKBENCH.read_text()

    assert "正在读取当前会话文件" in html
    assert "正在读取当前会话产物" in html
    assert "正在写入当前会话产物" in html


def test_workbench_deep_execution_toggle_is_in_visible_composer_toolbar() -> None:
    html = WORKBENCH.read_text()

    assert '<label class="small-action toggle-action" title="复杂任务使用 autonomous 模式，允许更多工具循环">' in html
    assert '<input id="deepExecution" type="checkbox">' in html
    assert '<label class="tool-pill" title="复杂任务使用 autonomous 模式，允许更多工具循环">' not in html


def test_workbench_does_not_restore_old_artifacts_after_thread_switch() -> None:
    html = WORKBENCH.read_text()

    assert "const threadId = ensureThread();" in html
    assert "if (ensureThread() !== threadId) return;" in html
    assert "const previousThread = els.threadMini.textContent;" in html
    assert "resetThreadUiState({ clearCapabilities: true });" in html
