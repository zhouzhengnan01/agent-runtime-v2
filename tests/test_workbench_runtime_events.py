from pathlib import Path


WORKBENCH = Path("static/workbench.html")
VUE_STORE = Path("frontend/workbench/src/composables/useWorkbenchStore.ts")


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
    assert '<input id="yoloExecution" type="checkbox">' in html
    assert "runtimeOptions.mode = 'yolo';" in html
    assert "runtimeOptions.config_options = { max_tool_rounds: 16 };" in html
    assert "runtimeOptions.configOptions = { max_tool_rounds: 16 };" in html
    assert '<label class="tool-pill" title="复杂任务使用 autonomous 模式，允许更多工具循环">' not in html


def test_workbench_does_not_restore_old_artifacts_after_thread_switch() -> None:
    html = WORKBENCH.read_text()

    assert "const threadId = ensureThread();" in html
    assert "if (ensureThread() !== threadId) return;" in html
    assert "const previousThread = els.threadMini.textContent;" in html
    assert "resetThreadUiState({ clearCapabilities: true });" in html


def test_workbench_keeps_inspector_synced_with_runtime_outputs() -> None:
    html = WORKBENCH.read_text()

    assert "function renderInspectorTabs()" in html
    assert "function selectInspectorTab(tabName)" in html
    assert "if (state.activeTab === 'artifacts') renderPanel();" in html
    assert "state.artifacts = data.artifacts || [];" in html
    assert "fetchOptions({ cache: 'no-store' })" in html
    assert "renderInspectorTabs();" in html


def test_workbench_sends_acp_runtime_options_on_session_new() -> None:
    html = WORKBENCH.read_text()

    assert "const runtimeOptions = buildAcpRuntimeOptions(threadId, workflow);" in html
    assert "runtimeOptions" in html[html.index("rpc('new_session'") : html.index("rpc('prompt'")]
    assert "runtimeOptions.appTemplateName = state.selectedAppTemplateName;" in html
    assert "runtimeOptions.modelType = 'chat';" in html


def test_workbench_sends_app_template_name_for_config_app_models() -> None:
    html = WORKBENCH.read_text()

    assert "runtimeOptions.app_template_name = state.selectedAppTemplateName;" in html
    assert "runtimeOptions.model_type = 'chat';" in html
    assert "app_template_name: template.name" in html


def test_workbench_does_not_restore_artifacts_from_recent_runs() -> None:
    html = WORKBENCH.read_text()

    assert "restoreRecentArtifacts" not in html
    assert "/api/agents/runs/recent?limit=20" not in html
    assert "localStorage.setItem(THREAD_STORAGE_KEY, candidate)" not in html


def test_workbench_can_open_clean_thread_from_url() -> None:
    html = WORKBENCH.read_text()

    assert "function initialThreadFromUrl()" in html
    assert "params.get('newThread')" in html
    assert "params.get('thread_id')" in html
    assert "initialThreadFromUrl() || localStorage.getItem(THREAD_STORAGE_KEY)" in html


def test_vue_workbench_keeps_event_artifacts_until_backend_refresh() -> None:
    store = VUE_STORE.read_text()

    assert "if (result.artifacts?.length) this.artifacts = result.artifacts;" in store
    assert "await this.refreshArtifacts().catch((error)" in store
    assert 'this.addTimeline("刷新文件失败", message);' in store


def test_vue_workbench_sends_selected_app_runtime_options() -> None:
    store = VUE_STORE.read_text()

    assert "state.appTemplates.find((item) => item.name === state.selectedAppTemplateName)" in store
    assert "...templateOptions" in store
    assert 'options.mode = "yolo";' in store
    assert "options.configOptions = { max_tool_rounds: 16 };" in store
