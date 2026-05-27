<script setup lang="ts">
import { computed } from "vue";

import { useWorkbenchStore } from "@/composables/useWorkbenchStore";

const store = useWorkbenchStore();

const tabs = [
  { key: "artifacts", label: "文件" },
  { key: "abilities", label: "能力" },
  { key: "acp", label: "ACP" },
  { key: "verification", label: "校验" },
  { key: "spec", label: "Spec" },
  { key: "timeline", label: "执行" }
] as const;

const specText = computed(() => (store.spec ? JSON.stringify(store.spec, null, 2) : ""));

function formatBytes(value?: number) {
  const size = Number(value || 0);
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / 1024 / 1024).toFixed(1)} MB`;
}
</script>

<template>
  <aside class="inspector">
    <header class="inspector-head">
      <div>
        <strong>输出</strong>
        <span>产物、Spec、校验和执行事件</span>
      </div>
      <button type="button" @click="store.refreshArtifacts()">刷新</button>
    </header>

    <div class="tabs">
      <button
        v-for="tab in tabs"
        :key="tab.key"
        :class="{ active: store.inspectorTab === tab.key }"
        type="button"
        @click="store.inspectorTab = tab.key"
      >
        {{ tab.label }}
      </button>
    </div>

    <div class="inspector-body">
      <section v-if="store.inspectorTab === 'artifacts'" class="panel-list">
        <div v-if="!store.artifacts.length" class="empty-panel">暂无文件输出</div>
        <article v-for="artifact in store.artifacts" :key="artifact.path" class="artifact-row">
          <strong>{{ artifact.name }}</strong>
          <span>{{ artifact.kind }} · {{ artifact.mime_type || '-' }} · {{ formatBytes(artifact.size) }}</span>
          <div class="row-actions">
            <a :href="artifact.preview_url" target="_blank" rel="noopener">预览</a>
            <a :href="artifact.download_url" target="_blank" rel="noopener">下载</a>
          </div>
        </article>
      </section>

      <section v-else-if="store.inspectorTab === 'verification'" class="panel-list">
        <div v-if="!store.verification" class="empty-panel">暂无校验结果</div>
        <template v-else>
          <div class="verify" :class="{ failed: !store.verification.passed }">
            {{ store.verification.passed ? "校验通过" : "校验未通过" }}
          </div>
          <div v-for="check in store.verification.checks || []" :key="check.name" class="check-row">
            <span>{{ check.passed ? "✓" : "×" }}</span>
            <div>
              <strong>{{ check.name }}</strong>
              <p>{{ check.detail || "" }}</p>
            </div>
          </div>
        </template>
      </section>

      <section v-else-if="store.inspectorTab === 'spec'" class="panel-list">
        <pre v-if="specText" class="code-block">{{ specText }}</pre>
        <div v-else class="empty-panel">暂无 Spec</div>
      </section>

      <section v-else-if="store.inspectorTab === 'timeline'" class="timeline-list">
        <div v-if="!store.timeline.length" class="empty-panel">等待任务</div>
        <div v-for="item in store.timeline" :key="item.id" class="timeline-item">
          <strong>{{ item.name }}</strong>
          <span>{{ item.detail }}</span>
        </div>
      </section>

      <section v-else-if="store.inspectorTab === 'acp'" class="acp-tools">
        <div class="acp-status">
          <strong>{{ store.acpConnected ? 'ACP 已连接' : 'ACP 未连接' }}</strong>
          <span>session: {{ store.acpSessionId || '-' }}</span>
          <span v-if="store.acpToolStatus">{{ store.acpToolStatus }}</span>
          <span v-if="store.acpToolError" class="error-text">{{ store.acpToolError }}</span>
        </div>

        <div class="tool-row">
          <button type="button" @click="store.connectAcpTools()">连接/建会话</button>
          <button type="button" @click="store.listAcpSessions()">列会话</button>
          <button type="button" @click="store.cancelAcpSession()">取消</button>
          <button type="button" @click="store.closeAcpSession()">关闭会话</button>
          <button type="button" @click="store.closeAcpTools()">断开</button>
        </div>

        <section class="tool-section">
          <div class="tool-title">
            <strong>FS</strong>
            <span>读取/写入当前 ACP session 工作区文件</span>
          </div>
          <input v-model="store.fsPath" aria-label="ACP file path" />
          <textarea v-model="store.fsContent" class="tool-textarea code" spellcheck="false"></textarea>
          <div class="tool-row">
            <button type="button" @click="store.readAcpFile()">读取</button>
            <button type="button" @click="store.writeAcpFile()">写入</button>
          </div>
        </section>

        <section class="tool-section">
          <div class="tool-title">
            <strong>Terminal</strong>
            <span>terminal: {{ store.terminalId || '-' }}</span>
          </div>
          <input v-model="store.terminalCommand" aria-label="Terminal command" />
          <input v-model="store.terminalArgs" aria-label="Terminal args" />
          <div class="tool-row">
            <button type="button" @click="store.createAcpTerminal()">创建</button>
            <button type="button" @click="store.readAcpTerminalOutput()">输出</button>
            <button type="button" @click="store.waitAcpTerminal()">等待</button>
            <button type="button" @click="store.killAcpTerminal()">Kill</button>
            <button type="button" @click="store.releaseAcpTerminal()">释放</button>
          </div>
          <pre class="code-block terminal-output">{{ store.terminalOutput || '暂无输出' }}</pre>
        </section>

        <section class="tool-section">
          <div class="tool-title">
            <strong>Raw RPC</strong>
            <span>调用 dispatcher 已支持的 ACP 方法</span>
          </div>
          <input v-model="store.rawRpcMethod" aria-label="RPC method" />
          <textarea v-model="store.rawRpcParams" class="tool-textarea code" spellcheck="false"></textarea>
          <div class="tool-row">
            <button type="button" @click="store.runRawAcpRpc()">执行 RPC</button>
          </div>
          <pre class="code-block">{{ store.acpLastResult ? JSON.stringify(store.acpLastResult, null, 2) : '暂无结果' }}</pre>
        </section>
      </section>

      <section v-else class="panel-list">
        <div class="empty-panel">能力管理会在后续阶段迁移。当前仍可使用旧 Workbench 的 Skills、Tools 和 Workflows 页面。</div>
      </section>
    </div>
  </aside>
</template>
