<script setup lang="ts">
import { Bot, Boxes, CalendarClock, FileText, Hammer, MessageSquare, Plus, RefreshCw, Workflow } from "lucide-vue-next";
import { computed } from "vue";

import { useWorkbenchStore } from "@/composables/useWorkbenchStore";
import AppCenter from "@/features/apps/AppCenter.vue";
import ChatWorkspace from "@/features/chat/ChatWorkspace.vue";
import InspectorPanel from "@/features/inspector/InspectorPanel.vue";
import ModulePlaceholder from "@/features/modules/ModulePlaceholder.vue";

const store = useWorkbenchStore();

const navItems = [
  { key: "chat", label: "聊天", icon: MessageSquare },
  { key: "apps", label: "应用", icon: Boxes },
  { key: "skills", label: "Skills", icon: Bot },
  { key: "tools", label: "Tools", icon: Hammer },
  { key: "workflows", label: "Workflows", icon: Workflow },
  { key: "cron", label: "Cron", icon: CalendarClock }
] as const;

const moduleTitle = computed(() => {
  const match = navItems.find((item) => item.key === store.view);
  return match?.label || "Workbench";
});

const sandboxText = computed(() => {
  if (!store.sandboxStatus) return "检测中";
  if (store.sandboxStatus.error) return "异常";
  if (store.sandboxStatus.available || store.sandboxStatus.enabled) return "可用";
  return "未启用";
});
</script>

<template>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-mark">JL</div>
        <div>
          <strong>JetLinks Agent</strong>
          <span>Runtime Workbench · upload-fix</span>
        </div>
      </div>

      <button class="new-task" type="button" @click="store.newThread()">
        <Plus :size="17" />
        新建任务
      </button>

      <nav class="nav" aria-label="主导航">
        <button
          v-for="item in navItems"
          :key="item.key"
          :class="{ active: store.view === item.key }"
          type="button"
          @click="store.setView(item.key)"
        >
          <component :is="item.icon" :size="17" />
          <span>{{ item.label }}</span>
        </button>
        <a href="/static/api-docs.html" target="_blank" rel="noopener">
          <FileText :size="17" />
          <span>接口文档</span>
        </a>
      </nav>

      <section class="agent-list">
        <div class="side-title">Agents</div>
        <button
          v-for="agent in store.agents"
          :key="agent.name"
          :class="{ active: store.selectedAgent === agent.name }"
          type="button"
          @click="store.selectedAgent = agent.name"
        >
          <strong>{{ agent.display_name || agent.name }}</strong>
          <span>{{ agent.description || agent.name }}</span>
        </button>
        <div v-if="!store.agents.length" class="empty-mini">暂无 Agent</div>
      </section>

      <section class="token-box">
        <label for="adminToken">Admin Token</label>
        <input
          id="adminToken"
          :value="store.adminToken"
          type="password"
          placeholder="RUNTIME_API_TOKEN"
          autocomplete="off"
          @change="store.setAdminToken(($event.target as HTMLInputElement).value)"
        />
        <span>{{ store.adminToken ? "管理接口将附带 token" : "未设置 token" }}</span>
      </section>
    </aside>

    <main class="main">
      <header class="topbar">
        <div class="topbar-title">
          <strong>{{ store.view === "chat" ? store.activeAgent?.display_name || store.selectedAgent : moduleTitle }}</strong>
          <span>{{ store.threadId }}</span>
        </div>
        <div class="topbar-actions">
          <button class="status-chip" type="button" @click="store.loadSandboxStatus()">
            <span class="dot" :class="{ ok: sandboxText === '可用', warn: sandboxText !== '可用' }"></span>
            Sandbox {{ sandboxText }}
          </button>
          <select v-model="store.selectedAgent" aria-label="Agent">
            <option v-for="agent in store.agents" :key="agent.name" :value="agent.name">
              {{ agent.display_name || agent.name }}
            </option>
          </select>
          <select v-model="store.transport" aria-label="Transport">
            <option value="acp-ws">ACP WS 优先</option>
            <option value="sse">SSE</option>
          </select>
          <input
            :value="store.threadId"
            aria-label="Thread ID"
            @change="store.setThreadId(($event.target as HTMLInputElement).value)"
          />
          <button class="ghost" type="button" @click="store.refreshArtifacts()">
            <RefreshCw :size="15" />
            刷新输出
          </button>
        </div>
      </header>

      <section class="workspace">
        <ChatWorkspace v-if="store.view === 'chat'" />
        <AppCenter v-else-if="store.view === 'apps'" />
        <ModulePlaceholder v-else :module="store.view" />
        <InspectorPanel />
      </section>
    </main>
  </div>
</template>
