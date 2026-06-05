<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { Boxes, FileJson, Play, Search, Send, Sparkles } from "lucide-vue-next";

import { useWorkbenchStore } from "@/composables/useWorkbenchStore";
import type { AppTemplate } from "@/api/types";

const store = useWorkbenchStore();
const query = ref("");

const categories = [
  { key: "all", label: "全部" },
  { key: "general", label: "通用" },
  { key: "generation", label: "生成" },
  { key: "vision", label: "视觉" },
  { key: "algorithm", label: "算法" }
];

const filteredTemplates = computed(() => {
  const keyword = query.value.trim().toLowerCase();
  return [...store.appTemplates]
    .sort((a, b) => String(a.title || a.name).localeCompare(String(b.title || b.name), "zh-CN"))
    .filter((template) => store.appFilter === "all" || (template.category || "general") === store.appFilter)
    .filter((template) => {
      if (!keyword) return true;
      return [template.name, template.title, template.description, ...(template.tags || [])]
        .filter(Boolean)
        .some((value) => String(value).toLowerCase().includes(keyword));
    });
});

const selectedTemplate = computed(() =>
  store.appTemplates.find((template) => template.name === store.selectedAppTemplateName)
);

onMounted(() => {
  if (!store.appTemplates.length) void store.loadAppTemplates();
});

function iconFor(template: AppTemplate) {
  const icons: Record<string, string> = {
    chat: "Chat",
    diagram: "Draw",
    slides: "PPT",
    table: "XLS",
    doc: "Doc",
    mindmap: "Map",
    shield: "Safe",
    package: "Pack",
    research: "Lab"
  };
  return icons[template.icon || ""] || "App";
}

function categoryLabel(category?: string) {
  if (category === "generation") return "生成";
  if (category === "vision") return "视觉";
  if (category === "algorithm") return "算法";
  return "通用";
}

</script>

<template>
  <section class="app-center">
    <header class="module-head">
      <div>
        <h1>应用中心</h1>
        <p>选择一个预置应用，自动带入 Agent、Workflow、Skills 和示例 Prompt，用来快速验证运行效果。</p>
      </div>
      <button class="module-action" type="button" @click="store.loadAppTemplates()">
        <Sparkles :size="16" />
        刷新模板
      </button>
    </header>

    <div class="app-toolbar">
      <div class="search-box">
        <Search :size="16" />
        <input v-model="query" placeholder="搜索应用、标签或描述" />
      </div>
      <div class="segmented">
        <button
          v-for="category in categories"
          :key="category.key"
          :class="{ active: store.appFilter === category.key }"
          type="button"
          @click="store.appFilter = category.key"
        >
          {{ category.label }}
        </button>
      </div>
    </div>

    <div class="app-layout">
      <div class="template-grid">
        <article
          v-for="template in filteredTemplates"
          :key="template.name"
          class="template-card"
          :class="{ active: store.selectedAppTemplateName === template.name }"
        >
          <button class="template-main" type="button" @click="store.selectedAppTemplateName = template.name">
            <span class="template-icon">{{ iconFor(template) }}</span>
            <span class="template-title">
              <strong>{{ template.title || template.name }}</strong>
              <span>{{ template.agent_name || 'default' }}{{ template.workflow ? ` · ${template.workflow}` : '' }}</span>
            </span>
          </button>
          <p>{{ template.description || "暂无描述" }}</p>
          <div class="tag-row">
            <span class="tag accent">{{ categoryLabel(template.category) }}</span>
            <span v-for="tag in (template.tags || []).slice(0, 4)" :key="tag" class="tag">{{ tag }}</span>
          </div>
          <div class="template-meta">
            Skills: {{ (template.selected_skills || []).join(', ') || '按 Agent 默认' }}
          </div>
          <div class="template-actions">
            <button type="button" @click="store.applyAppTemplate(template.name, { fillPrompt: true })">
              <Send :size="15" />
              填入
            </button>
            <button class="primary-inline" type="button" @click="store.applyAppTemplate(template.name, { fillPrompt: true, run: true })">
              <Play :size="15" />
              运行
            </button>
          </div>
        </article>

        <div v-if="!filteredTemplates.length" class="empty-panel">暂无匹配应用模板</div>
      </div>

      <aside class="template-detail">
        <div v-if="selectedTemplate" class="detail-inner">
          <div class="detail-kicker">
            <Boxes :size="16" />
            {{ categoryLabel(selectedTemplate.category) }}
          </div>
          <h2>{{ selectedTemplate.title || selectedTemplate.name }}</h2>
          <p>{{ selectedTemplate.description }}</p>
          <dl>
            <div>
              <dt>Agent</dt>
              <dd>{{ selectedTemplate.agent_name || 'default' }}</dd>
            </div>
            <div>
              <dt>Workflow</dt>
              <dd>{{ selectedTemplate.workflow || '-' }}</dd>
            </div>
            <div>
              <dt>Skills</dt>
              <dd>{{ (selectedTemplate.selected_skills || []).join(', ') || '-' }}</dd>
            </div>
          </dl>
          <section class="prompt-list">
            <strong>示例 Prompt</strong>
            <button
              v-for="prompt in selectedTemplate.prompt_examples || []"
              :key="prompt"
              type="button"
              @click="store.input = prompt; store.applyAppTemplate(selectedTemplate.name, { fillPrompt: false })"
            >
              {{ prompt }}
            </button>
          </section>
          <details>
            <summary>
              <FileJson :size="15" />
              JSON
            </summary>
            <pre class="code-block">{{ JSON.stringify(selectedTemplate, null, 2) }}</pre>
          </details>
        </div>
        <div v-else class="empty-panel">选择左侧模板查看详情</div>
      </aside>
    </div>
  </section>
</template>
