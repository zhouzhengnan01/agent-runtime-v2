<script setup lang="ts">
import { computed } from "vue";

const props = defineProps<{
  module: string;
}>();

const title = computed(() => {
  const labels: Record<string, string> = {
    apps: "应用中心",
    skills: "Skills 技能",
    tools: "Tools 工具",
    workflows: "Workflows 插件",
    cron: "Cron 计划"
  };
  return labels[props.module] || "模块";
});
</script>

<template>
  <section class="module-placeholder">
    <div>
      <h1>{{ title }}</h1>
      <p>该模块保留在旧 Workbench 中，本轮先迁移 Chat、运行事件和 Inspector。下一阶段会按模块逐步接入列表、编辑器和保存逻辑。</p>
    </div>
    <a href="/static/workbench.html">打开旧 Workbench</a>
  </section>
</template>
