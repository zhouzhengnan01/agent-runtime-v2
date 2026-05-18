<script setup lang="ts">
import { onMounted } from "vue";

import AppShell from "@/components/AppShell.vue";
import { useWorkbenchStore } from "@/composables/useWorkbenchStore";

const store = useWorkbenchStore();

onMounted(() => {
  void store.loadAgents();
  void store.loadAppTemplates();
  void store.loadSandboxStatus();
  void store.refreshArtifacts();
});
</script>

<template>
  <AppShell />
</template>
