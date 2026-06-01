<script setup lang="ts">
import { Bot, Boxes, FolderOpen, Paperclip, Send, Workflow, Zap } from "lucide-vue-next";
import { nextTick, ref, watch } from "vue";

import { useWorkbenchStore } from "@/composables/useWorkbenchStore";

const store = useWorkbenchStore();
const messageList = ref<HTMLElement | null>(null);
const fileInput = ref<HTMLInputElement | null>(null);
const folderInput = ref<HTMLInputElement | null>(null);
const pendingInputFile = ref<HTMLInputElement | null>(null);
const requiredInputFile = ref<HTMLInputElement | null>(null);

function setDeepExecution(event: Event) {
  store.setDeepExecution((event.target as HTMLInputElement).checked);
}

function setYoloExecution(event: Event) {
  store.setYoloExecution((event.target as HTMLInputElement).checked);
}

watch(
  () => store.messages.map((message) => message.content).join("|"),
  () => {
    void nextTick(() => {
      if (messageList.value) messageList.value.scrollTop = messageList.value.scrollHeight;
    });
  }
);

function submit() {
  void store.sendCurrentMessage();
}

function chooseFiles() {
  if (!fileInput.value) return;
  fileInput.value.value = "";
  fileInput.value.click();
}

function chooseFolder() {
  if (!folderInput.value) return;
  folderInput.value.value = "";
  folderInput.value.click();
}

function uploadSelectedFiles(event: Event) {
  const input = event.target as HTMLInputElement;
  void store.uploadFiles(input.files || []);
}

function choosePendingFiles() {
  if (!pendingInputFile.value) return;
  pendingInputFile.value.accept = store.pendingAttachmentRequirement?.accept || "";
  pendingInputFile.value.value = "";
  pendingInputFile.value.click();
}

function uploadPendingFiles(event: Event) {
  const input = event.target as HTMLInputElement;
  void store.handlePendingAttachmentFiles(input.files || []);
}

function chooseRequiredFiles() {
  if (!requiredInputFile.value) return;
  requiredInputFile.value.accept = store.requiredInputAccept === "*/*" ? "" : store.requiredInputAccept;
  requiredInputFile.value.value = "";
  requiredInputFile.value.click();
}

function uploadRequiredFiles(event: Event) {
  const input = event.target as HTMLInputElement;
  void store.handleRequiredInputFiles(input.files || []);
}
</script>

<template>
  <section class="chat-workspace">
    <div class="intro-row">
      <div>
        <h1>Agent Chat</h1>
        <p>对话、运行事件和产物预览已拆成独立链路，后续模块会按功能逐步迁入。</p>
      </div>
      <div class="run-state" :class="{ running: store.running }">
        <span class="dot" :class="{ warn: store.running, ok: !store.running }"></span>
        {{ store.running ? "Running" : "Ready" }}
      </div>
    </div>

    <div class="quick-actions">
      <span v-if="store.selectedAppTemplateName || store.selectedWorkflow || store.selectedSkills.length" class="route-pill">
        {{ store.selectedAppTemplateName ? `应用 ${store.selectedAppTemplateName}` : '自定义能力' }}
        <template v-if="store.selectedWorkflow"> · {{ store.selectedWorkflow }}</template>
        <template v-if="store.selectedSkills.length"> · Skill {{ store.selectedSkills.length }}</template>
      </span>
      <button type="button" @click="store.setView('skills')">
        <Bot :size="16" />
        Skills
      </button>
      <button type="button" @click="store.setView('workflows')">
        <Workflow :size="16" />
        Workflow
      </button>
      <button type="button" @click="store.setView('apps')">
        <Boxes :size="16" />
        应用
      </button>
    </div>

    <div ref="messageList" class="messages">
      <div v-if="!store.messages.length" class="message assistant">
        <div class="role">Agent</div>
        <div class="bubble">选择 Agent 后输入任务。运行过程会在右侧同步显示文件、Spec、校验和执行事件。</div>
      </div>
      <div v-for="message in store.messages" :key="message.id" class="message" :class="message.role">
        <div class="role">{{ message.role === "user" ? "You" : "Agent" }}</div>
        <div class="bubble" :class="{ streaming: message.streaming }">
          <pre>{{ message.content }}</pre>
        </div>
      </div>
    </div>

    <form class="composer" @submit.prevent="submit">
      <div v-if="store.pendingAttachmentRequirement" class="required-input-panel">
        <div>
          <strong>运行前需要{{ store.pendingAttachmentRequirement.label }}</strong>
          <span>{{ store.pendingAttachmentRequirement.reason }}</span>
        </div>
        <div class="required-actions">
          <button type="button" @click="store.clearPendingAttachmentRequirement()">取消</button>
          <button class="primary-inline" type="button" @click="choosePendingFiles">
            选择{{ store.pendingAttachmentRequirement.label }}
          </button>
        </div>
      </div>

      <div v-if="store.requiredInputs.length" class="required-input-panel">
        <div>
          <strong>需要上传{{ store.requiredInputLabel }}</strong>
          <span>{{ store.requiredInputAccept }}</span>
          <ul class="required-input-list">
            <li v-for="(item, index) in store.requiredInputs" :key="`${item.type || 'file'}-${index}`">
              {{ item.reason || item.description || item.name || item.type || "file" }}
            </li>
          </ul>
        </div>
        <div class="required-actions">
          <button type="button" @click="store.clearRequiredInputs()">稍后</button>
          <button class="primary-inline" type="button" @click="chooseRequiredFiles">选择文件</button>
        </div>
      </div>

      <textarea
        v-model="store.input"
        :disabled="store.running"
        placeholder="输入任务"
        @keydown.enter.exact.prevent="submit"
      ></textarea>
      <div v-if="store.uploadBatchSummary" class="upload-summary" :class="{ error: store.uploadBatchSummary.failed }">
        <strong>{{ store.uploadBatchSummary.total }} 个附件</strong>
        <span>
          已完成 {{ store.uploadBatchSummary.completed }}
          <template v-if="store.uploadBatchSummary.uploading"> · 上传中 {{ store.uploadBatchSummary.uploading }}</template>
          <template v-if="store.uploadBatchSummary.failed"> · 失败 {{ store.uploadBatchSummary.failed }}</template>
        </span>
      </div>
      <div v-if="store.attachments.length" class="attachments">
        <span
          v-for="file in store.visibleAttachments"
          :key="file.id"
          class="attachment"
          :class="{ uploading: file.uploading, error: file.error }"
        >
          {{ file.displayName || file.name }} · {{ file.uploading ? `上传中 ${Math.max(1, Math.min(99, file.progress || 0))}%` : file.error ? `失败：${file.error}` : "已上传" }}
          <button type="button" title="移除附件" @click="store.removeAttachment(file.id)">×</button>
        </span>
        <span v-if="store.hiddenAttachmentCount" class="attachment muted">
          另有 {{ store.hiddenAttachmentCount }} 个附件
        </span>
      </div>
      <div class="composer-bar">
        <div class="composer-tools">
          <button type="button" :disabled="store.running" @click="chooseFiles">
            <Paperclip :size="16" />
            上传
          </button>
          <button type="button" :disabled="store.running" @click="chooseFolder">
            <FolderOpen :size="16" />
            文件夹
          </button>
          <label class="composer-toggle" title="复杂任务使用 autonomous 模式，允许更多工具循环">
            <input
              :checked="store.deepExecution"
              type="checkbox"
              :disabled="store.running || store.yoloExecution"
              @change="setDeepExecution"
            />
            深度执行
          </label>
          <label class="composer-toggle" title="自动批准权限请求；除缺文件或缺参数外不再弹确认">
            <input
              :checked="store.yoloExecution"
              type="checkbox"
              :disabled="store.running"
              @change="setYoloExecution"
            />
            <Zap :size="14" />
            YOLO
          </label>
          <span v-if="store.error" class="error-text">{{ store.error }}</span>
        </div>
        <button class="send" type="submit" :disabled="store.running || !store.input.trim()">
          <Send :size="18" />
        </button>
      </div>
    </form>

    <input ref="fileInput" type="file" multiple hidden @change="uploadSelectedFiles" />
    <input ref="folderInput" type="file" multiple webkitdirectory directory hidden @change="uploadSelectedFiles" />
    <input ref="pendingInputFile" type="file" multiple hidden @change="uploadPendingFiles" />
    <input ref="requiredInputFile" type="file" multiple hidden @change="uploadRequiredFiles" />
  </section>
</template>
