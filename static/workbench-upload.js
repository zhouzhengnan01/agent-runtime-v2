(function () {
  function createWorkbenchUploadController(options) {
    const { state, els, config, hooks, utils } = options;
    const maxUploadBytes = config.maxUploadBytes;
    const uploadConcurrency = config.uploadConcurrency;
    const uploadTimeoutMs = config.uploadTimeoutMs || 120000;
    const activeRequests = new Map();
    const canceledUploads = new Set();
    let attachmentUpdateFrame = 0;
    let uploadSeq = 0;

    function renderAttachments() {
      els.attachments.innerHTML = state.files.map((file) => {
        const progress = Number(file.progress || 0);
        const status = file.error
          ? ` · 失败：${file.error}`
          : file.uploading
            ? ` · 上传中${progress ? ` ${Math.min(99, Math.max(1, Math.round(progress)))}%` : ''}`
            : ' · 已上传';
        return `
          <span class="attachment ${file.uploading ? 'uploading' : ''} ${file.error ? 'error' : ''}">
            ${utils.escapeHtml(file.name)} · ${utils.formatBytes(file.size)}${utils.escapeHtml(status)}
            <button type="button" title="移除附件" data-remove-attachment="${utils.escapeAttr(file.id || file.path || file.name)}">×</button>
          </span>
        `;
      }).join('');
      updateComposerControls();
    }

    function scheduleAttachmentUpdate() {
      if (attachmentUpdateFrame) return;
      attachmentUpdateFrame = requestAnimationFrame(() => {
        attachmentUpdateFrame = 0;
        renderAttachments();
      });
    }

    function cancelScheduledUpdate() {
      if (!attachmentUpdateFrame) return;
      cancelAnimationFrame(attachmentUpdateFrame);
      attachmentUpdateFrame = 0;
    }

    function hasUploadingAttachments() {
      return state.files.some((file) => file.uploading);
    }

    function hasFailedAttachments() {
      return state.files.some((file) => file.error);
    }

    function updateComposerControls() {
      const blockedByUpload = hasUploadingAttachments() || hasFailedAttachments();
      els.send.disabled = state.running || blockedByUpload;
      els.attach.disabled = state.running;
    }

    async function uploadChatFiles(files) {
      const selected = Array.from(files || []);
      if (!selected.length) return;
      const oversized = selected.filter((file) => file.size > maxUploadBytes);
      if (oversized.length) {
        const names = oversized.map((file) => `${file.name} (${utils.formatBytes(file.size)})`).join('、');
        hooks.finishAssistantMessage(`上传失败：${names} 超过 ${utils.formatBytes(maxUploadBytes)} 限制。`);
        hooks.addTimeline('上传失败', names);
        return;
      }
      const threadId = hooks.ensureThread();
      const pending = selected.map((file) => ({
        id: `upload-${Date.now().toString(36)}-${++uploadSeq}`,
        name: file.name,
        size: file.size,
        mime_type: file.type || null,
        uploading: true,
        progress: 0
      }));
      state.files = state.files.concat(pending);
      renderAttachments();
      hooks.addTimeline('上传附件', selected.map((file) => `${file.name} ${utils.formatBytes(file.size)}`).join('、'));

      const tasks = selected.map((file, index) => () => uploadChatFile(threadId, file, pending[index]));
      const results = await runUploadQueue(tasks, uploadConcurrency);
      const uploaded = results.filter((item) => item?.ok).map((item) => item.file);
      const failed = results.filter((item) => !item?.ok && !item.error?.canceled);
      if (uploaded.length) hooks.addTimeline('附件已上传', uploaded.map((file) => file.name).join('、'));
      if (failed.length) hooks.addTimeline('上传失败', failed.map((item) => item.error?.message || 'unknown').join('、'));
      return { uploaded, failed, results };
    }

    async function runUploadQueue(tasks, limit) {
      const results = new Array(tasks.length);
      let next = 0;
      const workerCount = Math.max(1, Math.min(limit, tasks.length));
      const workers = Array.from({ length: workerCount }, async () => {
        while (next < tasks.length) {
          const index = next;
          next += 1;
          try {
            results[index] = { ok: true, file: await tasks[index]() };
          } catch (err) {
            results[index] = { ok: false, error: err };
          }
        }
      });
      await Promise.all(workers);
      return results;
    }

    function replaceAttachment(id, patch) {
      const index = state.files.findIndex((item) => item.id === id);
      if (index < 0) return;
      state.files[index] = { ...state.files[index], ...patch };
      scheduleAttachmentUpdate();
    }

    async function uploadChatFile(threadId, file, pending) {
      const body = new FormData();
      body.append('file', file, file.name);
      try {
        const data = await uploadWithProgress(
          `/api/uploads/${encodeURIComponent(threadId)}`,
          body,
          (progress) => {
            replaceAttachment(pending.id, { progress });
          },
          pending.id
        );
        const uploaded = {
          id: pending.id,
          name: data.name || file.name,
          path: data.path,
          mime_type: data.mime_type || file.type || null,
          metadata: { size: Number(data.size || file.size), original_name: data.original_name || file.name },
          size: Number(data.size || file.size),
          uploading: false,
          progress: 100
        };
        replaceAttachment(pending.id, uploaded);
        return uploaded;
      } catch (err) {
        const wasCanceled = canceledUploads.delete(pending.id);
        if (!wasCanceled) replaceAttachment(pending.id, { uploading: false, error: err.message || '上传失败' });
        err.canceled = wasCanceled;
        throw err;
      }
    }

    function uploadWithProgress(url, body, onProgress, uploadId) {
      return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        const cleanup = () => {
          if (uploadId) activeRequests.delete(uploadId);
        };
        xhr.open('POST', url);
        xhr.timeout = uploadTimeoutMs;
        if (uploadId) activeRequests.set(uploadId, xhr);
        if (state.adminToken) xhr.setRequestHeader('Authorization', `Bearer ${state.adminToken}`);
        xhr.upload.addEventListener('progress', (event) => {
          if (!event.lengthComputable) return;
          onProgress((event.loaded / event.total) * 100);
        });
        xhr.addEventListener('load', () => {
          cleanup();
          let data = {};
          try {
            data = xhr.responseText ? JSON.parse(xhr.responseText) : {};
          } catch {
            reject(new Error('上传响应不是有效 JSON'));
            return;
          }
          if (xhr.status < 200 || xhr.status >= 300) {
            reject(new Error(data.detail || xhr.statusText || `HTTP ${xhr.status}`));
            return;
          }
          resolve(data);
        });
        xhr.addEventListener('error', () => {
          cleanup();
          reject(new Error('网络异常，上传失败'));
        });
        xhr.addEventListener('timeout', () => {
          cleanup();
          reject(new Error(`上传超时，请检查网络后重试（${Math.round(uploadTimeoutMs / 1000)} 秒）`));
        });
        xhr.addEventListener('abort', () => {
          cleanup();
          reject(new Error('上传已取消'));
        });
        xhr.send(body);
      });
    }

    async function attachmentsForRequest() {
      const uploading = state.files.filter((file) => file.uploading);
      if (uploading.length) {
        throw new Error('附件仍在上传，请稍后再发送。');
      }
      const failed = state.files.filter((file) => file.error);
      if (failed.length) {
        throw new Error('存在上传失败的附件，请移除后重试。');
      }
      return state.files.filter((file) => file.path).map((file) => ({
        name: file.name,
        path: file.path,
        mime_type: file.mime_type || null,
        metadata: file.metadata || { size: file.size }
      }));
    }

    function removeAttachment(id) {
      const active = activeRequests.get(id);
      if (active) {
        canceledUploads.add(id);
        active.abort();
      }
      state.files = state.files.filter((file) => (file.id || file.path || file.name) !== id);
      renderAttachments();
    }

    function cancelAllUploads() {
      for (const id of activeRequests.keys()) {
        canceledUploads.add(id);
      }
      for (const xhr of activeRequests.values()) {
        xhr.abort();
      }
      activeRequests.clear();
      canceledUploads.clear();
      cancelScheduledUpdate();
    }

    return {
      attachmentsForRequest,
      cancelAllUploads,
      cancelScheduledUpdate,
      hasFailedAttachments,
      hasUploadingAttachments,
      removeAttachment,
      renderAttachments,
      updateComposerControls,
      uploadChatFiles
    };
  }

  window.createWorkbenchUploadController = createWorkbenchUploadController;
})();
