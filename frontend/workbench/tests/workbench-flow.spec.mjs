import { expect, test } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';

const APP_URL = process.env.APP_URL || 'http://127.0.0.1:18012/static/workbench.html';
const RUNTIME_TOKEN = process.env.RUNTIME_API_TOKEN || '';
const WORK_DIR = process.env.WORKBENCH_UI_SMOKE_DIR || '/tmp/jetlinks-workbench-ui-smoke';
const IMAGE_PATH = path.join(WORK_DIR, 'smoke-upload.png');

function crc32(buffer) {
  let crc = 0xffffffff;
  for (const byte of buffer) {
    crc ^= byte;
    for (let index = 0; index < 8; index += 1) {
      crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
    }
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function pngChunk(type, data) {
  const typeBuffer = Buffer.from(type, 'ascii');
  const length = Buffer.alloc(4);
  length.writeUInt32BE(data.length, 0);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(Buffer.concat([typeBuffer, data])), 0);
  return Buffer.concat([length, typeBuffer, data, crc]);
}

function writeSmokePng(filePath) {
  const width = 640;
  const height = 480;
  const rowLength = 1 + width * 3;
  const pixels = Buffer.alloc(rowLength * height);
  for (let y = 0; y < height; y += 1) {
    const row = y * rowLength;
    pixels[row] = 0;
    for (let x = 0; x < width; x += 1) {
      const offset = row + 1 + x * 3;
      const person = x >= 120 && x <= 230 && y >= 120 && y <= 390;
      const helmet = x >= 145 && x <= 205 && y >= 70 && y <= 125;
      const car = x >= 340 && x <= 560 && y >= 250 && y <= 365;
      pixels[offset] = helmet ? 245 : car ? 210 : person ? 70 : 238;
      pixels[offset + 1] = helmet ? 190 : car ? 70 : person ? 130 : 242;
      pixels[offset + 2] = helmet ? 60 : car ? 70 : person ? 180 : 247;
    }
  }

  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8;
  ihdr[9] = 2;
  ihdr[10] = 0;
  ihdr[11] = 0;
  ihdr[12] = 0;

  const png = Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    pngChunk('IHDR', ihdr),
    pngChunk('IDAT', zlib.deflateSync(pixels)),
    pngChunk('IEND', Buffer.alloc(0)),
  ]);
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, png);
}

function sseFrame(type, data) {
  return `data: ${JSON.stringify({ type, data })}\n\n`;
}

function artifact(name, kind, mimeType, size = 256) {
  return {
    name,
    kind,
    mime_type: mimeType,
    size,
    path: `/mnt/user-data/outputs/${name}`,
    preview_url: `/api/artifacts/thread/preview/${name}`,
    download_url: `/api/artifacts/thread/download/${name}`,
  };
}

async function fulfillRunStream(route, requestIndex) {
  const artifacts =
    requestIndex === 0
      ? [artifact('annotations.coco.json', 'text', 'application/json', 512)]
      : [
          artifact('annotations.coco.json', 'text', 'application/json', 512),
          artifact('coco-summary.md', 'markdown', 'text/markdown', 320),
        ];
  const reply =
    requestIndex === 0
      ? '已自动标注完成，COCO JSON 文件已生成：annotations.coco.json'
      : '已读取当前会话已有的 annotations.coco.json，并生成 coco-summary.md';
  const result = {
    status: 'completed',
    reply,
    artifacts,
    metadata: { tool_rounds: requestIndex === 0 ? 2 : 4, tool_call_count: requestIndex === 0 ? 1 : 3, mode: 'autonomous' },
  };
  const body = [
    sseFrame('run.started', { workflow: requestIndex === 0 ? 'data-auto-annotation' : 'agent_loop', thread_id: 'ui-smoke' }),
    sseFrame('llm.request.started', { round: 1, tool_count: 3, tools: [{ name: 'local_read_file' }] }),
    sseFrame('tool.started', { tool_name: requestIndex === 0 ? 'data-auto-annotation' : 'artifact_read' }),
    sseFrame('tool.completed', { tool_name: requestIndex === 0 ? 'data-auto-annotation' : 'artifact_read', duration_ms: 12 }),
    ...artifacts.map((item) => sseFrame('artifact.created', { artifact: item })),
    sseFrame('agent.message.delta', { text: reply }),
    sseFrame('run.completed', { result }),
  ].join('');
  await route.fulfill({
    status: 200,
    headers: { 'Content-Type': 'text/event-stream; charset=utf-8', 'Cache-Control': 'no-store' },
    body,
  });
}

test('app center upload, skill execution, and same-thread continuation', async ({ page }) => {
  test.setTimeout(180000);
  writeSmokePng(IMAGE_PATH);

  const apiResponses = [];
  const runRequests = [];
  const consoleErrors = [];
  page.on('console', (message) => {
    if (message.type() === 'error') consoleErrors.push(message.text());
  });
  page.on('request', (request) => {
    if (request.url().includes('/api/agents/default/runs/stream')) {
      runRequests.push(request.postDataJSON());
    }
  });
  page.on('response', (response) => {
    const url = response.url();
    if (
      url.includes('/api/uploads/') ||
      url.includes('/api/agents/default/runs/stream') ||
      url.includes('/api/apps/templates') ||
      url.includes('/api/artifacts/')
    ) {
      apiResponses.push({ url, status: response.status() });
    }
  });
  await page.addInitScript(() => {
    localStorage.removeItem('jetlinks.runtime.threadId');
    const token = window.__WORKBENCH_UI_SMOKE_TOKEN__;
    if (token) localStorage.setItem('jetlinks.runtime.adminToken', token);
  });
  await page.exposeFunction('__workbenchUiSmokeToken', () => RUNTIME_TOKEN);
  await page.addInitScript(async () => {
    window.__WORKBENCH_UI_SMOKE_TOKEN__ = await window.__workbenchUiSmokeToken();
  });
  let runRequestCount = 0;
  await page.route('**/api/agents/default/runs/stream', async (route) => {
    const index = runRequestCount;
    runRequestCount += 1;
    await fulfillRunStream(route, index);
  });

  await page.goto(APP_URL, { waitUntil: 'domcontentloaded' });
  await page.locator('#transport').selectOption('sse');
  await page.locator('#thread').fill(`pw-ui-audit-${Date.now()}`);
  await expect(page.locator('#threadMini')).toContainText('pw-ui-audit-');
  await expect(page.locator('#runStatus')).toContainText('Ready');

  await page.locator('[data-view="apps"]').first().click();
  await expect(page.locator('#appCenter')).toBeVisible();
  await expect
    .poll(async () => page.locator('.app-template-card').count(), { timeout: 15000 })
    .toBeGreaterThanOrEqual(16);

  const appCard = page.locator('.app-template-card').filter({ hasText: 'data-auto-annotation' }).first();
  await expect(appCard).toBeVisible();
  await appCard.locator('[data-app-template-json="data-auto-annotation"]').click();
  await expect(appCard.locator('.app-template-json')).toContainText('selected_skills');
  await appCard.locator('[data-app-template-use="data-auto-annotation"]').click();

  await expect(page.locator('#chatFrame')).toBeVisible();
  await expect(page.locator('#input')).toHaveValue(/COCO|标注|图片/);
  await expect(page.locator('#capabilityTray')).toContainText('data-auto-annotation');

  await page.locator('#deepExecution').check();
  await page.locator('#fileInput').setInputFiles(IMAGE_PATH);
  await expect(page.locator('#attachments')).toContainText('已上传', { timeout: 20000 });
  await expect(page.locator('#attachments')).toContainText(/smoke-upload.*\.png/);

  await page.locator('#send').click();
  await expect(page.locator('#runStatus')).toContainText('Ready', { timeout: 90000 });
  await expect(page.locator('#messages')).toContainText('annotations.coco.json', { timeout: 15000 });
  await page.locator('[data-tab="artifacts"]').click();
  await expect(page.locator('#panel')).toContainText('annotations.coco.json', { timeout: 15000 });

  expect(runRequests.length).toBeGreaterThanOrEqual(1);
  const firstRequest = runRequests[0];
  expect(firstRequest.attachments?.[0]?.path).toMatch(/^\/mnt\/user-data\/uploads\//);
  expect(firstRequest.runtime_options?.selected_skills).toContain('data-auto-annotation');
  expect(firstRequest.runtime_options?.mode).toBe('autonomous');
  expect(firstRequest.runtime_options?.config_options?.max_tool_rounds).toBe(12);

  await page
    .locator('#input')
    .fill('请读取当前会话已有的 annotations.coco.json，生成一份 COCO 标注摘要 Markdown，保存为 coco-summary.md');
  await page.locator('#send').click();
  await expect(page.locator('#runStatus')).toContainText('Ready', { timeout: 90000 });
  await page.locator('[data-tab="artifacts"]').click();
  await expect(page.locator('#panel')).toContainText('coco-summary.md', { timeout: 15000 });

  expect(apiResponses.filter((item) => item.status >= 500)).toEqual([]);
  expect(consoleErrors).toEqual([]);
});
