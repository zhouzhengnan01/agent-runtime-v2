#!/usr/bin/env node
/**
 * 把一个 AI 组件目录打包成 zip。
 *
 * 用法:
 *   node build-zip.mjs <组件目录> [输出zip路径]
 *
 * 默认输出到 <组件目录>/../<目录名>.zip
 * 递归收集目录下所有文件(component.vue / config.mjs / Config.vue / 任意子文件)。
 * 依赖 fflate(从最近的 node_modules 解析;jetlinks 仓库 jetlinks-web-core 已装)。
 */
import { readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs'
import { join, basename, resolve, dirname } from 'node:path'

const srcArg = process.argv[2]
if (!srcArg) {
  console.error('用法: node build-zip.mjs <组件目录> [输出zip路径]')
  process.exit(1)
}
const srcDir = resolve(srcArg)
const name = basename(srcDir)
const outPath = process.argv[3] ? resolve(process.argv[3]) : join(dirname(srcDir), `${name}.zip`)

let zipSync, strToU8
try {
  ;({ zipSync, strToU8 } = await import('fflate'))
} catch {
  console.error('✗ 需要 fflate。请在含 fflate 的目录运行(如 jetlinks-web-core/),或先 `npm i fflate`')
  process.exit(1)
}

const TEXT = new Set(['vue', 'ts', 'js', 'mjs', 'cjs', 'json', 'css', 'less', 'scss', 'svg', 'md', 'txt'])

function collect(dir, base = dir, out = {}) {
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry)
    const rel = full.slice(base.length + 1).replace(/\\/g, '/')
    if (statSync(full).isDirectory()) collect(full, base, out)
    else {
      const ext = entry.split('.').pop()
      out[rel] = TEXT.has(ext) ? strToU8(readFileSync(full, 'utf8')) : readFileSync(full)
    }
  }
  return out
}

const files = collect(srcDir)
writeFileSync(outPath, zipSync(files, { level: 6 }))
console.log(`✓ ${outPath} (${(statSync(outPath).size / 1024).toFixed(1)} KB) — ${Object.keys(files).join(', ')}`)
