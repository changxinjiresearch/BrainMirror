'use strict';
/* U9 repository security / no-paid infra static policy regression. */
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const root=path.join(__dirname,'..','..'),read=file=>fs.readFileSync(path.join(root,file),'utf8');
const html=read('scept-web/universal.html'),sdk=read('scept-web/core/scept_sdk.js'),
 audit=read('scept-web/core/scept_audit.js'),consoleCode=read('scept-web/scept_m3_console.js'),
 pages=read('.github/workflows/scept-pages.yml'),work=read('.github/workflows/scept-browser.yml');
assert(html.includes('Content-Security-Policy'),'Universal core must declare a restrictive CSP');
assert(html.includes("object-src 'none'"));
assert(html.includes("base-uri 'none'"));
assert(html.includes("script-src 'self' https://cdn.jsdelivr.net 'wasm-unsafe-eval'"));
assert(!html.includes("'unsafe-eval'"),'No general eval permission');
assert(!html.includes("script-src 'unsafe-inline'"),'No inline JS permitted');
assert.match(html,/onnxruntime-web@1\.20\.1\/dist\/ort\.min\.js/);
for(const match of html.matchAll(/<script\\b([^>]*)>([\\s\\S]*?)<\\/script>/gi)){
 assert(/\\bsrc\\s*=/.test(match[1]),'Every JS element must be external');
 assert(!match[2].trim(),'External script tags must not include inline code');
}
for(const [name,code]of [['sdk',sdk],['audit',audit],['browser',consoleCode]]){
 assert(!/\beval\s*\(/.test(code),name+' must not evaluate uploaded JSON');
 assert(!/\bnew\s+Function\s*\(/.test(code),name+' must not construct executable JS');
 assert(!/fetch\s*\(/.test(code),name+' must not transmit experiments to a server');
}
assert(sdk.includes("NOT sandboxed"),'Must disclose trusted extension JS security model');
assert(audit.includes('AUDIT_HASH_MISMATCH'));
assert(audit.includes('INPUT_HASH_MISMATCH'));
assert(audit.includes('DEFINITION_MISMATCH'));
assert(audit.includes('redacted:true'));
assert(audit.includes('signature'), 'Checksum must not masquerade as a digital signature');
assert(consoleCode.includes('file.size>6000000'));
assert(!consoleCode.includes('localStorage.setItem'));
assert(pages.includes('github.event.workflow_run.conclusion == \'success\''));
assert(pages.includes("github.event.workflow_run.head_branch == 'main'"));
assert(pages.includes('enablement: true'));
assert(pages.includes('actions/deploy-pages@v4'));
assert(pages.includes('path: scept-web'));
assert(!pages.includes('vercel deploy'));
assert(work.includes('u9_m3_contract.cjs')&&work.includes('u9_m3_browser.cjs'));
console.log('PASS M3 security: no executed user JSON, CSP and disallowed script APIs, strict audit tamper/private data limits, free CI-gated static deploy');
