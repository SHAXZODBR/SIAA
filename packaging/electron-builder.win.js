// electron-builder configuration for the laptop / kit build (packaging/build_windows.ps1):
//
//     cd desktop-app
//     npm run build:win -- --publish never --config ../packaging/electron-builder.win.js
//
// It is desktop-app/package.json "build" verbatim (same NSIS target, icons, extraResources:
// backend exe + models + docs) with ONE change: npmRebuild=false. electron-builder would
// otherwise rebuild better-sqlite3 for the Electron ABI, and better-sqlite3 11.x ships no
// prebuilt binary for Electron 41 — the rebuild falls back to node-gyp and needs Visual
// Studio Build Tools, which a fresh laptop does not have. electron/main.js never loads
// better-sqlite3 (or any other native module), so skipping the rebuild changes nothing at
// runtime. A boolean cannot be passed on the CLI (-c.npmRebuild=false arrives as the
// string "false"), hence this file. Paths inside "build" stay relative to desktop-app/.
'use strict';
const base = require('../desktop-app/package.json').build;
module.exports = { ...base, npmRebuild: false };
