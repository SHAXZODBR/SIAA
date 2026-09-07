/**
 * Product identity shown in the UI and printed on reports.
 * Version comes from Electron (packaged build) or from package.json (dev/web).
 */

import { version as PKG_VERSION } from '../../package.json';

export const PRODUCT_NAME = 'Sentinel Medical AI';
export const VENDOR_NAME = 'SIA Medical AI';
export const VENDOR_SITE = 'siaa.uz';

let cachedVersion: string | null = null;

export function getPackageVersion(): string {
  return PKG_VERSION;
}

export async function getAppVersion(): Promise<string> {
  if (cachedVersion) return cachedVersion;
  try {
    const info = await window.electronAPI?.getAppInfo?.();
    if (info?.version) {
      cachedVersion = info.version;
      return cachedVersion;
    }
  } catch {
    // fall through to the package version
  }
  cachedVersion = PKG_VERSION;
  return cachedVersion;
}
