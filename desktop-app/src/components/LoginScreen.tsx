import React, { useState, useEffect } from 'react';
import { useAppStore } from '../store/appStore';
import { login, changePassword, describeApiError } from '../services/api';
import { useT, LANGS, langShort, langName } from '../i18n';
import { PRODUCT_NAME, getAppVersion } from '../services/appInfo';

/**
 * Login gate. Authenticates against POST /auth/login; the role comes from the
 * server. When the server says must_change_password, a change-password form is
 * shown before the workstation is unlocked.
 */
export default function LoginScreen() {
  const { setAuth, currentUser, mustChangePassword, settings, health, updateSettings } = useAppStore();
  const t = useT();
  const lang = settings.language;
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [version, setVersion] = useState('');

  useEffect(() => {
    const tm = setTimeout(() => setMounted(true), 50);
    getAppVersion().then(setVersion).catch(() => {});
    return () => clearTimeout(tm);
  }, []);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    if (!username || !password) {
      setError(t('login.enterBoth'));
      return;
    }
    setLoading(true);
    try {
      const res = await login(username.trim(), password);
      setAuth(res.token, res.user, res.mustChangePassword);
      setPassword('');
    } catch (err) {
      const info = describeApiError(err);
      console.error('Login failed:', info.code, info.detail);
      if (info.code === 'auth') setError(t('login.badCredentials'));
      else if (info.code === 'network') setError(t('health.aiServerDown'));
      else setError(t('health.aiServerError'));
    } finally {
      setLoading(false);
    }
  };

  const showChangePassword = !!currentUser && mustChangePassword;

  const serverState = !health || !health.reachable
    ? { label: t('common.offline'), color: 'bg-critical' }
    : health.status === 'ok'
      ? { label: t('common.online'), color: 'bg-normal' }
      : { label: t('common.degraded'), color: 'bg-moderate' };

  return (
    <div className="h-screen w-screen bg-ink-950 flex overflow-hidden">
      {/* Left: Login panel */}
      <div className="w-[420px] flex flex-col p-8 bg-ink-900 border-r border-ink-800 relative z-10">
        {/* Logo */}
        <div className={`flex items-center gap-3 mb-10 transition-all duration-500 ${mounted ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-2'}`}>
          <div className="w-10 h-10 bg-gradient-to-br from-accent-400 to-accent-700 rounded-xl flex items-center justify-center shadow-glow-accent">
            <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round"
                d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
          </div>
          <div>
            <div className="text-lg font-bold text-ink-100 tracking-tight">SENTINEL</div>
            <div className="text-[10px] text-ink-500 uppercase tracking-[0.2em]">{t('product.tagline')}</div>
          </div>
        </div>

        {/* Language switcher */}
        <div className="flex items-center gap-1 mb-6" role="group" aria-label={t('topbar.language')}>
          {LANGS.map((l) => (
            <button
              key={l}
              type="button"
              onClick={() => updateSettings({ language: l })}
              title={langName(l)}
              className={`px-2 py-0.5 text-[10px] font-semibold rounded ${
                lang === l ? 'bg-accent-600 text-white' : 'bg-ink-800 text-ink-400 hover:text-ink-200'
              }`}
            >
              {langShort(l)}
            </button>
          ))}
        </div>

        {showChangePassword ? (
          <ChangePasswordForm />
        ) : (
          <>
            {/* Welcome */}
            <div className={`mb-8 transition-all duration-500 delay-100 ${mounted ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2'}`}>
              <h1 className="text-2xl font-bold text-ink-100 mb-1">{t('login.title')}</h1>
              <p className="text-sm text-ink-400">{t('login.subtitle')}</p>
            </div>

            {/* Form */}
            <form onSubmit={handleLogin} className={`space-y-4 transition-all duration-500 delay-200 ${mounted ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2'}`}>
              <div>
                <label className="block text-[11px] font-semibold text-ink-400 uppercase tracking-wider mb-1.5">
                  {t('login.username')}
                </label>
                <input
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  autoFocus
                  autoComplete="username"
                  className="input-medical"
                />
              </div>

              <div>
                <label className="block text-[11px] font-semibold text-ink-400 uppercase tracking-wider mb-1.5">
                  {t('login.password')}
                </label>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  autoComplete="current-password"
                  className="input-medical"
                />
              </div>

              {error && <ErrorBox text={error} />}

              <button
                type="submit"
                disabled={loading}
                className="w-full py-2.5 bg-gradient-to-r from-accent-600 to-accent-500 hover:from-accent-500 hover:to-accent-400 text-white rounded-md font-semibold transition-all shadow-sm hover:shadow-glow-accent disabled:opacity-70 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <Spinner />
                    {t('login.authenticating')}
                  </>
                ) : (
                  <>
                    {t('login.signIn')}
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M17 8l4 4m0 0l-4 4m4-4H3" />
                    </svg>
                  </>
                )}
              </button>
            </form>
          </>
        )}

        {/* Spacer */}
        <div className="flex-1" />

        {/* Footer */}
        <div className="space-y-2 pt-4">
          <div className="flex items-center gap-4 text-[10px] text-ink-500">
            <span className="flex items-center gap-1">
              <div className={`w-1.5 h-1.5 rounded-full ${serverState.color}`} />
              {t('login.aiServer')}: {serverState.label}
            </span>
            <span className="font-mono truncate" title={`${t('login.serverUrl')}: ${settings.inferenceUrl}`}>{settings.inferenceUrl}</span>
          </div>
          <div className="text-[10px] text-ink-600 flex items-center justify-between">
            <span>{version ? `v${version}` : ''}</span>
            <span>{t('common.support')}: {settings.supportContact || t('login.supportPlaceholder')}</span>
          </div>
        </div>
      </div>

      {/* Right: Product identity panel */}
      <div className="flex-1 relative bg-mesh overflow-hidden">
        <div className="absolute inset-0 bg-grid" />
        <div className="absolute top-1/4 right-1/4 w-96 h-96 bg-accent-500/10 rounded-full blur-3xl" />

        <div className={`relative z-10 h-full flex flex-col justify-center px-16 transition-all duration-700 delay-300 ${mounted ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-8'}`}>
          <div className="mb-10">
            <h2 className="text-5xl font-bold text-ink-100 leading-tight mb-3 text-balance">
              {PRODUCT_NAME}
            </h2>
            {version && (
              <div className="text-sm font-mono text-ink-500 mb-6">{t('common.version')} {version}</div>
            )}
            <p className="text-lg text-ink-300 max-w-xl leading-relaxed border-l-2 border-accent-500 pl-4">
              {t('product.intendedUse')}
            </p>
          </div>

          <div className="max-w-xl p-4 bg-ink-900/50 backdrop-blur border border-ink-800 rounded-xl text-xs text-ink-400 leading-relaxed">
            {settings.clinicName || t('login.clinicPlaceholder')}
            {settings.clinicAddress ? ` · ${settings.clinicAddress}` : ''}
            {settings.clinicPhone ? ` · ${settings.clinicPhone}` : ''}
          </div>
        </div>
      </div>
    </div>
  );
}

function ChangePasswordForm() {
  const { setMustChangePassword } = useAppStore();
  const t = useT();
  const [oldPassword, setOldPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    if (newPassword.length < 8) { setError(t('login.passwordTooShort')); return; }
    if (newPassword !== confirm) { setError(t('login.passwordsDiffer')); return; }
    setLoading(true);
    try {
      await changePassword(oldPassword, newPassword);
      setMustChangePassword(false);
    } catch (err) {
      const info = describeApiError(err);
      console.error('Change password failed:', info.code, info.detail);
      setError(info.code === 'network' ? t('health.aiServerDown') : t('login.changeFailed'));
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-ink-100 mb-1">{t('login.changePassword')}</h1>
        <p className="text-sm text-ink-400">{t('login.changePasswordHint')}</p>
      </div>
      <form onSubmit={submit} className="space-y-4">
        <div>
          <label className="block text-[11px] font-semibold text-ink-400 uppercase tracking-wider mb-1.5">{t('login.currentPassword')}</label>
          <input type="password" value={oldPassword} onChange={(e) => setOldPassword(e.target.value)} autoFocus autoComplete="current-password" className="input-medical" />
        </div>
        <div>
          <label className="block text-[11px] font-semibold text-ink-400 uppercase tracking-wider mb-1.5">{t('login.newPassword')}</label>
          <input type="password" value={newPassword} onChange={(e) => setNewPassword(e.target.value)} autoComplete="new-password" className="input-medical" />
        </div>
        <div>
          <label className="block text-[11px] font-semibold text-ink-400 uppercase tracking-wider mb-1.5">{t('login.confirmPassword')}</label>
          <input type="password" value={confirm} onChange={(e) => setConfirm(e.target.value)} autoComplete="new-password" className="input-medical" />
        </div>
        {error && <ErrorBox text={error} />}
        <button
          type="submit"
          disabled={loading}
          className="w-full py-2.5 bg-gradient-to-r from-accent-600 to-accent-500 hover:from-accent-500 hover:to-accent-400 text-white rounded-md font-semibold transition-all disabled:opacity-70 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          {loading ? <Spinner /> : null}
          {t('common.save')}
        </button>
      </form>
    </>
  );
}

function ErrorBox({ text }: { text: string }) {
  return (
    <div className="p-3 bg-critical/10 border border-critical/30 rounded-md flex items-start gap-2 animate-fade-in">
      <svg className="w-4 h-4 text-critical flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
      </svg>
      <span className="text-xs text-critical">{text}</span>
    </div>
  );
}

function Spinner() {
  return (
    <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
    </svg>
  );
}
