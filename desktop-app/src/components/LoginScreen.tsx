import React, { useState, useEffect } from 'react';
import { useAppStore } from '../store/appStore';
import type { User } from '../types';

export default function LoginScreen() {
  const { setCurrentUser } = useAppStore();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setTimeout(() => setMounted(true), 50);
  }, []);

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    setTimeout(() => {
      if (username && password) {
        const user: User = {
          id: 'user-' + Date.now(),
          username,
          fullName: username === 'admin' ? 'Administrator' : `Dr. ${username}`,
          role: username === 'admin' ? 'admin' : 'radiologist',
        };
        setCurrentUser(user);
      } else {
        setError('Please enter username and password');
      }
      setLoading(false);
    }, 600);
  };

  return (
    <div className="h-screen w-screen bg-ink-950 flex overflow-hidden">
      {/* Left: Login panel */}
      <div className="w-[420px] flex flex-col p-8 bg-ink-900 border-r border-ink-800 relative z-10">
        {/* Logo */}
        <div className={`flex items-center gap-3 mb-12 transition-all duration-500 ${mounted ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-2'}`}>
          <div className="w-10 h-10 bg-gradient-to-br from-accent-400 to-accent-700 rounded-xl flex items-center justify-center shadow-glow-accent">
            <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round"
                d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
          </div>
          <div>
            <div className="text-lg font-bold text-ink-100 tracking-tight">SENTINEL</div>
            <div className="text-[10px] text-ink-500 uppercase tracking-[0.2em]">Medical AI</div>
          </div>
        </div>

        {/* Welcome */}
        <div className={`mb-8 transition-all duration-500 delay-100 ${mounted ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2'}`}>
          <h1 className="text-2xl font-bold text-ink-100 mb-1">Welcome back</h1>
          <p className="text-sm text-ink-400">Sign in to access the workstation</p>
        </div>

        {/* Form */}
        <form onSubmit={handleLogin} className={`space-y-4 transition-all duration-500 delay-200 ${mounted ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2'}`}>
          <div>
            <label className="block text-[11px] font-semibold text-ink-400 uppercase tracking-wider mb-1.5">
              Username
            </label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              autoFocus
              autoComplete="username"
              className="input-medical"
              placeholder="doctor.ivanov"
            />
          </div>

          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-[11px] font-semibold text-ink-400 uppercase tracking-wider">
                Password
              </label>
              <button type="button" className="text-[10px] text-accent-400 hover:text-accent-300">
                Forgot?
              </button>
            </div>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="current-password"
              className="input-medical"
              placeholder="••••••••"
            />
          </div>

          {error && (
            <div className="p-3 bg-critical/10 border border-critical/30 rounded-md flex items-start gap-2 animate-fade-in">
              <svg className="w-4 h-4 text-critical flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
              <span className="text-xs text-critical">{error}</span>
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full py-2.5 bg-gradient-to-r from-accent-600 to-accent-500 hover:from-accent-500 hover:to-accent-400 text-white rounded-md font-semibold transition-all shadow-sm hover:shadow-glow-accent disabled:opacity-70 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {loading ? (
              <>
                <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Authenticating...
              </>
            ) : (
              <>
                Sign In
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M17 8l4 4m0 0l-4 4m4-4H3" />
                </svg>
              </>
            )}
          </button>
        </form>

        {/* Divider */}
        <div className="flex items-center gap-3 my-6">
          <div className="flex-1 h-px bg-ink-800" />
          <span className="text-[10px] text-ink-600 uppercase tracking-wider">or</span>
          <div className="flex-1 h-px bg-ink-800" />
        </div>

        {/* Quick access */}
        <div className="space-y-2">
          <button className="w-full py-2 bg-ink-800/50 hover:bg-ink-800 border border-ink-700 rounded-md text-sm text-ink-200 flex items-center justify-center gap-2 transition-colors">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M7 11V7a5 5 0 0110 0v4M5 11h14a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6a1 1 0 011-1z" />
            </svg>
            Use Smart Card
          </button>
          <button className="w-full py-2 bg-ink-800/50 hover:bg-ink-800 border border-ink-700 rounded-md text-sm text-ink-200 flex items-center justify-center gap-2 transition-colors">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
            Biometric (Touch ID)
          </button>
        </div>

        {/* Spacer */}
        <div className="flex-1" />

        {/* Footer */}
        <div className="space-y-2 pt-4">
          <div className="flex items-center gap-4 text-[10px] text-ink-500">
            <span className="flex items-center gap-1">
              <div className="w-1.5 h-1.5 rounded-full bg-normal animate-pulse" />
              Secure Connection
            </span>
            <span className="flex items-center gap-1">
              <div className="w-1.5 h-1.5 rounded-full bg-normal" />
              Local · No Internet
            </span>
          </div>
          <div className="text-[10px] text-ink-600 flex items-center justify-between">
            <span>v1.0.0 · Build 20240101</span>
            <a href="#" className="text-ink-500 hover:text-ink-300">Support</a>
          </div>
        </div>
      </div>

      {/* Right: Marketing / Stats panel */}
      <div className="flex-1 relative bg-mesh overflow-hidden">
        {/* Grid pattern */}
        <div className="absolute inset-0 bg-grid" />

        {/* Glow orbs */}
        <div className="absolute top-1/4 right-1/4 w-96 h-96 bg-accent-500/10 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 left-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl" />

        {/* Content */}
        <div className={`relative z-10 h-full flex flex-col justify-center px-16 transition-all duration-700 delay-300 ${mounted ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-8'}`}>
          <div className="mb-12">
            <div className="inline-flex items-center gap-2 px-3 py-1 bg-accent-500/10 border border-accent-500/30 rounded-full mb-6">
              <div className="w-1.5 h-1.5 rounded-full bg-accent-400 animate-pulse" />
              <span className="text-[10px] font-semibold text-accent-300 uppercase tracking-wider">AI-Powered Radiology</span>
            </div>
            <h2 className="text-5xl font-bold text-ink-100 leading-tight mb-4 text-balance">
              On-premise AI<br />for Uzbek clinics
            </h2>
            <p className="text-lg text-ink-400 max-w-md leading-relaxed">
              Sentinel analyzes DICOM images in under 10 seconds, delivering diagnostic findings with 90%+ recall — all without leaving the clinic.
            </p>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-3 gap-4 max-w-2xl">
            <StatBox value="90%+" label="Recall Rate" color="text-normal" />
            <StatBox value="< 10s" label="Inference Time" color="text-accent-400" />
            <StatBox value="14" label="Pathologies" color="text-purple-400" />
            <StatBox value="RU/UZ/EN" label="Languages" color="text-yellow-400" small />
            <StatBox value="100%" label="On-premise" color="text-normal" small />
            <StatBox value="DenseNet121" label="Architecture" color="text-accent-400" small />
          </div>

          {/* Credits */}
          <div className="mt-16 flex items-center gap-8 text-[10px] text-ink-500 uppercase tracking-wider">
            <div className="flex items-center gap-2">
              <div className="w-6 h-1 bg-accent-500 rounded" />
              Stanford-validated
            </div>
            <div className="flex items-center gap-2">
              <div className="w-6 h-1 bg-normal rounded" />
              GDPR-compliant
            </div>
            <div className="flex items-center gap-2">
              <div className="w-6 h-1 bg-purple-500 rounded" />
              IT-Park Uzbekistan
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function StatBox({ value, label, color, small }: { value: string; label: string; color: string; small?: boolean }) {
  return (
    <div className="p-4 bg-ink-900/50 backdrop-blur border border-ink-800 rounded-xl hover:border-ink-700 transition-colors">
      <div className={`font-bold ${color} ${small ? 'text-lg' : 'text-3xl'} mb-1`}>{value}</div>
      <div className="text-[10px] text-ink-500 uppercase tracking-wider">{label}</div>
    </div>
  );
}
