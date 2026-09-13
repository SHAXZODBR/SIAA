import React, { useState, useRef, useEffect } from 'react';
import { useAppStore } from '../store/appStore';
import { askAIQuestion, describeApiError } from '../services/api';
import { useT, useLang, formatTime, langShort } from '../i18n';
import type { I18nKey } from '../i18n';

interface ChatMessage {
  id: string;
  role: 'user' | 'ai';
  text: string;
  timestamp: number;
}

const STUDY_SUGGESTIONS: I18nKey[] = ['chat.s1', 'chat.s2', 'chat.s3', 'chat.s4', 'chat.s5'];
const GENERAL_SUGGESTIONS: I18nKey[] = ['chat.g1', 'chat.g2', 'chat.g3', 'chat.g4', 'chat.g5'];

/**
 * Ask-AI panel — the doctor asks follow-up questions about the analysis.
 * Answered by the local report-assistant LLM (POST /report/ask); the backend
 * name comes from /health, nothing is hard-coded here.
 */
export default function AIChat() {
  const { selectedStudyId, aiResults, settings, health } = useAppStore();
  const t = useT();
  const lang = useLang();
  const result = selectedStudyId ? aiResults[selectedStudyId] : null;

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const currentSuggestions = (result ? STUDY_SUGGESTIONS : GENERAL_SUGGESTIONS).map((k) => t(k));
  const llmReady = !!health?.reachable && !!health.llm.reachable;
  const llmName = health?.llm.backend || 'LLM';

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    // Clear chat when study changes
    setMessages([]);
  }, [selectedStudyId]);

  const sendMessage = async (text: string) => {
    if (!text.trim()) return;

    const userMsg: ChatMessage = {
      id: Math.random().toString(36).slice(2),
      role: 'user',
      text: text.trim(),
      timestamp: Date.now(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      // If a study is selected, include findings as context. Otherwise, general Q&A.
      const findings = result?.findings.map((f) => ({
        class_name: f.className,
        confidence: f.confidence,
        location: f.location,
        positive: f.positive,
        status: f.status,
      })) || [];

      const context = result?.overallImpression || t('chat.noStudyContext');

      const answer = await askAIQuestion(
        text.trim(),
        findings,
        context,
        settings.language,
      );

      const aiMsg: ChatMessage = {
        id: Math.random().toString(36).slice(2),
        role: 'ai',
        text: answer,
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, aiMsg]);
    } catch (e) {
      const info = describeApiError(e);
      console.warn('Ask AI failed:', info.code, info.detail);
      setMessages((prev) => [...prev, {
        id: Math.random().toString(36).slice(2),
        role: 'ai',
        text: info.code === 'network' ? t('health.aiServerDown') : info.code === 'auth' ? t('health.sessionExpired') : t('chat.error'),
        timestamp: Date.now(),
      }]);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-3 py-2 border-b border-ink-800 bg-ink-950/30">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 bg-gradient-to-br from-accent-400 to-purple-600 rounded-md flex items-center justify-center">
            <svg className="w-3.5 h-3.5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
            </svg>
          </div>
          <div className="flex-1 min-w-0">
            <div className="text-xs font-semibold text-ink-100">{t('chat.title')}</div>
            <div className="text-[10px] text-ink-500 truncate">
              {t('chat.subtitle', { lang: langShort(lang) })}
              {health?.reachable ? ` · ${llmName}` : ''}
            </div>
          </div>
          <button
            onClick={() => setMessages([])}
            className="text-[10px] text-ink-500 hover:text-ink-300"
            title={t('chat.clear')}
          >
            {t('chat.clear')}
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {!llmReady && (
          <div className="px-3 py-2 rounded-lg border border-moderate/40 bg-moderate/10 text-[11px] text-ink-200 leading-snug">
            {health?.reachable ? t('chat.error') : t('health.aiServerDown')}
          </div>
        )}

        {messages.length === 0 && (
          <div className="space-y-3">
            <div className="text-[10px] font-semibold text-ink-500 uppercase tracking-wider px-1">
              {t('chat.suggested')}
            </div>
            {currentSuggestions.map((q, i) => (
              <button
                key={i}
                onClick={() => sendMessage(q)}
                className="w-full text-left p-3 bg-ink-850 hover:bg-ink-800 border border-ink-700 rounded-lg text-xs text-ink-200 transition-colors"
              >
                {q}
              </button>
            ))}
            <div className="text-[10px] text-ink-500 px-1 pt-2">
              {t('chat.hint')}
            </div>
          </div>
        )}

        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex gap-2 animate-fade-in ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            {msg.role === 'ai' && (
              <div className="w-7 h-7 flex-shrink-0 bg-gradient-to-br from-accent-500 to-purple-600 rounded-full flex items-center justify-center text-[10px] font-bold text-white">
                AI
              </div>
            )}
            <div
              className={`max-w-[85%] px-3 py-2 rounded-lg text-xs leading-relaxed whitespace-pre-wrap ${
                msg.role === 'user'
                  ? 'bg-accent-600 text-white rounded-br-sm'
                  : 'bg-ink-800 text-ink-100 rounded-bl-sm border border-ink-700'
              }`}
            >
              {msg.text}
              <div className="text-[9px] opacity-60 mt-1">
                {formatTime(msg.timestamp, lang)}
              </div>
            </div>
            {msg.role === 'user' && (
              <div className="w-7 h-7 flex-shrink-0 bg-ink-700 rounded-full flex items-center justify-center text-[10px] font-bold text-ink-200">
                {useAppStore.getState().currentUser?.fullName?.split(' ').map((n) => n[0]).join('').substring(0, 2) || '·'}
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="flex gap-2 animate-fade-in">
            <div className="w-7 h-7 flex-shrink-0 bg-gradient-to-br from-accent-500 to-purple-600 rounded-full flex items-center justify-center text-[10px] font-bold text-white">
              AI
            </div>
            <div className="px-3 py-2 bg-ink-800 border border-ink-700 rounded-lg text-xs">
              <div className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-ink-400 animate-pulse" style={{ animationDelay: '0ms' }} />
                <span className="w-1.5 h-1.5 rounded-full bg-ink-400 animate-pulse" style={{ animationDelay: '200ms' }} />
                <span className="w-1.5 h-1.5 rounded-full bg-ink-400 animate-pulse" style={{ animationDelay: '400ms' }} />
                <span className="text-ink-400 ml-1">{t('chat.thinking')}</span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-3 border-t border-ink-800 bg-ink-950/30">
        <div className="relative">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={t('chat.placeholder')}
            rows={2}
            className="w-full px-3 py-2 pr-10 text-xs bg-ink-900 border border-ink-700 rounded-lg text-ink-100 placeholder-ink-500 focus:outline-none focus:border-accent-500 resize-none"
          />
          <button
            onClick={() => sendMessage(input)}
            disabled={loading || !input.trim()}
            className="absolute right-2 bottom-2 p-1.5 bg-accent-600 hover:bg-accent-500 disabled:bg-ink-700 disabled:opacity-50 text-white rounded-md transition-colors"
          >
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
            </svg>
          </button>
        </div>
        <div className="text-[9px] text-ink-500 mt-1.5 flex items-center justify-between">
          <span>{t('chat.enterHint')}</span>
          <span className="flex items-center gap-1">
            <span className={`w-1 h-1 rounded-full ${llmReady ? 'bg-normal animate-pulse' : 'bg-ink-600'}`} />
            {t('chat.local')}
          </span>
        </div>
      </div>
    </div>
  );
}
