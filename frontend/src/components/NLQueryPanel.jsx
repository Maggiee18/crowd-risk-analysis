import React, { useState, useCallback, useRef, useEffect } from 'react';
import { MessageSquare, Send, X, Bot, User, Search } from 'lucide-react';

/**
 * Feature #9: Natural Language Query Interface
 * Pattern-matching NL query engine for crowd analytics data.
 */
export default function NLQueryPanel({ history = [], alerts = [], currentState, predictions }) {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([
    { role: 'bot', text: 'Hi! I\'m CrowdGuard AI. Ask me anything about the crowd data. Try:\n• "How many people are there?"\n• "What\'s the current risk?"\n• "When was the highest crowd?"\n• "Show me alert summary"' }
  ]);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const processQuery = useCallback((q) => {
    const ql = q.toLowerCase().trim();
    const features = currentState?.features || {};
    const risk = currentState?.risk_prediction || {};
    const anomaly = currentState?.anomaly_detection || {};
    const count = currentState?.people_count || 0;

    // Pattern matching
    if (ql.match(/how many (people|person|crowd)/)) {
      return `Currently detecting **${count} people** in the frame. Density is ${(features.density_per_frame || 0).toFixed(3)} people/m².`;
    }

    if (ql.match(/(current|what).*(risk|danger|threat)/)) {
      const rl = (risk.risk_level || 'unknown').toUpperCase();
      const conf = ((risk.confidence || 0) * 100).toFixed(0);
      return `Current risk level: **${rl}** (${conf}% confidence).\n${risk.risk_level === 'high' ? '⚠️ Immediate attention recommended!' : ''}`;
    }

    if (ql.match(/(highest|peak|max).*(crowd|people|count)/)) {
      if (history.length === 0) return 'No historical data available yet.';
      const peak = history.reduce((max, h) => h.people_count > max.people_count ? h : max, history[0]);
      const time = peak.timestamp ? new Date(peak.timestamp).toLocaleTimeString() : 'unknown';
      return `Peak crowd count was **${peak.people_count} people** at ${time}.`;
    }

    if (ql.match(/(lowest|min|minimum).*(crowd|people|count)/)) {
      if (history.length === 0) return 'No historical data available yet.';
      const min = history.reduce((m, h) => (h.people_count || 0) < (m.people_count || 0) ? h : m, history[0]);
      const time = min.timestamp ? new Date(min.timestamp).toLocaleTimeString() : 'unknown';
      return `Minimum crowd count was **${min.people_count || 0} people** at ${time}.`;
    }

    if (ql.match(/(average|avg|mean).*(crowd|people|count)/)) {
      if (history.length === 0) return 'No historical data available yet.';
      const avg = history.reduce((s, h) => s + (h.people_count || 0), 0) / history.length;
      return `Average crowd count across ${history.length} observations: **${avg.toFixed(1)} people**.`;
    }

    if (ql.match(/(alert|alarm|warning|incident)/)) {
      if (alerts.length === 0) return 'No active alerts at this time. ✅';
      return `There are **${alerts.length} active alert(s)**:\n${alerts.slice(0, 5).map((a, i) => `${i + 1}. ${a.message || a.type || 'Alert'}`).join('\n')}`;
    }

    if (ql.match(/(anomaly|abnormal|unusual)/)) {
      if (anomaly.is_anomaly) {
        return `⚠️ **Anomaly detected!** Score: ${(anomaly.anomaly_score || 0).toFixed(2)}. The current crowd behavior deviates significantly from normal patterns.`;
      }
      return 'No anomalies detected. Crowd behavior appears normal. ✅';
    }

    if (ql.match(/(speed|velocity|movement|flow)/)) {
      return `Average crowd speed: **${(features.avg_magnitude || 0).toFixed(1)} px/s**.\nDominant direction: ${(features.dominant_direction || 0).toFixed(0)}°.\nFlow uniformity: ${((features.direction_uniformity || 0) * 100).toFixed(0)}%.`;
    }

    if (ql.match(/(density|packed|crowded)/)) {
      const d = features.density_per_frame || 0;
      const level = d > 0.1 ? 'dangerously high' : d > 0.05 ? 'above normal' : d > 0.02 ? 'moderate' : 'low';
      return `Crowd density: **${d.toFixed(4)} people/m²** (${level}).`;
    }

    if (ql.match(/(predict|forecast|future|next)/)) {
      const preds = predictions?.predictions || [];
      if (preds.length === 0) return 'Not enough data for predictions yet. Need more observations.';
      return `Predicted crowd counts for next ${preds.length} steps:\n${preds.map((p, i) => `Step +${i + 1}: **${p} people**`).join('\n')}\nMethod: ${predictions.method || 'LSTM'} (${((predictions.confidence || 0) * 100).toFixed(0)}% confidence)`;
    }

    if (ql.match(/(summary|overview|status|report)/)) {
      return `**System Summary:**\n• People: ${count}\n• Risk: ${(risk.risk_level || 'unknown').toUpperCase()} (${((risk.confidence || 0) * 100).toFixed(0)}%)\n• Density: ${(features.density_per_frame || 0).toFixed(3)}\n• Speed: ${(features.avg_magnitude || 0).toFixed(1)} px/s\n• Anomaly: ${anomaly.is_anomaly ? '⚠️ Detected' : '✅ Normal'}\n• Alerts: ${alerts.length} active\n• History: ${history.length} data points`;
    }

    if (ql.match(/(help|what can you|commands|options)/)) {
      return 'I can answer questions about:\n• **People count** — "how many people?"\n• **Risk level** — "what\'s the risk?"\n• **Peak/min crowd** — "when was peak?"\n• **Alerts** — "show alerts"\n• **Anomalies** — "any anomalies?"\n• **Speed/flow** — "crowd speed?"\n• **Density** — "how crowded?"\n• **Predictions** — "forecast next steps"\n• **Summary** — "give me a summary"';
    }

    return 'I\'m not sure how to answer that. Try asking about people count, risk level, alerts, density, speed, predictions, or type "help" for all options.';
  }, [history, alerts, currentState, predictions]);

  const handleSend = useCallback(() => {
    if (!query.trim()) return;

    const userMsg = { role: 'user', text: query };
    const botResponse = processQuery(query);
    const botMsg = { role: 'bot', text: botResponse };

    setMessages(prev => [...prev, userMsg, botMsg]);
    setQuery('');
  }, [query, processQuery]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <>
      {/* Floating toggle button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="fixed bottom-6 right-6 z-50 w-14 h-14 rounded-2xl bg-brand-600 hover:bg-brand-500 text-white shadow-xl shadow-brand-600/30 flex items-center justify-center transition-all hover:scale-110 active:scale-95"
        id="nl-query-toggle"
        title="Ask CrowdGuard AI"
      >
        {isOpen ? <X size={22} /> : <MessageSquare size={22} />}
      </button>

      {/* Chat panel */}
      {isOpen && (
        <div className="fixed bottom-24 right-6 z-50 w-96 max-h-[500px] rounded-2xl shadow-2xl flex flex-col animate-slide-up overflow-hidden"
          style={{
            background: 'var(--card-bg)',
            border: '1px solid var(--card-border)',
            backdropFilter: 'blur(20px)',
          }}
        >
          {/* Header */}
          <div className="flex items-center gap-2 px-4 py-3" style={{ borderBottom: '1px solid var(--card-border)' }}>
            <Bot size={18} className="text-brand-400" />
            <h3 className="text-sm font-semibold t-heading flex-1">CrowdGuard AI</h3>
            <span className="text-[9px] t-muted bg-emerald-500/15 text-emerald-400 px-2 py-0.5 rounded-full">Online</span>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-3 space-y-3 max-h-[350px]">
            {messages.map((msg, i) => (
              <div key={i} className={`flex gap-2 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                {msg.role === 'bot' && (
                  <div className="w-6 h-6 rounded-lg bg-brand-500/20 flex items-center justify-center shrink-0 mt-0.5">
                    <Bot size={12} className="text-brand-400" />
                  </div>
                )}
                <div
                  className={`max-w-[80%] px-3 py-2 rounded-xl text-xs leading-relaxed ${
                    msg.role === 'user'
                      ? 'bg-brand-600 text-white rounded-br-sm'
                      : 'rounded-bl-sm'
                  }`}
                  style={msg.role === 'bot' ? { background: 'var(--panel-bg)', color: 'var(--color-text)' } : {}}
                >
                  {msg.text.split('\n').map((line, li) => (
                    <p key={li} className={li > 0 ? 'mt-1' : ''}>
                      {line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').split('<strong>').map((part, pi) => {
                        if (part.includes('</strong>')) {
                          const [bold, rest] = part.split('</strong>');
                          return <React.Fragment key={pi}><strong className="font-bold">{bold}</strong>{rest}</React.Fragment>;
                        }
                        return part;
                      })}
                    </p>
                  ))}
                </div>
                {msg.role === 'user' && (
                  <div className="w-6 h-6 rounded-lg bg-purple-500/20 flex items-center justify-center shrink-0 mt-0.5">
                    <User size={12} className="text-purple-400" />
                  </div>
                )}
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="p-3" style={{ borderTop: '1px solid var(--card-border)' }}>
            <div className="flex gap-2">
              <input
                ref={inputRef}
                value={query}
                onChange={e => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about crowd data…"
                className="input-field !py-2 !text-xs flex-1"
              />
              <button
                onClick={handleSend}
                disabled={!query.trim()}
                className="btn-primary !p-2.5 !rounded-xl"
              >
                <Send size={14} />
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
