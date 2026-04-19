import React, { useState, useMemo, useCallback } from 'react';
import { Sliders, Play, RotateCcw, TrendingUp, Users, AlertTriangle } from 'lucide-react';

/**
 * Feature #12: "What-If" Scenario Simulator
 * Lets users adjust parameters and see predicted risk impact.
 */
export default function WhatIfSimulator({ currentState, features: currentFeatures = {} }) {
  const baseline = {
    people: currentState?.people_count || 0,
    density: (currentFeatures.density_per_frame || 0) * 1000,
    speed: currentFeatures.avg_magnitude || 5,
    uniformity: (currentFeatures.direction_uniformity || 0.8) * 100,
  };

  const [params, setParams] = useState({
    additionalPeople: 0,
    densityMultiplier: 1.0,
    speedMultiplier: 1.0,
    counterFlowPct: 0, // % of people going opposite direction
    blockExit: false,
  });

  const [showResult, setShowResult] = useState(false);

  // Predict risk based on modified parameters
  const prediction = useMemo(() => {
    const totalPeople = baseline.people + params.additionalPeople;
    const effectiveDensity = baseline.density * params.densityMultiplier;
    const effectiveSpeed = baseline.speed * params.speedMultiplier;
    const effectiveUniformity = Math.max(0, baseline.uniformity - params.counterFlowPct);

    // Simple risk model
    let riskScore = 0;

    // People factor (0-30)
    riskScore += Math.min(30, totalPeople * 0.8);

    // Density factor (0-25)
    riskScore += Math.min(25, effectiveDensity * 0.5);

    // Speed factor — too fast OR too slow is risky
    if (effectiveSpeed < 2 && totalPeople > 5) riskScore += 20; // Congestion
    else if (effectiveSpeed > 15) riskScore += 20; // Panic
    else riskScore += Math.min(10, effectiveSpeed * 0.5);

    // Counter-flow (0-15)
    riskScore += (params.counterFlowPct / 100) * 15;

    // Exit blocked (0-15)
    if (params.blockExit) riskScore += 15;

    riskScore = Math.max(0, Math.min(100, riskScore));

    const riskLevel = riskScore < 30 ? 'low' : riskScore < 60 ? 'medium' : 'high';
    const riskColor = riskScore < 30 ? '#10b981' : riskScore < 60 ? '#f59e0b' : '#ef4444';

    // Time to critical
    const currentScore = Math.min(100, baseline.people * 0.8 + baseline.density * 0.5);
    const rateOfChange = riskScore - currentScore;
    const timeToCritical = rateOfChange <= 0 ? null : Math.ceil((80 - riskScore) / (rateOfChange * 0.1));

    return {
      totalPeople,
      effectiveDensity: effectiveDensity / 1000,
      effectiveSpeed,
      riskScore,
      riskLevel,
      riskColor,
      timeToCritical,
      deltaFromBaseline: riskScore - currentScore,
    };
  }, [params, baseline]);

  const resetParams = () => {
    setParams({
      additionalPeople: 0,
      densityMultiplier: 1.0,
      speedMultiplier: 1.0,
      counterFlowPct: 0,
      blockExit: false,
    });
    setShowResult(false);
  };

  return (
    <div className="glass-card" id="what-if-simulator">
      <div className="flex items-center gap-2 mb-4">
        <Sliders size={16} className="text-orange-400" />
        <h3 className="text-sm font-semibold t-heading">What-If Simulator</h3>
        <button onClick={resetParams} className="ml-auto p-1.5 rounded-lg hover:bg-white/5 t-muted transition-colors">
          <RotateCcw size={12} />
        </button>
      </div>

      {/* Parameter sliders */}
      <div className="space-y-3">
        {/* Additional people */}
        <div>
          <div className="flex justify-between text-[10px] mb-1">
            <span className="t-muted">Additional people entering</span>
            <span className="t-heading font-bold">+{params.additionalPeople}</span>
          </div>
          <input
            type="range" min={0} max={100} step={5}
            value={params.additionalPeople}
            onChange={e => setParams(p => ({ ...p, additionalPeople: +e.target.value }))}
            className="w-full h-1.5 rounded-full appearance-none cursor-pointer"
            style={{
              background: `linear-gradient(to right, #6366f1 ${params.additionalPeople}%, var(--panel-border) ${params.additionalPeople}%)`,
            }}
          />
        </div>

        {/* Density multiplier */}
        <div>
          <div className="flex justify-between text-[10px] mb-1">
            <span className="t-muted">Density multiplier</span>
            <span className="t-heading font-bold">{params.densityMultiplier.toFixed(1)}x</span>
          </div>
          <input
            type="range" min={0.5} max={5} step={0.1}
            value={params.densityMultiplier}
            onChange={e => setParams(p => ({ ...p, densityMultiplier: +e.target.value }))}
            className="w-full h-1.5 rounded-full appearance-none cursor-pointer"
            style={{
              background: `linear-gradient(to right, #f59e0b ${((params.densityMultiplier - 0.5) / 4.5) * 100}%, var(--panel-border) ${((params.densityMultiplier - 0.5) / 4.5) * 100}%)`,
            }}
          />
        </div>

        {/* Speed multiplier */}
        <div>
          <div className="flex justify-between text-[10px] mb-1">
            <span className="t-muted">Speed factor</span>
            <span className="t-heading font-bold">{params.speedMultiplier.toFixed(1)}x</span>
          </div>
          <input
            type="range" min={0} max={5} step={0.1}
            value={params.speedMultiplier}
            onChange={e => setParams(p => ({ ...p, speedMultiplier: +e.target.value }))}
            className="w-full h-1.5 rounded-full appearance-none cursor-pointer"
            style={{
              background: `linear-gradient(to right, #8b5cf6 ${(params.speedMultiplier / 5) * 100}%, var(--panel-border) ${(params.speedMultiplier / 5) * 100}%)`,
            }}
          />
        </div>

        {/* Counter-flow */}
        <div>
          <div className="flex justify-between text-[10px] mb-1">
            <span className="t-muted">Counter-flow percentage</span>
            <span className="t-heading font-bold">{params.counterFlowPct}%</span>
          </div>
          <input
            type="range" min={0} max={80} step={5}
            value={params.counterFlowPct}
            onChange={e => setParams(p => ({ ...p, counterFlowPct: +e.target.value }))}
            className="w-full h-1.5 rounded-full appearance-none cursor-pointer"
            style={{
              background: `linear-gradient(to right, #ef4444 ${(params.counterFlowPct / 80) * 100}%, var(--panel-border) ${(params.counterFlowPct / 80) * 100}%)`,
            }}
          />
        </div>

        {/* Exit blocked toggle */}
        <div className="flex items-center justify-between">
          <span className="text-[10px] t-muted">Block exit points</span>
          <button
            onClick={() => setParams(p => ({ ...p, blockExit: !p.blockExit }))}
            className={`px-3 py-1 rounded-lg text-[10px] font-semibold transition-all ${
              params.blockExit
                ? 'bg-red-500/20 text-red-400 border border-red-500/20'
                : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/20'
            }`}
          >
            {params.blockExit ? 'Blocked' : 'Open'}
          </button>
        </div>
      </div>

      {/* Simulate button */}
      <button
        onClick={() => setShowResult(true)}
        className="btn-primary w-full mt-4"
      >
        <Play size={14} /> Simulate Scenario
      </button>

      {/* Results */}
      {showResult && (
        <div className="mt-4 space-y-3 animate-fade-in">
          {/* Risk score */}
          <div className="text-center p-4 rounded-xl" style={{ background: prediction.riskColor + '15', border: `1px solid ${prediction.riskColor}30` }}>
            <div className="text-3xl font-bold" style={{ color: prediction.riskColor }}>
              {prediction.riskScore.toFixed(0)}%
            </div>
            <div className="text-xs font-semibold mt-1" style={{ color: prediction.riskColor }}>
              {prediction.riskLevel.toUpperCase()} RISK
            </div>
            <div className="text-[10px] t-muted mt-1">
              {prediction.deltaFromBaseline >= 0 ? '+' : ''}{prediction.deltaFromBaseline.toFixed(0)} from baseline
            </div>
          </div>

          {/* Predicted metrics */}
          <div className="grid grid-cols-2 gap-2">
            <div className="p-2.5 rounded-xl text-center" style={{ background: 'var(--panel-bg)', border: '1px solid var(--panel-border)' }}>
              <Users size={14} className="mx-auto mb-1 t-muted" />
              <div className="text-sm font-bold t-heading">{prediction.totalPeople}</div>
              <div className="text-[9px] t-muted">Total People</div>
            </div>
            <div className="p-2.5 rounded-xl text-center" style={{ background: 'var(--panel-bg)', border: '1px solid var(--panel-border)' }}>
              <TrendingUp size={14} className="mx-auto mb-1 t-muted" />
              <div className="text-sm font-bold t-heading">{prediction.effectiveDensity.toFixed(3)}</div>
              <div className="text-[9px] t-muted">Density (ppl/m²)</div>
            </div>
          </div>

          {/* Warnings */}
          {prediction.riskScore > 50 && (
            <div className="flex items-start gap-2 p-2.5 rounded-xl bg-red-500/10 border border-red-500/15 text-xs text-red-400">
              <AlertTriangle size={14} className="shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold">Scenario exceeds safe thresholds</p>
                <p className="text-[10px] opacity-70 mt-0.5">
                  {params.additionalPeople > 30 && 'Too many people entering simultaneously. '}
                  {params.blockExit && 'Blocked exits create dangerous bottlenecks. '}
                  {params.counterFlowPct > 30 && 'Counter-flow increases crush risk. '}
                  {prediction.timeToCritical && `Estimated time to critical: ~${prediction.timeToCritical} min`}
                </p>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
