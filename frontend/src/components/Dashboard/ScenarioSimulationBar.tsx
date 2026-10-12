'use client';

import React from 'react';
import { Sun, CloudRain, Moon, Radio, Zap, Eye } from 'lucide-react';

export type SimulationScenario = 'dry' | 'rain' | 'night';

interface ScenarioSimulationBarProps {
  scenario: SimulationScenario;
  onSelectScenario: (scenario: SimulationScenario) => void;
  activeSensor: 'lidar' | 'sonar';
  recordCount: number;
}

export const ScenarioSimulationBar: React.FC<ScenarioSimulationBarProps> = ({
  scenario,
  onSelectScenario,
  activeSensor,
  recordCount,
}) => {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        background: '#111520',
        border: '1px solid #1E2435',
        borderRadius: 8,
        padding: '7px 12px',
        flexWrap: 'wrap',
        gap: 8,
      }}
    >
      {/* ── Left: Scenario Mode Selector ─────────────────────── */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span
            style={{
              fontSize: '9px',
              fontWeight: 800,
              letterSpacing: '0.08em',
              padding: '2px 6px',
              borderRadius: 3,
              background: 'rgba(245, 158, 11, 0.15)',
              border: '1px solid rgba(245, 158, 11, 0.4)',
              color: '#FCD34D',
              fontFamily: 'monospace',
            }}
          >
            SIMULATION
          </span>
          <span style={{ fontSize: '11px', fontWeight: 700, color: '#94A3B8' }}>
            Environmental Scenario:
          </span>
        </div>

        <div
          style={{
            display: 'flex',
            background: '#0B0E14',
            padding: 2,
            borderRadius: 6,
            border: '1px solid #1E2435',
            gap: 2,
          }}
        >
          {/* Dry Scenario */}
          <button
            onClick={() => onSelectScenario('dry')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 5,
              padding: '4px 10px',
              borderRadius: 4,
              border: 'none',
              cursor: 'pointer',
              fontSize: '11px',
              fontWeight: 700,
              background: scenario === 'dry' ? '#1D4ED8' : 'transparent',
              color: scenario === 'dry' ? '#FFFFFF' : '#8892A4',
              transition: 'all 0.15s ease',
            }}
            title="Dry Road Daylight: 120 lx, Dry surface -> Optical LiDAR active"
          >
            <Sun style={{ width: 12, height: 12, color: scenario === 'dry' ? '#FDE047' : '#F59E0B' }} />
            <span>Dry (LiDAR)</span>
          </button>

          {/* Rain Scenario */}
          <button
            onClick={() => onSelectScenario('rain')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 5,
              padding: '4px 10px',
              borderRadius: 4,
              border: 'none',
              cursor: 'pointer',
              fontSize: '11px',
              fontWeight: 700,
              background: scenario === 'rain' ? '#0369A1' : 'transparent',
              color: scenario === 'rain' ? '#FFFFFF' : '#8892A4',
              transition: 'all 0.15s ease',
            }}
            title="Rain / Precipitation: Wet surface creates specular scatter -> Ultrasonic Sonar active"
          >
            <CloudRain style={{ width: 12, height: 12, color: scenario === 'rain' ? '#7DD3FC' : '#38BDF8' }} />
            <span>Rain (Sonar)</span>
          </button>

          {/* Night Scenario */}
          <button
            onClick={() => onSelectScenario('night')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 5,
              padding: '4px 10px',
              borderRadius: 4,
              border: 'none',
              cursor: 'pointer',
              fontSize: '11px',
              fontWeight: 700,
              background: scenario === 'night' ? '#581C87' : 'transparent',
              color: scenario === 'night' ? '#FFFFFF' : '#8892A4',
              transition: 'all 0.15s ease',
            }}
            title="Night Sub-40 Lux: Optical SNR drops below threshold -> Ultrasonic Sonar active"
          >
            <Moon style={{ width: 12, height: 12, color: scenario === 'night' ? '#DDD6FE' : '#A78BFA' }} />
            <span>Night (Sonar)</span>
          </button>
        </div>
      </div>

      {/* ── Right: Live Active Sensor Badge ──────────────────── */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '11px' }}>
        <span style={{ color: '#64748B', display: 'flex', alignItems: 'center', gap: 4 }}>
          <Radio style={{ width: 11, height: 11 }} />
          Active Modality:
        </span>
        <span
          style={{
            fontFamily: 'monospace',
            fontWeight: 800,
            fontSize: '10px',
            padding: '2px 8px',
            borderRadius: 4,
            background: activeSensor === 'lidar' ? 'rgba(59, 130, 246, 0.18)' : 'rgba(56, 189, 248, 0.18)',
            border: `1px solid ${activeSensor === 'lidar' ? 'rgba(59, 130, 246, 0.45)' : 'rgba(56, 189, 248, 0.45)'}`,
            color: activeSensor === 'lidar' ? '#93C5FD' : '#7DD3FC',
          }}
        >
          {activeSensor === 'lidar' ? 'Optical LiDAR (ToF)' : 'Ultrasonic Sonar (Acoustic)'}
        </span>
        <span style={{ color: '#4D5A72', fontSize: '10px', fontFamily: 'monospace' }}>
          ({recordCount} incidents updated live)
        </span>
      </div>
    </div>
  );
};
