'use client';

import React, { useEffect, useState, useCallback, useMemo, useRef } from 'react';
import dynamic from 'next/dynamic';
import { Header } from '@/components/Dashboard/Header';
import { SummaryPanel } from '@/components/Dashboard/SummaryPanel';
import { MapWrapper } from '@/components/Map/MapWrapper';
import { TelemetryTable } from '@/components/Dashboard/TelemetryTable';
import { IncidentDetailPanel } from '@/components/Dashboard/IncidentDetailPanel';
import { QueueOverviewPanel } from '@/components/Dashboard/QueueOverviewPanel';
import { ScenarioSimulationBar, SimulationScenario } from '@/components/Dashboard/ScenarioSimulationBar';
import { fetchTelemetry, fetchPrioritization, FALLBACK_DATA } from '@/lib/api';
import { TelemetryResponse, PriorityLevel, TelemetryRecord, WeightProfile } from '@/types/telemetry';

const DatasetExplorerPanel = dynamic(
  () => import('@/components/Dashboard/DatasetExplorerPanel').then((m) => m.DatasetExplorerPanel),
  {
    ssr: false,
    loading: () => (
      <div style={{ padding: 40, textAlign: 'center', color: '#94A3B8' }}>
        Loading Dataset Explorer...
      </div>
    ),
  }
);

const SensitivityPanel = dynamic(
  () => import('@/components/Dashboard/SensitivityPanel').then((m) => m.SensitivityPanel),
  {
    ssr: false,
    loading: () => (
      <div style={{ padding: 40, textAlign: 'center', color: '#94A3B8' }}>
        Loading Sensitivity Engine...
      </div>
    ),
  }
);

const FALLBACK_PROFILE_MAP: Record<
  WeightProfile,
  Record<string, { topsis_rank: number; topsis_score: number; vikor_rank: number; priority_level: PriorityLevel; vikor_Q: number }>
> = {
  'safety-first': {
    'SRMD-IND-000045-00': { topsis_rank: 1, topsis_score: 0.571, vikor_rank: 4, priority_level: 'Critical', vikor_Q: 0.437 },
    'SRMD-IND-000102-00': { topsis_rank: 3, topsis_score: 0.4081, vikor_rank: 3, priority_level: 'Medium', vikor_Q: 0.3842 },
    'SRMD-IND-000155-00': { topsis_rank: 5, topsis_score: 0.3825, vikor_rank: 6, priority_level: 'Low', vikor_Q: 0.9142 },
    'SRMD-IND-000210-00': { topsis_rank: 4, topsis_score: 0.3912, vikor_rank: 5, priority_level: 'Medium', vikor_Q: 0.6426 },
    'SRMD-IND-000278-00': { topsis_rank: 2, topsis_score: 0.5583, vikor_rank: 1, priority_level: 'High', vikor_Q: 0.0 },
    'SRMD-IND-000278-01': { topsis_rank: 6, topsis_score: 0.3752, vikor_rank: 2, priority_level: 'Low', vikor_Q: 0.3625 },
  },
  'traffic-first': {
    'SRMD-IND-000045-00': { topsis_rank: 1, topsis_score: 0.7625, vikor_rank: 1, priority_level: 'Critical', vikor_Q: 0.0 },
    'SRMD-IND-000102-00': { topsis_rank: 4, topsis_score: 0.4109, vikor_rank: 4, priority_level: 'Medium', vikor_Q: 0.4813 },
    'SRMD-IND-000155-00': { topsis_rank: 5, topsis_score: 0.2646, vikor_rank: 5, priority_level: 'Low', vikor_Q: 0.8078 },
    'SRMD-IND-000210-00': { topsis_rank: 6, topsis_score: 0.2241, vikor_rank: 6, priority_level: 'Low', vikor_Q: 1.0 },
    'SRMD-IND-000278-00': { topsis_rank: 2, topsis_score: 0.5916, vikor_rank: 2, priority_level: 'High', vikor_Q: 0.2267 },
    'SRMD-IND-000278-01': { topsis_rank: 3, topsis_score: 0.4506, vikor_rank: 3, priority_level: 'Medium', vikor_Q: 0.4429 },
  },
  'equity-first': {
    'SRMD-IND-000045-00': { topsis_rank: 6, topsis_score: 0.3527, vikor_rank: 6, priority_level: 'Low', vikor_Q: 0.9904 },
    'SRMD-IND-000102-00': { topsis_rank: 3, topsis_score: 0.5798, vikor_rank: 3, priority_level: 'Medium', vikor_Q: 0.3633 },
    'SRMD-IND-000155-00': { topsis_rank: 2, topsis_score: 0.6136, vikor_rank: 4, priority_level: 'Critical', vikor_Q: 0.4512 },
    'SRMD-IND-000210-00': { topsis_rank: 4, topsis_score: 0.5564, vikor_rank: 2, priority_level: 'Medium', vikor_Q: 0.2838 },
    'SRMD-IND-000278-00': { topsis_rank: 1, topsis_score: 0.7358, vikor_rank: 1, priority_level: 'Critical', vikor_Q: 0.0 },
    'SRMD-IND-000278-01': { topsis_rank: 5, topsis_score: 0.5042, vikor_rank: 5, priority_level: 'Low', vikor_Q: 0.7396 },
  },
};

export default function DashboardPage() {
  const [telemetry, setTelemetry] = useState<TelemetryResponse>(FALLBACK_DATA);
  const [weightProfile, setWeightProfile] = useState<WeightProfile>('safety-first');
  const [isLive, setIsLive] = useState<boolean>(false);
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);
  const [selectedPotholeId, setSelectedPotholeId] = useState<string | null>(null);
  const [filteredPriority, setFilteredPriority] = useState<PriorityLevel | 'All'>('All');
  const [autoRefresh, setAutoRefresh] = useState<boolean>(true);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);
  const [activeView, setActiveView] = useState<'twin' | 'explorer' | 'sensitivity'>('twin');
  const [simulationScenario, setSimulationScenario] = useState<SimulationScenario>('dry');
  const requestSeq = useRef(0);
  const lastPayloadKeyRef = useRef<string>('');

  const applyPrioritizationToData = useCallback((baseData: TelemetryRecord[], profile: WeightProfile, prioResults?: { id: string; topsis_rank?: number; rank: number; vikor_rank?: number; topsis_score?: number; score?: number; priority_level: PriorityLevel; vikor_Q?: number }[]) => {
    const prioMap = new Map<string, { topsis_rank?: number; vikor_rank?: number; topsis_score?: number; priority_level?: PriorityLevel; vikor_Q?: number }>();
    if (prioResults && prioResults.length > 0) {
      for (const res of prioResults) {
        prioMap.set(res.id, {
          topsis_rank: res.topsis_rank ?? res.rank,
          vikor_rank: res.vikor_rank,
          topsis_score: res.topsis_score ?? res.score,
          priority_level: res.priority_level,
          vikor_Q: res.vikor_Q,
        });
      }
    } else {
      const fallback = FALLBACK_PROFILE_MAP[profile];
      if (fallback) {
        for (const [id, val] of Object.entries(fallback)) {
          prioMap.set(id, val);
        }
      }
    }

    return baseData.map((rec) => {
      const match = prioMap.get(rec.id);
      if (!match) return rec;
      return {
        ...rec,
        topsis_rank: match.topsis_rank ?? rec.topsis_rank,
        vikor_rank: match.vikor_rank ?? rec.vikor_rank,
        topsis_score: match.topsis_score ?? rec.topsis_score,
        priority_level: (match.priority_level ?? rec.priority_level) as PriorityLevel,
        vikor_Q: match.vikor_Q ?? rec.vikor_Q,
      };
    });
  }, []);

  const loadData = useCallback(async (showLoadingSpinner = false, profileToUse = weightProfile) => {
    if (showLoadingSpinner) setIsRefreshing(true);
    const currentSeq = ++requestSeq.current;
    try {
      const [telemetryRes, prioRes] = await Promise.all([
        fetchTelemetry(),
        fetchPrioritization(profileToUse),
      ]);
      if (currentSeq === requestSeq.current) {
        const updatedRecords = applyPrioritizationToData(
          telemetryRes.data.data,
          profileToUse,
          prioRes.data?.results
        );

        // Compute fingerprint to avoid unnecessary re-render cascades
        const payloadKey = JSON.stringify({
          recs: updatedRecords.map((r) => [
            r.id,
            r.timestamp,
            r.topsis_score,
            r.topsis_rank,
            r.vikor_rank,
            r.processed_telemetry?.estimated_depth_cm,
          ]),
          live: telemetryRes.isLive,
        });

        // Only setState when payload changed or explicitly requested
        if (showLoadingSpinner || payloadKey !== lastPayloadKeyRef.current) {
          lastPayloadKeyRef.current = payloadKey;
          setTelemetry({
            ...telemetryRes.data,
            data: updatedRecords,
          });
          setIsLive(telemetryRes.isLive);
        }
        setLastUpdated(new Date().toISOString());
      }
    } catch (err) {
      console.error('Telemetry fetch error:', err);
    } finally {
      if (showLoadingSpinner) setIsRefreshing(false);
    }
  }, [weightProfile, applyPrioritizationToData]);

  const handleSelectWeightProfile = async (profile: WeightProfile) => {
    setWeightProfile(profile);
    // Immediately apply locally/fetch
    try {
      const { data: prioRes } = await fetchPrioritization(profile);
      setTelemetry((prev) => ({
        ...prev,
        data: applyPrioritizationToData(prev.data, profile, prioRes?.results),
      }));
    } catch (err) {
      console.warn('Re-prioritization failed, applying fallback:', err);
      setTelemetry((prev) => ({
        ...prev,
        data: applyPrioritizationToData(prev.data, profile),
      }));
    }
  };

  const handleSelectScenario = (scenario: SimulationScenario) => {
    setSimulationScenario(scenario);
    setTelemetry((prev) => {
      const isRain = scenario === 'rain';
      const isNight = scenario === 'night';
      const isSonar = isRain || isNight;
      const meanLux = isNight ? 12.5 : isRain ? 65.0 : 120.0;
      const activeModality = isSonar ? 'sonar_submerged_acoustic' : 'lidar_optical_tof';

      const updated = prev.data.map((rec) => {
        const rawDepth = isSonar ? rec.sonar_depth_cm : rec.lidar_depth_cm;
        const effectiveDepth = Math.max(0.5, Number((rawDepth - 30.0).toFixed(2)));
        const isPothole = (rec.damage_class || 'D40') === 'D40';
        const newVolume = isPothole
          ? Number(((2 / 3) * rec.surface_area_sqm * (effectiveDepth / 100) * 1000).toFixed(2))
          : null;

        return {
          ...rec,
          rain_detected: isRain,
          mean_luminance: meanLux,
          processed_telemetry: {
            ...rec.processed_telemetry,
            detection_modality: activeModality,
            is_submerged: isRain,
            estimated_depth_cm: effectiveDepth,
            calculated_volume_liters: newVolume,
          },
        };
      });

      const potholeVols = updated
        .map((r) => r.processed_telemetry?.calculated_volume_liters)
        .filter((v): v is number => v !== null && v !== undefined);
      const totalVol = Number(potholeVols.reduce((a, b) => a + b, 0).toFixed(2));
      const avgVol = potholeVols.length > 0 ? Number((totalVol / potholeVols.length).toFixed(2)) : 0;
      const submergedCount = updated.filter((r) => r.processed_telemetry?.is_submerged).length;

      return {
        ...prev,
        data: updated,
        summary: {
          ...prev.summary,
          total_volume_liters: totalVol,
          average_volume_liters: avgVol,
          weather_state: isRain ? 'Heavy Monsoon Rain' : isNight ? 'Clear Night (Dark)' : 'Clear Daylight',
          is_raining: isRain,
          submerged_defects: submergedCount,
          average_luminance: meanLux,
        },
      };
    });
  };

  useEffect(() => { loadData(true); }, [loadData]);

  useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(() => loadData(false), 3000);
    return () => clearInterval(interval);
  }, [autoRefresh, loadData]);

  const selectedRecord: TelemetryRecord | null = useMemo(
    () => telemetry.data.find((r) => r.id === selectedPotholeId) ?? null,
    [telemetry.data, selectedPotholeId]
  );

  const handleSelectPothole = (id: string) => {
    // Toggle deselect on double-click
    setSelectedPotholeId((prev) => (prev === id ? null : id));
  };

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        minHeight: '100vh',
        background: '#0B0E14',
        color: '#EEF2FF',
      }}
    >
      {/* ── Header ─────────────────────────────────────────── */}
      <Header
        isLive={isLive}
        isRefreshing={isRefreshing}
        lastUpdated={lastUpdated}
        onRefresh={() => loadData(true)}
        autoRefresh={autoRefresh}
        onToggleAutoRefresh={() => setAutoRefresh((p) => !p)}
        totalDefects={telemetry.data.length}
        activeView={activeView}
        onViewChange={(v) => setActiveView(v)}
      />

      {/* ── Main Content ────────────────────────────────────── */}
      <main
        style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          padding: '16px 20px 20px',
          gap: 12,
          maxWidth: 1600,
          width: '100%',
          margin: '0 auto',
          boxSizing: 'border-box',
        }}
      >
        {activeView === 'sensitivity' ? (
          <SensitivityPanel />
        ) : activeView === 'explorer' ? (
          <DatasetExplorerPanel
            records={telemetry.data}
            detector={telemetry.detector}
            isLive={isLive}
            isRefreshing={isRefreshing}
            onRefresh={() => loadData(true)}
            onSelectRecord={(id) => {
              setSelectedPotholeId(id);
              setActiveView('twin');
            }}
            onSwitchToMap={() => setActiveView('twin')}
          />
        ) : (
          <>
            {/* ── KPI Strip + Filters ──────────────────────────── */}
            <SummaryPanel
              summary={telemetry.summary}
              selectedPriority={filteredPriority}
              onSelectPriority={(priority) => setFilteredPriority(priority)}
            />

            {/* ── Scenario Simulation Control Bar (Dry / Rain / Night) ── */}
            <ScenarioSimulationBar
              scenario={simulationScenario}
              onSelectScenario={handleSelectScenario}
              activeSensor={simulationScenario === 'dry' ? 'lidar' : 'sonar'}
              recordCount={telemetry.data.length}
            />

            {/* ── Map Label Row ────────────────────────────────── */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                marginBottom: -4,
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span
                  style={{
                    width: 3,
                    height: 14,
                    background: '#3B82F6',
                    borderRadius: 2,
                    flexShrink: 0,
                    display: 'inline-block',
                  }}
                />
                <span
                  style={{
                    fontSize: '11px',
                    fontWeight: 700,
                    letterSpacing: '0.08em',
                    textTransform: 'uppercase',
                    color: '#8892A4',
                  }}
                >
                  Spatial Digital Twin — Mumbai Western Corridor
                </span>
              </div>
              <span
                style={{
                  fontSize: '10px',
                  color: '#4D5A72',
                  fontFamily: 'ui-monospace, monospace',
                }}
              >
                30 cm sensor baseline · CARTO dark_all basemap
              </span>
            </div>

            {/* ── Split Layout: Map (left) + Right Panel ────────── */}
            <div
              className="split-layout"
              style={{
                display: 'flex',
                gap: 12,
                flex: 1,
                minHeight: 'calc(100vh - 240px)',
              }}
            >
              {/* LEFT: Map panel — 65% */}
              <div
                className="map-panel"
                style={{
                  flex: '0 0 65%',
                  minWidth: 0,
                  position: 'relative',
                  borderRadius: 8,
                  overflow: 'hidden',
                  border: '1px solid #1E2435',
                }}
              >
                <MapWrapper
                  potholes={telemetry.data}
                  selectedPotholeId={selectedPotholeId}
                  onSelectPothole={handleSelectPothole}
                  filteredPriority={filteredPriority}
                />
              </div>

              {/* RIGHT: Incident list + detail — 35% */}
              <div
                style={{
                  flex: '0 0 35%',
                  minWidth: 0,
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 8,
                  minHeight: 0,
                }}
              >
                {/* Right panel: incident list (top) */}
                <div
                  style={{
                    flex: selectedRecord ? '0 0 42%' : '0 0 46%',
                    minHeight: 0,
                    display: 'flex',
                    flexDirection: 'column',
                  }}
                >
                  <TelemetryTable
                    records={telemetry.data}
                    selectedId={selectedPotholeId}
                    onSelect={handleSelectPothole}
                    filteredPriority={filteredPriority}
                    weightProfile={weightProfile}
                    onSelectWeightProfile={handleSelectWeightProfile}
                  />
                </div>

                {/* Right panel: bottom panel (Detail when selected, or Top-3 Queue + Priority Distribution when nothing selected) */}
                <div
                  style={{
                    flex: 1,
                    minHeight: 0,
                    display: 'flex',
                    flexDirection: 'column',
                    background: '#111520',
                    border: '1px solid #1E2435',
                    borderRadius: 8,
                    overflow: 'hidden',
                  }}
                >
                  {selectedRecord ? (
                    <>
                      {/* Detail panel header bar */}
                      <div
                        style={{
                          padding: '7px 14px',
                          borderBottom: '1px solid #1E2435',
                          background: '#0F1420',
                          flexShrink: 0,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                        }}
                      >
                        <span
                          style={{
                            fontSize: '10px',
                            fontWeight: 700,
                            letterSpacing: '0.08em',
                            textTransform: 'uppercase',
                            color: '#4D5A72',
                          }}
                        >
                          Incident Detail
                        </span>
                      </div>
                      <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
                        <IncidentDetailPanel
                          record={selectedRecord}
                          onClose={() => setSelectedPotholeId(null)}
                        />
                      </div>
                    </>
                  ) : (
                    <QueueOverviewPanel
                      records={telemetry.data}
                      onSelectRecord={handleSelectPothole}
                    />
                  )}
                </div>
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
