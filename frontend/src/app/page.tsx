'use client';

import React, { useEffect, useState, useCallback, useMemo } from 'react';
import { Header } from '@/components/Dashboard/Header';
import { SummaryPanel } from '@/components/Dashboard/SummaryPanel';
import { MapWrapper } from '@/components/Map/MapWrapper';
import { TelemetryTable } from '@/components/Dashboard/TelemetryTable';
import { IncidentDetailPanel } from '@/components/Dashboard/IncidentDetailPanel';
import { fetchTelemetry, FALLBACK_DATA } from '@/lib/api';
import { TelemetryResponse, PriorityLevel, TelemetryRecord } from '@/types/telemetry';

export default function DashboardPage() {
  const [telemetry, setTelemetry] = useState<TelemetryResponse>(FALLBACK_DATA);
  const [isLive, setIsLive] = useState<boolean>(false);
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);
  const [selectedPotholeId, setSelectedPotholeId] = useState<string | null>(null);
  const [filteredPriority, setFilteredPriority] = useState<PriorityLevel | 'All'>('All');
  const [autoRefresh, setAutoRefresh] = useState<boolean>(true);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);
  const requestSeq = React.useRef(0);

  const loadData = useCallback(async (showLoadingSpinner = false) => {
    if (showLoadingSpinner) setIsRefreshing(true);
    const currentSeq = ++requestSeq.current;
    try {
      const { data, isLive: liveStatus } = await fetchTelemetry();
      if (currentSeq === requestSeq.current) {
        setTelemetry(data);
        setIsLive(liveStatus);
        setLastUpdated(new Date().toISOString());
      }
    } catch (err) {
      console.error('Telemetry fetch error:', err);
    } finally {
      if (showLoadingSpinner) setIsRefreshing(false);
    }
  }, []);

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
        {/* ── KPI Strip + Filters ──────────────────────────── */}
        <SummaryPanel
          summary={telemetry.summary}
          selectedPriority={filteredPriority}
          onSelectPriority={(priority) => setFilteredPriority(priority)}
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
            30 cm sensor baseline · OSM tiles
          </span>
        </div>

        {/* ── Split Layout: Map (left) + Right Panel ────────── */}
        <div
          className="split-layout"
          style={{
            display: 'flex',
            gap: 12,
            flex: 1,
            /* Tall enough to be useful on screen */
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
                flex: selectedRecord ? '0 0 42%' : '1',
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
              />
            </div>

            {/* Right panel: incident detail (bottom, shown when selected) */}
            <div
              style={{
                flex: selectedRecord ? '1' : '0 0 0px',
                minHeight: 0,
                display: 'flex',
                flexDirection: 'column',
                background: '#111520',
                border: '1px solid #1E2435',
                borderRadius: 8,
                overflow: 'hidden',
                transition: 'flex 0.2s ease',
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
              ) : null}
            </div>

            {/* Empty detail placeholder (when nothing selected) */}
            {!selectedRecord && (
              <div
                style={{
                  flex: '0 0 auto',
                  background: '#111520',
                  border: '1px solid #1E2435',
                  borderRadius: 8,
                  padding: '20px 16px',
                  textAlign: 'center',
                }}
              >
                <p style={{ fontSize: '11px', color: '#374163', lineHeight: 1.5 }}>
                  Select an incident from the list above<br />or click a map marker to view details
                </p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
