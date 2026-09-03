'use client';

import React from 'react';
import dynamic from 'next/dynamic';
import { TelemetryRecord, PriorityLevel } from '@/types/telemetry';

// Dynamically import Leaflet map with SSR disabled
const PotholeMap = dynamic(
  () => import('./PotholeMap').then((mod) => mod.PotholeMap),
  {
    ssr: false,
    loading: () => (
      <div className="w-full h-[540px] rounded-2xl bg-slate-900 border border-slate-800 flex flex-col items-center justify-center text-slate-400 gap-3">
        <div className="w-10 h-10 border-4 border-cyan-500/20 border-t-cyan-500 rounded-full animate-spin" />
        <p className="text-sm font-medium tracking-wide">Initializing Leaflet Digital Twin Map...</p>
      </div>
    ),
  }
);

interface MapWrapperProps {
  potholes: TelemetryRecord[];
  selectedPotholeId: string | null;
  onSelectPothole: (id: string) => void;
  filteredPriority: PriorityLevel | 'All';
}

export const MapWrapper: React.FC<MapWrapperProps> = (props) => {
  return <PotholeMap {...props} />;
};
