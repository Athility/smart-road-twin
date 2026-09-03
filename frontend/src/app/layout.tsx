import type { Metadata } from 'next';
import 'leaflet/dist/leaflet.css';
import './globals.css';

export const metadata: Metadata = {
  title: 'Smart Road Digital Twin — Municipal Road Intelligence Platform',
  description: 'Real-time IoT pothole detection, AI depth estimation, and TOPSIS-based repair prioritization for municipal road infrastructure.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-[#0B0E14] text-[#EEF2FF] min-h-screen antialiased selection:bg-blue-600 selection:text-white">
        {children}
      </body>
    </html>
  );
}
