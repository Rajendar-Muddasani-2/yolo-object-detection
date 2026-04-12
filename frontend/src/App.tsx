import React, { useCallback, useEffect, useRef, useState } from 'react';
import axios from 'axios';

const API_BASE = import.meta.env.PROD ? '' : 'http://localhost:8080';

interface Detection {
  class_name: string;
  class_id: number;
  confidence: number;
  bbox: number[];
}

interface DetectionResponse {
  image_name: string;
  detections: Detection[];
  inference_time_ms: number;
  total_defects: number;
}

const COLORS = [
  '#ef4444', '#f97316', '#eab308', '#22c55e', '#06b6d4',
  '#3b82f6', '#8b5cf6', '#ec4899', '#f43f5e', '#14b8a6',
];

function App() {
  const [status, setStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [backend, setBackend] = useState('');
  const [result, setResult] = useState<DetectionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    axios.get(`${API_BASE}/health`)
      .then(r => { setStatus('online'); setBackend(r.data.backend); })
      .catch(() => setStatus('offline'));
  }, []);

  const drawDetections = useCallback((img: HTMLImageElement, detections: Detection[]) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.drawImage(img, 0, 0);

    detections.forEach(det => {
      const [x1, y1, x2, y2] = det.bbox;
      const color = COLORS[det.class_id % COLORS.length];

      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      const label = `${det.class_name} ${(det.confidence * 100).toFixed(1)}%`;
      ctx.font = 'bold 14px Inter, sans-serif';
      const textWidth = ctx.measureText(label).width;

      ctx.fillStyle = color;
      ctx.fillRect(x1, y1 - 22, textWidth + 12, 22);

      ctx.fillStyle = '#fff';
      ctx.fillText(label, x1 + 6, y1 - 6);
    });
  }, []);

  const handleFile = useCallback(async (file: File) => {
    setLoading(true);
    setResult(null);

    // Show preview
    const img = new Image();
    const url = URL.createObjectURL(file);
    img.src = url;
    img.onload = () => {
      imageRef.current = img;
      if (canvasRef.current) {
        const canvas = canvasRef.current;
        canvas.width = img.width;
        canvas.height = img.height;
        canvas.getContext('2d')?.drawImage(img, 0, 0);
      }
    };

    // Send to API
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post<DetectionResponse>(
        `${API_BASE}/detect`, formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      );
      setResult(response.data);

      // Wait for image to load then draw
      img.onload = () => {
        imageRef.current = img;
        drawDetections(img, response.data.detections);
      };
      if (img.complete && imageRef.current) {
        drawDetections(img, response.data.detections);
      }
    } catch (err) {
      console.error('Detection failed:', err);
    } finally {
      setLoading(false);
    }
  }, [drawDetections]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) handleFile(file);
  }, [handleFile]);

  const classCounts: Record<string, number> = {};
  result?.detections.forEach(d => {
    classCounts[d.class_name] = (classCounts[d.class_name] || 0) + 1;
  });

  return (
    <div className="app">
      <header className="header">
        <h1>Wafer Defect Inspector</h1>
        <div className="status">
          <span className={`status-dot ${status === 'online' ? '' : 'offline'}`} />
          {status === 'online' ? `Connected (${backend})` : status === 'checking' ? 'Connecting...' : 'Offline'}
        </div>
      </header>

      <div
        className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
        onClick={() => fileInputRef.current?.click()}
        onDragOver={e => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
      >
        <h3>Drop wafer image here or click to upload</h3>
        <p>Supports JPG, PNG. YOLOv8-Large model with 10 defect classes.</p>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          hidden
          onChange={e => {
            const file = e.target.files?.[0];
            if (file) handleFile(file);
          }}
        />
      </div>

      {loading && <div className="spinner" />}

      {result && (
        <div className="results-grid">
          <div className="detection-panel">
            <h3>Detection Result</h3>
            <canvas ref={canvasRef} className="detection-canvas" />
          </div>

          <div className="stats-panel">
            <h3>Inspection Summary</h3>

            <div className="stat-card">
              <div className="label">Total Defects</div>
              <div className="value" style={{ color: result.total_defects > 0 ? '#ef4444' : '#22c55e' }}>
                {result.total_defects}
              </div>
            </div>

            <div className="stat-card">
              <div className="label">Inference Time</div>
              <div className="value">{result.inference_time_ms.toFixed(1)}ms</div>
            </div>

            <div className="stat-card">
              <div className="label">Defect Breakdown</div>
              {Object.entries(classCounts).map(([cls, count]) => (
                <div key={cls} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0' }}>
                  <span>{cls}</span>
                  <span style={{ fontWeight: 700 }}>{count}</span>
                </div>
              ))}
            </div>

            <h3 style={{ marginTop: 16, marginBottom: 8 }}>Detections</h3>
            <ul className="detection-list">
              {result.detections.map((det, i) => (
                <li key={i} className="detection-item">
                  <span className="class-name" style={{ color: COLORS[det.class_id % COLORS.length] }}>
                    {det.class_name}
                  </span>
                  <span className="confidence">{(det.confidence * 100).toFixed(1)}%</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
