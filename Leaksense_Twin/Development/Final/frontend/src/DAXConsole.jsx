import React, { useState } from 'react';
import Editor from '@monaco-editor/react';

const API_BASE = 'http://localhost:8000';

const QUICK_QUERIES = [
  { label: 'Leak Rate Today', query: 'DIVIDE(COUNTX(FILTER(LeakEvents,[go_no_go]=\'NO-GO\'),1),COUNTX(SensorReadings,1),0)*100' },
  { label: 'Avg MAF', query: 'AVERAGEX(SensorReadings,[maf])' },
  { label: 'Zone 2 Leak Count', query: 'COUNTX(FILTER(LeakEvents,[zone] LIKE \'Zone 2%\'),1)' },
  { label: 'Top 10 Boost Drops', query: 'TOPN(10,SensorReadings,([map_boost]-218)/218,ASC)' },
  { label: 'Peak EF Deviation', query: 'MAXX(TwinResiduals,[ef_deviation])' },
  { label: 'Dataset: Avg RPM', query: 'AVERAGEX(EngineIQ_Twin_Dataset, [RPM])' },
  { label: 'Dataset: Leak Zones', query: 'SELECT Leak_Zone, COUNT(*) FROM EngineIQ_Twin_Dataset GROUP BY Leak_Zone' },
];

export default function DAXConsole() {
  const [query, setQuery] = useState('AVERAGEX(SensorReadings, [maf])');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const runQuery = async (q) => {
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const res = await fetch(`${API_BASE}/api/dax/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ expression: q || query }),
      });
      const data = await res.json();
      if (data.result?.success) {
        setResult(data.result);
      } else {
        setError(data.result?.error || 'Query failed');
      }
    } catch (err) {
      setError(err.message);
    }
    setLoading(false);
  };

  return (
    <div className="card" style={{ gridColumn: '1 / -1', minHeight: '600px' }}>
      <div className="card-header">
        <span className="card-title">DAX Query Console</span>
        <span className="card-icon"></span>
      </div>
      <div style={{ padding: '20px' }}>
        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', marginBottom: '15px' }}>
          {QUICK_QUERIES.map(q => (
            <button 
              key={q.label} 
              className="demo-btn" 
              style={{ padding: '5px 10px', fontSize: '12px' }}
              onClick={() => { setQuery(q.query); runQuery(q.query); }}
            >
              {q.label}
            </button>
          ))}
        </div>
        
        <div style={{ border: '1px solid var(--border-color)', borderRadius: '8px', overflow: 'hidden', height: '200px' }}>
          <Editor
            height="100%"
            defaultLanguage="sql"
            theme="vs-dark"
            value={query}
            onChange={(val) => setQuery(val)}
            options={{ minimap: { enabled: false }, fontSize: 14 }}
          />
        </div>
        
        <div style={{ marginTop: '15px', display: 'flex', gap: '10px' }}>
          <button className="demo-btn active" onClick={() => runQuery()} disabled={loading}>
            {loading ? 'Running...' : 'RUN Ctrl+Enter'}
          </button>
          <button className="demo-btn" onClick={() => setQuery('')}>CLEAR</button>
        </div>

        {error && <div style={{ color: 'var(--accent-red)', marginTop: '20px' }}>Error: {error}</div>}

        {result && (
          <div style={{ marginTop: '20px' }}>
            {result.is_scalar ? (
              <div style={{ fontSize: '48px', color: 'var(--accent-yellow)', fontFamily: 'Orbitron, sans-serif' }}>
                {result.scalar_value !== null ? result.scalar_value.toFixed(2) : 'No result'}
              </div>
            ) : (
              <div style={{ overflowX: 'auto' }}>
                <table className="history-table">
                  <thead>
                    <tr>
                      {result.columns.map(col => <th key={col}>{col}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {result.rows.slice(0, 50).map((row, i) => (
                      <tr key={i}>
                        {result.columns.map(col => (
                          <td key={col}>{typeof row[col] === 'number' ? row[col].toFixed(2) : row[col]}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
                {result.row_count > 50 && <div style={{ marginTop: '10px', color: 'var(--text-muted)' }}>Showing 50 of {result.row_count} rows.</div>}
              </div>
            )}
            <div style={{ marginTop: '10px', fontSize: '12px', color: 'var(--text-muted)' }}>
              {result.row_count} rows · {result.execution_ms}ms
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
