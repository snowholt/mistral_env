// Export utilities for structured report generation.
import { analyzeEvents, buildHeuristicSummary } from './debug_heuristics.js';

export function buildStructuredReport(meta, events){
  const analysis = analyzeEvents(events);
  const heur = buildHeuristicSummary(analysis);
  const metrics = computeMetrics(events);
  return {
    schema_version: 'stream_debug_v1',
    generated_utc: new Date().toISOString(),
    session: meta,
    metrics,
    heuristics: heur,
    events: events,
  };
}

function computeMetrics(events){
  const perf = events.filter(e=> e.type==='perf_cycle');
  const decodeMs = perf.map(p=> p.decode_ms).filter(n=> typeof n==='number');
  decodeMs.sort((a,b)=>a-b);
  function pct(p){ if(!decodeMs.length) return null; const idx = Math.min(decodeMs.length-1, Math.floor(p*decodeMs.length)); return decodeMs[idx]; }
  const firstPartial = events.find(e=> e.type==='partial');
  const firstFinal = events.find(e=> e.type==='final'|| e.type==='final_transcript');
  const firstAssistant = events.find(e=> e.type==='assistant_response');
  return {
    decode_ms: {
      count: decodeMs.length,
      p50: pct(0.5), p90: pct(0.9), max: decodeMs[decodeMs.length-1]||null
    },
    latency_ms: {
      first_partial: firstPartial? firstPartial.t_rel_ms: null,
      first_final: firstFinal? firstFinal.t_rel_ms: null,
      assistant_first: firstAssistant? firstAssistant.t_rel_ms: null
    },
    endpoint: {
      starts: events.filter(e=> e.type==='endpoint_event' && e.event==='start').length,
      finals: events.filter(e=> e.type==='endpoint_event' && e.event==='final').length
    }
  };
}

export function exportJson(struct, filename='offline_stream_report.json'){
  const blob = new Blob([JSON.stringify(struct,null,2)], {type:'application/json'});
  triggerDownload(blob, filename);
}

export function exportMarkdown(struct, filename='offline_stream_report.md'){
  const lines = [];
  lines.push(`# Streaming Voice Debug Report`);
  lines.push(`Generated: ${struct.generated_utc}`);
  if(struct.session){
    lines.push(`## Session`);
    lines.push('```');
    lines.push(JSON.stringify(struct.session,null,2));
    lines.push('```');
  }
  if(struct.metrics){
    lines.push(`## Metrics`);
    lines.push('```');
    lines.push(JSON.stringify(struct.metrics,null,2));
    lines.push('```');
  }
  lines.push(`## Heuristics`);
  lines.push('```');
  lines.push(JSON.stringify(struct.heuristics,null,2));
  lines.push('```');
  lines.push(`## Events (truncated if large)`);
  const ev = struct.events.slice(0, 1200); // cap
  lines.push('```');
  lines.push(JSON.stringify(ev,null,2));
  lines.push('```');
  const blob = new Blob([lines.join('\n')], {type:'text/markdown'});
  triggerDownload(blob, filename);
}

function triggerDownload(blob, filename){
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; document.body.appendChild(a); a.click(); a.remove();
  setTimeout(()=>URL.revokeObjectURL(url), 500);
}
