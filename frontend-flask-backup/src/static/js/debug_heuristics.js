// Debug Heuristics Module (Phase A/C)
// Provides client-side analysis of event log to detect phantom reactivation, duplication, repetition loops.

export function analyzeEvents(events) {
  const state = {
    finals: [],
    starts: [],
    frames: 0,
    lastFrameCountAtFinal: null,
    phantomStarts: 0,
    duplicateFinal: false,
    loopScore: 0,
    topPhrase: null,
  };
  const finalTexts = [];
  let frameCount = 0;
  for (const ev of events) {
    switch(ev.type) {
      case 'frame_sent': frameCount++; break;
      case 'final':
      case 'final_transcript':
        state.finals.push(ev);
        if (ev.text) finalTexts.push(normalize(ev.text));
        state.lastFrameCountAtFinal = frameCount;
        break;
      case 'endpoint_event':
        if (ev.event === 'start') {
          if (state.lastFrameCountAtFinal != null && frameCount === state.lastFrameCountAtFinal) {
            state.phantomStarts++;
          }
          state.starts.push(ev);
        }
        break;
    }
  }
  // Duplicate final detection
  const unique = new Set(finalTexts);
  state.duplicateFinal = unique.size < finalTexts.length;
  // Repetition loop scoring
  const lastFinal = finalTexts[finalTexts.length - 1] || '';
  const rep = repetitionScore(lastFinal);
  state.loopScore = rep.score;
  state.topPhrase = rep.topPhrase;
  return state;
}

function normalize(t){
  return t.trim().replace(/\s+/g,' ');
}

function repetitionScore(text){
  const tokens = text.split(/\s+/).filter(Boolean);
  if(tokens.length < 8) return {score:0, topPhrase:null};
  let best = {phrase:null, span:0, repeats:0};
  for(let n=3;n<=8;n++){
    for(let i=0;i + n <= tokens.length; i++){
      const seg = tokens.slice(i,i+n).join(' ');
      let repeats = 1; let j=i+n;
      while(j + n <= tokens.length && tokens.slice(j,j+n).join(' ') === seg){ repeats++; j+=n; }
      if(repeats >=2){ const span = repeats * n; if(span > best.span){ best={phrase:seg, span, repeats}; } }
    }
  }
  if(!best.phrase) return {score:0, topPhrase:null};
  const score = Math.min(100, Math.round((best.span / tokens.length) * 120));
  return {score, topPhrase:best.phrase};
}

export function buildHeuristicSummary(h){
  return {
    phantom_reactivation: h.phantomStarts > 0,
    phantom_start_count: h.phantomStarts,
    duplicate_final_text: h.duplicateFinal,
    repetition_loop_score: h.loopScore,
    top_repeated_phrase: h.topPhrase
  };
}
