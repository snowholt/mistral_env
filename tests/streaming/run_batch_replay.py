#!/usr/bin/env python3
"""Batch convert WebM test inputs to PCM and replay them against streaming endpoint.

Requirements: ffmpeg in PATH, websockets package installed in backend venv.

Outputs summary JSON lines and a final aggregate record.
"""
from __future__ import annotations
import asyncio, json, subprocess, sys, shutil
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
WEBM_DIR = ROOT / 'voice_tests' / 'input_test_questions' / 'webm'
PCM_DIR = ROOT / 'voice_tests' / 'input_test_questions' / 'pcm'
SCRIPT = Path(__file__).parent / 'ws_replay_pcm.py'
REFS = Path(__file__).parent / 'reference_transcripts.json'

def ensure_pcm():
    PCM_DIR.mkdir(parents=True, exist_ok=True)
    for webm in sorted(WEBM_DIR.glob('*.webm')):
        pcm = PCM_DIR / (webm.stem + '.pcm')
        if pcm.exists():
            continue
        cmd = [
            'ffmpeg','-v','error','-y','-i',str(webm),'-ar','16000','-ac','1','-f','s16le',str(pcm)
        ]
        subprocess.run(cmd, check=True)

async def run_replay(pcm_file: Path, language: str, reference: str | None):
    # Call ws_replay_pcm.py as a module (subprocess to isolate)
    cmd = [sys.executable, str(SCRIPT), '--file', str(pcm_file), '--language', language]
    if reference:
        cmd += ['--reference', reference]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await proc.communicate()
    lines = stdout.decode('utf-8', 'ignore').splitlines()
    summary = None
    for line in lines:
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get('event') == 'summary':
            summary = obj
    if not summary:
        summary = { 'file': pcm_file.name, 'error': 'no_summary' }
    if stderr:
        summary['stderr'] = stderr.decode('utf-8','ignore')
    return summary

async def main():
    if not shutil.which('ffmpeg'):
        print(json.dumps({'event':'error','message':'ffmpeg not found'}))
        return
    ensure_pcm()
    refs = json.loads(REFS.read_text(encoding='utf-8')) if REFS.exists() else {}
    summaries = []
    for pcm in sorted(PCM_DIR.glob('*.pcm')):
        # Heuristic language: Arabic if it starts with 'q' or ends with _ar
        if pcm.stem.endswith('_ar') or pcm.stem.startswith('q'):
            lang = 'ar'
        else:
            lang = 'en'
        reference = refs.get(pcm.name)
        print(json.dumps({'event':'run','file':pcm.name,'language':lang}))
        summary = await run_replay(pcm, lang, reference)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False))

    # Aggregate
    latencies = [s.get('first_partial_ms') for s in summaries if s.get('first_partial_ms') is not None]
    finals = [s.get('final_ms') for s in summaries if s.get('final_ms') is not None]
    wer_values = [s.get('wer') for s in summaries if s.get('wer') is not None]
    aggregate = {
        'event': 'aggregate',
        'files': len(summaries),
        'avg_first_partial_ms': int(mean(latencies)) if latencies else None,
        'avg_final_ms': int(mean(finals)) if finals else None,
        'avg_wer': round(mean(wer_values),3) if wer_values else None,
    }
    print(json.dumps(aggregate, ensure_ascii=False))

if __name__ == '__main__':
    asyncio.run(main())
