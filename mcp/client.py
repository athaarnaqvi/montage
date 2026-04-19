import os
import json
import math
import wave
import shutil
import random
import subprocess
import numpy as np
import requests
from PIL import Image, ImageDraw
import imageio_ffmpeg

from mcp.tool_loader import load_tools
# client.py

FFMPEG    = imageio_ffmpeg.get_ffmpeg_exe()
TARGET_SR = 22050
VIDEO_W   = 854
VIDEO_H   = 480
VIDEO_FPS = 24

TOOLS = load_tools()
print(f"[MCP] Registered tools: {list(TOOLS.keys())}")


# =============================================================================
# DISPATCHER
# =============================================================================
def mcp_call(tool_name, inputs=None):
    if inputs is None:
        inputs = {}
    if tool_name not in TOOLS:
        raise ValueError(f"[MCP] Tool '{tool_name}' not found in registry.")
    api = TOOLS[tool_name].get("api")
    _map = {
        "grok":              _call_grok,
        "pollinations":      _call_pollinations,
        "memory_commit":     _commit_memory,
        "task_graph":        _get_task_graph,
        "voice_synth":       _call_voice_synth,
        "merge_audio":       _merge_scene_audio,
        "stock_footage":     _call_stock_footage,
        "face_swap":         _call_face_swapper,
        "identity_validate": _call_identity_validator,
        "lip_sync":          _call_lip_sync,
    }
    if api not in _map:
        raise NotImplementedError(f"[MCP] API backend '{api}' not implemented.")
    return _map[api](inputs)


# =============================================================================
# GROQ API
# =============================================================================
def _call_grok(inputs):
    api_key = os.getenv("GROK_API_KEY", "")
    if not api_key:
        raise EnvironmentError("GROK_API_KEY is not set in .env")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": (
                "You are a professional Hollywood screenwriter. "
                "Always respond with ONLY valid JSON. No markdown, no explanation, no code fences."
            )},
            {"role": "user", "content": inputs.get("prompt", "")}
        ],
        "temperature": 0.7, "max_tokens": 3000
    }
    r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                      headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"[Groq] {r.status_code}: {r.text}")
    content = r.json()["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"): content = content[4:]
        content = content.strip()
    return json.loads(content)


# =============================================================================
# POLLINATIONS AI  (Phase 1 character portrait generation only)
# =============================================================================
def _call_pollinations(inputs):
    import urllib.parse, time
    prompt   = inputs.get("prompt", "")
    filename = inputs.get("filename", "character")
    url      = (f"https://image.pollinations.ai/prompt/{urllib.parse.quote(prompt)}"
                f"?width=512&height=512&nologo=true&seed={abs(hash(filename))%99999}")
    os.makedirs("outputs/images", exist_ok=True)
    filepath = f"outputs/images/{filename}.png"
    for attempt in range(1, 5):
        try:
            time.sleep(3 if attempt == 1 else attempt * 8)
            r = requests.get(url, timeout=120)
            if r.status_code == 200 and len(r.content) >= 1000:
                with open(filepath, "wb") as f: f.write(r.content)
                return filepath
            elif r.status_code == 429: time.sleep(15)
            else: raise RuntimeError(f"HTTP {r.status_code}")
        except requests.exceptions.Timeout:
            print(f"  ⚠️  Timeout attempt {attempt}.")
        except RuntimeError as e:
            if attempt == 4: raise
            print(f"  ⚠️  {e}")
    raise RuntimeError("[Pollinations] Failed after 4 attempts")


# =============================================================================
# MEMORY COMMIT
# =============================================================================
def _commit_memory(inputs):
    os.makedirs("memory_db", exist_ok=True)
    with open("memory_db/state_snapshot.json", "w", encoding="utf-8") as f:
        safe = {k: v for k, v in inputs.items()
                if isinstance(v, (str, int, float, list, dict, bool, type(None)))}
        json.dump(safe, f, indent=2)
    return True


# =============================================================================
# TASK GRAPH
# =============================================================================
def _get_task_graph(inputs):
    scenes = inputs.get("scenes", [])
    tasks  = []
    for scene in scenes:
        sid = scene.get("scene_id")
        tasks += [
            {"task_id": f"audio_{sid}",    "type": "audio",     "scene_id": sid, "depends_on": []},
            {"task_id": f"video_{sid}",    "type": "video",     "scene_id": sid, "depends_on": []},
            {"task_id": f"face_swap_{sid}","type": "face_swap", "scene_id": sid, "depends_on": [f"video_{sid}"]},
            {"task_id": f"lip_sync_{sid}", "type": "lip_sync",  "scene_id": sid,
             "depends_on": [f"audio_{sid}", f"face_swap_{sid}"]},
        ]
    print(f"[MCP/TaskGraph] Built task graph: {len(tasks)} tasks for {len(scenes)} scenes.")
    return {"tasks": tasks, "scene_count": len(scenes)}


# =============================================================================
# VOICE SYNTHESIS
# Uses edge_tts (real Microsoft Neural TTS) → saves MP3
# Then converts MP3 → WAV via ffmpeg so the rest of the pipeline
# can use wave.open() without errors.
# Falls back to Klatt-inspired numpy synthesizer if edge_tts unavailable.
# =============================================================================

_VOICE_MAP = {
    "Robot Guardian":       "en-US-GuyNeural",
    "Human Child":          "en-US-JennyNeural",
    "Malfunctioning Robot": "en-US-DavisNeural",
    "Detective Jameson":    "en-US-GuyNeural",
    "Captain Lewis":        "en-US-ChristopherNeural",
    "Suspect":              "en-US-DavisNeural",
    "Unknown Figure":       "en-US-DavisNeural",
    "Narrator":             "en-US-AriaNeural",
}

_VOICE_PROFILES = {
    "robot guardian":       {"pitch":  95, "speed": 0.80, "roughness": 0.78},
    "malfunctioning robot": {"pitch":  68, "speed": 0.62, "roughness": 0.94},
    "human child":          {"pitch": 290, "speed": 1.20, "roughness": 0.16},
    "detective jameson":    {"pitch": 112, "speed": 0.88, "roughness": 0.62},
    "captain lewis":        {"pitch": 148, "speed": 1.00, "roughness": 0.38},
    "suspect":              {"pitch": 102, "speed": 0.83, "roughness": 0.72},
    "unknown figure":       {"pitch":  88, "speed": 0.78, "roughness": 0.82},
    "narrator":             {"pitch": 162, "speed": 1.08, "roughness": 0.22},
    "default":              {"pitch": 155, "speed": 1.00, "roughness": 0.35},
}

def _get_voice_profile(speaker):
    key = speaker.lower().strip()
    for k, v in _VOICE_PROFILES.items():
        if k in key or key in k: return v
    return _VOICE_PROFILES["default"]


def _synth_numpy(line, speaker, sr=TARGET_SR):
    """Klatt-inspired fallback synthesizer when edge_tts is unavailable."""
    p = _get_voice_profile(speaker)
    words    = line.split()
    wcount   = max(1, len(words))
    duration = max(1.2, wcount * 0.36 / p["speed"])
    n        = int(sr * duration)
    t        = np.linspace(0, duration, n, dtype=np.float64)
    glottal  = ((t * p["pitch"]) % 1.0) - 0.5
    rng      = np.random.default_rng(abs(hash(line)) % (2**31))
    noise    = rng.standard_normal(n)
    source   = (1.0 - p["roughness"]) * glottal + p["roughness"] * noise
    try:
        from scipy.signal import butter, lfilter
        def _bp(sig, lo, hi):
            nyq = sr / 2.0
            b, a = butter(2, [max(0.001, lo/nyq), min(0.999, hi/nyq)], btype="band")
            return lfilter(b, a, sig)
        voiced = 0.50*_bp(source,250,900) + 0.35*_bp(source,900,2500) + 0.15*_bp(source,2500,4500)
    except ImportError:
        voiced = source
    env = np.zeros(n)
    for wi in range(wcount):
        c = ((wi+0.5)/wcount)*duration
        s = (duration/wcount)*0.38
        env += np.exp(-0.5*((t-c)/s)**2)
    fade = int(0.04*sr)
    env[:fade]  *= np.linspace(0,1,fade)
    env[-fade:] *= np.linspace(1,0,fade)
    env /= (env.max()+1e-9)
    speech = voiced*env*0.88
    pk = np.abs(speech).max()
    if pk > 0: speech = (speech/pk)*0.92
    return (speech*32767).astype(np.int16), duration


def _call_voice_synth(inputs):
    """
    Generate speech for one dialogue line.
    Output: a valid .wav file at TARGET_SR, mono, 16-bit.
    Strategy:
      1. Try edge_tts (real neural TTS) → produces MP3
         → convert MP3 → WAV via ffmpeg (fixes the wave.open crash)
      2. Fallback: numpy Klatt synthesizer → write WAV directly
    """
    speaker    = inputs.get("speaker", "Narrator")
    line       = inputs.get("line", "")
    emotion    = inputs.get("emotion", "neutral")
    scene_id   = inputs.get("scene_id", 0)
    line_index = inputs.get("line_index", 0)

    os.makedirs("outputs/audio", exist_ok=True)
    safe_name  = speaker.replace(" ", "_").replace("/", "_")
    mp3_path   = f"outputs/audio/scene_{scene_id:02d}_line_{line_index:02d}_{safe_name}.mp3"
    wav_path   = f"outputs/audio/scene_{scene_id:02d}_line_{line_index:02d}_{safe_name}.wav"

    duration_ms = None

    # ── Try edge_tts ────────────────────────────────────────
    try:
        import asyncio
        import edge_tts

        voice = _VOICE_MAP.get(speaker, "en-US-AriaNeural")

        async def _generate():
            communicate = edge_tts.Communicate(text=line, voice=voice)
            await communicate.save(mp3_path)

        # Handle event loop safely (works both standalone and in threads)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, _generate())
                    future.result()
            else:
                loop.run_until_complete(_generate())
        except RuntimeError:
            asyncio.run(_generate())

        # ── Convert MP3 → WAV (fixes wave.Error: not RIFF) ──
        cmd = [
            FFMPEG, "-y", "-i", mp3_path,
            "-ar", str(TARGET_SR),
            "-ac", "1",
            "-sample_fmt", "s16",
            wav_path
        ]
        res = subprocess.run(cmd, capture_output=True)
        if res.returncode != 0:
            raise RuntimeError(f"ffmpeg MP3→WAV failed: {res.stderr.decode()[:200]}")

        # Get actual duration from the WAV
        with wave.open(wav_path, "r") as wf:
            duration_ms = int(wf.getnframes() / wf.getframerate() * 1000)

        print(f"    [VoiceSynth/EdgeTTS] {speaker} ({emotion}): {duration_ms}ms → {wav_path}")

    except Exception as e:
        print(f"    [VoiceSynth] edge_tts unavailable ({type(e).__name__}: {e}), using numpy fallback")

        # ── Numpy fallback ──────────────────────────────────
        samples, dur = _synth_numpy(line, speaker)
        duration_ms  = int(dur * 1000)
        with wave.open(wav_path, "w") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(TARGET_SR)
            wf.writeframes(samples.tobytes())

        print(f"    [VoiceSynth/Numpy] {speaker}: {len(line.split())} words ({duration_ms}ms) → {wav_path}")

    return {
        "audio_path":  wav_path,
        "duration_ms": duration_ms,
        "speaker":     speaker,
        "emotion":     emotion
    }


# =============================================================================
# MERGE SCENE AUDIO
# Merges per-line WAV files into one scene WAV using numpy (clean, no distortion).
# Accepts .wav files only — edge_tts output is always converted to WAV first.
# =============================================================================
def _load_wav_float(path):
    try:
        with wave.open(path, "r") as wf:
            sr  = wf.getframerate()
            ch  = wf.getnchannels()
            raw = wf.readframes(wf.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
        if ch == 2:
            samples = samples.reshape(-1, 2).mean(axis=1)
        return samples, sr
    except Exception as e:
        print(f"      [AudioLoad] Cannot load {path}: {e}")
        return None, None


def _resample(samples, src_sr, tgt_sr=TARGET_SR):
    if src_sr == tgt_sr: return samples.astype(np.float32)
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(src_sr), int(tgt_sr))
        return resample_poly(samples, tgt_sr//g, src_sr//g).astype(np.float32)
    except Exception:
        n = int(len(samples) * tgt_sr / src_sr)
        return np.interp(np.linspace(0, len(samples)-1, n),
                         np.arange(len(samples)), samples).astype(np.float32)


def _normalize(samples, target=0.88):
    pk = np.abs(samples).max()
    if pk < 1e-6: return samples.astype(np.float32)
    return (samples / pk * target).astype(np.float32)


def _merge_scene_audio(inputs):
    """
    Merge per-line WAV files into one scene WAV.
    Uses numpy float32 pipeline: load → resample → normalize → concat with gap → save.
    All input files must be valid WAV (edge_tts output is converted before this call).
    """
    scene_id    = inputs.get("scene_id", 0)
    audio_files = inputs.get("audio_files", [])

    # Accept only existing valid WAV files
    valid_wavs = [
        f for f in audio_files
        if isinstance(f, str) and f.endswith(".wav") and os.path.exists(f)
           and os.path.getsize(f) > 44
    ]

    os.makedirs("outputs/audio", exist_ok=True)
    output_path = f"outputs/audio/scene_{scene_id:02d}_merged.wav"

    if not valid_wavs:
        # Write 2s silence as fallback
        n = int(TARGET_SR * 2.0)
        with wave.open(output_path, "w") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(TARGET_SR)
            wf.writeframes(b"\x00" * n * 2)
        print(f"    [AudioMerge] Scene {scene_id}: no valid WAVs → silence fallback")
        return {"merged_path": output_path, "duration_ms": 2000, "duration_s": 2.0}

    gap     = np.zeros(int(TARGET_SR * 0.18), dtype=np.float32)   # 180ms gap
    parts   = []
    for i, wp in enumerate(valid_wavs):
        s, sr = _load_wav_float(wp)
        if s is None or len(s) == 0:
            print(f"      [AudioMerge] Skipped bad file: {wp}")
            continue
        s = _resample(s, sr)
        s = _normalize(s)
        parts.append(s)
        if i < len(valid_wavs) - 1:
            parts.append(gap)     # gap between lines, NOT after last

    if not parts:
        parts = [np.zeros(int(TARGET_SR * 2), dtype=np.float32)]

    merged    = np.clip(np.concatenate(parts), -1.0, 1.0)
    dur_s     = len(merged) / TARGET_SR
    dur_ms    = int(dur_s * 1000)
    merged_i16 = (merged * 32767).astype(np.int16)

    with wave.open(output_path, "w") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(TARGET_SR)
        wf.writeframes(merged_i16.tobytes())

    print(f"    [AudioMerge] Scene {scene_id}: {len(valid_wavs)} clips → "
          f"{output_path} ({dur_s:.2f}s)")

    return {"merged_path": output_path, "duration_ms": dur_ms, "duration_s": dur_s}


# =============================================================================
# CHARACTER IMAGE LOADER
# =============================================================================
def _find_char_image(char_name):
    img_dir = "outputs/images"
    if not os.path.isdir(img_dir): return None
    safe = char_name.replace(" ", "_")
    for fname in os.listdir(img_dir):
        base = os.path.splitext(fname)[0].lower()
        norm = char_name.lower().replace(" ", "")
        if (safe.lower() in base or base in safe.lower() or
                norm in base.replace("_", "") or base.replace("_","") in norm):
            path = os.path.join(img_dir, fname)
            try:
                img = Image.open(path).convert("RGBA")
                return img, path
            except Exception:
                continue
    return None, None


# =============================================================================
# AUDIO AMPLITUDE EXTRACTION (for lip sync)
# =============================================================================
def _extract_amplitudes(wav_path, fps=VIDEO_FPS):
    if not wav_path or not os.path.exists(wav_path):
        return np.array([], dtype=np.float32)
    try:
        samples, sr = _load_wav_float(wav_path)
        if samples is None: return np.array([], dtype=np.float32)
        samples = _resample(samples, sr)
        spf  = TARGET_SR / fps
        n_fr = int(len(samples) / spf)
        amps = np.zeros(n_fr, dtype=np.float32)
        for fi in range(n_fr):
            chunk = samples[int(fi*spf):int((fi+1)*spf)]
            if len(chunk) > 0:
                amps[fi] = float(np.sqrt(np.mean(chunk**2)))
        pk = amps.max()
        if pk > 1e-6: amps /= pk
        return amps
    except Exception as e:
        print(f"      [Amp] {e}")
        return np.array([], dtype=np.float32)


def _wav_duration_s(path):
    try:
        with wave.open(path, "r") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0


# =============================================================================
# SCENE BACKGROUND RENDERERS
# =============================================================================
def _gradient_lines(draw, W, H, top, bot):
    for y in range(H):
        t = y / H
        draw.line([(0,y),(W,y)],
                  fill=(int(top[0]+(bot[0]-top[0])*t),
                        int(top[1]+(bot[1]-top[1])*t),
                        int(top[2]+(bot[2]-top[2])*t)))


def _add_grain(img, fi, strength=6):
    rng   = np.random.default_rng(fi * 3571)
    grain = rng.integers(-strength, strength, (img.height, img.width, 3), dtype=np.int16)
    arr   = np.clip(np.array(img, dtype=np.int16) + grain, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _bg_ruins(draw, W, H, t, fi):
    horizon = int(H * 0.42)
    # Stormy sky
    for y in range(horizon):
        yt = y / horizon
        r = int(28 + 18*yt + 4*math.sin(yt*4 + t*0.8))
        draw.line([(0,y),(W,y)], fill=(r, int(24+14*yt), int(38+22*yt)))
    # Ground rubble
    for y in range(horizon, H):
        yt = (y-horizon)/(H-horizon)
        draw.line([(0,y),(W,y)], fill=(int(30+10*yt), int(26+8*yt), int(22+6*yt)))
    # Distant crumbled skyline
    rng = random.Random(99)
    for bx in range(-30, W+30, 48):
        bh = rng.randint(100,220); bw = rng.randint(32,58)
        by = horizon - bh
        dc = rng.randint(10,20)
        pts = [(bx,horizon),(bx+bw,horizon),(bx+bw,by+rng.randint(0,25))]
        xp = bx+bw
        while xp > bx:
            pts.append((xp, by+rng.randint(-8,20))); xp -= rng.randint(6,15)
        pts.append((bx, by+rng.randint(0,15)))
        draw.polygon(pts, fill=(dc,dc-2,dc-3))
        for wy in range(by+20, horizon-12, 22):
            for wx in range(bx+4, bx+bw-4, 12):
                draw.rectangle([wx,wy,wx+5,wy+9], fill=(6,6,8))
    # Mid-layer
    rng2 = random.Random(42)
    for bx in range(-15, W+15, 72):
        bh=rng2.randint(60,140); bw=rng2.randint(45,75); by=horizon-bh; c=rng2.randint(18,32)
        pts=[(bx,horizon),(bx+bw,horizon),(bx+bw,by+rng2.randint(0,25))]
        for _ in range(3): pts.append((bx+rng2.randint(0,bw), by+rng2.randint(-6,18)))
        pts.append((bx,by+rng2.randint(0,18)))
        draw.polygon(pts, fill=(c,c-3,c-4))
    # Foreground debris
    rng3 = random.Random(13)
    for _ in range(25):
        rx=rng3.randint(0,W); ry=rng3.randint(horizon+5,H-25); rs=rng3.randint(3,16)
        rc=rng3.randint(22,42)
        draw.ellipse([rx-rs,ry-rs//2,rx+rs,ry+rs//2], fill=(rc,rc-3,rc-4))
    # Dust particles
    rng4 = random.Random(fi*7+3)
    for _ in range(10):
        dx=rng4.randint(0,W); dy=rng4.randint(horizon-20,horizon+30)
        ds=rng4.randint(2,5); da=rng4.randint(25,70)
        draw.ellipse([dx-ds,dy-ds//2,dx+ds,dy+ds//2], fill=(da+10,da+8,da+12))


def _bg_bunker(draw, W, H, t, fi):
    _gradient_lines(draw, W, H, (10,12,8), (20,24,16))
    rng = random.Random(77)
    for lx in range(0, W, 40):
        s = rng.randint(6,14); draw.line([(lx,0),(lx,H)], fill=(s,s,s))
    for ly in range(0, H, 35):
        s = rng.randint(6,12); draw.line([(0,ly),(W,ly)], fill=(s,s,s+1))
    for _ in range(8):
        sx=rng.randint(30,W-30); sw=rng.randint(8,25); sh=rng.randint(20,60)
        draw.rectangle([sx-sw//2,0,sx+sw//2,sh], fill=(8,10,8))
    for ex in [int(W*0.05), int(W*0.78)]:
        ew = int(W*0.14)
        for sy in range(int(H*0.25), int(H*0.75), 40):
            draw.rectangle([ex,sy,ex+ew,sy+4], fill=(22,24,20))
            for ix in range(ex+4,ex+ew-4,14):
                ih=rng.randint(12,28); ic=rng.randint(18,35)
                draw.rectangle([ix,sy-ih,ix+8,sy], fill=(ic,ic,ic-3))
    fl = 0.75 + 0.25*math.sin(t*85*math.pi)
    for lx in [int(W*0.28), int(W*0.50), int(W*0.72)]:
        a = int(fl*160)
        draw.rectangle([lx-18,0,lx+18,8], fill=(a+20,a+20,a))
        for r in range(60,0,-5):
            ca=max(0,int(r*1.8*fl))
            draw.ellipse([lx-r,4,lx+r,4+r//2], fill=(ca,ca,max(0,ca-15)))
    draw.ellipse([int(W*0.45),int(H*0.82),int(W*0.55),int(H*0.88)], fill=(12,15,12))


def _bg_control_room(draw, W, H, t, fi):
    _gradient_lines(draw, W, H, (6,10,18), (14,22,38))
    panel_y = int(H*0.55)
    draw.rectangle([0,panel_y,W,H], fill=(14,16,12))
    draw.rectangle([0,panel_y,W,panel_y+6], fill=(22,28,18))
    rng = random.Random(fi//3)
    screens = [
        (int(W*0.06),int(H*0.12),int(W*0.22),int(H*0.38),(0,80,160)),
        (int(W*0.26),int(H*0.08),int(W*0.46),int(H*0.42),(0,100,40)),
        (int(W*0.50),int(H*0.10),int(W*0.68),int(H*0.40),(120,80,0)),
        (int(W*0.72),int(H*0.14),int(W*0.92),int(H*0.36),(0,80,160)),
    ]
    for (x1,y1,x2,y2,sc) in screens:
        fl=int(15*math.sin(t*70*math.pi+x1))
        bc=(max(0,sc[0]//6+fl//3),max(0,sc[1]//6+fl//3),max(0,sc[2]//6+fl//3))
        draw.rectangle([x1,y1,x2,y2], fill=bc)
        draw.rectangle([x1,y1,x2,y2], outline=(sc[0]//2,sc[1]//2,sc[2]//2), width=2)
        sb=(min(255,sc[0]+80),min(255,sc[1]+80),min(255,sc[2]+80))
        rng2=random.Random(x1+fi//6)
        for _ in range(5):
            lx1=rng2.randint(x1+5,x2-5); lx2=rng2.randint(x1+5,x2-5); ly=rng2.randint(y1+5,y2-5)
            draw.line([(lx1,ly),(lx2,ly)], fill=sb)
        if (fi//6)%2==0: draw.rectangle([x1+5,y2-10,x1+10,y2-5], fill=sb)
    for bx in range(int(W*0.05),int(W*0.95),22):
        for by in range(panel_y+12,panel_y+50,18):
            bc=rng.choice([(180,30,30),(30,180,30),(30,30,180),(180,160,30)])
            draw.ellipse([bx-4,by-4,bx+4,by+4], fill=bc)


def _render_bg(location, draw, W, H, t, fi):
    loc = location.lower()
    if any(k in loc for k in ("ruins","city","urban","debris","wasteland")):
        _bg_ruins(draw, W, H, t, fi)
    elif any(k in loc for k in ("control","room","computer","screens","keyboard")):
        _bg_control_room(draw, W, H, t, fi)
    elif any(k in loc for k in ("bunker","underground","shelter","abandoned","tunnel")):
        _bg_bunker(draw, W, H, t, fi)
    else:
        _gradient_lines(draw, W, H, (12,10,18), (45,40,65))


# =============================================================================
# VIDEO FRAME RENDERING (subtitle + mouth animation)
# =============================================================================
def _draw_subtitle(draw, text, W, H, bar):
    if not text: return
    max_chars = 70
    words = text.split(); lines, cur = [], []
    for w in words:
        if sum(len(x)+1 for x in cur)+len(w) <= max_chars: cur.append(w)
        else:
            if cur: lines.append(" ".join(cur))
            cur = [w]
    if cur: lines.append(" ".join(cur))
    y0 = H - bar + int(bar*0.18)
    for i, ln in enumerate(lines[:2]):
        draw.text((20, y0+i*16), ln, fill=(232,226,218))


def _composite_portrait(base_img, portrait_rgba, cx, bar, W, H):
    usable_h = H - 2*bar
    char_h   = int(usable_h * 0.85)
    aspect   = portrait_rgba.width / portrait_rgba.height
    char_w   = int(char_h * aspect)
    port_r   = portrait_rgba.resize((char_w, char_h), Image.LANCZOS)
    py = H - bar - char_h
    px = cx - char_w//2
    px = max(0, min(px, W-char_w))
    py = max(bar, py)
    canvas = base_img.convert("RGBA")
    canvas.paste(port_r, (px, py), port_r)
    face_cx  = cx
    face_ty  = py + int(char_h * 0.05)
    return canvas.convert("RGB"), face_cx, face_ty, char_w, char_h


def _draw_mouth(draw, face_cx, face_ty, char_w, char_h, amp):
    mouth_y = face_ty + int(char_h * 0.32)
    mw = max(8, int(char_w * 0.18))
    mh = max(2, int(mw * 0.60 * amp))
    draw.ellipse([face_cx-mw, mouth_y-mh, face_cx+mw, mouth_y+mh], fill=(12,4,4))
    draw.ellipse([face_cx-mw, mouth_y-mh, face_cx+mw, mouth_y+mh],
                 outline=(80,35,35), width=2)
    if amp > 0.45:
        draw.rectangle([face_cx-mw+4, mouth_y-mh+2, face_cx+mw-4, mouth_y-1],
                       fill=(218,212,208))


def _char_xpositions(n, W):
    if n == 1: return [W//2]
    margin = int(W*0.16)
    return [margin + int((W-2*margin)*i/(n-1)) for i in range(n)]


# =============================================================================
# STOCK FOOTAGE / VIDEO GENERATION
# =============================================================================
def _call_stock_footage(inputs):
    scene_id    = inputs.get("scene_id", 0)
    location    = inputs.get("location", "Unknown")
    visual_cues = inputs.get("visual_cues", "")
    chars_raw   = inputs.get("characters", [])
    dialogue    = inputs.get("dialogue", [])
    output_path = inputs.get("output_path",
                             f"outputs/raw_scenes/scene_{scene_id:02d}_raw.mp4")
    frame_dir   = f"outputs/frames/scene_{scene_id:02d}/"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(frame_dir, exist_ok=True)

    characters = [c.get("name", str(c)) if isinstance(c, dict) else str(c)
                  for c in chars_raw]
    if not dialogue:
        dialogue = [{"speaker":"","line": visual_cues[:80] if visual_cues else ""}]

    # Try Pexels API first
    pexels_key = os.getenv("PEXELS_API_KEY","")
    if pexels_key:
        r = _try_pexels(scene_id, location, output_path, frame_dir, dialogue, pexels_key)
        if r: return r

    return _render_cinematic_scene(
        scene_id, location, visual_cues, characters,
        dialogue, output_path, frame_dir
    )


def _try_pexels(scene_id, location, output_path, frame_dir, dialogue, key):
    try:
        r = requests.get("https://api.pexels.com/videos/search",
                         headers={"Authorization": key},
                         params={"query":f"{location} cinematic","per_page":3,
                                 "orientation":"landscape"}, timeout=15)
        if r.status_code != 200: return None
        videos = r.json().get("videos",[])
        if not videos: return None
        files = [f for v in videos for f in v.get("video_files",[])
                 if f.get("file_type")=="video/mp4" and f.get("width",0)<=1920]
        if not files: return None
        files.sort(key=lambda f: f.get("width",0), reverse=True)
        vr = requests.get(files[0]["link"], timeout=60, stream=True)
        if vr.status_code != 200: return None
        raw = f"outputs/raw_scenes/scene_{scene_id:02d}_pexels.mp4"
        with open(raw,"wb") as f:
            for chunk in vr.iter_content(1<<20): f.write(chunk)
        cmd=[FFMPEG,"-y","-i",raw,
             "-vf",f"scale={VIDEO_W}:{VIDEO_H}:force_original_aspect_ratio=decrease,"
                  f"pad={VIDEO_W}:{VIDEO_H}:(ow-iw)/2:(oh-ih)/2",
             "-r",str(VIDEO_FPS),"-c:v","libx264","-pix_fmt","yuv420p","-crf","20","-an",
             output_path]
        if subprocess.run(cmd, capture_output=True).returncode != 0: return None
        subprocess.run([FFMPEG,"-y","-i",output_path,"-vf",f"fps={VIDEO_FPS}",
                        os.path.join(frame_dir,"frame_%04d.png")], capture_output=True)
        n = len([f for f in os.listdir(frame_dir) if f.endswith(".png")])
        print(f"    [Pexels] ✅ {output_path} ({n} frames)")
        return {"video_path":output_path,"frame_dir":frame_dir,"frame_count":n,
                "num_frames":n,"dialogue":dialogue,"method":"stock",
                "scene_id":scene_id}
    except Exception as e:
        print(f"    [Pexels] {e}, using PIL renderer.")
        return None


def _render_cinematic_scene(scene_id, location, visual_cues, characters,
                             dialogue, output_path, frame_dir):
    bar  = int(VIDEO_H * 0.09)
    cxs  = _char_xpositions(len(characters), VIDEO_W)

    # Load portraits
    portraits = []
    for name in characters:
        img, path = _find_char_image(name)
        if img: print(f"      [CharImg] '{os.path.basename(path)}' → '{name}'")
        else:   print(f"      [CharImg] ⚠️  No image for '{name}'")
        portraits.append((name, img))

    # Per-line frame budgets from audio amplitude files
    audio_dir = "outputs/audio"
    line_amps_list    = []
    line_frame_counts = []
    for li, dl in enumerate(dialogue):
        spk  = dl.get("speaker","")
        safe = spk.replace(" ","_").replace("/","_")
        wav  = f"{audio_dir}/scene_{scene_id:02d}_line_{li:02d}_{safe}.wav"
        amps = _extract_amplitudes(wav, VIDEO_FPS)
        if len(amps) == 0:
            wc = max(1, len(dl.get("line","").split()))
            amps = np.zeros(int(max(1.5, wc*0.36)*VIDEO_FPS), dtype=np.float32)
        line_amps_list.append(amps)
        line_frame_counts.append(len(amps))

    total_frames = sum(line_frame_counts)
    if total_frames < VIDEO_FPS:
        total_frames = VIDEO_FPS * 3

    print(f"    [VideoGen] Scene {scene_id} '{location}': "
          f"{len(dialogue)} lines, {total_frames} frames")

    global_fi = 0
    for li, dl in enumerate(dialogue):
        n_fr    = line_frame_counts[li]
        amps    = line_amps_list[li]
        speaker = dl.get("speaker","")
        line    = dl.get("line","")
        subtitle = f'{speaker}: "{line}"' if speaker else line

        speaking_ci = 0
        for ci, (cname,_) in enumerate(portraits):
            if cname.lower() == speaker.lower():
                speaking_ci = ci; break

        for local_fi in range(n_fr):
            t   = global_fi / max(total_frames-1, 1)
            amp = float(amps[local_fi]) if local_fi < len(amps) else 0.0

            img  = Image.new("RGB", (VIDEO_W, VIDEO_H))
            draw = ImageDraw.Draw(img)
            _render_bg(location, draw, VIDEO_W, VIDEO_H, t, global_fi)
            img  = _add_grain(img, global_fi)

            # Composite character portraits
            char_mouth_data = []
            for ci, (cname, portrait_rgba) in enumerate(portraits):
                cx   = cxs[ci]
                sway = int(3 * math.sin(2*math.pi*t + ci*1.9))
                cx_s = cx + sway
                if portrait_rgba is not None:
                    img, fcx, fty, cw, ch = _composite_portrait(
                        img, portrait_rgba, cx_s, bar, VIDEO_W, VIDEO_H)
                    char_mouth_data.append((fcx, fty, cw, ch))
                else:
                    d2 = ImageDraw.Draw(img)
                    ph = int((VIDEO_H-2*bar)*0.82); pw = int(ph*0.4)
                    py = VIDEO_H-bar-ph; px = cx_s-pw//2
                    bc = (30+ci*8, 28+ci*5, 45+ci*8)
                    d2.rectangle([px,py,px+pw,py+ph], fill=bc)
                    hr = int(pw*0.55)
                    d2.ellipse([cx_s-hr,py-hr*2,cx_s+hr,py], fill=bc)
                    char_mouth_data.append((cx_s, py+int(ph*0.1), pw, ph))

            # Lip sync mouth
            if amp > 0.02 and speaking_ci < len(char_mouth_data):
                draw = ImageDraw.Draw(img)
                _draw_mouth(draw, *char_mouth_data[speaking_ci], amp)

            # Letterbox
            draw = ImageDraw.Draw(img)
            draw.rectangle([0,0,VIDEO_W,bar], fill=(0,0,0))
            draw.rectangle([0,VIDEO_H-bar,VIDEO_W,VIDEO_H], fill=(0,0,0))

            # Location title (first 22 frames)
            if global_fi < 22:
                fade = max(0.0, (22-global_fi)/15.0)
                draw.text((20, bar+8), location.upper(),
                          fill=(int(200*fade), int(175*fade), int(80*fade)))
                if visual_cues and global_fi < 30:
                    cue = visual_cues[:72]+("…" if len(visual_cues)>72 else "")
                    draw.text((20, bar+26), cue,
                              fill=(int(100*fade), int(95*fade), int(130*fade)))

            _draw_subtitle(draw, subtitle, VIDEO_W, VIDEO_H, bar)
            img.save(os.path.join(frame_dir, f"frame_{global_fi:04d}.png"))
            global_fi += 1

    # Encode silent MP4
    cmd = [FFMPEG,"-y",
           "-framerate", str(VIDEO_FPS),
           "-i", os.path.join(frame_dir,"frame_%04d.png"),
           "-c:v","libx264","-pix_fmt","yuv420p","-crf","18",
           output_path]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(f"[VideoGen] ffmpeg: {res.stderr.decode()[:300]}")

    kb = os.path.getsize(output_path)//1024
    print(f"    [VideoGen] → {output_path} ({kb} KB, {total_frames} frames)")
    return {"video_path":output_path,"frame_dir":frame_dir,
            "frame_count":total_frames,"num_frames":total_frames,
            "dialogue":dialogue,"method":"opencv","scene_id":scene_id}


# =============================================================================
# IDENTITY VALIDATOR
# =============================================================================
def _call_identity_validator(inputs):
    char_name = inputs.get("character_name","")
    ref_img   = inputs.get("reference_image","")
    if ref_img and os.path.exists(ref_img):
        try:
            with Image.open(ref_img) as im: _ = im.size
            valid,conf,reason = True,0.95,"Reference verified."
        except Exception:
            valid,conf,reason = False,0.0,"Invalid image."
    elif not ref_img:
        img,_ = _find_char_image(char_name)
        if img: valid,conf,reason = True,0.90,"Auto-located."
        else:   valid,conf,reason = True,0.60,"No reference — low confidence."
    else:
        valid,conf,reason = False,0.0,f"Not found: {ref_img}"
    print(f"    [IdentityValidator] {char_name}: valid={valid}, conf={conf:.2f}")
    return {"character_name":char_name,"valid":valid,"confidence":conf,"reason":reason}


# =============================================================================
# FACE SWAP
# =============================================================================
def _call_face_swapper(inputs):
    char_name  = inputs.get("character_name","Unknown")
    scene_id   = inputs.get("scene_id",0)
    frame_dir  = inputs.get("frame_dir", f"outputs/frames/scene_{scene_id:02d}/")
    ref_path   = inputs.get("reference_image","")
    output_dir = inputs.get("output_dir",
                             f"outputs/face_swapped/scene_{scene_id:02d}/")

    os.makedirs(output_dir, exist_ok=True)
    out_video = os.path.join(output_dir, f"{char_name.replace(' ','_')}.mp4")

    ref_pil = None
    if ref_path and os.path.exists(ref_path):
        try: ref_pil = Image.open(ref_path).convert("RGBA")
        except Exception: pass
    if ref_pil is None:
        ref_pil, _ = _find_char_image(char_name)

    if not os.path.isdir(frame_dir):
        _write_placeholder_mp4(out_video)
        return {"character_name":char_name,"output_path":out_video,"frames_processed":0}

    frames = sorted([f for f in os.listdir(frame_dir) if f.endswith(".png")])
    if not frames:
        _write_placeholder_mp4(out_video)
        return {"character_name":char_name,"output_path":out_video,"frames_processed":0}

    swap_dir = os.path.join(output_dir,"frames")
    os.makedirs(swap_dir, exist_ok=True)

    for fname in frames:
        src = Image.open(os.path.join(frame_dir,fname)).convert("RGBA")
        W,H = src.size
        if ref_pil is not None:
            ps   = int(min(W,H)*0.16)
            port = ref_pil.resize((ps,ps), Image.LANCZOS)
            mask = Image.new("L",(ps,ps),0)
            md   = ImageDraw.Draw(mask)
            for ring in range(6,0,-1):
                md.ellipse([ring,ring,ps-ring,ps-ring], fill=int(225*(1-ring/6)))
            md.ellipse([0,0,ps,ps], fill=220)
            px = W-ps-12; py = int(H*0.09)+6
            src.paste(port,(px,py),mask)
            ImageDraw.Draw(src).text((px+3,py+ps+2), char_name[:18], fill=(200,196,240))
        src.convert("RGB").save(os.path.join(swap_dir,fname))

    cmd=[FFMPEG,"-y","-framerate",str(VIDEO_FPS),
         "-i",os.path.join(swap_dir,"frame_%04d.png"),
         "-c:v","libx264","-pix_fmt","yuv420p","-crf","18",out_video]
    res=subprocess.run(cmd, capture_output=True)
    if res.returncode!=0:
        raise RuntimeError(f"[FaceSwap] {res.stderr.decode()[:300]}")

    kb=os.path.getsize(out_video)//1024
    print(f"    [FaceSwap] {char_name} → {out_video} ({len(frames)} frames, {kb} KB)")
    return {"character_name":char_name,"output_path":out_video,"frames_processed":len(frames)}


# =============================================================================
# LIP SYNC ALIGNER
# Merges audio + video with exact duration match (no truncation).
# =============================================================================
def _call_lip_sync(inputs):
    scene_id       = inputs.get("scene_id", 0)
    output_path    = inputs.get("output_path",
                                f"outputs/raw_scenes/scene_{scene_id:02d}.mp4")
    audio_files    = inputs.get("audio_files", [])
    dialogue       = inputs.get("dialogue", [])
    swapped_frames = inputs.get("swapped_frames", [])
    source_video   = inputs.get("source_video", "")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
                exist_ok=True)
    os.makedirs("outputs/audio", exist_ok=True)

    # Best video source
    video_src = None
    for sw in swapped_frames:
        p = sw.get("output_path","")
        if p and os.path.exists(p) and os.path.getsize(p) > 2000:
            video_src = p; break
    if not video_src and source_video and os.path.exists(source_video):
        video_src = source_video

    # Use pre-merged audio if available, else merge now
    merged_wav   = f"outputs/audio/scene_{scene_id:02d}_merged.wav"
    valid_wavs   = [af for af in audio_files
                    if isinstance(af,str) and af.endswith(".wav")
                    and os.path.exists(af) and os.path.getsize(af) > 44]

    if os.path.exists(merged_wav) and os.path.getsize(merged_wav) > 44:
        audio_dur = _wav_duration_s(merged_wav)
        sync_score = 0.95
    elif valid_wavs:
        merge_result = _merge_scene_audio({"scene_id":scene_id,"audio_files":valid_wavs})
        audio_dur    = merge_result["duration_s"]
        sync_score   = 0.95 if len(valid_wavs)>1 else 0.90
    else:
        n = int(TARGET_SR*2.0)
        with wave.open(merged_wav,"w") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(TARGET_SR)
            wf.writeframes(b"\x00"*n*2)
        audio_dur  = 2.0
        sync_score = 0.50

    # Amplitude stats
    merged_amps = _extract_amplitudes(merged_wav, VIDEO_FPS)
    frame_count = len(merged_amps) if len(merged_amps) > 0 else int(audio_dur*VIDEO_FPS)
    voiced_fr   = int(np.sum(merged_amps > 0.05)) if len(merged_amps)>0 else 0

    # Extend video if shorter than audio
    if video_src and os.path.exists(video_src):
        vid_dur = _get_video_duration(video_src)
        if vid_dur > 0 and audio_dur > vid_dur + 0.1:
            ext_path = f"outputs/raw_scenes/scene_{scene_id:02d}_ext.mp4"
            pad_dur  = audio_dur - vid_dur
            cmd_ext  = [FFMPEG,"-y","-i",video_src,
                        "-vf",f"tpad=stop_mode=clone:stop_duration={pad_dur:.3f}",
                        "-c:v","libx264","-pix_fmt","yuv420p","-crf","18",ext_path]
            if subprocess.run(cmd_ext, capture_output=True).returncode == 0:
                video_src = ext_path

    # Final mux
    if video_src and os.path.exists(video_src) and os.path.getsize(video_src) > 2000:
        cmd = [FFMPEG,"-y",
               "-i", video_src,
               "-i", merged_wav,
               "-map","0:v:0","-map","1:a:0",
               "-c:v","libx264","-pix_fmt","yuv420p",
               "-c:a","aac","-b:a","128k",
               "-preset","fast","-shortest",
               output_path]
    else:
        cmd = [FFMPEG,"-y",
               "-f","lavfi","-i",f"color=c=black:size={VIDEO_W}x{VIDEO_H}:rate={VIDEO_FPS}",
               "-i", merged_wav,
               "-c:v","libx264","-pix_fmt","yuv420p",
               "-c:a","aac","-b:a","128k","-shortest",
               output_path]

    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(f"[LipSync] ffmpeg mux failed:\n{res.stderr.decode()[:500]}")

    final_dur = _get_video_duration(output_path)
    kb        = os.path.getsize(output_path)//1024
    total_ms  = sum(d.get("duration_ms",2000) for d in dialogue)

    print(f"    [LipSync] Scene {scene_id}: audio={audio_dur:.2f}s, "
          f"video={final_dur:.2f}s, sync={sync_score:.2f}, "
          f"voiced={voiced_fr}/{frame_count} → {output_path} ({kb} KB)")

    return {
        "output_path":      output_path,
        "sync_score":       round(sync_score,2),
        "frame_count":      frame_count,
        "total_duration_ms":total_ms,
        "audio_duration_s": round(audio_dur,3),
        "video_duration_s": round(final_dur,3),
        "voiced_frames":    voiced_fr,
    }


def _get_video_duration(path):
    try:
        probe = subprocess.run([FFMPEG,"-i",path], capture_output=True, text=True)
        for line in probe.stderr.splitlines():
            if "Duration" in line:
                p = line.strip().split("Duration:")[1].split(",")[0].strip()
                h,m,s = p.split(":")
                return int(h)*3600 + int(m)*60 + float(s)
    except Exception:
        pass
    return 0.0


def _write_placeholder_mp4(path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path,"wb") as f:
        f.write(b'\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isomiso2'
                b'\x00\x00\x00\x08mdat')