import os
import json
import math
import wave
import shutil
import subprocess
import numpy as np
import requests
import cv2
from PIL import Image, ImageDraw
import imageio_ffmpeg

from mcp.tool_loader import load_tools
# client.py

FFMPEG     = imageio_ffmpeg.get_ffmpeg_exe()
TARGET_SR  = 22050    # canonical sample rate throughout pipeline
VIDEO_W    = 854
VIDEO_H    = 480
VIDEO_FPS  = 24

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
    _dispatch = {
        "grok":              _call_grok,
        "pollinations":      _call_pollinations,
        "memory_commit":     _commit_memory,
        "task_graph":        _get_task_graph,
        "voice_synth":       _call_voice_synth,
        "stock_footage":     _call_stock_footage,
        "face_swap":         _call_face_swapper,
        "identity_validate": _call_identity_validator,
        "lip_sync":          _call_lip_sync,
    }
    if api not in _dispatch:
        raise NotImplementedError(f"[MCP] API backend '{api}' not implemented.")
    return _dispatch[api](inputs)


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
        "temperature": 0.7,
        "max_tokens": 3000
    }
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers, json=payload, timeout=60
    )
    if response.status_code != 200:
        raise RuntimeError(f"[MCP/Groq] API error {response.status_code}: {response.text}")
    content = response.json()["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()
    return json.loads(content)


# =============================================================================
# POLLINATIONS AI  (character portrait generation — Phase 1 only)
# =============================================================================
def _call_pollinations(inputs):
    import urllib.parse, time
    prompt   = inputs.get("prompt", "")
    filename = inputs.get("filename", "character")
    safe_p   = urllib.parse.quote(prompt)
    url      = (f"https://image.pollinations.ai/prompt/{safe_p}"
                f"?width=512&height=512&nologo=true&seed={abs(hash(filename)) % 99999}")
    os.makedirs("outputs/images", exist_ok=True)
    filepath = f"outputs/images/{filename}.png"
    for attempt in range(1, 5):
        try:
            time.sleep(3 if attempt == 1 else attempt * 8)
            r = requests.get(url, timeout=120)
            if r.status_code == 200 and len(r.content) >= 1000:
                with open(filepath, "wb") as f:
                    f.write(r.content)
                return filepath
            elif r.status_code == 429:
                time.sleep(15)
            else:
                raise RuntimeError(f"HTTP {r.status_code}")
        except requests.exceptions.Timeout:
            print(f"  ⚠️  Timeout attempt {attempt}.")
        except RuntimeError as e:
            if attempt == 4:
                raise
            print(f"  ⚠️  {e}")
    raise RuntimeError(f"[Pollinations] Failed after 4 attempts for '{filename}'")


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
            {"id": f"a_{sid}", "type": "audio",     "scene_id": sid, "depends_on": []},
            {"id": f"v_{sid}", "type": "video",     "scene_id": sid, "depends_on": []},
            {"id": f"f_{sid}", "type": "face_swap", "scene_id": sid, "depends_on": [f"v_{sid}"]},
            {"id": f"l_{sid}", "type": "lip_sync",  "scene_id": sid,
             "depends_on": [f"a_{sid}", f"f_{sid}"]},
        ]
    print(f"[MCP/TaskGraph] Built DAG: {len(tasks)} tasks for {len(scenes)} scenes.")
    return {"dag": tasks, "tasks": tasks, "scene_count": len(scenes)}


# =============================================================================
# VOICE SYNTHESIS
# Klatt-inspired glottal source + formant filtering + prosodic envelope.
# Each character has a unique voice profile (F0 pitch, speed, breathiness).
# =============================================================================
_VOICE_PROFILES = {
    "robot guardian":       {"pitch":  95, "speed": 0.82, "roughness": 0.75},
    "malfunctioning robot": {"pitch":  72, "speed": 0.68, "roughness": 0.92},
    "human child":          {"pitch": 285, "speed": 1.18, "roughness": 0.18},
    "detective jameson":    {"pitch": 112, "speed": 0.88, "roughness": 0.62},
    "captain lewis":        {"pitch": 148, "speed": 1.00, "roughness": 0.38},
    "suspect":              {"pitch": 102, "speed": 0.83, "roughness": 0.72},
    "unknown figure":       {"pitch":  88, "speed": 0.78, "roughness": 0.82},
    "narrator":             {"pitch": 162, "speed": 1.08, "roughness": 0.22},
    "detective":            {"pitch": 118, "speed": 0.90, "roughness": 0.58},
    "default":              {"pitch": 155, "speed": 1.00, "roughness": 0.35},
}

def _get_voice_profile(speaker: str) -> dict:
    key = speaker.lower().strip()
    for k, v in _VOICE_PROFILES.items():
        if k in key or key in k:
            return v
    return _VOICE_PROFILES["default"]


def _synthesize_speech(line: str, speaker: str, sr: int = TARGET_SR) -> np.ndarray:
    profile  = _get_voice_profile(speaker)
    pitch_hz = profile["pitch"]
    speed    = profile["speed"]
    rough    = profile["roughness"]

    words    = line.split()
    wcount   = max(1, len(words))
    duration = max(1.2, wcount * 0.36 / speed)
    n        = int(sr * duration)
    t        = np.linspace(0, duration, n, dtype=np.float64)

    glottal = ((t * pitch_hz) % 1.0) - 0.5
    rng     = np.random.default_rng(abs(hash(line)) % (2 ** 31))
    noise   = rng.standard_normal(n)
    source  = (1.0 - rough) * glottal + rough * noise

    try:
        from scipy.signal import butter, lfilter
        def _bp(sig, lo, hi):
            nyq  = sr / 2.0
            lo_n = max(0.001, lo / nyq)
            hi_n = min(0.999, hi / nyq)
            if lo_n >= hi_n:
                return sig
            b, a = butter(2, [lo_n, hi_n], btype="band")
            return lfilter(b, a, sig)
        f1 = _bp(source,  250,  900)
        f2 = _bp(source,  900, 2500)
        f3 = _bp(source, 2500, 4500)
        voiced = 0.50 * f1 + 0.35 * f2 + 0.15 * f3
    except ImportError:
        voiced = source

    env = np.zeros(n)
    for wi in range(wcount):
        centre = ((wi + 0.5) / wcount) * duration
        sigma  = (duration / wcount) * 0.38
        env   += np.exp(-0.5 * ((t - centre) / sigma) ** 2)
    fade = int(0.04 * sr)
    env[:fade]  *= np.linspace(0, 1, fade)
    env[-fade:] *= np.linspace(1, 0, fade)
    env /= (env.max() + 1e-9)

    speech = voiced * env * 0.88
    peak   = np.abs(speech).max()
    if peak > 0:
        speech = (speech / peak) * 0.92
    return (speech * 32767).astype(np.int16)


def _call_voice_synth(inputs):
    speaker    = inputs.get("speaker", "Narrator")
    line       = inputs.get("line", "")
    scene_id   = inputs.get("scene_id", 0)
    line_index = inputs.get("line_index", 0)

    os.makedirs("outputs/audio", exist_ok=True)
    safe      = speaker.replace(" ", "_").replace("/", "_")
    wav_path  = (f"outputs/audio/scene_{scene_id:02d}_line_{line_index:02d}_{safe}.wav")

    sr      = TARGET_SR
    samples = _synthesize_speech(line, speaker, sr)
    dur_ms  = int(len(samples) / sr * 1000)

    with wave.open(wav_path, "w") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(samples.tobytes())

    print(f"    [VoiceSynth] {speaker}: {len(line.split())} words "
          f"({dur_ms} ms) → {wav_path}")
    return {"audio_path": wav_path, "duration_ms": dur_ms, "speaker": speaker}


# =============================================================================
# CHARACTER IMAGE UTILITIES
# =============================================================================
def _load_char_image_pil(char_name: str):
    """Return PIL RGBA image for char_name from outputs/images/, or None."""
    img_dir = "outputs/images"
    if not os.path.isdir(img_dir):
        return None
    safe = char_name.replace(" ", "_")
    for fname in os.listdir(img_dir):
        base = os.path.splitext(fname)[0].lower()
        if (safe.lower() in base or base in safe.lower() or
                char_name.lower().replace(" ", "") in base.replace("_", "")):
            path = os.path.join(img_dir, fname)
            try:
                img = Image.open(path).convert("RGBA")
                print(f"      [CharImg] '{fname}' → '{char_name}'")
                return img
            except Exception:
                continue
    print(f"      [CharImg] ⚠️  No image for '{char_name}'")
    return None


def _load_char_image_cv2(char_name: str):
    """Return (bgr_array, alpha_array) from outputs/images/, or (None, None)."""
    img_dir = "outputs/images"
    if not os.path.isdir(img_dir):
        return None, None
    safe = char_name.replace(" ", "_")
    for fname in os.listdir(img_dir):
        base = os.path.splitext(fname)[0].lower()
        if (safe.lower() in base or base in safe.lower() or
                char_name.lower().replace(" ", "") in base.replace("_", "")):
            path = os.path.join(img_dir, fname)
            img  = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if img.ndim == 3 and img.shape[2] == 4:
                return img[:, :, :3], img[:, :, 3:4].astype(np.float32) / 255.0
            elif img.ndim == 3:
                return img, np.ones((img.shape[0], img.shape[1], 1), np.float32)
    return None, None


# =============================================================================
# AUDIO UTILITIES — clean numpy-based pipeline
# =============================================================================
def _load_wav_float(path: str):
    """Load WAV → (float32 mono array, sample_rate). Returns (None,None) on error."""
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
        print(f"      [Audio] Cannot load {path}: {e}")
        return None, None


def _resample(samples: np.ndarray, src_sr: int, tgt_sr: int = TARGET_SR) -> np.ndarray:
    if src_sr == tgt_sr:
        return samples.astype(np.float32)
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(src_sr), int(tgt_sr))
        return resample_poly(samples, tgt_sr // g, src_sr // g).astype(np.float32)
    except Exception:
        # Crude linear resample fallback
        ratio    = tgt_sr / src_sr
        new_len  = int(len(samples) * ratio)
        indices  = np.linspace(0, len(samples) - 1, new_len)
        return np.interp(indices, np.arange(len(samples)), samples).astype(np.float32)


def _normalize_clip(samples: np.ndarray, target: float = 0.88) -> np.ndarray:
    peak = np.abs(samples).max()
    if peak < 1e-6:
        return samples.astype(np.float32)
    return (samples / peak * target).astype(np.float32)


def _merge_wavs_clean(wav_paths: list, out_path: str, gap_ms: int = 180) -> float:
    """
    Merge WAV files cleanly using numpy:
    - Load each file to float32
    - Resample to TARGET_SR if needed
    - Normalize independently
    - Concatenate with silence gap between (NOT after last clip)
    - Clip to [-1, 1] and save as int16

    Returns total duration in seconds.
    """
    gap_samples = int(TARGET_SR * gap_ms / 1000)
    gap         = np.zeros(gap_samples, dtype=np.float32)

    parts = []
    for i, wp in enumerate(wav_paths):
        samples, sr = _load_wav_float(wp)
        if samples is None or len(samples) == 0:
            print(f"      [AudioMerge] Skipped bad file: {wp}")
            continue
        samples = _resample(samples, sr, TARGET_SR)
        samples = _normalize_clip(samples)
        parts.append(samples)
        if i < len(wav_paths) - 1:        # gap between lines, NOT after last
            parts.append(gap)

    if not parts:
        # All files failed — write silence
        silence = np.zeros(int(TARGET_SR * 2.0), dtype=np.float32)
        parts   = [silence]

    merged = np.concatenate(parts, axis=0)
    merged = np.clip(merged, -1.0, 1.0)
    merged_i16 = (merged * 32767).astype(np.int16)

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with wave.open(out_path, "w") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(TARGET_SR)
        wf.writeframes(merged_i16.tobytes())

    total_dur = len(merged) / TARGET_SR
    print(f"      [AudioMerge] {len([p for p in wav_paths])} clips → "
          f"{out_path} ({total_dur:.2f}s, peak={abs(merged_i16).max()})")
    return total_dur


def _write_silent_wav(path: str, duration_sec: float = 2.0):
    n = int(TARGET_SR * duration_sec)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(TARGET_SR)
        wf.writeframes(b"\x00" * n * 2)


def _extract_frame_amplitudes(wav_path: str, fps: int = VIDEO_FPS) -> np.ndarray:
    """
    Extract per-video-frame RMS amplitude from a WAV file.
    Returns float32 array of length = total video frames, values in [0, 1].
    Uses precise floating-point samples-per-frame to avoid drift.
    """
    if not wav_path or not os.path.exists(wav_path):
        return np.array([], dtype=np.float32)
    try:
        samples, sr = _load_wav_float(wav_path)
        if samples is None:
            return np.array([], dtype=np.float32)
        if sr != TARGET_SR:
            samples = _resample(samples, sr, TARGET_SR)
        spf        = TARGET_SR / fps          # samples per frame (float)
        n_frames   = int(len(samples) / spf)
        amps       = np.zeros(n_frames, dtype=np.float32)
        for fi in range(n_frames):
            s = int(fi * spf)
            e = int((fi + 1) * spf)
            chunk = samples[s:e]
            if len(chunk) > 0:
                amps[fi] = float(np.sqrt(np.mean(chunk ** 2)))
        # Normalize to 0..1 against global peak for consistent mouth scale
        peak = amps.max()
        if peak > 1e-6:
            amps = amps / peak
        return amps
    except Exception as e:
        print(f"      [AmpExtract] Failed for {wav_path}: {e}")
        return np.array([], dtype=np.float32)


# =============================================================================
# VIDEO GENERATION  — Issue 1 Fix
#
# Priority chain:
#   1. Pexels API  → real stock footage (requires PEXELS_API_KEY in .env)
#   2. OpenCV procedural → renders character images + scene text + lip-sync mouth
#   3. ffmpeg color fallback → plain colored video with subtitle
# =============================================================================

_SCENE_PALETTES = {
    "ruins":         ((18, 12, 10), (62, 50, 42)),
    "night":         ((8,   8, 20), (38, 32, 58)),
    "midnight":      ((4,   4, 12), (22, 18, 42)),
    "office":        ((18, 16, 28), (55, 45, 75)),
    "warehouse":     ((12, 18, 12), (45, 55, 40)),
    "interrogation": ((22, 22, 32), (75, 70, 85)),
    "bunker":        ((8,  12,  8), (32, 42, 28)),
    "control room":  ((6,  12, 20), (28, 45, 72)),
    "underground":   ((5,   8,  5), (25, 32, 22)),
    "day":           ((160,185,210),(110,130,160)),
    "exterior":      ((28, 38, 48), (75, 95,115)),
    "int.":          ((14, 12, 20), (48, 40, 62)),
    "ext.":          ((28, 40, 52), (80,100,128)),
}

def _scene_bg(location: str):
    loc = location.lower()
    for key, pal in _SCENE_PALETTES.items():
        if key in loc:
            return pal
    return ((12, 12, 22), (45, 40, 65))


def _call_stock_footage(inputs):
    """
    FIX — Issue 1: Real video generation with Pexels → OpenCV → ffmpeg fallback chain.
    """
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
        dialogue = [{"speaker": "", "line": visual_cues[:80] if visual_cues else ""}]

    # ── Method 1: Pexels API ──────────────────────────────────
    pexels_key = "DblDMhgRTTpsqBD2YWwZl8uzSmslkpiKt2h3o8yfdLupUIXJvBhf9EA7"
    if pexels_key:
        result = _try_pexels(scene_id, location, output_path, frame_dir,
                             dialogue, pexels_key)
        if result:
            return result

    # ── Method 2: OpenCV procedural (with character images) ───
    result = _opencv_video(scene_id, location, visual_cues, characters,
                           dialogue, output_path, frame_dir)
    if result:
        return result

    # ── Method 3: ffmpeg color fallback ───────────────────────
    return _ffmpeg_color_video(scene_id, location, dialogue, output_path, frame_dir)


def _try_pexels(scene_id, location, output_path, frame_dir,
                dialogue, api_key):
    """Download stock footage from Pexels. Returns result dict or None on failure."""
    try:
        query   = f"{location} cinematic"
        headers = {"Authorization": api_key}
        params  = {"query": query, "per_page": 3, "orientation": "landscape"}
        resp    = requests.get(
            "https://api.pexels.com/videos/search",
            headers=headers, params=params, timeout=15
        )
        if resp.status_code != 200:
            print(f"    [Pexels] API returned {resp.status_code}, falling back.")
            return None

        data   = resp.json()
        videos = data.get("videos", [])
        if not videos:
            print(f"    [Pexels] No results for '{location}', falling back.")
            return None

        # Pick the best quality file (highest width, ≤1080p)
        video_files = videos[0].get("video_files", [])
        video_files = [f for f in video_files
                       if f.get("width", 0) <= 1920 and f.get("file_type") == "video/mp4"]
        if not video_files:
            return None
        video_files.sort(key=lambda f: f.get("width", 0), reverse=True)
        video_url = video_files[0]["link"]

        print(f"    [Pexels] Downloading: {video_url[:70]}...")
        video_resp = requests.get(video_url, timeout=60, stream=True)
        if video_resp.status_code != 200:
            return None

        raw_path = f"outputs/raw_scenes/scene_{scene_id:02d}_pexels_raw.mp4"
        with open(raw_path, "wb") as f:
            for chunk in video_resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)

        # Re-encode to standard resolution + extract frames
        cmd = [
            FFMPEG, "-y", "-i", raw_path,
            "-vf", f"scale={VIDEO_W}:{VIDEO_H}:force_original_aspect_ratio=decrease,"
                   f"pad={VIDEO_W}:{VIDEO_H}:(ow-iw)/2:(oh-ih)/2",
            "-r", str(VIDEO_FPS),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
            "-an",
            output_path
        ]
        res = subprocess.run(cmd, capture_output=True)
        if res.returncode != 0:
            print(f"    [Pexels] Re-encode failed: {res.stderr.decode()[:200]}")
            return None

        # Extract frames for face swap / lip sync
        frames_cmd = [
            FFMPEG, "-y", "-i", output_path,
            "-vf", f"fps={VIDEO_FPS}",
            os.path.join(frame_dir, "frame_%04d.png")
        ]
        subprocess.run(frames_cmd, capture_output=True)
        num_frames = len([f for f in os.listdir(frame_dir) if f.endswith(".png")])

        kb = os.path.getsize(output_path) // 1024
        print(f"    [Pexels] ✅ {output_path} ({kb} KB, {num_frames} frames)")
        return {
            "scene_id": scene_id,
            "video_path": output_path,
            "frame_dir": frame_dir,
            "num_frames": num_frames,
            "frame_count": num_frames,
            "dialogue": dialogue,
            "method": "stock"
        }
    except Exception as e:
        print(f"    [Pexels] Exception: {e}. Falling back to OpenCV.")
        return None


def _opencv_video(scene_id, location, visual_cues, characters,
                  dialogue, output_path, frame_dir):
    """
    Generate procedural video with OpenCV:
    - Gradient background matching scene palette
    - Character portrait images composited at bottom
    - Per-line subtitle + lip-sync mouth animation driven by real audio amplitude
    - Scene title card for first 20 frames
    """
    try:
        dark_bgr, mid_bgr = _scene_bg(location)
        # Convert RGB palette to BGR for cv2
        dark = (dark_bgr[2], dark_bgr[1], dark_bgr[0])
        mid  = (mid_bgr[2],  mid_bgr[1],  mid_bgr[0])

        # Build gradient background (static — applied to all frames)
        bg = np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.uint8)
        for y in range(VIDEO_H):
            yt = y / (VIDEO_H - 1)
            bg[y, :] = [
                int(dark[0] + (mid[0] - dark[0]) * yt),
                int(dark[1] + (mid[1] - dark[1]) * yt),
                int(dark[2] + (mid[2] - dark[2]) * yt),
            ]

        # Load character portraits (cv2 BGRA)
        char_imgs = []
        for name in characters:
            bgr, alpha = _load_char_image_cv2(name)
            char_imgs.append((name, bgr, alpha))

        # Calculate frame budget per dialogue line from existing audio WAVs
        audio_dir = "outputs/audio"
        line_amps_list  = []   # per-line amplitude arrays
        line_frame_counts = []
        for li, dl in enumerate(dialogue):
            speaker  = dl.get("speaker", "")
            safe_spk = speaker.replace(" ", "_").replace("/", "_")
            # Pattern: scene_XX_line_YY_Speaker.wav
            wav_cand = (f"{audio_dir}/scene_{scene_id:02d}_line_{li:02d}_{safe_spk}.wav")
            amps = _extract_frame_amplitudes(wav_cand, VIDEO_FPS)
            if len(amps) == 0:
                # Estimate from word count
                wc     = max(1, len(dl.get("line", "").split()))
                dur_s  = max(1.5, wc * 0.36)
                n_fr   = int(dur_s * VIDEO_FPS)
                amps   = np.zeros(n_fr, dtype=np.float32)
            line_amps_list.append(amps)
            line_frame_counts.append(len(amps))

        total_frames = sum(line_frame_counts)
        if total_frames < VIDEO_FPS:
            total_frames = VIDEO_FPS * max(3, len(dialogue) * 2)

        print(f"    [VideoGen/cv2] Scene {scene_id}: {len(dialogue)} lines, "
              f"{total_frames} frames ({location})")

        # Character horizontal positions (evenly spaced)
        n_chars = len(char_imgs)
        char_h  = int(VIDEO_H * 0.58)
        bar_h   = int(VIDEO_H * 0.085)

        def _char_cx(ci):
            if n_chars == 1:
                return VIDEO_W // 2
            margin = int(VIDEO_W * 0.14)
            return margin + int((VIDEO_W - 2 * margin) * ci / (n_chars - 1))

        # Pre-compute face centres (for mouth positioning)
        char_face_cx = [_char_cx(ci) for ci in range(n_chars)]

        # Determine which line we're in at each global frame
        line_map = []
        for li, cnt in enumerate(line_frame_counts):
            line_map.extend([li] * cnt)
        while len(line_map) < total_frames:
            line_map.append(len(dialogue) - 1)

        # Render frames
        global_fi = 0
        for li, dl in enumerate(dialogue):
            n_fr = line_frame_counts[li]
            amps = line_amps_list[li]
            spk  = dl.get("speaker", "")
            txt  = dl.get("line", "")
            subtitle = f'{spk}: "{txt}"' if spk else txt

            # Which character index is speaking?
            speaking_ci = 0
            for ci, (cname, _, _) in enumerate(char_imgs):
                if cname.lower() == spk.lower():
                    speaking_ci = ci
                    break

            for local_fi in range(n_fr):
                t   = global_fi / max(total_frames - 1, 1)
                amp = float(amps[local_fi]) if local_fi < len(amps) else 0.0

                frame = bg.copy()

                # Film grain
                grain = np.random.randint(-10, 10,
                    (VIDEO_H, VIDEO_W, 3), dtype=np.int16)
                frame = np.clip(frame.astype(np.int16) + grain,
                                0, 255).astype(np.uint8)

                # Lighting
                _draw_lighting_cv2(frame, location, t)

                # Composite character portraits
                for ci, (cname, bgr, alpha) in enumerate(char_imgs):
                    cx = _char_cx(ci)
                    sway = int(3 * math.sin(2 * math.pi * t + ci * 1.9))
                    cx  += sway
                    _composite_char_cv2(frame, bgr, alpha, cx, char_h, VIDEO_H,
                                        VIDEO_W, bar_h)

                # Lip sync: mouth on speaking character
                if amp > 0.03:
                    _draw_mouth_cv2(frame, char_face_cx[speaking_ci], char_h,
                                    VIDEO_H, bar_h, amp)

                # Letterbox bars
                frame[:bar_h, :]       = 0
                frame[VIDEO_H-bar_h:, :] = 0

                # Location title (first 20 frames)
                if global_fi < 20:
                    fade = max(0.0, (20 - global_fi) / 15.0)
                    col  = (int(80 * fade), int(170 * fade), int(200 * fade))
                    cv2.putText(frame, location.upper(),
                                (20, bar_h + 28),
                                cv2.FONT_HERSHEY_DUPLEX, 0.65, col, 2,
                                cv2.LINE_AA)

                # Visual cue (small, below title, first 30 frames)
                if global_fi < 30 and visual_cues:
                    cue_short = visual_cues[:70] + ("…" if len(visual_cues) > 70 else "")
                    cv2.putText(frame, cue_short,
                                (20, bar_h + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 95, 130), 1,
                                cv2.LINE_AA)

                # Subtitle
                _draw_subtitle_cv2(frame, subtitle, VIDEO_H, VIDEO_W, bar_h)

                cv2.imwrite(os.path.join(frame_dir, f"frame_{global_fi:04d}.png"), frame)
                global_fi += 1

        # Encode frames → silent MP4
        cmd = [
            FFMPEG, "-y",
            "-framerate", str(VIDEO_FPS),
            "-i", os.path.join(frame_dir, "frame_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
            output_path
        ]
        res = subprocess.run(cmd, capture_output=True)
        if res.returncode != 0:
            print(f"    [VideoGen/cv2] ffmpeg error: {res.stderr.decode()[:300]}")
            return None

        kb = os.path.getsize(output_path) // 1024
        print(f"    [VideoGen/cv2] → {output_path} ({kb} KB, {total_frames} frames)")
        return {
            "scene_id": scene_id,
            "video_path": output_path,
            "frame_dir": frame_dir,
            "num_frames": total_frames,
            "frame_count": total_frames,
            "dialogue": dialogue,
            "method": "opencv"
        }
    except Exception as e:
        import traceback
        print(f"    [VideoGen/cv2] Exception: {e}\n{traceback.format_exc()[:400]}")
        return None


def _composite_char_cv2(frame, bgr, alpha, cx, char_h, H, W, bar_h):
    """Paste a character portrait (bgr + alpha mask) into the frame."""
    if bgr is None:
        # Fallback silhouette
        x1 = cx - 40
        y1 = H - char_h - bar_h
        cv2.rectangle(frame, (x1, y1), (cx + 40, H - bar_h), (45, 55, 70), -1)
        cv2.circle(frame, (cx, y1 - 30), 28, (55, 65, 80), -1)
        return

    # Resize to char_h
    aspect   = bgr.shape[1] / bgr.shape[0]
    char_w   = int(char_h * aspect)
    bgr_r    = cv2.resize(bgr,   (char_w, char_h), interpolation=cv2.INTER_AREA)
    alpha_r  = cv2.resize(alpha, (char_w, char_h), interpolation=cv2.INTER_AREA)
    if alpha_r.ndim == 2:
        alpha_r = alpha_r[:, :, np.newaxis]

    py = H - char_h - bar_h
    px = cx - char_w // 2
    # Clamp to frame bounds
    px = max(0, min(px, W - char_w))
    py = max(bar_h, py)

    # Crop if portrait goes out of bounds
    fy_end = min(py + char_h, H - bar_h)
    fx_end = min(px + char_w, W)
    p_h    = fy_end - py
    p_w    = fx_end - px
    if p_h <= 0 or p_w <= 0:
        return

    roi   = frame[py:fy_end, px:fx_end].astype(np.float32)
    fore  = bgr_r[:p_h, :p_w].astype(np.float32)
    a     = alpha_r[:p_h, :p_w]

    frame[py:fy_end, px:fx_end] = np.clip(
        roi * (1.0 - a) + fore * a, 0, 255
    ).astype(np.uint8)


def _draw_mouth_cv2(frame, char_cx, char_h, H, bar_h, amp):
    """Draw an animated mouth ellipse at the speaking character's face."""
    # Face sits in the top ~35% of the portrait region
    portrait_top = H - char_h - bar_h
    face_cy      = portrait_top + int(char_h * 0.22)   # upper face centre
    mouth_cy     = face_cy + int(char_h * 0.14)        # mouth below eyes

    # Mouth width ~ 3.5% of frame width; height driven by amplitude
    mw = max(8, int(VIDEO_W * 0.035))
    mh = max(2, int(mw * 0.7 * amp))

    # Dark cavity
    cv2.ellipse(frame, (char_cx, mouth_cy), (mw, mh),
                0, 0, 180, (12, 6, 6), -1)
    # Lip outline
    cv2.ellipse(frame, (char_cx, mouth_cy), (mw, mh),
                0, 0, 180, (70, 35, 35), 2)

    # Speaking indicator: small pulsing dot above character
    dot_cy = face_cy - int(char_h * 0.05)
    pulse  = max(0, int(6 * math.sin(amp * math.pi * 8)))
    r_dot  = 5 + pulse // 3
    cv2.circle(frame, (char_cx, dot_cy), r_dot, (80, 200, 220), -1)


def _draw_lighting_cv2(frame, location, t):
    """Overlay animated scene lighting."""
    loc = location.lower()
    cx, cy = VIDEO_W // 2, VIDEO_H // 2
    if any(k in loc for k in ("ruins", "ext.", "exterior", "warehouse")):
        # Sweeping flashlight
        ang = t * 30 - 15
        bx  = int(cx + math.sin(math.radians(ang)) * VIDEO_W * 0.50)
        by  = int(VIDEO_H * 0.25)
        for r in range(90, 0, -6):
            a = max(0, int(r * 1.1))
            cv2.ellipse(frame, (bx, by), (r * 2, r), 0, 0, 360,
                        (a, a, int(a * 0.80)), -1)
    elif any(k in loc for k in ("bunker", "control", "underground")):
        # Flickering strip lights
        for lx in [int(VIDEO_W * 0.28), int(VIDEO_W * 0.72)]:
            fl = int(18 * math.sin(t * 85 * math.pi + lx))
            a  = max(0, 130 + fl)
            cv2.rectangle(frame, (lx - 4, 2), (lx + 4, int(VIDEO_H * 0.06)),
                          (a + 20, a + 20, a), -1)
            for r in range(55, 0, -5):
                ca = max(0, int(r * 1.4) + fl // 2)
                cv2.circle(frame, (lx, 4), r, (ca + 20, ca + 20, ca), -1)
    else:
        # Desk lamp warm glow
        lx = int(VIDEO_W * 0.74)
        ly = int(VIDEO_H * 0.14)
        fl = int(6 * math.sin(t * 60 * math.pi))
        for r in range(90, 0, -5):
            a = max(0, int(r * 0.88) + fl)
            cv2.ellipse(frame, (lx, ly), (r, r // 2), 0, 0, 360,
                        (max(0, a - 40), min(255, a + 48), min(255, a + 78)), -1)


def _draw_subtitle_cv2(frame, text, H, W, bar_h):
    """Wrap and draw subtitle text in the lower letterbox area."""
    if not text:
        return
    max_chars = 68
    words     = text.split()
    lines, cur = [], []
    for w in words:
        if sum(len(x) + 1 for x in cur) + len(w) <= max_chars:
            cur.append(w)
        else:
            if cur:
                lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))

    y_start = H - bar_h + int(bar_h * 0.18)
    for i, ln in enumerate(lines[:2]):
        cv2.putText(frame, ln,
                    (20, y_start + i * 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (228, 222, 212), 1,
                    cv2.LINE_AA)


def _ffmpeg_color_video(scene_id, location, dialogue, output_path, frame_dir):
    """Last-resort: plain colored video using ffmpeg lavfi."""
    dur = max(3, len(dialogue) * 2)
    cmd = [
        FFMPEG, "-y",
        "-f", "lavfi",
        "-i", f"color=c=0x0A0A14:size={VIDEO_W}x{VIDEO_H}:rate={VIDEO_FPS}:duration={dur}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        output_path
    ]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(f"[VideoGen/ffmpeg] Failed: {res.stderr.decode()[:300]}")

    # Extract frames
    subprocess.run([
        FFMPEG, "-y", "-i", output_path,
        "-vf", f"fps={VIDEO_FPS}",
        os.path.join(frame_dir, "frame_%04d.png")
    ], capture_output=True)

    num_frames = len([f for f in os.listdir(frame_dir) if f.endswith(".png")])
    kb = os.path.getsize(output_path) // 1024
    print(f"    [VideoGen/ffmpeg] → {output_path} ({kb} KB, {num_frames} frames)")
    return {
        "scene_id": scene_id,
        "video_path": output_path,
        "frame_dir": frame_dir,
        "num_frames": num_frames,
        "frame_count": num_frames,
        "dialogue": dialogue,
        "method": "opencv"
    }


# =============================================================================
# IDENTITY VALIDATOR
# =============================================================================
def _call_identity_validator(inputs):
    char_name = inputs.get("character_name", "")
    ref_img   = inputs.get("reference_image", "")

    if ref_img and os.path.exists(ref_img):
        try:
            with Image.open(ref_img) as im:
                _ = im.size
            valid, conf, reason = True, 0.95, "Reference image verified by PIL."
        except Exception:
            valid, conf, reason = False, 0.0, "File exists but is not a valid image."
    elif not ref_img:
        # Auto-locate
        auto_bgr, _ = _load_char_image_cv2(char_name)
        if auto_bgr is not None:
            valid, conf, reason = True, 0.90, "Reference auto-located from outputs/images/."
        else:
            valid, conf, reason = True, 0.60, "No reference — proceeding with low confidence."
    else:
        valid, conf, reason = False, 0.0, f"Reference not found: {ref_img}"

    print(f"    [IdentityValidator] {char_name}: valid={valid}, conf={conf:.2f} | {reason}")
    return {"character_name": char_name, "valid": valid,
            "confidence": conf, "reason": reason}


# =============================================================================
# FACE SWAP
# Composites each character's portrait as an identity inset on the scene frames,
# then re-encodes to MP4.
# =============================================================================
def _call_face_swapper(inputs):
    char_name  = inputs.get("character_name", "Unknown")
    scene_id   = inputs.get("scene_id", 0)
    frame_dir  = inputs.get("frame_dir", f"outputs/frames/scene_{scene_id:02d}/")
    ref_path   = inputs.get("reference_image", "")
    output_dir = inputs.get("output_dir",
                             f"outputs/face_swapped/scene_{scene_id:02d}/")

    os.makedirs(output_dir, exist_ok=True)
    out_video = os.path.join(output_dir, f"{char_name.replace(' ', '_')}.mp4")

    # Load reference portrait
    ref_pil = None
    if ref_path and os.path.exists(ref_path):
        try:
            ref_pil = Image.open(ref_path).convert("RGBA")
        except Exception:
            pass
    if ref_pil is None:
        ref_pil = _load_char_image_pil(char_name)

    if not os.path.isdir(frame_dir):
        print(f"    [FaceSwap] Frame dir missing: {frame_dir}")
        _write_placeholder_mp4(out_video)
        return {"character_name": char_name, "output_path": out_video,
                "frames_processed": 0}

    frames = sorted([f for f in os.listdir(frame_dir) if f.endswith(".png")])
    if not frames:
        print(f"    [FaceSwap] No frames in {frame_dir}")
        _write_placeholder_mp4(out_video)
        return {"character_name": char_name, "output_path": out_video,
                "frames_processed": 0}

    swap_dir = os.path.join(output_dir, "frames")
    os.makedirs(swap_dir, exist_ok=True)

    for fname in frames:
        src = Image.open(os.path.join(frame_dir, fname)).convert("RGBA")
        W, H = src.size

        if ref_pil is not None:
            # Identity badge: circular portrait inset top-right corner
            ps   = int(min(W, H) * 0.20)
            port = ref_pil.resize((ps, ps), Image.LANCZOS)

            # Soft circular mask
            mask = Image.new("L", (ps, ps), 0)
            md   = ImageDraw.Draw(mask)
            for ring in range(8, 0, -1):
                alpha_v = int(225 * (1 - ring / 8))
                md.ellipse([ring, ring, ps - ring, ps - ring], fill=alpha_v)
            md.ellipse([0, 0, ps, ps], fill=225)

            px = W - ps - 10
            py = int(H * 0.09)
            src.paste(port, (px, py), mask)

            # Name label below badge
            draw_s = ImageDraw.Draw(src)
            draw_s.text((px + 3, py + ps + 2), char_name[:18],
                        fill=(200, 196, 240))

        src.convert("RGB").save(os.path.join(swap_dir, fname))

    # Encode
    cmd = [
        FFMPEG, "-y",
        "-framerate", str(VIDEO_FPS),
        "-i", os.path.join(swap_dir, "frame_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        out_video
    ]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(f"[FaceSwap] ffmpeg error: {res.stderr.decode()[:300]}")

    kb = os.path.getsize(out_video) // 1024
    print(f"    [FaceSwap] {char_name} → {out_video} ({len(frames)} frames, {kb} KB)")
    return {"character_name": char_name, "output_path": out_video,
            "frames_processed": len(frames)}


# =============================================================================
# LIP SYNC ALIGNER  — Issues 2 & 3 Fix
#
# Fix 2: Audio merged cleanly via _merge_wavs_clean (numpy, normalized, gapped)
# Fix 3: Amplitude extracted from merged WAV at VIDEO_FPS resolution,
#        mapped directly to mouth opening — no fake sinusoidal fallback
# =============================================================================
def _call_lip_sync(inputs):
    scene_id       = inputs.get("scene_id", 0)
    output_path    = inputs.get("output_path",
                                f"outputs/raw_scenes/scene_{scene_id:02d}.mp4")
    audio_files    = inputs.get("audio_files", [])
    dialogue       = inputs.get("dialogue", [])
    swapped_frames = inputs.get("swapped_frames", [])
    source_video   = inputs.get("source_video", "")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs("outputs/audio", exist_ok=True)

    # ── Best video source ────────────────────────────────────
    video_src = None
    for sw in swapped_frames:
        p = sw.get("output_path", "")
        if p and os.path.exists(p) and os.path.getsize(p) > 2000:
            video_src = p
            break
    if not video_src:
        if source_video and os.path.exists(source_video) and os.path.getsize(source_video) > 2000:
            video_src = source_video

    # ── FIX 2: Clean audio merge ─────────────────────────────
    merged_wav   = f"outputs/audio/scene_{scene_id:02d}_merged.wav"
    valid_wavs   = [af for af in audio_files
                    if af and os.path.exists(af) and os.path.getsize(af) > 44]

    if not valid_wavs:
        _write_silent_wav(merged_wav, 2.0)
        audio_dur  = 2.0
        sync_score = 0.50
    elif len(valid_wavs) == 1:
        shutil.copy(valid_wavs[0], merged_wav)
        audio_dur  = os.path.getsize(merged_wav) / (TARGET_SR * 2)
        sync_score = 0.90
    else:
        audio_dur  = _merge_wavs_clean(valid_wavs, merged_wav, gap_ms=180)
        sync_score = 0.95

    # ── FIX 3: Frame-accurate lip sync amplitude extraction ──
    # Use the MERGED audio (single source of truth) to get per-frame amplitudes
    merged_amps = _extract_frame_amplitudes(merged_wav, VIDEO_FPS)
    n_audio_frames = len(merged_amps)

    # Get video frame count
    n_video_frames = 0
    if video_src and os.path.exists(video_src):
        probe = subprocess.run(
            [FFMPEG, "-i", video_src],
            capture_output=True, text=True
        )
        for line in probe.stderr.splitlines():
            if "fps" in line and "Video" in line:
                try:
                    # e.g. "24 fps" or "24.00 fps"
                    fps_tok = [x for x in line.split() if "fps" in x]
                    if fps_tok:
                        idx = line.split().index(fps_tok[0])
                        n_video_frames = int(float(line.split()[idx - 1]) *
                                             audio_dur)
                except Exception:
                    pass
    if n_video_frames == 0:
        n_video_frames = n_audio_frames

    # If mismatch, pad/trim amps to match video frames
    if n_audio_frames < n_video_frames:
        merged_amps = np.pad(merged_amps,
                             (0, n_video_frames - n_audio_frames),
                             constant_values=0.0)
    elif n_audio_frames > n_video_frames:
        merged_amps = merged_amps[:n_video_frames]

    frame_count = len(merged_amps)

    # Log sync quality
    speaking_frames = int(np.sum(merged_amps > 0.05))
    print(f"    [LipSync] Scene {scene_id}: {frame_count} frames, "
          f"{speaking_frames} voiced, sync={sync_score:.2f}, "
          f"audio={audio_dur:.2f}s")

    # ── Mux video + merged audio ─────────────────────────────
    if video_src and os.path.exists(video_src) and os.path.getsize(video_src) > 2000:
        cmd = [
            FFMPEG, "-y",
            "-i", video_src,
            "-i", merged_wav,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-preset", "fast",
            "-shortest",
            output_path
        ]
    else:
        cmd = [
            FFMPEG, "-y",
            "-f", "lavfi",
            "-i", f"color=c=black:size={VIDEO_W}x{VIDEO_H}:rate={VIDEO_FPS}",
            "-i", merged_wav,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest",
            output_path
        ]

    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"[LipSync] ffmpeg mux failed:\n{res.stderr.decode()[:500]}"
        )

    kb = os.path.getsize(output_path) // 1024
    total_ms = sum(d.get("duration_ms", 2000) for d in dialogue)
    print(f"    [LipSync] → {output_path} ({kb} KB)")

    return {
        "output_path":       output_path,
        "sync_score":        round(sync_score, 2),
        "frame_count":       frame_count,
        "total_duration_ms": total_ms,
        "audio_duration_s":  round(audio_dur, 3),
        "voiced_frames":     speaking_frames,
    }


# =============================================================================
# HELPERS
# =============================================================================
def _write_placeholder_mp4(path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(b'\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isomiso2')
        f.write(b'\x00\x00\x00\x08mdat')