"""
agents/scene_composer.py

Timeline-driven Scene Composer.
Reads: audio_tracks, scenes (dialogue + characters), video_outputs, images
Builds: per-scene timeline  →  single composed final video per scene
Output: outputs/final_scenes/scene_XX.mp4

Architecture:
  1. Build timeline  →  [{speaker, text, start_s, end_s, frame_start, frame_end, audio_path}]
  2. Open background video (from raw_scenes)
  3. Load all character portraits for that scene
  4. For each frame:
       a. Pull background frame (loop if BG is shorter)
       b. Determine active speaker from timeline
       c. Composite all characters (brighten + border on speaker)
       d. Draw mouth animation on speaker using audio amplitude
       e. Draw subtitle bar
       f. Write frame via ffmpeg pipe
  5. Mux with merged audio
"""

import os
import cv2
import math
import wave
import shutil
import subprocess
import numpy as np
from PIL import Image, ImageDraw

import imageio_ffmpeg

FFMPEG  = imageio_ffmpeg.get_ffmpeg_exe()
VIDEO_W = 854
VIDEO_H = 480
FPS     = 24
BAR_H   = int(VIDEO_H * 0.09)   # letterbox bar height


# =============================================================================
# MAIN ENTRY POINT (called from lip_sync_node or directly from pipeline)
# =============================================================================
def scene_composer_node(state):
    print("\n[Scene Composer] Building timeline-driven scene videos...")

    scenes        = state.get("scenes", [])
    audio_tracks  = state.get("audio_tracks", [])
    video_outputs = state.get("video_outputs", [])
    images        = state.get("images", [])

    os.makedirs("outputs/final_scenes", exist_ok=True)

    # Build lookup maps
    audio_map = {a["scene_id"]: a for a in audio_tracks}
    video_map = {v["scene_id"]: v for v in video_outputs}

    # Character name → image path
    char_img_map = {}
    for img_path in images:
        base      = os.path.splitext(os.path.basename(img_path))[0]
        char_name = base.replace("_", " ")
        char_img_map[char_name] = img_path

    final_scenes = []

    for scene in scenes:
        scene_id   = scene["scene_id"]
        characters = scene.get("characters", [])
        dialogue   = scene.get("dialogue", [])

        audio_data = audio_map.get(scene_id, {})
        video_data = video_map.get(scene_id, {})

        audio_files  = audio_data.get("audio_files", [])
        merged_audio = audio_data.get("merged_audio", "")
        bg_video     = video_data.get("video_path", "")

        print(f"\n  → Scene {scene_id}: {len(characters)} characters, "
              f"{len(dialogue)} dialogue lines")

        try:
            output_path = _compose_scene(
                scene_id   = scene_id,
                dialogue   = dialogue,
                characters = characters,
                audio_files= audio_files,
                merged_audio = merged_audio,
                bg_video   = bg_video,
                char_img_map = char_img_map,
            )
            kb = os.path.getsize(output_path) // 1024
            print(f"    ✅ Composed: {output_path} ({kb} KB)")
            final_scenes.append({
                "scene_id":   scene_id,
                "output_path":output_path,
            })
        except Exception as e:
            import traceback
            print(f"    ⚠️  Scene {scene_id} composition failed: {e}")
            traceback.print_exc()

    state["final_scenes"] = final_scenes
    print(f"\n[Scene Composer] ✅ {len(final_scenes)} scenes composed.")
    return state


# =============================================================================
# TIMELINE BUILDER
# =============================================================================
def _build_timeline(dialogue, audio_files, gap_s=0.18):
    """
    Returns list of timeline entries, each:
    {speaker, text, visual_cue, start_s, end_s, duration_s, audio_path,
     frame_start, frame_end}
    """
    timeline = []
    cursor   = 0.0

    for i, dl in enumerate(dialogue):
        speaker    = dl.get("speaker", "")
        text       = dl.get("line", "")
        visual_cue = dl.get("visual_cue", "")

        # Audio file for this line
        wav = audio_files[i] if i < len(audio_files) else ""

        # Get real duration from WAV
        dur = _wav_duration(wav) if wav and os.path.exists(wav) else \
              max(1.5, len(text.split()) * 0.36)

        timeline.append({
            "speaker":     speaker,
            "text":        text,
            "visual_cue":  visual_cue,
            "start_s":     cursor,
            "end_s":       cursor + dur,
            "duration_s":  dur,
            "audio_path":  wav,
            "frame_start": round(cursor * FPS),
            "frame_end":   round((cursor + dur) * FPS),
        })
        cursor += dur
        if i < len(dialogue) - 1:
            cursor += gap_s   # silence gap between lines, NOT after last

    return timeline, cursor   # cursor = total duration


def _wav_duration(path):
    try:
        with wave.open(path, "r") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 2.0


# =============================================================================
# AMPLITUDE ENVELOPE CACHE
# Per-frame RMS amplitude extracted from each line's WAV.
# Used to animate the speaker's mouth.
# =============================================================================
def _extract_amps(wav_path, fps=FPS):
    if not wav_path or not os.path.exists(wav_path):
        return np.array([], dtype=np.float32)
    try:
        with wave.open(wav_path, "r") as wf:
            sr  = wf.getframerate()
            ch  = wf.getnchannels()
            raw = wf.readframes(wf.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
        if ch == 2:
            samples = samples.reshape(-1, 2).mean(axis=1)
        # Resample to TARGET_SR if needed
        if sr != 22050:
            ratio = 22050 / sr
            n_new = int(len(samples) * ratio)
            samples = np.interp(np.linspace(0, len(samples)-1, n_new),
                                np.arange(len(samples)), samples)
        spf  = 22050 / fps
        n_fr = int(len(samples) / spf)
        amps = np.zeros(n_fr, dtype=np.float32)
        for fi in range(n_fr):
            chunk = samples[int(fi*spf):int((fi+1)*spf)]
            if len(chunk) > 0:
                amps[fi] = float(np.sqrt(np.mean(chunk**2)))
        pk = amps.max()
        if pk > 1e-6:
            amps /= pk
        return amps
    except Exception as e:
        print(f"      [Amp] {e}")
        return np.array([], dtype=np.float32)


# =============================================================================
# CHARACTER PORTRAIT LOADER
# =============================================================================
def _load_portrait(char_name, char_img_map):
    """Return PIL RGBA portrait or None."""
    # Direct lookup
    path = char_img_map.get(char_name, "")
    if not path:
        # Fuzzy lookup in outputs/images/
        img_dir = "outputs/images"
        if os.path.isdir(img_dir):
            safe = char_name.replace(" ", "_")
            for fname in os.listdir(img_dir):
                base = os.path.splitext(fname)[0].lower()
                if (safe.lower() in base or base in safe.lower() or
                        char_name.lower().replace(" ","") in base.replace("_","")):
                    path = os.path.join(img_dir, fname)
                    break
    if path and os.path.exists(path):
        try:
            img = Image.open(path).convert("RGBA")
            print(f"      [Portrait] '{os.path.basename(path)}' → '{char_name}'")
            return img
        except Exception:
            pass
    print(f"      [Portrait] ⚠️  No image for '{char_name}'")
    return None


# =============================================================================
# BACKGROUND VIDEO READER  (looping)
# =============================================================================
class _BgReader:
    def __init__(self, video_path):
        self._path  = video_path
        self._cap   = None
        self._blank = np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.uint8)
        self._blank[:, :] = [20, 18, 30]
        if video_path and os.path.exists(video_path):
            self._cap = cv2.VideoCapture(video_path)
        self._total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self._cap else 0

    def read(self, fi):
        """Read frame fi (loops if BG is shorter than total scene)."""
        if not self._cap or self._total == 0:
            return self._blank.copy()
        seek = fi % self._total
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, seek)
        ret, frame = self._cap.read()
        if not ret:
            return self._blank.copy()
        # Resize to standard resolution
        if frame.shape[:2] != (VIDEO_H, VIDEO_W):
            frame = cv2.resize(frame, (VIDEO_W, VIDEO_H), interpolation=cv2.INTER_AREA)
        return frame

    def close(self):
        if self._cap:
            self._cap.release()


# =============================================================================
# CHARACTER LAYOUT  (horizontal, evenly spaced)
# =============================================================================
def _char_cx_positions(n, W=VIDEO_W):
    if n == 1: return [W // 2]
    margin = int(W * 0.14)
    return [margin + int((W - 2 * margin) * i / (n - 1)) for i in range(n)]


# =============================================================================
# FRAME COMPOSER
# Composites one complete frame (BGR numpy array).
# =============================================================================
def _compose_frame(
    bg_frame,          # (H, W, 3) numpy BGR
    portraits,         # list of (char_name, PIL RGBA | None)
    cx_positions,      # list of int  (centre-x per character)
    speaking_ci,       # index of currently speaking character
    amp,               # amplitude 0..1 for mouth animation
    subtitle,          # text to render in lower bar
    title_text,        # location title (shown first 20 frames)
    show_title,        # bool
):
    W, H = VIDEO_W, VIDEO_H
    bar  = BAR_H

    # ── Convert bg to PIL for compositing ────────────────────
    frame_pil = Image.fromarray(cv2.cvtColor(bg_frame, cv2.COLOR_BGR2RGB)).convert("RGBA")

    # ── Composite character portraits ─────────────────────────
    usable_h = H - 2 * bar
    char_h   = int(usable_h * 0.85)
    mouth_data = []   # (face_cx, face_ty, char_w, char_h) per character

    for ci, (cname, portrait) in enumerate(portraits):
        cx      = cx_positions[ci]
        speaking = (ci == speaking_ci)

        if portrait is not None:
            aspect = portrait.width / portrait.height
            char_w = int(char_h * aspect)
            port_r = portrait.resize((char_w, char_h), Image.LANCZOS)

            # ── Speaking: brighten + slight scale up ──────────
            if speaking:
                scale   = 1.06
                new_h   = int(char_h * scale)
                new_w   = int(char_w * scale)
                port_r  = portrait.resize((new_w, new_h), Image.LANCZOS)
                char_w, char_h_eff = new_w, new_h
            else:
                char_h_eff = char_h
                # Darken non-speakers slightly
                arr = np.array(port_r, dtype=np.float32)
                arr[:, :, :3] = np.clip(arr[:, :, :3] * 0.72, 0, 255)
                port_r = Image.fromarray(arr.astype(np.uint8))

            py = H - bar - char_h_eff
            px = cx - char_w // 2
            px = max(0, min(px, W - char_w))
            py = max(bar, py)

            frame_pil.paste(port_r, (px, py), port_r)
            mouth_data.append((cx, py + int(char_h_eff * 0.05), char_w, char_h_eff))

            # ── Speaking highlight border ─────────────────────
            if speaking:
                draw = ImageDraw.Draw(frame_pil)
                draw.rectangle(
                    [px - 3, py - 3, px + char_w + 3, py + char_h_eff + 3],
                    outline=(220, 200, 60, 200),
                    width=3
                )
                # "SPEAKING" dot indicator above character
                dot_cx = cx
                dot_cy = py - 14
                pulse  = max(0, int(6 * math.sin(amp * math.pi * 8)))
                r_dot  = 6 + pulse // 2
                draw.ellipse(
                    [dot_cx - r_dot, dot_cy - r_dot,
                     dot_cx + r_dot, dot_cy + r_dot],
                    fill=(220, 200, 60, 220)
                )
        else:
            # Silhouette fallback
            ph = int(usable_h * 0.82)
            pw = int(ph * 0.40)
            py = H - bar - ph
            pxl = cx - pw // 2
            bc  = (30 + ci * 8, 28 + ci * 5, 45 + ci * 8, 255)
            draw = ImageDraw.Draw(frame_pil)
            draw.rectangle([pxl, py, pxl + pw, py + ph], fill=bc)
            hr = int(pw * 0.55)
            draw.ellipse([cx - hr, py - hr * 2, cx + hr, py], fill=bc)
            mouth_data.append((cx, py + int(ph * 0.10), pw, ph))

    # ── Mouth animation on active speaker ─────────────────────
    if amp > 0.03 and 0 <= speaking_ci < len(mouth_data):
        draw = ImageDraw.Draw(frame_pil)
        fcx, fty, cw, ch_eff = mouth_data[speaking_ci]
        mouth_y = fty + int(ch_eff * 0.32)
        mw = max(8, int(cw * 0.18))
        mh = max(2, int(mw * 0.60 * amp))
        draw.ellipse([fcx - mw, mouth_y - mh, fcx + mw, mouth_y + mh],
                     fill=(12, 4, 4, 255))
        draw.ellipse([fcx - mw, mouth_y - mh, fcx + mw, mouth_y + mh],
                     outline=(80, 35, 35, 255), width=2)
        if amp > 0.45:
            draw.rectangle([fcx - mw + 4, mouth_y - mh + 2,
                            fcx + mw - 4, mouth_y - 1],
                           fill=(218, 212, 208, 255))

    # ── Letterbox bars ─────────────────────────────────────────
    draw = ImageDraw.Draw(frame_pil)
    draw.rectangle([0, 0, W, bar],       fill=(0, 0, 0, 255))
    draw.rectangle([0, H - bar, W, H],   fill=(0, 0, 0, 255))

    # ── Location title (first 20 frames) ──────────────────────
    if show_title and title_text:
        draw.text((20, bar + 8),  title_text.upper(), fill=(200, 175, 80))

    # ── Speaker name label ─────────────────────────────────────
    if subtitle:
        _draw_subtitle(draw, subtitle, W, H, bar)

    # ── Convert back to BGR numpy ──────────────────────────────
    result = cv2.cvtColor(np.array(frame_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    return result


def _draw_subtitle(draw, text, W, H, bar):
    """Wrap and centre-draw dialogue subtitle in lower letterbox."""
    if not text:
        return
    max_chars = 72
    words     = text.split()
    lines, cur = [], []
    for w in words:
        if sum(len(x) + 1 for x in cur) + len(w) <= max_chars:
            cur.append(w)
        else:
            if cur: lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    y0 = H - bar + int(bar * 0.15)
    for i, ln in enumerate(lines[:2]):
        tw = len(ln) * 7          # approximate pixel width
        x  = max(10, (W - tw) // 2)
        draw.text((x, y0 + i * 16), ln, fill=(232, 226, 218))


# =============================================================================
# SCENE COMPOSITION PIPELINE
# =============================================================================
def _compose_scene(scene_id, dialogue, characters, audio_files,
                   merged_audio, bg_video, char_img_map):
    """
    Compose one final scene video:
      1. Build timeline from dialogue + per-line audio durations
      2. Pre-load portraits for all characters
      3. Pre-extract amplitude envelopes per timeline entry
      4. Stream frames to ffmpeg pipe
      5. Mux with merged audio
    Returns path to final composed video.
    """
    output_path = f"outputs/final_scenes/scene_{scene_id:02d}.mp4"
    os.makedirs("outputs/final_scenes", exist_ok=True)

    # ── Step 1: Timeline ──────────────────────────────────────
    timeline, total_dur = _build_timeline(dialogue, audio_files)
    total_frames = max(1, round(total_dur * FPS))
    print(f"      [Composer] Timeline: {len(timeline)} entries, "
          f"{total_dur:.2f}s, {total_frames} frames")
    for e in timeline:
        print(f"        [{e['start_s']:.2f}→{e['end_s']:.2f}s] "
              f"{e['speaker']}: \"{e['text'][:45]}{'…' if len(e['text'])>45 else ''}\"")

    # ── Step 2: Load portraits ────────────────────────────────
    portraits  = [(name, _load_portrait(name, char_img_map)) for name in characters]
    cx_pos     = _char_cx_positions(len(portraits))

    # Map character name → index
    char_index = {name: i for i, (name, _) in enumerate(portraits)}

    # ── Step 3: Pre-extract amplitudes per timeline entry ──────
    for entry in timeline:
        entry["amps"] = _extract_amps(entry["audio_path"], FPS)

    # ── Step 4: Open background video ────────────────────────
    bg = _BgReader(bg_video)

    # ── Step 5: Open ffmpeg pipe ──────────────────────────────
    tmp_video = f"outputs/final_scenes/scene_{scene_id:02d}_noaudio.mp4"
    pipe_cmd  = [
        FFMPEG, "-y",
        "-f",       "rawvideo",
        "-pix_fmt", "bgr24",
        "-s",       f"{VIDEO_W}x{VIDEO_H}",
        "-r",       str(FPS),
        "-i",       "pipe:0",
        "-c:v",     "libx264",
        "-pix_fmt", "yuv420p",
        "-crf",     "18",
        "-preset",  "fast",
        tmp_video
    ]
    proc = subprocess.Popen(pipe_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # ── Step 6: Render each frame ─────────────────────────────
    for fi in range(total_frames):
        t_sec = fi / FPS

        # Find active timeline entry
        active = None
        for entry in timeline:
            if entry["frame_start"] <= fi < entry["frame_end"]:
                active = entry
                break
        if active is None:
            active = timeline[-1]   # last entry covers any remainder

        # Speaking character index
        speaker_name = active["speaker"]
        speaking_ci  = char_index.get(speaker_name, 0)

        # Amplitude for this frame (relative position within the entry)
        amps    = active.get("amps", np.array([]))
        local_f = fi - active["frame_start"]
        amp     = float(amps[local_f]) if local_f < len(amps) else 0.0

        # Subtitle = "Speaker: line"
        subtitle = f'{active["speaker"]}: "{active["text"]}"'

        # Background frame (loops if BG is shorter)
        bg_frame = bg.read(fi)

        # Compose frame
        frame = _compose_frame(
            bg_frame    = bg_frame,
            portraits   = portraits,
            cx_positions= cx_pos,
            speaking_ci = speaking_ci,
            amp         = amp,
            subtitle    = subtitle,
            title_text  = None,
            show_title  = fi < 20,
        )

        proc.stdin.write(frame.tobytes())

    # ── Finish video pipe ─────────────────────────────────────
    proc.stdin.close()
    proc.wait()
    bg.close()

    # ── Step 7: Validate or create merged audio ───────────────
    if not merged_audio or not os.path.exists(merged_audio):
        merged_audio = _build_merged_audio(audio_files, scene_id, total_dur)

    # ── Step 8: Extend video to match audio if needed ─────────
    audio_dur = _wav_duration(merged_audio) if merged_audio else total_dur
    vid_dur   = _get_video_duration(tmp_video)

    if audio_dur > vid_dur + 0.1:
        ext = f"outputs/final_scenes/scene_{scene_id:02d}_ext.mp4"
        subprocess.run([
            FFMPEG, "-y", "-i", tmp_video,
            "-vf", f"tpad=stop_mode=clone:stop_duration={audio_dur-vid_dur:.3f}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            ext
        ], capture_output=True)
        tmp_video = ext

    # ── Step 9: Mux video + merged audio ─────────────────────
    mux_cmd = [
        FFMPEG, "-y",
        "-i", tmp_video,
        "-i", merged_audio,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-preset", "fast",
        "-shortest",
        output_path
    ]
    res = subprocess.run(mux_cmd, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(f"[Composer] Mux failed: {res.stderr.decode()[:300]}")

    # Cleanup temp file
    for f in [f"outputs/final_scenes/scene_{scene_id:02d}_noaudio.mp4",
              f"outputs/final_scenes/scene_{scene_id:02d}_ext.mp4"]:
        if os.path.exists(f) and f != output_path:
            try: os.remove(f)
            except Exception: pass

    return output_path


# =============================================================================
# HELPERS
# =============================================================================
def _get_video_duration(path):
    try:
        probe = subprocess.run([FFMPEG, "-i", path], capture_output=True, text=True)
        for line in probe.stderr.splitlines():
            if "Duration" in line:
                p = line.strip().split("Duration:")[1].split(",")[0].strip()
                h, m, s = p.split(":")
                return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        pass
    return 0.0


def _build_merged_audio(wav_paths, scene_id, total_dur_s):
    """Build merged WAV if it doesn't exist yet."""
    import numpy as _np
    out_path = f"outputs/audio/scene_{scene_id:02d}_merged.wav"
    if os.path.exists(out_path):
        return out_path

    valid = [p for p in wav_paths
             if p and os.path.exists(p) and os.path.getsize(p) > 44]
    if not valid:
        # Write silence
        n = int(22050 * max(total_dur_s, 2.0))
        with wave.open(out_path, "w") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(22050)
            wf.writeframes(b"\x00" * n * 2)
        return out_path

    gap = _np.zeros(int(22050 * 0.18), dtype=_np.float32)
    parts = []
    for i, wp in enumerate(valid):
        try:
            with wave.open(wp, "r") as wf:
                sr  = wf.getframerate()
                ch  = wf.getnchannels()
                raw = wf.readframes(wf.getnframes())
            smp = _np.frombuffer(raw, dtype=_np.int16).astype(_np.float32) / 32767.0
            if ch == 2: smp = smp.reshape(-1,2).mean(axis=1)
            pk = abs(smp).max()
            if pk > 1e-6: smp = smp / pk * 0.88
            parts.append(smp.astype(_np.float32))
            if i < len(valid) - 1: parts.append(gap)
        except Exception as e:
            print(f"      [AudioMerge] Skipped {wp}: {e}")

    if not parts:
        n = int(22050 * max(total_dur_s, 2.0))
        with wave.open(out_path, "w") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(22050)
            wf.writeframes(b"\x00" * n * 2)
        return out_path

    merged = _np.clip(_np.concatenate(parts), -1.0, 1.0)
    i16    = (merged * 32767).astype(_np.int16)
    with wave.open(out_path, "w") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(22050)
        wf.writeframes(i16.tobytes())
    return out_path