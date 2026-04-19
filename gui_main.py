import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import json
import os
import threading
from workflows.langgraph_flow import build_graph

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

ACCENT   = "#7C6FF7"
ACCENT2  = "#5A54C4"
SUCCESS  = "#2ECC71"
WARNING  = "#F39C12"
DANGER   = "#E74C3C"
BG_DARK  = "#0F0F13"
BG_CARD  = "#1A1A24"
BG_INPUT = "#13131A"
BORDER   = "#2A2A3A"
TEXT     = "#E8E8F0"
MUTED    = "#7A7A9A"


class MontageApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("PROJECT MONTAGE — Phase 1")
        self.geometry("960x780")
        self.minsize(800, 680)
        self.configure(fg_color=BG_DARK)

        self._graph = None
        self._hitl_event = threading.Event()
        self._hitl_approved = False

        self._build_ui()

    # ─────────────────────────────────────────────────────────
    # UI CONSTRUCTION
    # ─────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Top bar ──────────────────────────────────────────
        top = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=0, height=64)
        top.pack(fill="x")
        top.pack_propagate(False)

        ctk.CTkLabel(
            top,
            text="◈  PROJECT MONTAGE",
            font=ctk.CTkFont("Courier", 22, "bold"),
            text_color=ACCENT
        ).pack(side="left", padx=24, pady=16)

        ctk.CTkLabel(
            top,
            text="Phase 1 · Writer's Room",
            font=ctk.CTkFont(size=13),
            text_color=MUTED
        ).pack(side="left", pady=16)

        # ── Main content area ─────────────────────────────────
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=20, pady=16)
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=1)
        content.grid_rowconfigure(1, weight=1)

        # ── Left panel: config ────────────────────────────────
        left = ctk.CTkFrame(content, fg_color=BG_CARD, corner_radius=12)
        left.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 8))

        self._build_left_panel(left)

        # ── Right panel: input + log ──────────────────────────
        right = ctk.CTkFrame(content, fg_color="transparent")
        right.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=(8, 0))
        right.grid_rowconfigure(0, weight=2)
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self._build_input_panel(right)
        self._build_log_panel(right)

        # ── Bottom bar ────────────────────────────────────────
        self._build_bottom_bar()

    def _build_left_panel(self, parent):
        parent.grid_columnconfigure(0, weight=1)

        # Section: Mode
        self._section_label(parent, "01  MODE", row=0)

        self.mode_var = tk.StringVar(value="auto")

        mode_auto = ctk.CTkRadioButton(
            parent, text="Auto — generate from prompt",
            variable=self.mode_var, value="auto",
            font=ctk.CTkFont(size=13),
            fg_color=ACCENT, hover_color=ACCENT2,
            command=self._on_mode_change
        )
        mode_auto.grid(row=1, column=0, sticky="w", padx=20, pady=(4, 2))

        mode_manual = ctk.CTkRadioButton(
            parent, text="Manual — paste screenplay",
            variable=self.mode_var, value="manual",
            font=ctk.CTkFont(size=13),
            fg_color=ACCENT, hover_color=ACCENT2,
            command=self._on_mode_change
        )
        mode_manual.grid(row=2, column=0, sticky="w", padx=20, pady=(2, 12))

        # Divider
        self._divider(parent, row=3)

        # Section: Options
        self._section_label(parent, "02  OPTIONS", row=4)

        self.hitl_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            parent, text="Human review checkpoint",
            variable=self.hitl_var,
            font=ctk.CTkFont(size=13),
            fg_color=ACCENT, hover_color=ACCENT2
        ).grid(row=5, column=0, sticky="w", padx=20, pady=(4, 2))

        self.save_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            parent, text="Save outputs to disk",
            variable=self.save_var,
            font=ctk.CTkFont(size=13),
            fg_color=ACCENT, hover_color=ACCENT2
        ).grid(row=6, column=0, sticky="w", padx=20, pady=(2, 12))

        # Divider
        self._divider(parent, row=7)

        # Section: Output preview
        self._section_label(parent, "03  OUTPUT FILES", row=8)

        self.file_labels = {}
        files = [
            ("script",  "scene_manifest.json", "📄"),
            ("chars",   "character_db.json",   "👥"),
            ("images",  "outputs/images/",     "🖼"),
            ("memory",  "memory_db/",          "🧠"),
        ]
        for i, (key, name, icon) in enumerate(files):
            row_frame = ctk.CTkFrame(parent, fg_color="transparent")
            row_frame.grid(row=9+i, column=0, sticky="ew", padx=20, pady=2)
            row_frame.grid_columnconfigure(1, weight=1)

            ctk.CTkLabel(row_frame, text=icon, width=24, font=ctk.CTkFont(size=14)).grid(row=0, column=0)
            lbl = ctk.CTkLabel(
                row_frame, text=name,
                font=ctk.CTkFont("Courier", 11),
                text_color=MUTED, anchor="w"
            )
            lbl.grid(row=0, column=1, sticky="w", padx=(6, 0))
            self.file_labels[key] = lbl

        # Status indicator
        self._divider(parent, row=14)

        self.status_dot = ctk.CTkLabel(parent, text="● IDLE", font=ctk.CTkFont(size=12, weight="bold"), text_color=MUTED)
        self.status_dot.grid(row=15, column=0, sticky="w", padx=20, pady=12)

    def _build_input_panel(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=12)
        frame.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 0))
        header.grid_columnconfigure(1, weight=1)

        self.input_title = ctk.CTkLabel(
            header, text="STORY PROMPT",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=MUTED
        )
        self.input_title.grid(row=0, column=0, sticky="w")

        self.char_count = ctk.CTkLabel(
            header, text="0 chars",
            font=ctk.CTkFont(size=11),
            text_color=MUTED
        )
        self.char_count.grid(row=0, column=1, sticky="e")

        self.input_box = ctk.CTkTextbox(
            frame,
            fg_color=BG_INPUT,
            border_color=BORDER,
            border_width=1,
            corner_radius=8,
            font=ctk.CTkFont(size=14),
            text_color=TEXT,
            wrap="word"
        )
        self.input_box.grid(row=1, column=0, sticky="nsew", padx=12, pady=12)
        self.input_box.bind("<KeyRelease>", self._update_char_count)

        self._set_placeholder()

    def _build_log_panel(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=12)
        frame.grid(row=1, column=0, sticky="nsew")
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            frame, text="PIPELINE LOG",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=MUTED
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(12, 0))

        self.log_box = ctk.CTkTextbox(
            frame,
            fg_color=BG_INPUT,
            border_color=BORDER,
            border_width=1,
            corner_radius=8,
            font=ctk.CTkFont("Courier", 12),
            text_color="#A0A0C0",
            state="disabled",
            wrap="word"
        )
        self.log_box.grid(row=1, column=0, sticky="nsew", padx=12, pady=12)

    def _build_bottom_bar(self):
        # Review bar (shown during HITL, hidden otherwise)
        self.review_bar = ctk.CTkFrame(self, fg_color="#1E1A10", corner_radius=0, height=56)
        self.review_bar.pack(fill="x", side="bottom")
        self.review_bar.pack_propagate(False)
        self.review_bar.pack_forget()

        # Row 1: approve / reject
        self.hitl_approve_row = ctk.CTkFrame(self.review_bar, fg_color="transparent")
        self.hitl_approve_row.pack(fill="x", padx=20, pady=(8, 0))

        ctk.CTkLabel(
            self.hitl_approve_row,
            text="Review checkpoint — approve the generated script?",
            font=ctk.CTkFont(size=13),
            text_color=WARNING
        ).pack(side="left", padx=(0, 16))

        self.approve_btn = ctk.CTkButton(
            self.hitl_approve_row, text="✓  Approve", width=110, height=32,
            fg_color=SUCCESS, hover_color="#27AE60",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._hitl_approve
        )
        self.approve_btn.pack(side="left", padx=(0, 8))

        self.reject_btn = ctk.CTkButton(
            self.hitl_approve_row, text="✕  Reject", width=110, height=32,
            fg_color=DANGER, hover_color="#C0392B",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._hitl_show_feedback
        )
        self.reject_btn.pack(side="left")

        # Row 2: feedback input (shown after reject)
        self.feedback_row = ctk.CTkFrame(self.review_bar, fg_color="transparent")

        ctk.CTkLabel(
            self.feedback_row,
            text="Feedback:",
            font=ctk.CTkFont(size=12),
            text_color=MUTED
        ).pack(side="left", padx=(0, 8))

        self.feedback_entry = ctk.CTkEntry(
            self.feedback_row,
            placeholder_text="Describe changes (leave blank to stop pipeline)...",
            width=420, height=30,
            font=ctk.CTkFont(size=12),
            fg_color=BG_INPUT, border_color=BORDER
        )
        self.feedback_entry.pack(side="left", padx=(0, 8))
        self.feedback_entry.bind("<Return>", lambda e: self._hitl_submit_feedback())

        ctk.CTkButton(
            self.feedback_row, text="Regenerate", width=110, height=30,
            fg_color=ACCENT, hover_color=ACCENT2,
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._hitl_submit_feedback
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            self.feedback_row, text="Cancel", width=70, height=30,
            fg_color="transparent", border_width=1, border_color=BORDER,
            font=ctk.CTkFont(size=12),
            command=self._hitl_cancel_feedback
        ).pack(side="left")

        # Action bar
        bar = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=0, height=72)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        self.run_btn = ctk.CTkButton(
            bar, text="▶  Run Pipeline", width=160, height=42,
            fg_color=ACCENT, hover_color=ACCENT2,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._run_pipeline
        )
        self.run_btn.pack(side="right", padx=20, pady=15)

        self.progress = ctk.CTkProgressBar(bar, width=300, height=6, fg_color=BORDER, progress_color=ACCENT)
        self.progress.pack(side="right", padx=(0, 20), pady=0)
        self.progress.set(0)

    # ─────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────
    def _section_label(self, parent, text, row):
        ctk.CTkLabel(
            parent, text=text,
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=MUTED
        ).grid(row=row, column=0, sticky="w", padx=20, pady=(16, 2))

    def _divider(self, parent, row):
        ctk.CTkFrame(parent, fg_color=BORDER, height=1, corner_radius=0).grid(
            row=row, column=0, sticky="ew", padx=20, pady=4
        )

    def _set_placeholder(self):
        placeholder = (
            "A detective receives an anonymous tip about a murder that hasn't happened yet. "
            "She has 24 hours to find the killer before the victim dies."
        )
        self.input_box.delete("1.0", "end")
        self.input_box.insert("1.0", placeholder)

    def _on_mode_change(self):
        mode = self.mode_var.get()
        if mode == "auto":
            self.input_title.configure(text="STORY PROMPT")
            self._set_placeholder()
        else:
            self.input_title.configure(text="SCREENPLAY SCRIPT")
            self.input_box.delete("1.0", "end")
            self.input_box.insert("1.0",
                "Scene 1 - INT. DETECTIVE'S OFFICE - NIGHT\n\n"
                "DETECTIVE: Something feels off about this case.\n\n"
                "Scene 2 - EXT. WAREHOUSE - MIDNIGHT\n\n"
                "DETECTIVE: Show yourself!\n"
            )
        self._update_char_count()

    def _update_char_count(self, *_):
        text = self.input_box.get("1.0", "end-1c")
        self.char_count.configure(text=f"{len(text)} chars")

    def _log(self, msg, color=None):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.configure(state="disabled")
        self.log_box.see("end")

    def _set_status(self, text, color=MUTED):
        self.status_dot.configure(text=f"● {text}", text_color=color)

    def _mark_file_done(self, key):
        if key in self.file_labels:
            self.file_labels[key].configure(text_color=SUCCESS)

    # ─────────────────────────────────────────────────────────
    # PIPELINE
    # ─────────────────────────────────────────────────────────
    def _run_pipeline(self):
        user_input = self.input_box.get("1.0", "end-1c").strip()
        if not user_input:
            messagebox.showerror("Error", "Please enter a prompt or script.")
            return

        # Reset UI
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")
        for lbl in self.file_labels.values():
            lbl.configure(text_color=MUTED)
        self.progress.set(0)
        self.run_btn.configure(state="disabled", text="Running…")
        self._set_status("RUNNING", ACCENT)

        state = {
            "mode":      self.mode_var.get(),
            "input":     user_input,
            "_gui_log":  self._log,
            "_gui_hitl": self._gui_hitl_checkpoint,
            "_gui_mark": self._mark_file_done,
            "_gui_prog": self._set_progress,
        }

        threading.Thread(target=self._pipeline_thread, args=(state,), daemon=True).start()

    def _pipeline_thread(self, state):
        try:
            graph = build_graph()
            result = graph.invoke(state)

            if self.save_var.get():
                os.makedirs("outputs/images", exist_ok=True)
                os.makedirs("memory_db", exist_ok=True)

                with open("outputs/scene_manifest.json", "w", encoding="utf-8") as f:
                    json.dump(result.get("scene_manifest", {}), f, indent=2)
                with open("outputs/character_db.json", "w", encoding="utf-8") as f:
                    json.dump(result.get("character_db", []), f, indent=2)

            self.after(0, self._on_success)

        except SystemExit as e:
            self.after(0, lambda: self._on_hitl_rejected(str(e)))
        except Exception as e:
            self.after(0, lambda: self._on_error(str(e)))

    def _set_progress(self, val):
        self.after(0, lambda: self.progress.set(val))

    # ─────────────────────────────────────────────────────────
    # HITL — called from agent thread, blocks until GUI responds
    # Returns: "approved" | "regenerate:<feedback>" | "halt"
    # ─────────────────────────────────────────────────────────
    def _gui_hitl_checkpoint(self, script_json):
        self._hitl_event.clear()
        self._hitl_result = None
        self.after(0, lambda: self._show_hitl(script_json))
        self._hitl_event.wait()
        return self._hitl_result   # "approved" | "regenerate:<fb>" | "halt"

    def _show_hitl(self, script_json):
        self._log("\n── HUMAN REVIEW CHECKPOINT ──────────────────", "#F39C12")
        self._log(json.dumps(script_json, indent=2), "#8888AA")
        self._log("─────────────────────────────────────────────\n", "#F39C12")
        self.review_bar.pack(fill="x", side="bottom", before=self.run_btn.master)
        self.review_bar.pack(fill="x", side="bottom")
        self.review_bar.lift()

    def _hitl_approve(self):
        self._log("[HITL] ✓ Script approved. Continuing pipeline...\n", SUCCESS)
        self.review_bar.pack_forget()
        self.feedback_row.pack_forget()
        self._hitl_result = "approved"
        self._hitl_event.set()

    def _hitl_show_feedback(self):
        """Expand review bar to show feedback row."""
        self.review_bar.configure(height=96)
        self.feedback_row.pack(fill="x", padx=20, pady=(4, 8))
        self.feedback_entry.delete(0, "end")
        self.feedback_entry.focus()

    def _hitl_submit_feedback(self):
        feedback = self.feedback_entry.get().strip()
        self.review_bar.pack_forget()
        self.feedback_row.pack_forget()
        self.review_bar.configure(height=56)
        if feedback:
            self._log(f"[HITL] ✕ Rejected. Regenerating with feedback: \"{feedback}\"\n", WARNING)
            self._hitl_result = f"regenerate:{feedback}"
        else:
            self._log("[HITL] ✕ Rejected with no feedback. Halting pipeline.\n", DANGER)
            self._hitl_result = "halt"
        self._hitl_event.set()

    def _hitl_cancel_feedback(self):
        """Go back to approve/reject view without submitting."""
        self.feedback_row.pack_forget()
        self.review_bar.configure(height=56)

    # ─────────────────────────────────────────────────────────
    # PIPELINE OUTCOMES
    # ─────────────────────────────────────────────────────────
    def _on_success(self):
        self.progress.set(1)
        self._set_status("COMPLETE", SUCCESS)
        self.run_btn.configure(state="normal", text="▶  Run Pipeline")
        for key in self.file_labels:
            self._mark_file_done(key)
        self._log("\n══════════════════════════════════════════════", SUCCESS)
        self._log("  ✓ Phase 1 completed successfully!", SUCCESS)
        self._log("══════════════════════════════════════════════\n", SUCCESS)

    def _on_error(self, msg):
        self.progress.set(0)
        self._set_status("ERROR", DANGER)
        self.run_btn.configure(state="normal", text="▶  Run Pipeline")
        self._log(f"\n✕ Error: {msg}\n", DANGER)
        messagebox.showerror("Pipeline Error", msg)

    def _on_hitl_rejected(self, msg):
        self.progress.set(0)
        self._set_status("HALTED", WARNING)
        self.run_btn.configure(state="normal", text="▶  Run Pipeline")


# ─────────────────────────────────────────────────────────────
# PATCH: make agents GUI-aware
# ─────────────────────────────────────────────────────────────

def _patch_agents():
    """
    Monkey-patch the agents so they use GUI callbacks when available,
    falling back to terminal I/O otherwise.
    """
    import agents.hitl as hitl_mod
    import agents.scriptwriter as sw_mod
    import agents.character_designer as cd_mod
    import agents.image_synthesizer as is_mod
    import agents.memory_agent as ma_mod

    _orig_hitl = hitl_mod.human_checkpoint
    def _gui_hitl(state):
        cb = state.get("_gui_hitl")
        log = state.get("_gui_log")
        prog = state.get("_gui_prog")
        if prog: prog(0.35)
        if not cb:
            return _orig_hitl(state)

        # GUI feedback loop
        from agents.hitl import _regenerate_with_feedback
        while True:
            result = cb(state.get("scene_manifest", {}))

            if result == "approved":
                return state

            elif result == "halt":
                raise SystemExit("Pipeline halted — script rejected with no feedback.")

            elif result and result.startswith("regenerate:"):
                feedback = result[len("regenerate:"):]
                if log: log(f"\n[Scriptwriter] Regenerating with feedback: \"{feedback}\"...")
                if prog: prog(0.2)
                state = _regenerate_with_feedback(state, feedback)
                n = len(state.get("scene_manifest", {}).get("scenes", []))
                if log: log(f"[Scriptwriter] ✓ Regenerated {n} scenes. Returning to review...")
                if prog: prog(0.35)
                # loop — show new script to user again

            else:
                raise SystemExit("Pipeline halted.")

    hitl_mod.human_checkpoint = _gui_hitl

    _orig_sw = sw_mod.scriptwriter
    def _gui_sw(state):
        log = state.get("_gui_log")
        prog = state.get("_gui_prog")
        if log: log("[Scriptwriter] Generating script via Groq…")
        if prog: prog(0.1)
        result = _orig_sw(state)
        n = len(result.get("scene_manifest", {}).get("scenes", []))
        if log: log(f"[Scriptwriter] ✓ Generated {n} scenes.")
        if prog: prog(0.25)
        return result
    sw_mod.scriptwriter = _gui_sw

    _orig_cd = cd_mod.character_designer
    def _gui_cd(state):
        log = state.get("_gui_log")
        prog = state.get("_gui_prog")
        if log: log("\n[Character Designer] Extracting character identities…")
        if prog: prog(0.5)
        result = _orig_cd(state)
        n = len(result.get("character_db", []))
        if log: log(f"[Character Designer] ✓ {n} characters processed.")
        if prog: prog(0.65)
        return result
    cd_mod.character_designer = _gui_cd

    _orig_is = is_mod.image_generator
    def _gui_is(state):
        log = state.get("_gui_log")
        prog = state.get("_gui_prog")
        mark = state.get("_gui_mark")
        if log: log("\n[Image Synthesizer] Generating character visuals…")
        if prog: prog(0.7)
        result = _orig_is(state)
        n = len(result.get("images", []))
        if log: log(f"[Image Synthesizer] ✓ {n} images saved.")
        if prog: prog(0.9)
        if mark: mark("images")
        return result
    is_mod.image_generator = _gui_is

    _orig_ma = ma_mod.memory_commit
    def _gui_ma(state):
        log = state.get("_gui_log")
        mark = state.get("_gui_mark")
        prog = state.get("_gui_prog")
        if log: log("\n[Memory] Committing state to memory store…")
        result = _orig_ma(state)
        if log: log("[Memory] ✓ State saved to memory_db/")
        if prog: prog(0.98)
        if mark:
            mark("script"); mark("chars"); mark("memory")
        return result
    ma_mod.memory_commit = _gui_ma


if __name__ == "__main__":
    _patch_agents()
    app = MontageApp()
    app.mainloop()