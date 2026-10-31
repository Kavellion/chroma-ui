"""Chroma UI -- text-to-image only, for lodestones/Chroma1-HD.

Same shape as the Kandinsky, Sana, Z-Image, SD 3.5, PixArt, and
FLUX.1 UIs in this family: no Jupyter, no Colab. Run it, a page
opens in your browser, type a prompt, hit Generate, a picture
appears underneath. Chroma1-HD is an 8.9B text-to-image model
built from FLUX.1 [schnell] and heavily reworked by its makers:
the giant timestep-embedding layer FLUX carries was replaced by
a small guidance approximator, CLIP was dropped entirely (T5-XXL
is the only text encoder), and -- unlike schnell -- true
classifier-free guidance was trained back in, so the negative
prompt genuinely works. The model card's own example runs
40 steps at guidance 3.0, and this UI defaults to exactly that.

The model is Apache 2.0 and UNGATED. There is no license to
accept and no token anywhere in this app: first Generate just
downloads and loads.

How to run (on a Windows PC with an NVIDIA GPU):

    install.bat      (once -- sets up pip in the bundled Python
                      3.12 and installs torch from cu130 + the
                      packages)
    run.bat          (starts the app)

Python is INCLUDED in this package, in the python_embeded folder
(Python 3.12, ComfyUI portable style) -- there is nothing to
install first and no Python on your system is touched or needed.
Python 3.12 is required because Diffusers 0.42 dropped Python
3.10.

The app file lives in the chroma-ui/ subfolder; run.bat runs it
from the folder above, so pictures land in images_out/ next to
run.bat -- one subfolder per batch name, reused across runs,
frames numbered continuously, and settings.txt appended once
per run, exactly like the other apps in the family.

First Generate takes a while: the model downloads from Hugging
Face (about 17.3 GB on the default quantized path) and loads
once. Repeat generations reuse what is already loaded.

How it works (Diffusers 0.42.0, ChromaPipeline -- call shapes
verified against the installed 0.42.0 source and the model
card):
  1. The transformer loads from a quantized GGUF file
     (silveroxides/Chroma1-HD-GGUF) through
     ChromaTransformer2DModel.from_single_file with a
     GGUFQuantizationConfig -- the exact pattern in the
     diffusers GGUF documentation, and the same loader shape
     the FLUX.1 [dev] and [schnell] UIs already proved on this
     PC. The text encoder (T5-XXL), the VAE, the tokenizer,
     and the scheduler load from lodestones/Chroma1-HD with
     the pipeline; the 17.8 GB BF16 transformer shards in that
     repo are never downloaded, because the pipeline receives
     the GGUF transformer instead.
  2. The call passes num_inference_steps, guidance_scale,
     negative_prompt, max_sequence_length=512, and a CPU
     generator. The pipeline class defaults in 0.42.0 are
     35 steps / guidance 5.0; this app passes the model card's
     40 / 3.0 explicitly on every call, the same way the
     schnell UI passes its card values over the Flux class
     defaults.

Fitting on a 12 GB card (RTX 4070, 16 GB system RAM):
  - BF16 weights total about 27.5 GB (transformer 17.80,
    T5-XXL encoder 9.52, VAE 0.17). The transformer alone is
    bigger than the whole 12 GB card, and whole-model CPU
    offload cannot hold a stage that does not fit the GPU at
    all, so full BF16 is NOT offered -- not even as an explicit
    choice. The quantized path is the only path.
  - The DEFAULT is the GGUF Q6_K transformer (7.65 GB).
    Chroma's transformer is 8.9B against FLUX's 12B, so one
    rung above the FLUX default fits the same envelope.
    Unique weight bytes on the default path total about
    17.3 GB, split across VRAM and RAM by CPU offload: at any
    moment only one stage's weights sit on the GPU. While
    denoising, the T5 encoder parks in RAM (about 9.5 GB, the
    same T5 parking this PC already runs for SD 3.5 and
    FLUX.1 [dev]); while encoding, the transformer parks in
    RAM instead.
  - Q5_K_S (6.51 GB) and Q4_K_S (5.43 GB) are offered as
    lighter choices with more headroom and a little less
    detail. Q8_0 (9.74 GB) is NOT offered: weights plus
    activations plus dequantization scratch would crowd a
    12 GB card.
  - CPU offload defaults ON and VAE tiling is enabled, both
    standard diffusers memory savers for this pipeline.
  - If a chosen GGUF load fails, the loader falls back to the
    Q4_K_S file with offload forced ON and says so in the
    status. It never falls back to full BF16 -- there is no
    BF16 path in this app at all.

About the negative prompt: Chroma has REAL classifier-free
guidance. The pipeline runs the CFG pass whenever guidance is
above 1, so at the default 3.0 the negative prompt genuinely
steers. Set guidance to 1 or below and the negative has no
effect (the pipeline skips the second pass) -- the app passes
the box through either way and settings.txt records what was
typed.

About prompt length: there is no CLIP in this model, so there
is no 77-token wall. T5 hears the full prompt, up to the
pipeline's max_sequence_length of 512 tokens, which this app
passes on every call.

About the model's character: Chroma1-HD is intentionally a
BASE model -- its makers describe it as a strong, neutral
foundation meant as a starting point for finetuning. Expect
base-model manners (a notch less finish and prompt polish than
a tuned release); that is the model, not a loader problem.

No LoRA slots in v1, matching the FLUX UIs: LoRA-on-GGUF under
CPU offload is not a verified combination in this build, so v1
leaves them out rather than shipping a slot that might silently
do nothing. No animation, no upscaling, no editing, no
ControlNet -- v1 is text-to-image only, kept simple on purpose.

STATUS: CODE VERIFIED ONLY. This build has not run on a real
PC yet. See README.md.
"""

import gc
import os
import random
import re
import shutil
from datetime import datetime

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)  # folder holding run.bat
IMAGES_DIR = os.path.join(ROOT_DIR, "images_out")
FAVORITES_DIR = os.path.join(IMAGES_DIR, "favorites")

MODEL_ID = "lodestones/Chroma1-HD"
GGUF_REPO = "silveroxides/Chroma1-HD-GGUF"
# Every filename below was verified live in the GGUF repo
# listing on 2026-10-31 (file sizes in BUILD-NOTES.txt). The
# loader only ever asks for a name from this table.
GGUF_FILES = {
    "gguf-q6": "Chroma1-HD-Q6_K.gguf",
    "gguf-q5": "Chroma1-HD-Q5_K_S.gguf",
    "gguf-q4": "Chroma1-HD-Q4_K_S.gguf",
}
GGUF_FALLBACK_MODE = "gguf-q4"

# The model card's recommended values for Chroma1-HD:
# 40 steps, guidance 3.0. The real ChromaPipeline.__call__ in
# Diffusers 0.42.0 defaults to 35 steps / guidance 5.0 /
# max_sequence_length 512; the app passes the card values
# explicitly on every call.
DEFAULT_STEPS = 40
DEFAULT_GUIDANCE = 3.0
MAX_SEQUENCE_LENGTH = 512

DEFAULT_PROMPT = (
    "A lighthouse on a rocky cliff during a storm, waves "
    "crashing, dramatic sky, detailed"
)

_pipe = None
_pipe_key = None  # (precision, offload)
_pipe_note = ""


def _safe_batch_name(name):
    """Strip characters Windows forbids in folder names. Falls back
    to a timestamped name when the box is empty."""
    name = (name or "").strip()
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = name.strip(" .")
    if not name:
        name = datetime.now().strftime("chroma-%Y%m%d-%H%M%S")
    return name


def _batch_folder(base):
    """The one folder for a batch name: images_out/<base>, created
    if missing, reused if it already exists. Repeat runs never get
    a _2/_3 folder -- frames just keep numbering upward."""
    return os.path.join(IMAGES_DIR, base)


def _unique_folder(base, root=IMAGES_DIR):
    """<root>/<base>, or <base>_2, <base>_3... if it already exists.
    Used only by Star batch; generations reuse one folder per batch
    name (see _batch_folder) instead of numbered folders."""
    folder = os.path.join(root, base)
    if not os.path.exists(folder):
        return folder
    i = 2
    while True:
        candidate = "{}_{}".format(folder, i)
        if not os.path.exists(candidate):
            return candidate
        i += 1


_FRAME_RE = re.compile(r"^(\d{6})\.png$")


def _next_frame_number(folder):
    """First unused frame number in a folder: one past the highest
    existing 000001-style frame, or 1 for a fresh folder."""
    highest = 0
    if os.path.isdir(folder):
        for name in os.listdir(folder):
            match = _FRAME_RE.match(name)
            if match:
                highest = max(highest, int(match.group(1)))
    return highest + 1


def _frame_range_text(start, count):
    """'frame 000006' for one image, 'frames 000006-000008' for
    several -- used in status lines and settings headers."""
    if count == 1:
        return "frame {:06d}".format(start)
    return "frames {:06d}-{:06d}".format(start, start + count - 1)


def _seeds_for(seed, count):
    """Resolve the seed box to one seed per image. -1 (or empty)
    draws a fresh random seed per image; a fixed seed S gives S,
    S+1, S+2... so a multi-image run with a fixed seed stays
    reproducible."""
    if seed is None:
        seed = -1
    seed = int(seed)
    if seed < 0:
        return [random.randint(0, 2**31 - 1) for _ in range(count)]
    return [seed + i for i in range(count)]


# Dynamic prompts: single level only, one random pick per group.
# "{a|b|c}" becomes a, b, or c. Groups do not nest; a group with no
# pipe is just its own text with the braces removed.
_TEMPLATE_RE = re.compile(r"\{([^{}]*)\}")


def _expand_templates(prompt):
    """Expand {a|b|c} groups, one random pick each. Plain prompts
    (no braces) come back unchanged."""
    return _TEMPLATE_RE.sub(
        lambda match: random.choice(match.group(1).split("|")), prompt
    )


def _append_settings(folder, header, entries):
    """Append one run's block to settings.txt: a header line (date,
    time, frame range) followed by the run's 'key: value' lines and
    a blank line. Creates the file on first use; earlier runs'
    blocks are never touched. A missing trailing newline is
    repaired so blocks never run together."""
    path = os.path.join(folder, "settings.txt")
    prefix = ""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as handle:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                prefix = "\n"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(prefix + header + "\n")
        handle.write("\n".join(entries) + "\n\n")


def _run_header(start, count):
    """The settings.txt header for one run: date, time, and the
    frame-number range it produced."""
    return "Run {}, {}".format(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        _frame_range_text(start, count))


def _seed_entry(seeds):
    """The settings.txt seed line: 'seed: X' for one image,
    'seeds: X, Y' for several."""
    if len(seeds) == 1:
        return "seed: {}".format(seeds[0])
    return "seeds: {}".format(", ".join(str(s) for s in seeds))


def _expanded_entries(expanded_list):
    """The settings.txt expanded-prompt line(s): one line for one
    image, a comma-joined list for several."""
    if len(expanded_list) == 1:
        return ["expanded_prompt: {}".format(expanded_list[0])]
    return ["expanded_prompts: {}".format(", ".join(expanded_list))]


def _round16(value):
    """Nearest multiple of 16, at least 16. The real Chroma
    pipeline only warns and resizes when a size is not divisible
    by 16 (vae_scale_factor 8 x 2); rounding here keeps sizes
    exact and the warning unreachable from the UI."""
    return max(16, int(round(int(value) / 16.0)) * 16)


def _free_memory():
    """Drop cached GPU memory after unloading a model. Best effort:
    never raises, so an unload step cannot break the next load."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 -- freeing memory must not fail
        pass


def _unload_pipe():
    """Drop the loaded pipeline so the next Generate loads fresh
    (used when precision/offload change, and by Unload)."""
    global _pipe, _pipe_key, _pipe_note
    _pipe = None
    _pipe_key = None
    _pipe_note = ""
    _free_memory()


def _load_pipe(precision, offload):
    """Load (or reuse) the Chroma1-HD pipeline. Returns
    (pipe, note). Precision is one of the GGUF_FILES keys
    (default "gguf-q6"). Offload True uses the standard
    enable_model_cpu_offload(); False loads everything onto
    the GPU. There is no token and no BF16 path: the model is
    ungated, and the BF16 transformer (17.8 GB) cannot fit a
    12 GB card even with whole-model offload.

    If the chosen GGUF load fails, this falls back to the
    Q4_K_S file WITH offload forced on and the note says so
    plainly. If even that fails, it raises a RuntimeError with
    a plain sentence for the status box."""
    global _pipe, _pipe_key, _pipe_note
    key = (precision, bool(offload))
    if _pipe is not None and _pipe_key == key:
        return _pipe, _pipe_note
    _unload_pipe()

    import torch
    from diffusers import (
        ChromaPipeline,
        ChromaTransformer2DModel,
        GGUFQuantizationConfig,
    )

    note = ""
    pipe = None

    def _gguf_pipe(mode):
        from huggingface_hub import hf_hub_download
        picked = GGUF_FILES[mode]
        local = hf_hub_download(GGUF_REPO, filename=picked)
        transformer = ChromaTransformer2DModel.from_single_file(
            local,
            config=MODEL_ID,
            subfolder="transformer",
            quantization_config=GGUFQuantizationConfig(
                compute_dtype=torch.bfloat16),
            dtype=torch.bfloat16)
        loaded = ChromaPipeline.from_pretrained(
            MODEL_ID, transformer=transformer,
            dtype=torch.bfloat16)
        return loaded, picked

    if precision in GGUF_FILES:
        try:
            pipe, picked = _gguf_pipe(precision)
            note = "Quantized transformer: {} from {}.".format(
                picked, GGUF_REPO)
        except Exception as exc:  # noqa: BLE001 -- fall back, loudly
            if precision != GGUF_FALLBACK_MODE:
                try:
                    pipe, picked = _gguf_pipe(GGUF_FALLBACK_MODE)
                    note = (
                        "Quantized load of {} failed ({}); fell "
                        "back to {} with CPU offload on.".format(
                            GGUF_FILES[precision], exc, picked))
                    offload = True
                    key = (precision, True)
                except Exception:  # noqa: BLE001 -- report below
                    pipe = None
            if pipe is None:
                raise RuntimeError(
                    "Could not load Chroma1-HD ({}). The model "
                    "is ungated, so this is not a license or "
                    "token problem -- check the internet "
                    "connection and try Generate again.".format(exc))
    else:
        raise RuntimeError(
            "Unknown precision mode: {}.".format(precision))
    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    try:
        pipe.vae.enable_tiling()
    except Exception:  # noqa: BLE001 -- tiling is an optimization
        pass
    _pipe = pipe
    _pipe_key = key
    _pipe_note = note
    return pipe, note


def _history_add(history, prompt):
    """Add a prompt to the session history (most recent last, no
    duplicates, last 10 kept). Returns the new history list."""
    history = [item for item in (history or []) if item != prompt]
    history.append(prompt)
    return history[-10:]


def _history_update(history):
    """The history dropdown update: choices are the history, most
    recent first, value cleared so picking the same prompt twice
    still fires."""
    import gradio as gr
    return gr.update(choices=list(reversed(history or [])), value=None)


def _early(message, last_folder, history):
    """A single yield for an early exit (bad input, load failure):
    empty gallery, the message as status, history untouched."""
    yield [], message, _history_update(history), last_folder, history


def _preset_size(preset):
    """Size preset -> (width, height). Custom leaves the boxes
    alone (returns them unchanged)."""
    import gradio as gr
    sizes = {
        "Square 1024x1024": (1024, 1024),
        "Portrait 768x1024": (768, 1024),
        "Landscape 1024x768": (1024, 768),
        "Small square 768x768": (768, 768),
    }
    if preset in sizes:
        return sizes[preset]
    return gr.update(), gr.update()


def _noop():
    """Stop-button target: cancels the Generate event, does
    nothing itself."""
    return None


_SUBJECTS = [
    "a lighthouse", "a fox", "a mountain cabin", "a castle",
    "a harbor town", "a red panda", "a sailing ship",
    "a robot gardener", "a treehouse village", "a steam train",
]
_SETTINGS = [
    "a rocky cliff", "a misty forest", "desert dunes",
    "a snowy village", "a neon city street", "a quiet lake",
    "a sunflower field", "a space station", "a coral reef",
    "an old library",
]
_LIGHTS = [
    "golden hour light", "storm light", "moonlight",
    "neon dusk", "soft morning fog", "candlelight",
    "sunset clouds", "starlight",
]
_STYLES = [
    "photorealistic", "oil painting", "anime style",
    "watercolor", "cyberpunk", "storybook illustration",
]


def random_prompt():
    """One random prompt from the built-in word banks."""
    return "A {} in {}, {}, {}, detailed.".format(
        random.choice(_SUBJECTS),
        random.choice(_SETTINGS),
        random.choice(_LIGHTS),
        random.choice(_STYLES),
    )


def _precision_mode(choice):
    """Map a Precision radio label to a loader mode key. Order
    matters: check the specific GGUF labels before the Q6
    default."""
    text = str(choice or "")
    if text.startswith("Quantized GGUF Q5"):
        return "gguf-q5"
    if text.startswith("Quantized GGUF Q4"):
        return "gguf-q4"
    return "gguf-q6"


def generate(prompt_text, negative_text, queue_text, batch_name,
             width, height, steps, guidance, seed, count,
             precision_choice, offload,
             last_folder, history):
    """Text-to-image generation. A generator function: it yields
    (gallery images, status, history dropdown update, last batch
    folder, history list) after every image, so the page updates
    live, and the Stop button can cancel it mid-run.

    Queue beats the single prompt: when the queue box has lines,
    each line is one job; otherwise the prompt box is the one job.
    Every job renders `count` images. Dynamics {a|b|c} are
    re-expanded for every single image -- in the negative prompt
    as well as the prompt -- so no two images in a run share an
    expansion by accident. All jobs in one click share one batch
    folder, frames numbered continuously.

    Steps and guidance are the live slider values, passed to the
    pipeline on every call along with max_sequence_length=512.
    The negative prompt is passed as None when the box is empty
    (the pipeline's own default; Chroma turns it into an empty
    string for the CFG pass, which runs whenever guidance is
    above 1). Memory is freed after every image, not just at
    unload, so a long queue does not creep upward."""
    import torch

    count = int(count) if count else 1
    count = max(1, min(10, count))
    history = history or []

    try:
        steps = int(steps)
    except (TypeError, ValueError):
        steps = DEFAULT_STEPS
    steps = max(1, min(100, steps))
    try:
        guidance = float(guidance)
    except (TypeError, ValueError):
        guidance = DEFAULT_GUIDANCE

    lines = [line.strip() for line in (queue_text or "").splitlines()
             if line.strip()]
    if lines:
        templates = lines
    elif prompt_text and prompt_text.strip():
        templates = [prompt_text.strip()]
    else:
        yield from _early("Type a prompt first, or fill the queue.",
                          last_folder, history)
        return

    negative_template = (negative_text or "").strip()

    try:
        width = _round16(max(16, min(2048, int(width))))
        height = _round16(max(16, min(2048, int(height))))
    except (TypeError, ValueError):
        yield from _early("Width and height must be numbers.",
                          last_folder, history)
        return

    precision = _precision_mode(precision_choice)

    base = _safe_batch_name(batch_name)
    folder = _batch_folder(base)
    os.makedirs(folder, exist_ok=True)

    try:
        pipe, note = _load_pipe(precision, bool(offload))
    except RuntimeError as exc:
        yield from _early(str(exc), last_folder, history)
        return

    gallery = []
    status = ""
    for template in templates:
        start = _next_frame_number(folder)
        seeds = _seeds_for(seed, count)
        expanded_list = []
        expanded_negatives = []
        for i in range(count):
            expanded = _expand_templates(template)
            expanded_list.append(expanded)
            negative = _expand_templates(negative_template)
            expanded_negatives.append(negative)
            try:
                image = pipe(
                    prompt=expanded,
                    negative_prompt=negative if negative else None,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    max_sequence_length=MAX_SEQUENCE_LENGTH,
                    generator=torch.Generator("cpu").manual_seed(
                        seeds[i]),
                ).images[0]
            except Exception as exc:  # noqa: BLE001 -- plain status
                yield (gallery,
                       "Generation failed on {}: {}".format(
                           _frame_range_text(start + i, 1), exc),
                       _history_update(history), folder, history)
                return
            frame = start + i
            image.save(os.path.join(folder, "{:06d}.png".format(frame)))
            gallery.append(image)
            status = (
                "Saved {} in {} ({}x{}, steps {}, guidance {}). "
                "Seeds: {}. {}".format(
                    _frame_range_text(start, i + 1), folder,
                    width, height, steps, guidance,
                    ", ".join(str(s) for s in seeds[:i + 1]),
                    note))
            yield (gallery, status, _history_update(history),
                   folder, history)
            # Free cached memory between images. Without this, a
            # long queue creeps upward in RAM/VRAM until Windows
            # kills the process mid-batch (proven on the Turbo app,
            # Oct 27, 2026). _free_memory is best-effort, never
            # raises, so cleanup cannot break a run.
            _free_memory()
        entries = [
            "model: {}".format(MODEL_ID),
            "precision: {}".format(precision),
            "gguf_file: {}".format(
                GGUF_FILES.get(precision, "none")),
            "offload: {}".format(bool(offload)),
            "vae_tiling: True",
            "size: {}x{}".format(width, height),
            "steps: {}".format(steps),
            "guidance: {}".format(guidance),
            "max_sequence_length: {}".format(MAX_SEQUENCE_LENGTH),
            "prompt: {}".format(template),
            "negative_prompt: {}".format(negative_template),
            _seed_entry(seeds),
        ] + _expanded_entries(expanded_list)
        if negative_template:
            if len(expanded_negatives) == 1:
                entries.append(
                    "expanded_negative: {}".format(expanded_negatives[0]))
            else:
                entries.append(
                    "expanded_negatives: {}".format(
                        ", ".join(expanded_negatives)))
        _append_settings(folder, _run_header(start, count), entries)
        history = _history_add(history, template)
        status = (
            "Done: {} in {} ({}x{}, steps {}, guidance {}). "
            "Seeds: {}. {}".format(
                _frame_range_text(start, count), folder,
                width, height, steps, guidance,
                ", ".join(str(s) for s in seeds), note))
        yield (gallery, status, _history_update(history),
               folder, history)


def star_batch(last_folder):
    """Copy the most recently generated batch folder into
    images_out/favorites/ under a unique name (never overwrite).
    If nothing has run yet this session, say so in the status."""
    if not last_folder or not os.path.isdir(last_folder):
        return "No batch has run yet in this session."
    os.makedirs(FAVORITES_DIR, exist_ok=True)
    base = os.path.basename(os.path.normpath(last_folder))
    dest = _unique_folder(base, root=FAVORITES_DIR)
    shutil.copytree(last_folder, dest)
    return "Starred to {}".format(dest)


def unload_models():
    """Free the generator (VRAM and RAM)."""
    _unload_pipe()
    return "Models unloaded -- VRAM and RAM freed."


def build_ui():
    """The Gradio page: one tab, text-to-image only, kept simple
    on purpose. Chroma differences from the rest of the family:
    no token box (the model is ungated), a negative prompt that
    genuinely works (real CFG, no True CFG slider), and a
    Precision radio whose default is the Q6_K GGUF because the
    8.9B transformer leaves room for one rung above the FLUX
    default. There is no BF16 option at all."""
    import gradio as gr

    with gr.Blocks(title="Chroma UI") as demo:
        gr.Markdown(
            "# Chroma UI\n"
            "Type a prompt, hit Generate, pictures appear below. "
            "Defaults are the model card's own: 40 steps, "
            "guidance 3.0. Chroma1-HD is Apache 2.0 and "
            "ungated -- no license to accept, no token to paste. "
            "It is a full-CFG model, so expect minutes per "
            "image, not schnell's seconds. It is also a base "
            "model by design: a strong, neutral foundation, a "
            "notch less polished than a tuned release.")
        with gr.Row():
            prompt_box = gr.Textbox(
                label="Prompt", value=DEFAULT_PROMPT, lines=3)
            batch_box = gr.Textbox(
                label="Batch name (one folder per name)",
                value="chroma-pictures")
        negative_box = gr.Textbox(
            label="Negative prompt (real CFG -- it works at "
                  "guidance above 1; at guidance 1 or below the "
                  "pipeline skips the CFG pass and it has no "
                  "effect)",
            lines=2)
        with gr.Row():
            preset = gr.Dropdown(
                label="Size preset",
                choices=["Square 1024x1024", "Portrait 768x1024",
                         "Landscape 1024x768", "Small square 768x768",
                         "Custom"],
                value="Square 1024x1024")
            width_box = gr.Number(label="Width", value=1024,
                                  precision=0)
            height_box = gr.Number(label="Height", value=1024,
                                   precision=0)
        with gr.Row():
            steps_box = gr.Slider(
                label="Steps (model card: 40)",
                minimum=1, maximum=100, value=DEFAULT_STEPS, step=1)
            guidance_box = gr.Slider(
                label="Guidance (model card: 3.0)",
                minimum=0.0, maximum=10.0, value=DEFAULT_GUIDANCE,
                step=0.1)
        with gr.Row():
            seed_box = gr.Number(label="Seed (-1 = random)",
                                 value=-1, precision=0)
            count_box = gr.Slider(label="Images per prompt",
                                  minimum=1, maximum=10, value=1,
                                  step=1)
        with gr.Row():
            precision_box = gr.Radio(
                label="Precision",
                choices=[
                    "Quantized GGUF Q6_K -- recommended for "
                    "12 GB",
                    "Quantized GGUF Q5_K_S -- lighter, a little "
                    "less detail",
                    "Quantized GGUF Q4_K_S -- lightest, most "
                    "headroom",
                ],
                value="Quantized GGUF Q6_K -- recommended for "
                      "12 GB")
            offload_box = gr.Checkbox(
                label="CPU offload (keep ON for a 12 GB card)",
                value=True)
        queue_box = gr.Textbox(
            label="Prompt queue (one prompt per line -- when this "
                  "has lines, Generate runs the queue instead of "
                  "the prompt box; all lines share one batch folder)",
            lines=4)
        with gr.Row():
            random_btn = gr.Button("Random prompt")
            history_dd = gr.Dropdown(
                label="History (last 10 -- pick one to reload it)",
                choices=[])
        with gr.Row():
            generate_btn = gr.Button("Generate", variant="primary")
            stop_btn = gr.Button("Stop")
            star_btn = gr.Button("Star batch")
            unload_btn = gr.Button("Unload models")
        status_box = gr.Textbox(label="Status", lines=4)
        gallery = gr.Gallery(label="Pictures", columns=3)

        last_folder_state = gr.State("")
        history_state = gr.State([])

        preset.change(fn=_preset_size, inputs=preset,
                      outputs=[width_box, height_box])
        random_btn.click(fn=random_prompt, outputs=prompt_box)
        history_dd.change(fn=lambda choice: choice,
                          inputs=history_dd, outputs=prompt_box)
        generate_event = generate_btn.click(
            fn=generate,
            inputs=[prompt_box, negative_box, queue_box, batch_box,
                    width_box, height_box, steps_box, guidance_box,
                    seed_box, count_box,
                    precision_box, offload_box,
                    last_folder_state, history_state],
            outputs=[gallery, status_box, history_dd,
                     last_folder_state, history_state])
        stop_btn.click(fn=_noop, cancels=[generate_event])
        star_btn.click(fn=star_batch, inputs=last_folder_state,
                       outputs=status_box)
        unload_btn.click(fn=unload_models, outputs=status_box)
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", inbrowser=True)
