# One Dial or Many? Mapping Caution Inside Gemma-3-4B

IE624 (Generative & Agentic AI) · Course project · Core approach document · Nov 2026

> Status: this is the **main project file**. `IE624 Project.md` is the earlier proposal and is kept for reference only.

## 0. Progress and findings so far (as of 2026-11-26)

| Notebook | Status | Key result |
| --- | --- | --- |
| `00_initial_checks` | done | fp32 across 2× T4 works; activations reach \~71,000 (**fp16 would overflow**); \~2.9 s per generation |
| `01_select_direction_v2` | done | Raw Arditi directions put up to 97% of their weight on \~46 **massive dimensions**; ablating them breaks Gemma 3 (KL 15–96, with or without `<bos>`). **Masking those dims fixes it**: selected direction = layer 17, `<end_of_turn>`, bypass −47.8, induce +5.9, KL 0.018 (neighbouring layers 13–19 also pass). Behaviour (fast score): strict prompt raises refusal starts on Hard-1K **7% → 67%**; XSTest 6% → 26% (reproduces the pilot) |
| `02_map_and_shift` | done | **RQ1:** category directions share one dominant dial (cos with A 0.78–0.96) but are more distinct than "one dial + noise" (PC1 share 0.84 vs null 0.98, p < 0.01 at every layer); residues cluster into ethical-wrongdoing vs graphic-content groups (topic confound to be tested causally). **RQ2:** cos(strict − neutral shift, A) = **0.16** [0.14, 0.18] — below the random-contrast floor (0.41); the instruction moves benign prompts only **9%** of the harmful-harmless gap along A, but compresses them toward a common level (a caution "set point"); no category-specific push (p = 0.999). Without masking the headline flips sign (−0.20) |
| `03_causal_tests` | done | **Abliteration × instruction:** strict − neutral effect on Hard-1K = +0.59 [0.56, 0.63] normally, **+0.005** when abliterated (fast score) — but the judge still finds **22% redirects** under abliteration+strict (vs 0% neutral): abliteration kills refusal *openers*, lectures and hard refusals, not soft redirection. **Shift decomposition (L17, 5 template tokens):** removing the mean shift from strict prompts removes **96%** of the effect; adding it to neutral prompts recovers 50%. The along-A part (only 2–5% of the shift's squared norm) and the orthogonal part each carry about half (removal 49% / 45%; recovery 21% / 16%); random vector −4%. Recovery grows with layer (L6 ≈ 0 → L24 47%). **Categories:** category directions induce refusal on harmless prompts (41–97%, A: 92%); their residues induce **0%** → the sub-dials are A + topic, not separate refusal mechanisms. **Judge vs fast score:** agreement 87%, AUROC 0.92, precision 0.94, recall 0.64 (fast score misses redirects). Strict style: 63% redirect, 15% lecture, 3% hard refuse |
| `04_instruction_atlas` | done | 16 instructions × held-out prompts. **Across instructions, the mean push along A (layer 17) does not predict over-refusal**: Spearman +0.01 [−0.52, +0.54] with the fast score, +0.44 [−0.05, +0.75] with judged rates (n = 15 instructions, 29 judged replies each — inconclusive, not negative). All pushes are small (−0.11 … +0.12 of the harmful-harmless gap); e.g. `strict_v2` moves 0.03 yet refuses 86%. **Within an instruction, the projection ranks prompts well** (AUROC 0.82–0.97) → A reads "how risky the prompt looks"; the instruction changes the threshold somewhere else. **1-D vs k-D** (10,592 pairs, held-out instructions): A alone 0.77, A + 4 category PCs 0.86, A + 4 random directions 0.75, 64-PC upper bound 0.95 → several dimensions predict clearly better (predictive, not causal: 03 showed residues do not induce refusal). **Category profiles:** rank-1 share 0.97, but leftover is 2.9× sampling noise. **Judged check:** the fast score fails for personas — librarian 0% vs 62% judged, compliance 28% vs 69%, travel 3% vs 38% (overall agreement 80%). Cautious personas cause over-refusal with no safety rule (mostly redirects). "Answer directly, no disclaimers" *raises* refusal (17% → 38% judged; 10% hard refusals): with hedging forbidden, replies split into plain compliance or refusal. `very_strict` refuses 80% of harmless Alpaca prompts |
| `05_judged_grid` | done | All judged (2,600 replies). **Abliteration grid, 150 benign Hard-1K prompts** — over-refusal vs neutral, normal → abliterated: `strict` +0.69 [0.62, 0.77] → **+0.22** [0.15, 0.29] (**32% survives** [20%, 44%]); `librarian` +0.57 → **+0.19** (**34%** [22%, 47%]); `compliance` +0.41 → +0.04 (10% [−4%, 24%]). **Every surviving refusal is a redirect**: lecture + hard refuse drop to 0–1.3% in all abliterated conditions (strict normal: 24%). Harmful prompts (n = 50): refusal 88% → **4%** when abliterated, but the strict instruction brings it back to **26%** without the dial (all redirects). Survivors look like "here is the general process, but I won't give specifics" or (librarian) the request retold as a children's activity. **Judged atlas (16 instructions × 100 prompts):** push along A vs judged over-refusal Spearman **+0.30 [−0.24, +0.69]** (`<end_of_turn>`), +0.23 (final token) → no reliable relation across instructions; within an instruction the projection ranks prompts well (AUROC 0.69–0.93). E.g. `strict_v2` pushes 0.016 yet refuses 96%; `concise` pushes −0.11 yet adds +22 points. Replications: "no disclaimers" (`direct`) raises over-refusal +0.21 [0.10, 0.31]; `very_strict` switches style to lecture/hard refuse (63%); permissive prompts (`expert`, `no_refuse`) do not push refusal below neutral (−0.02) |

**Methods policy adopted from these results:** all geometry and all ablations use directions with the 46 massive dimensions zeroed (`data/massive_dims.json`); raw (unmasked) values are reported alongside as a sensitivity check. The fast refusal score is used for screening; final behavioural numbers come from the blind LLM judge (`judge.py`, `hyde/glm-5.3-flash`), validated against \~100 hand labels. The fast score is precise (0.94) but misses about a third of judged refusals (redirects that do not start with "I"/"Sorry"), so any claim about *soft* refusal — in particular what survives abliteration — must use judged generations.

**Answer to the title question:** *one dial for "no", a second route for "not quite".* The trained refusal direction is necessary for every lecture and hard refusal, for harmful and benign prompts and under every instruction; categories are that dial plus topic. But about a third of instruction-induced over-refusal (safety rule or cautious persona) survives its removal, entirely as soft redirection, and an instruction's average push along the dial does not tell how much it will over-refuse. The dial reads how risky the prompt looks; the instruction decides how much risk is tolerated, largely somewhere else.

**Judge validation (2026-11-25, `IE624/runs/03_causal_tests/handlabel/agreement.json`):** refusal vs not — human vs GLM **86%** (n = 29), Claude vs GLM 88% (n = 50, blind second judge). Exact six-way style — human vs GLM only 59%, Claude vs GLM 84%: the human uses `comply` where GLM uses `comply_warning`, and `hard_refuse` where GLM uses `redirect`. → Report refusal rates from the judge with confidence; report styles only in coarse groups (comply / redirect / lecture + hard refuse), noting that style boundaries are subjective.

**After 04:** the fast score is not valid for comparing *instructions* (personas change the opening words), so instruction-level claims must use judged replies; 04's fast-score atlas (Parts A–C) is reported as exploratory.

**Interactive page: done (2026-11-26)** — [`page/index.html`](page/index.html), one self-contained offline file built by `python page/build_page.py` from the notebook outputs (§7).

**Remaining:** slides → report → Q&A prep sheet. Optional (one \~20-min GPU run): token heatmap and steering-slider panels.

## 1. The idea in one paragraph

When a chat model refuses, is there a single internal "caution dial" that gets turned up, or many separate mechanisms — one per kind of danger, one for instructions, one for training? Prior work disagrees: refusal of harmful requests looks roughly one-dimensional (Arditi et al., 2024; Wang et al., 2026), but refusal directions differ across categories (Joad et al., 2026) and over-refusal on benign prompts looks task-specific and higher-dimensional (Maskey et al., 2026). We test this directly on Gemma-3-4B-it. We **map** how many dimensions caution has, measure where a **safety instruction** pushes the model in that space, and **remove** the main refusal direction (abliteration) to see what kinds of refusal survive. Every result is shown in an **interactive, explorable web page**.

## 2. Research questions and hypotheses

| # | Question | Hypothesis (from prior work) |
| --- | --- | --- |
| RQ1 | **How many dimensions does caution have?** Do the refusal directions for different harm categories point the same way? | Mostly one dial for clearly harmful requests: high cross-category cosine, PC1 dominant — with some category-specific residue. |
| RQ2 | **Where does a safety instruction push the model?** Is the instruction-induced shift aligned with the trained refusal direction? | Partially: moderate, layer-dependent cosine, peaking at mid-to-late layers. |
| RQ3 | **Does one axis predict instruction-induced over-refusal?** How much better do several directions do? | One axis explains a large part, but several directions predict over-refusal clearly better (over-refusal is less one-dimensional than harmful refusal). |
| RQ4 | **What survives abliteration?** With the main refusal direction removed, which refusals remain, in which categories and styles? | The safety instruction restores *some* refusal, mostly as soft styles (redirects, warnings) rather than hard refusals. |
| RQ5 | **Does the instruction change *whether* the model refuses or *how*?** | Mostly *how*: a shift toward redirect / safe-alternative styles (cf. Joad et al.: directions change style more than rate). |

Framing: a focused **verification-and-extension** on one model, not a novelty claim. To our knowledge, no prior work directly measures the alignment between an instruction-induced shift and the trained refusal direction, or tests what an instruction can still do after that direction is removed.

## 3. Pilot evidence (already obtained)

XSTest through Gemma-3-4B (Ollama, quantized), with and without the strict safety prefix, hand-labelled:

| Prompt type | No prefix | Strict prefix |
| --- | --- | --- |
| Benign (over-refusal) | 5.0% | \~25% |
| Harmful (appropriate refusal) | \~92% | \~95% |

The prefix refuses hardest on fiction, games, and privacy-of-fictional-characters prompts (Lion King plot, Counter-Strike, Minnie Mouse's address), while word-level triggers ("kill a Python process") stay answered — refusal keys on topic, not surface words. This already hints at "many dials": the instruction does not just raise refusal uniformly.

Caveats: single run, \~40 prompts per cell, quantized model. All reported numbers are re-measured on the full-precision (fp32) Hugging Face model.

## 4. Setup

### Model

`google/gemma-3-4b-it`, Hugging Face `transformers`. 34 layers, d_model 2560, d_mlp 10240. Gated repo (accept the licence early; on Kaggle, store the HF token as a secret). Loads as a multimodal model; we use the text path only. Precision: **fp32, split across Kaggle's two T4s** (see §8). Avoid fp16 — Gemma overflows to inf/NaN in fp16 activations — and avoid quantization, which distorts the activations we measure.

### Input format

Gemma 3 has no system role, so instructions are **prepended to the user turn ourselves** (never via a tool's template), keeping the text identical across Ollama, Hugging Face and the pilot:

```
<start_of_turn>user
{instruction}

{user question}<end_of_turn>
<start_of_turn>model
```

With no instruction, the user turn is just the question. Activations are read at the five **post-instruction template tokens** (`<end_of_turn>`, `\n`, `<start_of_turn>`, `model`, `\n`); the selected refusal direction lives at **`<end_of_turn>`, layer 17**, and the final token is reported alongside.

### Instruction conditions

| Condition | Text | Purpose |
| --- | --- | --- |
| None | — | Baseline |
| Neutral | Length-matched, safety-free ("You are a helpful assistant. Write clearly, use short paragraphs…") | Controls for "a long instruction is present" |
| Strict | The pilot's safety prompt | The main manipulation |
| Atlas set | 15–20 further instructions (Part 2b) | Maps the space of instructions |

The instruction-induced shift is always **instruction − neutral** (length-matched), never instruction − none.

### Datasets

Few datasets, each with one job. Forward passes are cheap, so size matters little for the direction work; the real cost is full generations, which are capped separately (§8).

| Dataset | Job | Size used |
| --- | --- | --- |
| **OR-Bench Toxic** | The **only** harmful source: Direction A and the per-category directions (it carries category labels) | 655 (fit 325 / eval 330) |
| **Alpaca** | Harmless baseline | 600 (300 / 300) |
| **OR-Bench Hard-1K** | The **only** benign over-refusal set; same categories as Toxic, so category X's refusal direction and category X's over-refusal refer to the same thing | 1,319 (657 / 662) |
| **XSTest** | Showcase prompts for the page (Lion King, Minnie Mouse, …) and pilot continuity | \~50 hand-picked |

Every dataset is split into a **direction-fitting half** and a **held-out evaluation half**.

**Measured sizes (notebook 00):** OR-Bench Toxic 655 prompts in 10 categories; Hard-1K 1,319 (dominated by "illegal", 527). The catch-all "harmful" category has only 30 toxic prompts, so it is kept for Direction A but excluded from per-category analyses in both datasets.

**Dropped, optional if time remains (forward passes only):** JailbreakBench — check that its refusal direction matches ours (cos ≈ 1), restoring comparability with Arditi et al.; FalseReject — check that over-refusal findings generalise beyond OR-Bench. Single-benchmark scope is listed as a limitation.

### Measuring refusal

- **Fast score (no generation):** Arditi et al.'s first-token refusal score — the log-odds that the reply starts with a refusal token ("I", "Sorry", …). Used wherever we need thousands of measurements (atlas, prediction).
- **Judged generations:** full responses (≤200 tokens, temperature 0 — refusal and its style are clear within the first few sentences) labelled by a blind LLM judge (sees only question + reply) into **five styles** plus `other` (empty/garbled); redirect, lecture and hard refuse count as refusal:

  | Style | Description |
  | --- | --- |
  | comply | answers normally |
  | comply-with-warning | answers with a disclaimer |
  | redirect | declines the literal request, offers a safer alternative |
  | lecture | moralises without helping |
  | hard refuse | "I can't help with that" |

- **Validation:** hand-label a random \~100 judged responses (report agreement), and report how well the fast score separates judged refusals from compliances. *Done for the fast score (notebook 03, 440 replies): AUROC 0.92, precision 0.94, recall 0.64. Hand labels pending.*
- **Division of labour:** the fast score is used for the atlas, the 1-D vs k-D prediction, Direction B's refused/answered labels and all patching/steering sweeps. Because it misses redirects, generations + judge are used for the behavioural tables, the abliteration grid, style labels and the steering slider.

## 5. Experiments

### Part 1 — The map: how many dials? (RQ1)

1. **Direction A (global refusal direction).** Per layer and post-instruction token, difference-in-means: harmful − harmless, with the 46 massive dimensions zeroed. Pick the candidate as Arditi et al. do (largest refusal drop when ablated, induces refusal when added, KL < 0.1 on harmless prompts, layer < 0.8 L). *Selected: layer 17, `<end_of_turn>`.*
2. **Category directions.** For each OR-Bench Toxic category (9 used; the 30-prompt "harmful" catch-all is excluded), harmful-in-category − harmless.
   - *Gotcha:* all category directions share the same harmless baseline, which inflates their cosines. Report both raw cosines and cosines **after projecting out Direction A** (the category-specific residue).
3. **Cosine matrix** between category directions, per layer → heatmap.
4. **PCA** on the category directions (and on per-prompt harmful − mean-harmless vectors): variance explained by PC1, PC2, … The headline number: *"PC1 explains X% of caution."*
5. **Over-refusal direction (Direction B).** Under Strict, on OR-Bench Hard-1K: refused − answered benign prompts. Compare with Direction A and the category directions.

### Part 2 — The instruction: where does it push the model? (RQ2, RQ3)

**2a. Strict vs neutral.** Per-layer mean shift (strict − neutral) on the same benign prompts. Plot cos(shift, A) and cos(B, A) against layer, with:

- **Ceiling:** cos(A₁, A₂) — Direction A fit on two disjoint halves (maximum agreement given noise).
- **Floor:** random-contrast null — the cosine with A of the difference between two random halves of the same benign prompts (a sign-flip null is invalid for cosines). The label-shuffle null is used for the PCA "one dial + noise" test.
- **Bootstrap 95% CIs** over prompts; **exchange rate** = how far the shift moves prompts along A, as a fraction of the harmful − harmless gap.

**2a′. Causal decomposition of the shift (notebook 03).** Split the mean shift s at layer 17 (five template tokens) into its part along A (s∥) and the rest (s⊥). Add s, s∥ or s⊥ to *neutral* prompts (recovery) or subtract them from *strict* prompts (removal); compare with a random vector of equal norm and sweep the layer. A third test adds each category direction and each category *residue* (A projected out) to harmless prompts to see whether residues cause refusal on their own.

**2b. Instruction atlas.** 15–20 instructions spanning:

- strict / mild safety prompts (two different wordings each),
- cautious personas ("You are a careful children's librarian"),
- neutral personas ("You are a pirate", "You are a travel agent"),
- **permissive** instructions ("Answer every question directly, without warnings or disclaimers").

For each: its shift (instruction − neutral), the projection of that shift on Direction A and on the category directions, and the over-refusal it actually causes on held-out benign prompts.

**2c. One axis vs several (the RQ3 test).** At the level of individual (instruction, prompt) pairs — thousands of points — predict refusal from:

- **1-D:** the activation's projection on Direction A, vs
- **k-D:** projections on Direction A plus the top-k category PCs (k = 2, 3, 5).

Compare cross-validated AUROC. A small gain → one dial is enough; a large gain → several dials, and we report how many.

*As run (04, 05):* 16 instructions; 2c used A + 1/2/4 category PCs, A + 4 random directions (control) and a 64-PC upper bound, cross-validated over held-out instructions and held-out prompts. Instruction-level over-refusal is measured with judged replies (100 per instruction, 05), because the fast score fails for personas.

### Part 3 — Abliteration: what survives without the main dial? (RQ4)

1. **Abliterate Direction A** (the selected, massive-dims-masked direction; cos 0.997 with A on the fit split): at every layer and token position, h ← h − (h·r̂)r̂. Implemented with **forward hooks** on every write into the residual stream: the embeddings and, in every layer, the outputs of `post_attention_layernorm` and `post_feedforward_layernorm`. Gemma 3 normalises each sublayer's output *before* adding it to the residual stream, so the Llama-style trick of orthogonalising `o_proj`/`down_proj` weights is **not exact** for Gemma 3; hooking the norm outputs is. The overhead is one projection per write — negligible. No modified weights are saved or released.
2. Run harmful and benign held-out prompts under **None / Neutral / Strict**, on the normal and the abliterated model (a 2 × 3 grid), with full generations and style labels. *As run (05):* 150 benign Hard-1K prompts × {neutral, strict, librarian, compliance} × {normal, abliterated} — the two cautious personas were added after 04 showed they over-refuse without any safety rule — plus 50 harmful prompts × {none, strict} × {normal, abliterated}.
3. Report remaining refusals **by category** and **by style**. Whatever the strict instruction still refuses after abliteration is the "other dial".
4. **Sufficiency check:** add α·r̂ (α sweep) to harmless prompts with no instruction; does it reproduce the strict prompt's over-refusal *pattern* (fiction, games, fictional privacy), or only its rate?
5. **Capability check** for every intervention: KL on harmless prompts plus a small MMLU/ARC sanity set, so a "refusal drop" is not just a broken model. *As run:* KL on harmless prompts (0.018 for the selected direction, 01) and the judge's `other` label (garbled replies: ≤ 1% in every abliterated condition); the MMLU/ARC set was not run.
6. *(Should; not run)* **Subspace ablation curve:** ablate the top 1, 2, 4, 8 directions from Part 1 and plot remaining strict-induced refusal vs k. A flat curve means the instruction works through something outside the refusal subspace.

### Part 4 — Refusal styles (RQ5)

Using the five-style labels from Parts 2–3, compare the style distribution under None / Neutral / Strict, normal vs abliterated. *As reported:* three coarse groups (comply / redirect / lecture + hard refuse), because the hand-label check showed finer style boundaries are subjective. Question: does the instruction change the *rate* of refusal, the *style*, or both?

## 6. How we "observe" refusal inside the model

Refusal is not stored in single neurons: individual neurons respond to many unrelated things and features overlap. It shows up as **directions in the residual stream**. We observe it in four ways:

1. **Projection.** For any input, the scalar h_l · r̂_l is its refusal score at layer l. Histograms and maps of this score are the basic view.
2. **Token-level heatmap.** Project *every* token position onto r̂ to see which words push toward refusal, and how the instruction changes that.
3. **Sublayer attribution.** Project the attention output (`o_proj`) and the MLP output of each layer separately onto r̂ to see which component writes the refusal signal, and measure how much the final token attends to the instruction tokens.
4. **Intervention.** Abliteration (remove r̂) and steering (add α·r̂) confirm the observed signal is causal.

*Stretch:* rank MLP neurons by activation difference (reported as correlated units only), or use Gemma Scope 2 sparse-autoencoder features, if available for the 4B model.

## 7. The interactive results page (required deliverable) — built

[`page/index.html`](page/index.html): a single static file (\~1.3 MB) with all data inlined, so it opens offline on any laptop or projector. `page/template.html` holds the layout and code; `page/build_page.py` reads the results under `IE624/runs/` and writes the page, so no number is typed by hand. Light and dark themes, hover tooltips, a table view for the atlas.

The page is written as a guided explanation, not only a results dashboard: each section explains the idea, then shows the evidence.

| Section | What it shows | Interaction | Data |
| --- | --- | --- | --- |
| Hero | The question, the one-line answer, 4 headline numbers | — | 05 |
| How it works | What the refusal direction is; find / remove / add | 2-D sketch (illustrative, labelled as such): slider removes the dial, button steers harmless prompts | — |
| Setup | Model, datasets, instructions, fast score vs judge, judge validation | Expandable instruction texts | 03, hand labels |
| RQ1 · One dial? | Category cosine matrix; category directions vs residues added to harmless prompts | Full / residue toggle, hover | 02, 03 |
| RQ2 · The instruction | Caution meter (planned panel 2, as a 1-D strip per prompt set); refusal rate vs dial position, neutral vs strict (threshold shift); shift decomposition | None / Neutral / Strict toggle, hover | 02, 03 |
| RQ4/5 · Remove the dial | Style bars normal vs abliterated for 4 instructions; over-refusal effect with CIs and survival; harmful check | Hover | 05 |
| Explorer (planned panel 5) | Real replies to 150 benign prompts: 4 instructions × normal / abliterated, judge labels | Prompt picker, random, "still refused without the dial" filter | 05 |
| RQ3 · Predicting (planned panel 4) | Push along the dial vs judged over-refusal, 16 instructions, CIs | Position toggle, hover with instruction text, table view | 05 |
| Surprises, Method, Answer | Unexpected results, pipeline, rigour, limitations, answer per RQ | — | 04, 05 |

**Planned but not built:** panel 1 (token heatmap) and panel 6 (steering slider) need a new GPU run (all-token projections for \~50 showcase prompts; a small α × prompt generation grid, \~90 replies). The planned 2-D caution map became a 1-D strip because only the projection on A was exported. Harmful-prompt replies appear nowhere on the page.

## 8. Compute

**Platform: Kaggle, 2× T4 (16 GB each).** \~30 GPU-hours/week quota, 12-hour session limit.

**Precision on T4.** The T4 has no native bf16, and vLLM refuses bf16 on it. fp16 is native but Gemma 3's activations overflow it — fatal when activations are what we measure. Quantization distorts activations. So: **fp32 (\~17 GB), split across both T4s with `device_map="auto"`**. Slower, but numerically exact.

| Item | Requirement |
| --- | --- |
| Weights | \~17 GB in fp32 across 2× T4 (includes the vision tower) |
| Activation cache | A few positions per prompt (34 × 2560 × 4 B ≈ 350 KB per position); notebook 03's template-token stats are \~600 MB (recomputable in \~5 min, not downloaded). All token positions only for the \~50 showcase prompts on the page |
| Forward passes (maps, atlas, fast score) | Under \~1 hour total |
| Generations | HF `generate` only (no vLLM on T4): **\~4–8 GPU-hours** for the budget below, spread over 2–3 sessions |
| Judge | `judge.py` → `hyde/glm-5.3-flash` via `https://inference.hyde.tech/v1` (key in the `GLM_API_KEY` secret / env var, never in code); blind to condition; resumable |
| Disk / RAM | Model + datasets + caches fit in Kaggle's disk; outputs to `/kaggle/working` (20 GB) or a Kaggle dataset |

**Generation budget — planned \~2,800; actually used \~3,500, all judged:**

| Notebook | What | Replies |
| --- | --- | --- |
| 03 | Key conditions (instruction × abliteration × patching) on 60 benign + 40 harmful prompts | 440 |
| 04 | 16 instructions × 29 benign prompts | 464 |
| 05 | Abliteration grid 4 instructions × 2 × 150 benign; harmful check 2 × 2 × 50; judged atlas 12 more instructions × 100 | 2,600 |
| **Total** | | **3,504** |

Measured speed: \~3 s per reply in fp32 on 2× T4 with batch 16; the judge labels \~500 replies in 5 minutes.

**Cost-saving rules:**

1. Fast first-token score instead of generations wherever validated (§4).
2. Abliteration via lightweight forward hooks on the post-sublayer norms — exact for Gemma 3 and nearly free (§5, Part 3).
3. Generations capped at \~200 tokens.
4. **Checkpoint constantly:** write activations and generations to disk in chunks so a dead session costs minutes, not a day.

**Day-1 sanity test** (settles precision before anything else):

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
print(torch.cuda.is_bf16_supported(), torch.cuda.device_count())

tok = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-4b-it", dtype=torch.float32, device_map="auto")  # torch_dtype= on transformers 4.x
x = tok.apply_chat_template([{"role": "user", "content": "Hello"}],
                            add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model(x, output_hidden_states=True)
print(model.hf_device_map)
print([h.float().abs().max().item() for h in out.hidden_states][::5])  # must be finite
```

Fallback if behind schedule: \~$10 of Colab Pro on an L4 (bf16, vLLM) makes the generation batches 3–5× faster.

Everything is forward-pass only — no fine-tuning, no backprop.

## 9. Priorities

| Priority | Items |
| --- | --- |
| **Must** | ✅ Direction A; ✅ category directions + cosine matrix + PCA; ✅ strict − neutral shift with ceiling/floor/CIs; ✅ abliteration × instruction grid with style labels; page panels ✅ 2, ✅ 5, ❌ 1 (token heatmap) |
| **Should** | ✅ Instruction atlas + 1-D vs k-D prediction; ❌ Direction B; ✅ sufficiency (category / A induction in 03); ❌ subspace ablation curve; page panels ✅ 3, ✅ 4, ❌ 6 |
| **Stretch** | ❌ Sublayer attribution; ❌ neuron / SAE view; ❌ live Gradio demo |
| **Added along the way** | ✅ massive-dimension fix (01); ✅ shift decomposition by causal patching (03); ✅ fast score vs judge validation (03, 04); ✅ hand-label + second-judge check |

## 10. Timeline (actual)

| Date | Done |
| --- | --- |
| 2026-11-09 | Approach fixed: Gemma-3-4B only, instructions prepended, abliteration as the causal test |
| 2026-11-10 | `00` checks (fp32 on 2× T4); `01` direction selection, massive-dimension problem found and fixed; `02` map and shift |
| 2026-11-14 | `03` causal tests + first judged replies |
| 2026-11-16 | `04` instruction atlas |
| 2026-11-25 | Hand labels + second-judge check |
| 2026-11-26 | `05` judged grid (2,600 replies); interactive page |

**Still to do:** slides, report, Q&A prep sheet. Optional: token heatmap and steering slider (one short GPU run).

## 11. Possible outcomes

| Finding | Meaning |
| --- | --- |
| PC1 dominant, instruction shift aligned with A, abliteration removes instruction-induced refusal | **One dial.** Training and instructions use the same mechanism. |
| Category residues large, 1-D prediction ≪ k-D, instruction still refuses after abliteration | **Many dials.** Instructions recruit mechanisms beyond the trained direction. |
| One dial for harmful refusal, several for instruction-induced over-refusal (predicted) | **Both, depending on context.** Consistent with Wang et al. and Maskey et al., now shown for instructions specifically. |

Any of these is a reportable result.

**What we got:** closest to the third row, with a twist. One dial for refusal of harmful prompts and for every hard "no" (categories are that dial plus topic); instruction-induced over-refusal is *not* one-dimensional: an instruction's push along the dial does not predict its over-refusal, more directions predict it better, and about a third of it survives abliteration as soft redirection. See §0.

## 12. Ethics

Abliteration is used only as an inference-time probe; no modified weights are saved or released. Harmful-prompt outputs are reported only in aggregate and are not shown on the interactive page (the abliteration panel uses benign and borderline prompts only). Permissive atlas instructions are simple plain-language instructions, not jailbreak templates.

## 13. References

Core technique:

- Arditi et al. (2024), [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717).
- Zou et al. (2023), [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405).

Refusal geometry and instructions:

- Wang, Miehling, Wei, Natesan Ramamurthy & Zhang (2026), [On the Steering Dimensionality of Refusal in Language Models](https://arxiv.org/abs/2610.04245).
- Joad, Hawasly, Boughorbel, Durrani & Sencar (2026), [There Is More to Refusal in Large Language Models than a Single Direction](https://arxiv.org/abs/2602.02132), EMNLP 2026.
- Maskey, Dras & Naseem (2026), [Over-Refusal and Representation Subspaces: A Mechanistic Analysis of Task-Conditioned Refusal in Aligned LLMs](https://arxiv.org/abs/2603.27518), EMNLP 2026.
- Zheng et al. (2024), *On Prompt-Driven Safeguarding for Large Language Models*, ICML 2024.
- Marbut, Olson & Wheeler (2026), [Decomposing Refusal Steering in Mixture-of-Experts Models](https://arxiv.org/abs/2606.04160).
- Li, Fastowski, Zaradoukas, Prenkaj & Kasneci (2026), [Analysing the Safety Pitfalls of Steering Vectors](https://arxiv.org/abs/2603.24543).
- Xue, Qi, Liu, Chen & Pedarsani (2026), [Deactivating Refusal Triggers: Understanding and Mitigating Overrefusal in Safety Alignment](https://arxiv.org/abs/2603.11388).
- [COSMIC: Generalized Refusal Direction Identification in LLM Activations](https://arxiv.org/abs/2506.00085) (2025).

Benchmarks:

- Cui et al. (2025), [OR-Bench](https://arxiv.org/abs/2405.20947); Röttger et al. (2024), [XSTest](https://arxiv.org/abs/2308.01263); Zhang et al. (2025), [FalseReject](https://arxiv.org/abs/2505.08054); Mazeika et al. (2024), [HarmBench](https://arxiv.org/abs/2402.04249); Chao et al. (2024), [JailbreakBench](https://arxiv.org/abs/2404.01318).

*To confirm before submission: Zheng et al. (2024) arXiv link; OR-Bench Toxic category labels and counts per category (need \~50+ per category; merge small categories if not); whether Gemma Scope 2 covers the 4B model.*
