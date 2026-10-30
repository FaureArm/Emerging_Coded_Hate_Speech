# Emerging_Coded_Hate_Speech

This repository concern's a project for the course IA717 of Télécom Paris' Mastère Spécialisé Intelligence Artificielle et Multimodale.

The aim of this project is to reproduce the findings fo Patick Wu's team, as explained in the paper "A Computational Approach to Discovering Emergent Hate Speech on Social Media", published in 2026.

## Run the complete pipeline

Edit the settings at the top of **`run_pipeline.py`**, then run from the repository root in WSL:

```bash
.venv/bin/python run_pipeline.py --check  # Optional: checks paths/settings, no files or remote calls.
.venv/bin/python run_pipeline.py
```

This is the only file you need to configure and execute. Select `HATE_CATEGORY` at the top, or override it with `--category`. Both retrieval seeds and reference anchors are read from `src/utils/vocabulary.py`; there is no separate seed list in the launcher or worker. Supported categories are `antisemitism`, `anti_black`, `islamophobia`, `anti_asian`, and `anti_lgbtq`.

```bash
.venv/bin/python run_pipeline.py --category islamophobia --check
.venv/bin/python run_pipeline.py --category islamophobia
```

The internal discovery worker also accepts `--category`, for example `.venv/bin/python src/run_reproduction.py --category islamophobia --reference-bank artifacts/reference_banks/islamophobia_v1`. Its category defaults to `antisemitism` when called directly; the launcher always supplies the selected category. The former `--seed-words` override has been removed; edit the category's seed list in `vocabulary.py` instead.

The launcher uses your installed, authenticated Colab CLI; it detects `colab` on PATH or `~/.local/bin/colab`. Set `COLAB_EXECUTABLE` if installed elsewhere.

| Setting | Purpose |
|---|---|
| `HATE_CATEGORY` / `--category` | Category key used to select both seeds and anchors from `vocabulary.py` |
| `REFERENCE_SOURCE` | Reference NDJSON input, or output path when automatic slicing is enabled |
| `DISCOVERY_SOURCE` | Discovery NDJSON input/output path; must differ from the reference path when automatic slicing is enabled |
| `PREPARE_SLICES_FROM_ARCHIVE` | `False` keeps using existing files; `True` creates/verifies consecutive slices before reference preparation |
| `RAW_ARCHIVE_PATH`, `RAW_ARCHIVE_MEMBER` | Full compressed Raiders archive and its NDJSON member name |
| `REFERENCE_SLICE_SIZE_GB`, `DISCOVERY_SLICE_SIZE_GB` | Maximum uncompressed decimal GB for the reference prefix and following discovery slice |
| `REFERENCE_VERSION` | `None` uses `<category>_v1`; set an unused name when reference or model settings change |
| `MAX_REFERENCES_PER_ANCHOR` | Maximum distinct contexts per anchor; `None` removes the cap |
| `REFERENCE_RANDOM_SEED` | Reproducible selection of reference contexts |
| `MAX_DISCOVERY_POSTS` | Maximum eligible seed-matching posts processed, not the number of raw posts scanned |
| `CANDIDATE_BATCH_SIZE` | Candidate embedding batch size on the GPU |
| `RUN_NAME` | New experiment name; `None` generates a unique timestamped name |
| `BERT_CHECKPOINT`, `MODEL_REVISION`, `CONTEXT_TOKENS_EACH_SIDE` | Reference and candidate embedding settings; defaults retain the first completed run's model revision and 64 surrounding subwords per side |
| `EXISTING_COLAB_SESSION` | `None` creates a fresh session; set a name such as `"hmapp"` to reuse an existing session |
| `COLAB_GPU` | GPU requested for a fresh session, default `"T4"` |
| `REMOTE_STEP_TIMEOUT_SECONDS` | Time allowed for each remote execution step, default 7200 |
| `TRANSFER_MAX_ATTEMPTS` | Total attempts per upload/download, default 5 |
| `TRANSFER_TIMEOUT_SECONDS` | Time limit per transfer attempt, default 180 seconds |
| `TRANSFER_RETRY_DELAY_SECONDS` | Initial retry delay, default 5 seconds; doubles up to 60 seconds |

The current selection is **`islamophobia`, up to 100 contexts per anchor, random seed 42**, using the new `islamophobia_v1` reference version. The older directory named `islamophobia_v3` was built with antisemitism anchors and cannot be reused for an Islamophobia experiment. Existing saved references are checked against the selected category and anchor dictionary before Colab starts. Earlier completed results are preserved; correcting the category requires a newly built bank and a fresh discovery run. Raising the reference cap cannot create contexts absent from the source.

### What happens automatically

1. If enabled, prepare/verify the raw reference and discovery slices locally. Then prepare reference text and exact spans, or validate/reload a matching saved version. Existing reference versions are never silently resampled or overwritten.
2. Reuse a validated local reference bank, including extracting an existing `artifacts/reference_bank_<version>.zip` if needed. Mismatched selections or BERT settings cause an explicit error before Colab starts.
3. Package only Python code, the selected discovery slice, and either the prepared references or the encoded bank. Create/reuse a Colab session, install libraries and require a CUDA GPU.
4. Upload in **16 MiB parts**, verify hashes, reassemble and extract remotely. This incorporates the user's confirmed workaround for HTTP 500 on a single 208 MiB upload with CLI v0.7.5.
5. If the bank does not exist, encode it on Colab, download it and validate/extract it locally **before** starting discovery. Later experiments reuse it.
6. Run discovery, candidate embedding and scoring; download results in chunks; verify every artifact checksum and extract the run locally. Local checkpoint files and transfer parts are removed after success and retained after failure.

The launcher checks a success marker after each remote step because the installed CLI can print remote Python errors without returning a failing exit status. Transfers use unique remote directories and exact part manifests, so stale chunks cannot be combined accidentally. ZIP extraction checks paths and publishes an output folder only after validation.

A newly created Colab session is stopped after successful local extraction. An existing session is left running. On failure, the remote session/workspace remains available for inspection; an already downloaded bank remains reusable. Failed discovery is not published as a completed local run.

### Retry and resume interrupted transfers

Uploads and downloads retry up to five times, with a 180-second limit per attempt and delays of 5, 10, 20 and 40 seconds. Session creation and BERT execution are not automatically retried.

If retries are exhausted or you interrupt the launcher, it retains the frozen input archive, parts and progress in `artifacts/.pipeline-<id>/` and prints the exact recovery command:

```bash
.venv/bin/python run_pipeline.py --category islamophobia --resume artifacts/.pipeline-<id>
```

Use the printed checkpoint path with the original Colab session still alive. Resume retains the same run name and workspace, verifies uploaded/downloaded parts by size and SHA-256, and transfers only missing or damaged parts. Completed reference encoding and discovery stages are reused. A failed BERT job has no embedding-level checkpoint; a partial remote output directory may still require manual recovery.

Keep the experiment settings, selected category, vocabulary and source files unchanged; resume rejects mismatches and a missing/corrupt saved archive. The printed resume command includes the saved category. Transfer retry/timeout settings may be adjusted. `--check --resume <path>` validates the checkpoint configuration without remote calls. Older checkpoints without the category/vocabulary snapshot must start a new run.

### Results and subsequent runs

Completed outputs are saved to:

- `data/raiders/reference_corpora/<version>/`: frozen reference posts and spans.
- `artifacts/reference_banks/<version>/` and `artifacts/reference_bank_<version>.zip`: reusable encoded bank and backup.
- `artifacts/<run_name>/` and `artifacts/<run_name>.zip`: experiment outputs and backup.

Each experiment also contains `pipeline_settings.json` and a copy of the launcher used, `pipeline_launcher.py`. The runner's `run_info.json` records the computational settings, model revision, source-code hashes and provenance.

Open **`src/Results.ipynb`** and set `RUN_DIRECTORY` to the path printed by the launcher. It reads saved tables locally, without BERT, downloads or the reference bank. Existing notebook outputs are left untouched.

For another experiment, change the discovery settings and use a new `RUN_NAME` (or keep `None`). Keep the same category and reference settings/version to reuse the bank. To use the earlier antisemitism v1 bank, select `antisemitism` and set **both** `REFERENCE_VERSION = "antisemitism_v1"` and `MAX_REFERENCES_PER_ANCHOR = 15`.

### Change the raw slices in the same launcher

Automatic extraction is **off by default**, preserving the working shared-slice setup. To keep the first 1 GB as your reference source and use the next 3 GB for discovery, edit these settings in `run_pipeline.py`:

```python
PREPARE_SLICES_FROM_ARCHIVE = True
REFERENCE_SLICE_SIZE_GB = 1.0
DISCOVERY_SLICE_SIZE_GB = 3.0
REFERENCE_SOURCE = "data/raiders/raiders_first_1GB.ndjson"
DISCOVERY_SOURCE = "data/raiders/raiders_next_3GB.ndjson"
```

Then run the same launcher command. It streams the local `.tar.zst` archive without unpacking the full dataset. Slices contain consecutive **whole thread records**, so actual sizes can be slightly below their limits. Discovery begins with the very next complete record after the reference prefix; these are not random samples. The runner continues to apply reference thread/text exclusions.

The existing first-1-GB file is verified byte-for-byte against the requested archive prefix and kept intact. A completed pair is recorded in `raiders_next_3GB.slices.json`, alongside the discovery file. Later runs verify both slices' SHA-256 hashes and reuse them without decompressing the archive. The archive itself is identified by path, size and modification time; it is not fully hashed on every run. The slice manifest is also included in the experiment's `pipeline_settings.json`.

- To change only discovery size, change `DISCOVERY_SLICE_SIZE_GB` and choose a new `DISCOVERY_SOURCE` filename, such as `raiders_next_5GB.ndjson`. Keep the reference source/version/settings to reuse its bank.
- To change reference size, change `REFERENCE_SLICE_SIZE_GB`, use a new `REFERENCE_SOURCE` filename and a new `REFERENCE_VERSION`. Also choose a new discovery filename: changing the reference boundary changes where discovery starts.
- An existing discovery file without the matching slice manifest is not overwritten or assumed compatible. Use a new filename. Interrupted extractions leave `.partial` files and no completed manifest; those paths are not silently reused.
- A slice size is an **uncompressed on-disk budget**, not a BERT processing limit. `MAX_DISCOVERY_POSTS` still controls how many eligible posts are processed; `MAX_REFERENCES_PER_ANCHOR` still caps selected reference contexts.

`--check` validates the archive path and distinct output paths without extracting data; missing planned slice outputs are allowed. The full archive member, existing reference prefix and cached slice contents are verified during the actual run. No real multi-GB extraction is performed just by editing these settings.

### Files retained for later runs

- `run_pipeline.py`: the configurable local launcher.
- `src/run_reproduction.py` and `src/build_reference_bank.py`: internal remote workers invoked by the launcher; no manual editing or separate terminal sequence is required.
- `src/Results.ipynb`: offline result inspection.
- `src/utils/`: implementations shared by the workers and launcher.
- `src/utils/split_raiders.py`: internal streaming slice helper invoked by the launcher; its duplicate standalone settings/entry point have been removed.
- `tests/`, dependency files and `Documentation/`: validation, environment and research references.

The superseded `Prepare_Reference.ipynb` has been removed; its preparation logic remains in `utils/reference_review.py`, called by the launcher. Previously generated datasets, completed artifacts and ignored archives are retained.

### Independence and source identity

For the available starter corpus, the frozen holdout is **every entire thread represented by a selected post in the saved reference bank**. All posts from those threads are excluded from discovery, regardless of seed matches. Full-text copies of reference posts are also excluded after case/whitespace normalization. The exclusion set is fixed for a bank, so changing seeds does not reshuffle references. No random split is created during discovery.

This is the smallest split compatible with the prepared bank. It does **not** reserve the entire 1 GB pool from which references were sampled: other threads in that pool remain eligible. A subsequent archive slice remains an optional alternative: set `DISCOVERY_SOURCE = "data/raiders/raiders_next_3GB.ndjson"` with `PREPARE_SLICES_FROM_ARCHIVE = True` in the launcher. Either input still undergoes the fixed-bank thread and text checks. The optional launcher step extracts slices locally before the remote discovery worker starts.

Identity is `4chan:pol:<source id>` for posts and threads; the reader rejects reference banks from another board. Numeric IDs are original archive IDs, never reset DataFrame row numbers. Saved posts include thread identity, timestamp, matched seeds, raw source JSON and the exact readable text. `resto` is used only to verify thread membership; it is not treated as a direct reply parent. No conversation graph or context model is implemented.

Reproducibility records include the consumed complete-line prefix SHA-256, source path/size, bank-manifest identifier, seeds, software versions, code hashes/Git status, random seed and timings. The untouched suffix is not read merely to hash the entire input. Retain the original input and commit the source version used for a reported experiment.

### Preserved methodology and explicit corrections

- Reused lexical cleanup, sentence-level POS tagging, unigram/bigram/trigram construction and the existing provisional noun/adjective filters. The exact POS grammar is not verified against Wu's implementation.
- TF-IDF selection stays **strictly above the mean of per-term maximum TF-IDF values**. No median or top-K replacement.
- Stopwords are still removed before n-grams to preserve the old baseline. `is_contiguous`, `word_spans` and occurrence counts expose artificial/gapped phrases; no automatic correction or filtering is applied.
- Numeric `>>post_id` markers are removed by the existing reference parser. Quoted language, punctuation, capitalization and newlines remain. The old runner's extra `ftfy` pass is not applied: reference and discovery now use the same parser and exact saved text for offsets.
- BERT pools only subwords belonging to target words, not the entire post. Repeated occurrences are averaged per candidate/post before averaging relevance across posts. The reference preparation's existing one-selected-occurrence-per-anchor/thread policy is preserved; it can differ from using every occurrence in Wu's reference collection.
- The old pilot generated anchors from discovery itself. This runner loads an independent frozen bank instead. Same-thread and copied-reference-text exclusions happen before candidate extraction; scoring rejects residual overlap rather than using same-post masking as a substitute.
- Each post score averages exactly the five nearest reference vectors, allowing multiple vectors of one anchor. Fewer than five references is an error. Candidate relevance and the cutoff use means. Tied ranks use candidate text for deterministic ordering.
- The main ranking retains singletons and seed/anchor phrases. Frequency, seed and anchor exclusions are notebook views only; `original_rank` and raw scores remain intact.
- Reference matches are assumed suitable contexts, not manually validated labels. Copied passages/paraphrases are not detected by full-text hashes. A historical archive prefix and seed-conditioned retrieval are not representative sampling; distinct posts/threads do not prove independent adoption. No precision/recall/F1 or novelty claims are produced.

### Outputs and inspection

Each completed run writes:

| File | Contents |
|---|---|
| `run_info.json` | Completion marker, settings, provenance, timings, thresholds and artifact checksums |
| `discovery_posts.csv` | Source identities, exact text, timestamp, seeds and original source post JSON |
| `reference_coverage.csv` | Frozen bank's per-anchor coverage, including missing anchors |
| `retrieval_summary.csv` | Selected-post counts per retrieval seed |
| `candidate_stage_counts.json` | POS/ngram counts, removal examples, TF-IDF survival, contiguity and vector counts |
| `tfidf_candidates.csv` | All POS-surviving terms, maximum TF-IDF and selection flags |
| `candidate_occurrences.csv` | Source-linked spans, constituent word offsets and contiguity |
| `candidate_embeddings.npy` | One saved vector per candidate/post; no redundant occurrence-vector export |
| `candidate_metadata.csv` | Row-aligned vector identities and occurrence counts |
| `candidate_post_scores.csv` | The same identities with per-post relevance |
| `ranked_candidates.csv` | Original ranks, relevance, post/thread/occurrence counts and threshold decisions |

`run_info.json` is written last. A partial directory without it is not a completed run. The bank's arrays are not copied into each run. The notebook reads only small tables, preserving literal text such as `null`, and uses saved word offsets to highlight candidate words in their original post context.

### Lightweight validation

```bash
.venv/bin/python -B -m unittest discover -s tests -v
```

Tests exercise reference/thread/text exclusion, source identities, phrase matching, text offsets, target-token pooling, repeated-occurrence aggregation, top-five scoring and failure guards. A small complete orchestration test uses **synthetic reference vectors and a fake encoder**, then executes every results-notebook cell in a fresh process with torch/transformers/NLTK imports forbidden. This validates wiring and artifact inspection, not actual BERT quality or a real GPU run. The reference bank is checked for unchanged contents after testing. Launcher tests simulate Colab and exercise new-bank creation, directory/ZIP cache reuse, changed-setting rejection, chunked transfers, corrupt/missing parts, extraction safety, remote failures and no-side-effect check mode. Slice tests use actual tiny tar/zstd archives to check consecutive boundaries, existing-prefix reuse, cache hashes, changed-size/corruption errors and integration before reference preparation. These tests do not create a real Colab session.

Additional checks on 2026-10-27: CPU retrieval/candidate extraction selected 1,000 real posts from 222 threads after excluding 4 reference threads (872 posts) in the consumed prefix. It produced 4,822 TF-IDF-selected candidates, 13,518 occurrences and 1,454 flagged noncontiguous occurrences. No candidate embeddings were computed for that sample. Top-five scoring on 256 eligible rows of the older saved matrices exactly matched the old per-post calculation; synthetic thread keys were used only for that numerical comparison, not to claim independence of the historical corpus.
