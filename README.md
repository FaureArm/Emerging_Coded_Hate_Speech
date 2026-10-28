# Emerging_Coded_Hate_Speech

This repository concern's a project for the course IA717 of Télécom Paris' Mastère Spécialisé Intelligence Artificielle et Multimodale.

The aim of this project is to reproduce the findings fo Patick Wu's team, as explained in the paper "A Computational Approach to Discovering Emergent Hate Speech on Social Media", published in 2026.

## Current Raiders workflow

The two main entry points are:

- `src/run_reproduction.py`: retrieve posts, extract candidates, encode them, score against a **saved independent reference bank**, and export one completed run.
- `src/Results.ipynb`: change `RUN_DIRECTORY` and inspect that run locally. No BERT, GPU, downloads, or reference-bank directory is needed for inspection. Charts use the same plain HTML approach as the reference notebook.

Run from the repository root, or pass the absolute script path from elsewhere:

```bash
.venv/bin/python src/run_reproduction.py \
  --input data/raiders/raiders_first_1GB.ndjson \
  --reference-bank artifacts/reference_banks/antisemitism_v1 \
  --output artifacts/raiders_antisemitism_broad_seeds \
  --max-posts 1000
```

Relative CLI paths resolve against the repository, not the current working directory. Omitting `--output` creates a timestamped name. Existing run directories are never overwritten. `--batch-size` defaults to 16; `--device` defaults to `auto` (CUDA if available, otherwise CPU). Use `--device cuda` on Colab to fail rather than accidentally run on its CPU.

Edit the antisemitism entry of `seed_words_by_hate_category` in `src/utils/vocabulary.py` to change retrieval. The initial seeds are `jew`, `jews`, `jewish`, `zionist`, `zionism`, `antisemitism`, `israel`. These are experimental seeds, not Wu's published seed list. The published anchor dictionary is unchanged. Retrieval streams records and stops as soon as the requested number of eligible matching posts has been collected. It may examine more posts to find those matches, but does not parse the unused suffix.

### Reference setup (occasional, not part of every experiment)

The existing `src/Prepare_Reference.ipynb` prepares exact text/spans; `src/build_reference_bank.py` encodes that saved corpus once. The refactor reuses both, with explicit embedding arguments replacing global model configuration.

On 2026-10-27, CPU-only preparation generated `data/raiders/reference_corpora/antisemitism_v1`: **113 posts/selected occurrences, 13 of 20 anchors represented**, using the existing 15-context cap and random seed 42. The bank itself has **not** been encoded locally. No real BERT run was performed during this refactor. A missing bank causes an explicit error before model loading; there is no same-corpus fallback.

The bank contains `bank_manifest.json`, `reference_vectors.npy`, aligned `reference_metadata.csv`, reference source posts/spans, coverage and checksums. The runner checks file hashes, row alignment, source identities, parser version, vocabulary, target-token pooling and model/tokenizer revision. It inherits the bank's checkpoint, pinned revision, context size and float32 dtype. Current reference settings are BERT base uncased and **64 surrounding subword tokens on each side**.

### Independence and source identity

For the available starter corpus, the frozen holdout is **every entire thread represented by a selected post in the saved reference bank**. All posts from those threads are excluded from discovery, regardless of seed matches. Full-text copies of reference posts are also excluded after case/whitespace normalization. The exclusion set is fixed for a bank, so changing seeds does not reshuffle references. No random split is created during discovery.

This is the smallest split compatible with the prepared bank. It does **not** reserve the entire 1 GB pool from which references were sampled: other threads in that pool remain eligible. Earlier reference-notebook prose described using a subsequent archive slice; that remains an optional alternative via `--input data/raiders/raiders_next_3GB.ndjson` if generated using the existing `src/utils/split_raiders.py`. Either input still undergoes the fixed-bank thread and text checks. No new archive slice is extracted by the discovery runner.

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

The superseded preparation/reproduction/evaluation notebooks, conversation helpers, duplicate anchor module, legacy scoring/export helpers and old pilot artifacts have been removed from this branch. Their tracked versions remain available on `Base_eval_Raiders`. The current workflow keeps `Prepare_Reference.ipynb` and `build_reference_bank.py` for reference setup, and `utils/split_raiders.py` for optional archive slicing. The research papers, original Raiders archive, starter slice and newly prepared reference corpus are retained.

### Colab CLI: encode the reference once, then run discovery

Run these commands yourself from the local repository root. No 114 GB extraction is needed. The upload contains the 1 GB starter file and the small prepared reference corpus. The executable scripts run as Python processes inside Colab, avoiding notebook-kernel argument/path assumptions.

```bash
colab new -s hmapp --gpu T4
colab install -s hmapp pandas numpy nltk ftfy emoji torch transformers scikit-learn tqdm

zip -r hmapp_inputs.zip src data/raiders/raiders_first_1GB.ndjson \
  data/raiders/reference_corpora/antisemitism_v1 -x '*/__pycache__/*'
colab upload -s hmapp hmapp_inputs.zip content/hmapp_inputs.zip
echo 'import zipfile; zipfile.ZipFile("/content/hmapp_inputs.zip").extractall("/content")' | colab exec -s hmapp

# Once only: use the existing prepared text/spans to encode the reference bank.
echo 'import subprocess; subprocess.run(["python", "/content/src/build_reference_bank.py"], cwd="/content", check=True)' | colab exec -s hmapp --timeout 7200
colab download -s hmapp content/artifacts/reference_banks/antisemitism_v1.zip artifacts/reference_bank_antisemitism_v1.zip

# One complete discovery experiment. Pick a new output name for subsequent runs.
echo 'import subprocess; subprocess.run(["python", "/content/src/run_reproduction.py", "--input", "data/raiders/raiders_first_1GB.ndjson", "--reference-bank", "artifacts/reference_banks/antisemitism_v1", "--output", "artifacts/raiders_antisemitism_broad_seeds", "--max-posts", "1000", "--device", "cuda"], check=True)' | colab exec -s hmapp --timeout 7200

echo 'import shutil; shutil.make_archive("/content/raiders_antisemitism_broad_seeds", "zip", root_dir="/content/artifacts", base_dir="raiders_antisemitism_broad_seeds")' | colab exec -s hmapp
colab download -s hmapp content/raiders_antisemitism_broad_seeds.zip artifacts/raiders_antisemitism_broad_seeds.zip
```

Extract the downloaded run ZIP into local `artifacts/`, then open `src/Results.ipynb` and set its one run path. Preserve the reference-bank ZIP for future sessions; upload it and extract under `/content/artifacts/reference_banks/` instead of rerunning the encoding step. Scripts refuse to overwrite an existing bank or experiment directory. NLTK resources and the pinned model download only during an explicit computational run if absent from cache; they are never needed by the results notebook.

### Lightweight validation

```bash
.venv/bin/python -B -m unittest discover -s tests -v
```

Tests exercise reference/thread/text exclusion, source identities, phrase matching, text offsets, target-token pooling, repeated-occurrence aggregation, top-five scoring and failure guards. A small complete orchestration test uses **synthetic reference vectors and a fake encoder**, then executes every results-notebook cell in a fresh process with torch/transformers/NLTK imports forbidden. This validates wiring and artifact inspection, not actual BERT quality or a real GPU run. The reference bank is checked for unchanged contents after testing.

Additional checks on 2026-10-27: CPU retrieval/candidate extraction selected 1,000 real posts from 222 threads after excluding 4 reference threads (872 posts) in the consumed prefix. It produced 4,822 TF-IDF-selected candidates, 13,518 occurrences and 1,454 flagged noncontiguous occurrences. No candidate embeddings were computed for that sample. Top-five scoring on 256 eligible rows of the older saved matrices exactly matched the old per-post calculation; synthetic thread keys were used only for that numerical comparison, not to claim independence of the historical corpus.
