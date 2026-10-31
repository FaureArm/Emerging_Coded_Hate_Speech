"""Small CPU fixtures. No model downloads or actual BERT inference."""

from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from nltk.tokenize import NLTKWordTokenizer, PunktSentenceTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_reproduction
from utils import discovery, embeddings
from utils.candidates import build_ngrams, build_pos_ngrams, apply_pos_filter, find_candidate_occurrences
from utils.discovery import retrieve_discovery_posts, normalized_text_hash
from utils.preprocessing import cleanup, tokenize_with_positions
from utils.reference_bank import save_reference_bank, load_reference_bank, validate_discovery_bank
from utils.reference_review import prepare_references, extract_readable_post_text, compile_phrase_pattern
from utils.results import load_run, inspection_view, highlight_word_spans
from utils.scoring import score_fixed_reference_bank, summarize_rankings
from utils.vocabulary import HATE_CATEGORIES, anchor_terms_by_hate_category, seed_words_by_hate_category


class TinyTokenizer:
    is_fast = True

    def __call__(self, text, add_special_tokens=True, return_tensors=None, **kwargs):
        passages = [text] if isinstance(text, str) else text
        encoded = []
        for passage in passages:
            spans = []
            for match in re.finditer(r"\w+|[^\w\s]", passage):
                if len(match.group()) > 6:
                    spans.extend([(match.start(), match.start() + 3), (match.start() + 3, match.end())])
                else:
                    spans.append(match.span())
            token_ids = [int(sha256(passage[start:end].lower().encode()).hexdigest()[:4], 16) % 97 + 1
                         for start, end in spans]
            if add_special_tokens:
                token_ids, spans = [101] + token_ids + [102], [(0, 0)] + spans + [(0, 0)]
            encoded.append((token_ids, spans))
        if return_tensors is None:
            return {"input_ids": encoded[0][0], "offset_mapping": encoded[0][1]}
        maximum_length = max(len(token_ids) for token_ids, _ in encoded)
        return {
            "input_ids": torch.tensor([token_ids + [0] * (maximum_length - len(token_ids)) for token_ids, _ in encoded]),
            "attention_mask": torch.tensor([[1] * len(token_ids) + [0] * (maximum_length - len(token_ids)) for token_ids, _ in encoded]),
            "offset_mapping": torch.tensor([spans + [(0, 0)] * (maximum_length - len(spans)) for _, spans in encoded]),
        }


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(hidden_size=4, max_position_embeddings=512)

    def forward(self, input_ids, attention_mask):
        values = input_ids.float()
        context = (values * attention_mask).sum(1) / attention_mask.sum(1)
        vectors = torch.stack([values / 100, values.remainder(7) / 7,
                               torch.ones_like(values), values.remainder(11) / 11], dim=-1)
        return SimpleNamespace(last_hidden_state=vectors + context[:, None, None] / 1000)


def make_record(thread_id, texts):
    return {"posts": [{"no": thread_id + index, "resto": 0 if index == 0 else thread_id,
                       "time": 1467500000 + index, "com": text}
                      for index, text in enumerate(texts)]}


def write_records(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def build_fixture_bank(directory, category="antisemitism"):
    anchors = anchor_terms_by_hate_category[category]
    first_anchor = next(iter(anchors.values()))[0]
    source = directory / "reference.ndjson"
    write_records(source, [make_record(100 + index * 10, [f"{first_anchor} reference context {index}."])
                           for index in range(6)])
    with redirect_stdout(StringIO()):
        posts, occurrences, _, preparation = prepare_references(
            source, directory / "corpus", anchors, category, max_per_anchor=15, random_seed=42)
    configuration = {
        "procedure": embeddings.EMBEDDING_PROCEDURE,
        "checkpoint": "fixture-model", "requested_revision": "fixture-revision",
        "resolved_revision": "fixture-revision", "tokenizer_checkpoint": "fixture-model",
        "tokenizer_revision": "fixture-revision", "context_tokens_each_side": 64,
        "layer": "last_hidden_state",
        "pooling": "mean of subword tokens overlapping the selected anchor span; special tokens excluded",
        "aggregation": "mean of selected occurrence vectors per (anchor, post_id)",
        "text_transform": "none; exact saved reference text", "model_eval": True,
        "model_dtype": "torch.float32", "hidden_size": 4, "max_position_embeddings": 512,
        "embedding_helper_sha256": sha256(Path(embeddings.__file__).read_bytes()).hexdigest(),
    }
    vectors = np.random.default_rng(42).normal(size=(len(occurrences), 4)).astype(np.float32)
    reference_metadata = occurrences[["anchor", "post_id"]].copy()
    reference_metadata["n_occurrences"] = 1
    bank = directory / "bank"
    save_reference_bank(bank, directory / "corpus", preparation, posts, occurrences, vectors,
                        reference_metadata, vectors, configuration, {"fixture": True})
    return bank


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="hmapp-test-")
        self.directory = Path(self.temporary.name)
        self.bank = build_fixture_bank(self.directory)
        self.vectors, self.references, self.reference_posts, self.manifest = load_reference_bank(self.bank)
        self.references = validate_discovery_bank(self.vectors, self.references, self.reference_posts,
                                                 self.manifest, anchor_terms_by_hate_category["antisemitism"], category="antisemitism")

    def tearDown(self):
        self.temporary.cleanup()

    def test_category_cli_defaults_follow_selected_category(self):
        for category in HATE_CATEGORIES:
            with self.subTest(category=category):
                arguments = run_reproduction.parse_arguments(["--category", category])
                self.assertEqual(arguments.category, category)
                self.assertEqual(arguments.reference_bank.name, f"{category}_v1")
                self.assertTrue(arguments.output.name.startswith(f"raiders_{category}_"))

    def test_wrong_category_bank_is_rejected_before_retrieval_or_embedding(self):
        arguments = run_reproduction.parse_arguments([
            "--category", "islamophobia", "--reference-bank", str(self.bank),
            "--input", str(self.directory / "not_read.ndjson"),
            "--output", str(self.directory / "not_created"),
        ])
        with patch("utils.discovery.retrieve_discovery_posts") as retrieve, \
             patch("utils.embeddings.load_embedding_model") as model, redirect_stdout(StringIO()):
            with self.assertRaisesRegex(ValueError, "category mismatch.*islamophobia.*antisemitism"):
                run_reproduction.run_experiment(arguments)
        retrieve.assert_not_called()
        model.assert_not_called()
        self.assertFalse(arguments.output.exists())

    def test_retrieval_separation_and_limit(self):
        source = self.directory / "discovery.ndjson"
        copied_text = self.reference_posts.iloc[0]["text"]
        write_records(source, [make_record(100, ["jewish blocked", "jewish sibling"]),
                               make_record(200, [copied_text.upper(), "jewish survivor"]),
                               make_record(300, ['Jewish &amp; <b>text</b><br>&gt;&gt;123\n&gt;quoted words']),
                               make_record(400, ["jewish unread"])])
        posts, summary, info = retrieve_discovery_posts(source, ["jewish"], self.reference_posts, 2)
        self.assertEqual(posts.post_id.tolist(), [201, 300])
        self.assertEqual(posts.thread_key.tolist(), ["4chan:pol:200", "4chan:pol:300"])
        self.assertEqual(posts.loc[300, "text"], "Jewish & text\n\n>quoted words")
        self.assertEqual(info["counts"]["reference_threads_excluded"], 1)
        self.assertEqual(info["counts"]["reference_thread_posts_excluded"], 2)
        self.assertEqual(info["counts"]["reference_text_posts_excluded"], 1)
        self.assertEqual(info["counts"]["source_records_read"], 3)
        self.assertEqual(summary.selected_posts.tolist(), [2])
        repeated, _, repeated_info = retrieve_discovery_posts(source, ["jewish"], self.reference_posts, 2)
        pd.testing.assert_frame_equal(posts, repeated)
        self.assertEqual(info, repeated_info)
        self.assertIsNotNone(compile_phrase_pattern("jew").search("JEW"))
        self.assertIsNone(compile_phrase_pattern("jew").search("jewellery"))
        self.assertIsNotNone(compile_phrase_pattern("great replacement").search("great \n replacement"))

    def dated_discovery_source(self, counts_by_day):
        records = []
        for day_offset, count in enumerate(counts_by_day):
            for _ in range(count):
                record = make_record(1000 + len(records) * 10, ["jewish fictional sample"])
                record["posts"][0]["time"] = 1467331200 + day_offset * 86400
                records.append(record)
        # Archive order need not match post timestamps.
        source = self.directory / "dated.ndjson"
        write_records(source, list(reversed(records)))
        return source

    def test_first_matching_still_stops_before_unread_suffix(self):
        source = self.dated_discovery_source([10, 10])
        valid_lines = source.read_bytes().splitlines(keepends=True)
        with source.open("ab") as output:
            output.write(b"invalid unread JSON\n")
        posts, _, info = retrieve_discovery_posts(source, ["jewish"], self.reference_posts, 3)
        self.assertEqual(posts.post_id.tolist(), [1190, 1180, 1170])
        self.assertEqual(info["stop_reason"], "maximum_posts")
        self.assertEqual(info["sampling_method"], "first_matching")
        self.assertIsNone(info["sampling_random_seed"])
        self.assertEqual(info["scan_passes"], 1)
        self.assertEqual(info["consumed_prefix_sha256"], sha256(b"".join(valid_lines[:3])).hexdigest())

    def test_uniform_time_balances_days_and_redistributes_sparse_capacity(self):
        source = self.dated_discovery_source([1, 10, 10])
        posts, summary, info = retrieve_discovery_posts(
            source, ["jewish"], self.reference_posts, 9, "uniform_time", 42)
        self.assertEqual([row["eligible_posts"] for row in info["daily_counts"]], [1, 10, 10])
        self.assertEqual([row["selected_posts"] for row in info["daily_counts"]], [1, 4, 4])
        actual_days = pd.to_datetime(posts.timestamp, unit="s", utc=True).dt.strftime("%Y-%m-%d")
        self.assertEqual(actual_days.value_counts().sort_index().tolist(), [1, 4, 4])
        self.assertTrue(posts.source_line.is_monotonic_increasing)
        self.assertEqual(summary.selected_posts.tolist(), [9])
        self.assertEqual(info["counts"]["source_records_read"], 21)
        self.assertEqual(info["counts"]["seed_matched_posts"], 21)
        self.assertEqual(info["counts"]["selected_posts"], 9)
        self.assertEqual(info["sampling_random_seed"], 42)
        self.assertEqual(info["scan_passes"], 2)
        self.assertEqual(info["stop_reason"], "end_of_file")
        self.assertEqual(info["consumed_bytes"], source.stat().st_size)
        self.assertEqual(info["consumed_prefix_sha256"], sha256(source.read_bytes()).hexdigest())
        json.dumps(info)  # Saved metadata contains only JSON-native values.

    def test_uniform_time_reproducibility_and_limits(self):
        source = self.dated_discovery_source([12, 12, 12])
        arguments = (source, ["jewish"], self.reference_posts, 8, "uniform_time")
        first, _, first_info = retrieve_discovery_posts(*arguments, 123)
        repeated, _, repeated_info = retrieve_discovery_posts(*arguments, 123)
        different, _, _ = retrieve_discovery_posts(*arguments, 456)
        pd.testing.assert_frame_equal(first, repeated)
        self.assertEqual(first_info, repeated_info)
        self.assertNotEqual(first.post_id.tolist(), different.post_id.tolist())
        self.assertEqual(sorted(row["selected_posts"] for row in first_info["daily_counts"]), [2, 3, 3])
        all_posts, _, _ = retrieve_discovery_posts(source, ["jewish"], self.reference_posts, 40, "uniform_time")
        original_order, _, _ = retrieve_discovery_posts(source, ["jewish"], self.reference_posts, 40)
        pd.testing.assert_frame_equal(all_posts, original_order)
        few_posts, _, info = retrieve_discovery_posts(source, ["jewish"], self.reference_posts, 2, "uniform_time")
        self.assertEqual(len(few_posts), 2)
        self.assertEqual(sorted(row["selected_posts"] for row in info["daily_counts"]), [0, 1, 1])

    def test_uniform_time_uses_post_dates_and_preserves_reference_exclusions(self):
        source = self.directory / "exclusions.ndjson"
        copied_text = self.reference_posts.iloc[0]["text"]
        records = [make_record(100, ["jewish excluded thread"]), make_record(900, [copied_text.upper()]),
                   make_record(1000, ["unrelated topic"]),
                   make_record(1100, ["jewish before midnight", "jewish after midnight"])]
        for record in records[:3]:
            record["posts"][0].pop("time")  # Excluded/nonmatching posts do not require dates.
        records[3]["posts"][0]["time"] = 1467331200 - 1
        records[3]["posts"][1]["time"] = 1467331200
        write_records(source, records)
        posts, _, info = retrieve_discovery_posts(source, ["jewish"], self.reference_posts, 10, "uniform_time")
        self.assertEqual(posts.post_id.tolist(), [1100, 1101])
        self.assertEqual([row["date_utc"] for row in info["daily_counts"]], ["2016-06-30", "2016-07-01"])
        self.assertEqual([row["selected_posts"] for row in info["daily_counts"]], [1, 1])
        self.assertEqual(info["counts"]["reference_threads_excluded"], 1)
        self.assertEqual(info["counts"]["reference_text_posts_excluded"], 1)

    def test_uniform_time_rejects_bad_dates_and_empty_selection(self):
        source = self.directory / "bad_dates.ndjson"
        for timestamp in (None, "1467331200", True, 0, -1, 10**30):
            with self.subTest(timestamp=timestamp):
                record = make_record(900, ["jewish sample"])
                record["posts"][0]["time"] = timestamp
                write_records(source, [record])
                with self.assertRaisesRegex(ValueError, "timestamp"):
                    retrieve_discovery_posts(source, ["jewish"], self.reference_posts, 2, "uniform_time")
        record["posts"][0]["time"] = None
        write_records(source, [record])
        posts, _, _ = retrieve_discovery_posts(source, ["jewish"], self.reference_posts, 2)
        self.assertEqual(len(posts), 1)  # Existing mode remains permissive about missing time.
        with self.assertRaisesRegex(ValueError, "No discovery posts"):
            retrieve_discovery_posts(source, ["nonmatching"], self.reference_posts, 2, "uniform_time")

    def test_uniform_time_detects_changed_source_between_passes(self):
        source = self.dated_discovery_source([4, 4])
        original_stat = source.stat()
        allocate_quotas = discovery._allocate_daily_quotas

        def change_text(*arguments):
            source.write_bytes(source.read_bytes().replace(b"sample", b"edited", 1))
            os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
            return allocate_quotas(*arguments)

        with patch("utils.discovery._allocate_daily_quotas", side_effect=change_text):
            with self.assertRaisesRegex(ValueError, "changed between"):
                retrieve_discovery_posts(source, ["jewish"], self.reference_posts, 4, "uniform_time")

    def test_offsets_and_legacy_noncontiguous_candidate(self):
        text = extract_readable_post_text('Generations after the Holocaust.<br>"Quoted", &amp; words http://exam<wbr>ple.org')
        with patch("nltk.pos_tag", side_effect=lambda words: [(word, "NN") for word in words]):
            tokens = tokenize_with_positions(cleanup(text), {"after", "the"}, PunktSentenceTokenizer(), NLTKWordTokenizer())
        data = pd.DataFrame({"Text for BERT": [text], "Word Tokens": [tokens]}, index=[900])
        with redirect_stdout(StringIO()):
            data = apply_pos_filter(build_pos_ngrams(build_ngrams(data)))
        _, _, _, occurrences, _, _ = find_candidate_occurrences(data, ["generations holocaust"])
        self.assertEqual(len(occurrences), 1)
        row = occurrences.iloc[0]
        self.assertFalse(row.is_contiguous)
        self.assertEqual(text[row.start:row.end], row.matched_text)
        self.assertEqual([text[start:end].lower() for start, end in row.word_spans], ["generations", "holocaust"])
        highlighted = highlight_word_spans(text, row.word_spans)
        self.assertIn("<mark>Generations</mark> after the <mark>Holocaust</mark>", highlighted)
        self.assertIn("&amp;", highlighted)

    def test_embedding_alignment_target_pooling_and_repeats(self):
        model, tokenizer, device = TinyModel(), TinyTokenizer(), torch.device("cpu")
        text = "alpha context alpha"
        occurrences = pd.DataFrame([
            {"post_id": 900, "candidate": "alpha", "matched_text": "alpha", "start": start,
             "end": start + 5, "word_spans": ((start, start + 5),), "is_contiguous": True}
            for start in [0, 14]
        ])
        posts = pd.DataFrame({"Text for BERT": [text]}, index=[900])
        metadata, occurrence_vectors, _, _, vectors, averaged = embeddings.embed_candidates(
            occurrences, posts, 16, model, tokenizer, device, 64)
        self.assertEqual(metadata.start.tolist(), [0, 14])
        self.assertEqual(averaged.n_occurrences.tolist(), [2])
        np.testing.assert_allclose(vectors[0], occurrence_vectors.mean(axis=0))
        anchor_vector = embeddings.embed_anchor_occurrence(text, 0, 5, model, tokenizer, device, 64)
        np.testing.assert_allclose(anchor_vector, occurrence_vectors[0])
        offsets = torch.tensor([[0, 0], [0, 5], [6, 11], [12, 17], [0, 0]])
        token_vectors = torch.tensor([[0., 0.], [2., 2.], [99., 99.], [6., 6.], [0., 0.]])
        np.testing.assert_array_equal(embeddings.average_candidate_tokens(
            token_vectors, offsets, [(0, 5), (12, 17)], 0), [4., 4.])

    def test_top_five_scoring_and_guards(self):
        candidate_vectors = np.random.default_rng(2).normal(size=(3, 4)).astype(np.float32)
        metadata = pd.DataFrame({"candidate": ["sample", "sample", "other"], "post_id": [900, 901, 902],
                                 "n_occurrences": [2, 1, 1], "post_key": ["4chan:pol:900", "4chan:pol:901", "4chan:pol:902"],
                                 "thread_key": ["4chan:pol:900", "4chan:pol:901", "4chan:pol:902"],
                                 "normalized_text_sha256": [normalized_text_hash(text) for text in ["a", "b", "c"]]})
        scores = score_fixed_reference_bank(candidate_vectors, metadata, self.vectors, self.references, batch_size=2)
        # Independent scalar cosine calculation checks the top-five definition.
        expected_scores = []
        for candidate_vector in candidate_vectors:
            similarities = [float(np.dot(candidate_vector, reference_vector) /
                                  (np.linalg.norm(candidate_vector) * np.linalg.norm(reference_vector)))
                            for reference_vector in self.vectors]
            expected_scores.append(sum(sorted(similarities, reverse=True)[:5]) / 5)
        np.testing.assert_allclose(scores.relevance, expected_scores, rtol=1e-6, atol=1e-7)
        occurrence_rows = pd.DataFrame({"candidate": ["sample", "sample", "sample", "other"],
                                         "is_contiguous": [True, False, True, True]})
        rankings, threshold = summarize_rankings(scores, occurrence_rows)
        sample = rankings.set_index("candidate").loc["sample"]
        self.assertEqual((sample.n_posts, sample.n_occurrences, sample.n_threads), (2, 3, 2))
        self.assertAlmostEqual(sample.relevance, scores.relevance.iloc[:2].mean(), places=6)
        self.assertEqual(threshold, float(rankings.relevance.mean()))
        view = inspection_view(rankings, excluded_phrases=["other"])
        self.assertEqual(view.original_rank.tolist(), rankings.loc[rankings.candidate.eq("sample"), "original_rank"].tolist())
        self.assertEqual(view.display_position.tolist(), [1])
        with self.assertRaisesRegex(ValueError, "at least five"):
            score_fixed_reference_bank(candidate_vectors, metadata, self.vectors[:4], self.references.iloc[:4])
        for column in ["post_key", "thread_key", "normalized_text_sha256"]:
            contaminated = metadata.copy()
            contaminated.loc[0, column] = self.references.iloc[0][column]
            with self.assertRaisesRegex(ValueError, "overlap"):
                score_fixed_reference_bank(candidate_vectors, contaminated, self.vectors, self.references)

    def test_bank_validation_and_phrase_aware_views(self):
        bad_configuration = json.loads(json.dumps(self.manifest))
        bad_configuration["embedding_config"]["tokenizer_revision"] = "another-revision"
        with self.assertRaisesRegex(ValueError, "resolved revision"):
            validate_discovery_bank(self.vectors, self.references, self.reference_posts,
                                    bad_configuration, anchor_terms_by_hate_category["antisemitism"], category="antisemitism")
        misaligned = self.references.copy()
        misaligned.loc[0, "thread_id"] = 99999
        with self.assertRaisesRegex(ValueError, "metadata disagrees"):
            validate_discovery_bank(self.vectors, misaligned, self.reference_posts,
                                    self.manifest, anchor_terms_by_hate_category["antisemitism"], category="antisemitism")
        metadata_path = self.bank / "reference_metadata.csv"
        metadata_path.write_text(metadata_path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            load_reference_bank(self.bank)
        rankings = pd.DataFrame({"candidate": ["jewellery", "jew", "the jew", "great replacement theory", "greatness"],
                                 "original_rank": [1, 2, 3, 4, 5], "n_posts": [1] * 5, "n_threads": [1] * 5})
        view = inspection_view(rankings, ["jew", "great replacement"])
        self.assertEqual(view.candidate.tolist(), ["jewellery", "greatness"])
        self.assertEqual(view.original_rank.tolist(), [1, 5])
        self.assertEqual(view.display_position.tolist(), [1, 2])

    def test_full_orchestration_with_fake_encoder_and_offline_notebook(self):
        source = self.directory / "discovery.ndjson"
        write_records(source, [make_record(900 + index * 10, [text]) for index, text in enumerate([
            "Jewish fictional silver moon garden.", "Jewish fictional golden garden.",
            "Jewish generations after the Holocaust.", "Jewish quiet town and silver tree.",
        ])])
        output = self.directory / "run"
        arguments = SimpleNamespace(category="antisemitism", input=source, reference_bank=self.bank, output=output,
                                    max_posts=4, batch_size=16, device="cpu",
                                    sampling_method="first_matching", sampling_random_seed=42)
        bank_checksums = {path.name: sha256(path.read_bytes()).hexdigest() for path in self.bank.iterdir()}
        with patch("utils.preprocessing.load_nlp_tools", return_value=({"after", "the", "and"}, PunktSentenceTokenizer(), NLTKWordTokenizer())), \
             patch("nltk.pos_tag", side_effect=lambda words: [(word, "NN") for word in words]), \
             patch("utils.embeddings.load_embedding_model", return_value=(TinyModel(), TinyTokenizer(), torch.device("cpu"))), \
             redirect_stdout(StringIO()):
            run_reproduction.run_experiment(arguments)
        info, tables, counts = load_run(output)
        self.assertEqual(info["retrieval"]["counts"]["selected_posts"], 4)
        saved_vectors = np.load(output / "candidate_embeddings.npy", allow_pickle=False)
        self.assertEqual(saved_vectors.shape[0], len(tables["candidate_metadata"]))
        self.assertEqual(tables["candidate_metadata"].embedding_row.tolist(), list(range(len(saved_vectors))))
        pd.testing.assert_frame_equal(tables["candidate_metadata"], tables["candidate_post_scores"].drop(columns="relevance"))
        for occurrence in tables["candidate_occurrences"].itertuples(index=False):
            text = tables["discovery_posts"].set_index("post_key").loc[occurrence.post_key, "text"]
            self.assertEqual(text[occurrence.start:occurrence.end], occurrence.matched_text)
        self.assertEqual(bank_checksums, {path.name: sha256(path.read_bytes()).hexdigest() for path in self.bank.iterdir()})
        with self.assertRaises(FileExistsError):
            run_reproduction.run_experiment(arguments)
        # Execute every notebook cell in a fresh process with heavy model imports forbidden.
        script = '''
import builtins, json, os
from pathlib import Path
original_import = builtins.__import__
def checked_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers', 'nltk'}:
        raise AssertionError('Results notebook imported a model/NLP module: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = checked_import
os.chdir(ROOT_PATH)
notebook = json.loads(Path('src/Results.ipynb').read_text())
namespace = {}
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        source = '\\n'.join(
            'RUN_DIRECTORY = Path(' + repr(RUN_PATH) + ')' if line.startswith('RUN_DIRECTORY = ') else line
            for line in source.split('\\n'))
        exec(compile(source, cell['id'], 'exec'), namespace)
print('OFFLINE_NOTEBOOK_OK')
'''
        completed = subprocess.run([sys.executable, "-B", "-c",
                                    f"ROOT_PATH={str(ROOT)!r}\nRUN_PATH={str(output)!r}\n" + script],
                                   capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("OFFLINE_NOTEBOOK_OK", completed.stdout)


if __name__ == "__main__":
    unittest.main()
