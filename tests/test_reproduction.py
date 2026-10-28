"""Small CPU fixtures. No model downloads or actual BERT inference."""

from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
import json
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
from utils import embeddings
from utils.candidates import build_ngrams, build_pos_ngrams, apply_pos_filter, find_candidate_occurrences
from utils.discovery import retrieve_discovery_posts, normalized_text_hash
from utils.preprocessing import cleanup, tokenize_with_positions
from utils.reference_bank import save_reference_bank, load_reference_bank, validate_discovery_bank
from utils.reference_review import prepare_references, extract_readable_post_text, compile_phrase_pattern
from utils.results import load_run, inspection_view, highlight_word_spans
from utils.scoring import score_fixed_reference_bank, summarize_rankings
from utils.vocabulary import anchor_terms_by_hate_category


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


def build_fixture_bank(directory):
    anchors = anchor_terms_by_hate_category["antisemitism"]
    source = directory / "reference.ndjson"
    write_records(source, [make_record(100 + index * 10, [f"kike reference context {index}."])
                           for index in range(6)])
    with redirect_stdout(StringIO()):
        posts, occurrences, _, preparation = prepare_references(
            source, directory / "corpus", anchors, "antisemitism", max_per_anchor=15, random_seed=42)
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
                                                 self.manifest, anchor_terms_by_hate_category["antisemitism"])

    def tearDown(self):
        self.temporary.cleanup()

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
                                    bad_configuration, anchor_terms_by_hate_category["antisemitism"])
        misaligned = self.references.copy()
        misaligned.loc[0, "thread_id"] = 99999
        with self.assertRaisesRegex(ValueError, "metadata disagrees"):
            validate_discovery_bank(self.vectors, misaligned, self.reference_posts,
                                    self.manifest, anchor_terms_by_hate_category["antisemitism"])
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
        arguments = SimpleNamespace(input=source, reference_bank=self.bank, output=output,
                                    max_posts=4, batch_size=16, device="cpu")
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
        source = ''.join(cell['source']).replace(
            'RUN_DIRECTORY = PROJECT_ROOT / "artifacts/raiders_antisemitism_broad_seeds"',
            'RUN_DIRECTORY = Path(' + repr(RUN_PATH) + ')')
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
