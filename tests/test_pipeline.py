"""Launcher tests: synthetic data and a local Colab simulation, no network or BERT."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch
from nltk.tokenize import NLTKWordTokenizer, PunktSentenceTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import run_pipeline as pipeline
import build_reference_bank
import run_reproduction
from test_reproduction import TinyModel, TinyTokenizer, make_record, write_records, build_fixture_bank
from utils.reference_bank import load_reference_corpus, load_reference_bank, save_reference_bank
from utils.results import load_run
from utils.vocabulary import HATE_CATEGORIES, anchor_terms_by_hate_category, seed_words_by_hate_category


class LocalColab:
    """Execute transfer code locally and replace only GPU work with fake encoders."""

    def __init__(self, remote_directory, template_bank):
        self.remote_directory = remote_directory
        self.template_bank = template_bank
        self.commands = []
        self.reference_builds = 0
        self.discovery_runs = 0
        self.fail_discovery = False

    def remote_path(self, path):
        return self.remote_directory / str(path).removeprefix("/")

    def command(self, action, *arguments):
        self.commands.append((action, *map(str, arguments)))
        if action == "upload":
            shutil.copyfile(arguments[0], self.remote_path(arguments[1]))
        elif action == "download":
            shutil.copyfile(self.remote_path(arguments[0]), arguments[1])
        elif action not in {"new", "install", "stop"}:
            raise AssertionError(action)

    def execute(self, code):
        if "torch.cuda.is_available()" in code:
            return
        code = code.replace("/content/", str(self.remote_directory / "content") + "/")
        with patch("subprocess.run", side_effect=self.run_worker):
            exec(compile(code, "<simulated-colab>", "exec"), {})

    def run_worker(self, arguments, **kwargs):
        if arguments[0] == "git":
            return subprocess.CompletedProcess(arguments, 1, "", "")
        if arguments[1].endswith("build_reference_bank.py"):
            self.reference_builds += 1
            parsed = build_reference_bank.parse_arguments(arguments[2:])
            posts, occurrences, preparation = load_reference_corpus(parsed.reference_corpus)
            _, _, _, template = load_reference_bank(self.template_bank)
            configuration = template["embedding_config"]
            configuration.update(checkpoint=parsed.checkpoint, tokenizer_checkpoint=parsed.checkpoint,
                                 requested_revision=parsed.revision, resolved_revision=parsed.revision,
                                 tokenizer_revision=parsed.revision, context_tokens_each_side=parsed.context_tokens)
            vectors = np.random.default_rng(42).normal(size=(len(occurrences), 4)).astype(np.float32)
            metadata = occurrences[["anchor", "post_id"]].copy()
            metadata["n_occurrences"] = 1
            save_reference_bank(parsed.output, parsed.reference_corpus, preparation, posts, occurrences,
                                vectors, metadata, vectors, configuration, {"test": True})
            shutil.make_archive(str(parsed.output), "zip", root_dir=parsed.output.parent, base_dir=parsed.output.name)
        elif arguments[1].endswith("run_reproduction.py"):
            self.discovery_runs += 1
            if self.fail_discovery:
                raise RuntimeError("Simulated remote discovery failure")
            parsed = run_reproduction.parse_arguments(arguments[2:])
            parsed.device = "cpu"
            with patch("utils.preprocessing.load_nlp_tools", return_value=(set(), PunktSentenceTokenizer(), NLTKWordTokenizer())), \
                 patch("nltk.pos_tag", side_effect=lambda words: [(word, "NN") for word in words]), \
                 patch("utils.embeddings.load_embedding_model", return_value=(TinyModel(), TinyTokenizer(), torch.device("cpu"))):
                run_reproduction.run_experiment(parsed)
        else:
            raise AssertionError(arguments)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="hmapp-launcher-test-")
        self.directory = Path(self.temporary.name)
        self.project = self.directory / "project"
        self.project.mkdir()
        shutil.copytree(ROOT / "src", self.project / "src", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copyfile(ROOT / "run_pipeline.py", self.project / "run_pipeline.py")
        source = self.project / "reference.ndjson"
        write_records(source, [make_record(100 + index * 10, [f"kike reference context {index}."])
                               for index in range(6)])
        discovery = self.project / "discovery.ndjson"
        write_records(discovery, [make_record(900 + index * 10, [text]) for index, text in enumerate([
            "Jewish fictional silver moon garden.", "Jewish fictional golden garden.",
            "Jewish generations after the Holocaust.", "Jewish quiet town and silver tree.",
        ])])
        template_directory = self.directory / "template"
        template_directory.mkdir()
        self.client = LocalColab(self.directory / "remote", build_fixture_bank(template_directory))
        self.settings = patch.multiple(
            pipeline, PROJECT_ROOT=self.project, REFERENCE_SOURCE="reference.ndjson", DISCOVERY_SOURCE="discovery.ndjson",
            HATE_CATEGORY="antisemitism", REFERENCE_VERSION="antisemitism_test", MAX_REFERENCES_PER_ANCHOR=15,
            MAX_DISCOVERY_POSTS=4, RUN_NAME="first_run",
            DISCOVERY_SAMPLING_METHOD="first_matching", DISCOVERY_RANDOM_SEED=42,
            COLAB_EXECUTABLE=sys.executable, EXISTING_COLAB_SESSION=None, TRANSFER_CHUNK_BYTES=8192,
            PREPARE_SLICES_FROM_ARCHIVE=False,
        )
        self.settings.start()
        self.client_patch = patch.object(pipeline, "ColabClient", return_value=self.client)
        self.client_patch.start()
        self.quiet = redirect_stdout(StringIO())
        self.quiet.__enter__()

    def tearDown(self):
        self.quiet.__exit__(None, None, None)
        self.client_patch.stop()
        self.settings.stop()
        self.temporary.cleanup()

    def test_complete_run_reuses_bank_and_preserves_settings(self):
        output = pipeline.run_pipeline()
        info, tables, _ = load_run(output)
        self.assertEqual(info["seed_words"], seed_words_by_hate_category["antisemitism"])
        self.assertEqual(len(tables["discovery_posts"]), 4)
        self.assertEqual(self.client.reference_builds, 1)
        self.assertTrue((output / "pipeline_settings.json").is_file())
        self.assertTrue((output / "pipeline_launcher.py").is_file())
        self.assertTrue(output.with_suffix(".zip").is_file())
        self.assertEqual(self.client.commands[-1][0], "stop")
        self.assertFalse(list((self.project / "artifacts").glob(".pipeline-*")))
        bank = self.project / "artifacts/reference_banks/antisemitism_test"
        bank_digest = pipeline.file_digest(bank / "bank_manifest.json")
        with patch.object(pipeline, "RUN_NAME", "second_run"), patch.object(pipeline, "EXISTING_COLAB_SESSION", "existing-session"):
            command_count = len(self.client.commands)
            pipeline.run_pipeline()
        self.assertEqual(self.client.reference_builds, 1)
        self.assertEqual(self.client.discovery_runs, 2)
        self.assertEqual(pipeline.file_digest(bank / "bank_manifest.json"), bank_digest)
        self.assertNotIn("stop", [command[0] for command in self.client.commands[command_count:]])
        with self.assertRaises(FileExistsError):
            pipeline.run_pipeline()
        with patch.object(pipeline, "RUN_NAME", "different_context"), patch.object(pipeline, "CONTEXT_TOKENS_EACH_SIDE", 32):
            with self.assertRaisesRegex(ValueError, "different BERT"):
                pipeline.run_pipeline()
        with patch.object(pipeline, "RUN_NAME", "different_cap"), patch.object(pipeline, "MAX_REFERENCES_PER_ANCHOR", 20):
            with self.assertRaisesRegex(ValueError, "settings differ"):
                pipeline.run_pipeline()

    def test_uniform_time_setting_reaches_worker_and_reuses_existing_bank(self):
        pipeline.run_pipeline()
        bank_manifest = self.project / "artifacts/reference_banks/antisemitism_test/bank_manifest.json"
        bank_digest = pipeline.file_digest(bank_manifest)
        records = [make_record(1000 + index * 10, [f"Jewish fictional silver garden {index}."])
                   for index in range(20)]
        for index, record in enumerate(records):
            record["posts"][0]["time"] = 1467331200 + (index // 10) * 86400
        write_records(self.project / "discovery.ndjson", records)
        with patch.multiple(pipeline, RUN_NAME="uniform_run", DISCOVERY_SAMPLING_METHOD="uniform_time", DISCOVERY_RANDOM_SEED=17):
            output = pipeline.run_pipeline()
            info, tables, _ = load_run(output)
            settings = json.loads((output / "pipeline_settings.json").read_text())
            self.assertEqual(settings["discovery_sampling_method"], "uniform_time")
            self.assertEqual(settings["discovery_random_seed"], 17)
            self.assertEqual(info["retrieval"]["sampling_method"], "uniform_time")
            self.assertEqual(info["retrieval"]["sampling_random_seed"], 17)
            self.assertEqual([row["selected_posts"] for row in info["retrieval"]["daily_counts"]], [2, 2])
            self.assertEqual(len(tables["discovery_posts"]), 4)
            self.assertTrue(settings["reference_bank_reused"])
        self.assertEqual(self.client.reference_builds, 1)
        self.assertEqual(pipeline.file_digest(bank_manifest), bank_digest)

    def test_sampling_settings_fail_fast_and_check_does_not_scan(self):
        for setting, value in [("DISCOVERY_SAMPLING_METHOD", "typo"), ("DISCOVERY_RANDOM_SEED", 1.5)]:
            with self.subTest(setting=setting), patch.object(pipeline, setting, value):
                with self.assertRaisesRegex(ValueError, "sampling"):
                    pipeline.run_pipeline(check_only=True)
        with patch.object(pipeline, "DISCOVERY_SAMPLING_METHOD", "uniform_time"), \
             patch("utils.discovery._iter_eligible_posts", side_effect=AssertionError("Should not scan")):
            pipeline.run_pipeline(check_only=True)
        self.assertEqual(self.client.commands, [])
        self.assertFalse((self.project / "artifacts").exists())

    def test_resume_rejects_changed_sampling_settings(self):
        checkpoint, _ = self.interrupt_upload()
        command_count = len(self.client.commands)
        for setting, value in [("DISCOVERY_SAMPLING_METHOD", "uniform_time"), ("DISCOVERY_RANDOM_SEED", 7)]:
            with self.subTest(setting=setting), patch.object(pipeline, setting, value):
                with self.assertRaisesRegex(ValueError, "Settings or source files changed"):
                    pipeline.run_pipeline(resume=checkpoint)
        self.assertEqual(len(self.client.commands), command_count)

    def test_failed_discovery_keeps_downloaded_bank_and_does_not_publish_run(self):
        self.client.fail_discovery = True
        with self.assertRaisesRegex(RuntimeError, "Simulated remote"):
            pipeline.run_pipeline()
        bank = self.project / "artifacts/reference_banks/antisemitism_test"
        self.assertTrue((bank / "bank_manifest.json").is_file())
        self.assertFalse((self.project / "artifacts/first_run").exists())
        self.assertNotIn("stop", [command[0] for command in self.client.commands])

    def test_all_categories_use_shared_seeds_and_anchors_end_to_end(self):
        for category in HATE_CATEGORIES:
            with self.subTest(category=category):
                anchors = anchor_terms_by_hate_category[category]
                anchor = next(iter(anchors.values()))[0]
                seed = seed_words_by_hate_category[category][0]
                unrelated_seed = "muslim" if category == "antisemitism" else "jewish"
                write_records(self.project / "reference.ndjson", [
                    make_record(100 + index * 10, [f"{anchor} reference context {index}."])
                    for index in range(6)
                ])
                write_records(self.project / "discovery.ndjson", [
                    make_record(900, [f"{seed} fictional silver moon."]),
                    make_record(910, [f"{seed} fictional golden garden."]),
                    make_record(920, [f"{unrelated_seed} unrelated topical post."]),
                ])
                with patch.multiple(pipeline, HATE_CATEGORY=category, REFERENCE_VERSION=None, RUN_NAME=None):
                    output = pipeline.run_pipeline()
                info, tables, _ = load_run(output)
                settings = json.loads((output / "pipeline_settings.json").read_text())
                bank = self.project / "artifacts/reference_banks" / f"{category}_v1"
                _, metadata, _, manifest = load_reference_bank(bank)
                self.assertEqual(info["category"], category)
                self.assertEqual(info["anchor_terms"], anchors)
                self.assertEqual(info["seed_words"], seed_words_by_hate_category[category])
                self.assertEqual(settings["category"], category)
                self.assertEqual(settings["seed_words"], seed_words_by_hate_category[category])
                self.assertEqual(manifest["category"], category)
                self.assertEqual(manifest["preparation_config"]["anchors"], anchors)
                self.assertEqual(set(metadata["anchor"]), {next(iter(anchors))})
                self.assertEqual(set(tables["reference_coverage"]["anchor"]), set(anchors))
                self.assertEqual(set(tables["discovery_posts"]["post_id"]), {900, 910})
                self.assertTrue(output.name.startswith(f"raiders_{category}_"))

    def test_mismatched_reference_category_is_rejected_before_remote_work(self):
        pipeline.run_pipeline()
        command_count = len(self.client.commands)
        with patch.multiple(pipeline, HATE_CATEGORY="islamophobia", RUN_NAME="wrong_category"):
            with self.assertRaisesRegex(ValueError, "Saved reference vocabulary.*islamophobia"):
                pipeline.run_pipeline()
        self.assertEqual(len(self.client.commands), command_count)

    def test_unknown_category_is_rejected(self):
        with patch.object(pipeline, "HATE_CATEGORY", "unknown"):
            with self.assertRaisesRegex(ValueError, "Unknown hate category"):
                pipeline.run_pipeline(check_only=True)
        self.assertEqual(self.client.commands, [])

    def interrupt_upload(self):
        original = self.client.command
        uploaded = []

        def fail_third_upload(action, *arguments):
            if action == "upload" and len(uploaded) == 2:
                # A timeout may leave an incomplete remote file.
                self.client.remote_path(arguments[1]).write_bytes(b"partial upload")
                raise subprocess.CalledProcessError(1, ["colab", "upload"])
            original(action, *arguments)
            if action == "upload":
                uploaded.append(str(arguments[1]))

        with patch.object(self.client, "command", side_effect=fail_third_upload), redirect_stderr(StringIO()):
            with self.assertRaises(subprocess.CalledProcessError):
                pipeline.run_pipeline()
        checkpoint, = (self.project / "artifacts").glob(".pipeline-*")
        return checkpoint, uploaded

    def test_resume_upload_reuses_verified_parts_and_repairs_corrupt_parts(self):
        # None must retain the original generated run name when resumed later.
        with patch.object(pipeline, "RUN_NAME", None):
            checkpoint, uploaded = self.interrupt_upload()
            state = json.loads((checkpoint / "checkpoint.json").read_text())
            self.assertTrue((checkpoint / "inputs.zip").is_file())
            self.assertNotIn("upload", state["completed_steps"])
            self.client.remote_path(uploaded[0]).write_bytes(b"corrupted after upload")
            output = pipeline.run_pipeline(resume=checkpoint)
        uploads = [command[2] for command in self.client.commands if command[0] == "upload"]
        self.assertEqual(uploads.count(uploaded[0]), 2)  # Corrupt part uploaded again.
        self.assertEqual(uploads.count(uploaded[1]), 1)  # Valid part skipped.
        self.assertEqual(output.name, state["settings"]["run_name"])
        self.assertEqual(sum(c[0] == "new" for c in self.client.commands), 1)
        self.assertEqual(self.client.reference_builds, 1)
        self.assertEqual(self.client.discovery_runs, 1)
        self.assertFalse(checkpoint.exists())
        load_run(output)

    def test_resume_rejects_changed_settings_and_corrupt_archive_without_remote_calls(self):
        checkpoint, _ = self.interrupt_upload()
        command_count = len(self.client.commands)
        with patch.object(pipeline, "MAX_DISCOVERY_POSTS", 9):
            with self.assertRaisesRegex(ValueError, "Settings or source files changed"):
                pipeline.run_pipeline(resume=checkpoint)
        with patch.dict(seed_words_by_hate_category, {"antisemitism": ["changed vocabulary"]}):
            with self.assertRaisesRegex(ValueError, "Settings or source files changed"):
                pipeline.run_pipeline(resume=checkpoint)
        self.assertEqual(len(self.client.commands), command_count)
        pipeline.run_pipeline(check_only=True, resume=checkpoint)
        self.assertEqual(len(self.client.commands), command_count)
        (checkpoint / "inputs.zip").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "archive is missing or corrupt"):
            pipeline.run_pipeline(resume=checkpoint)
        self.assertEqual(len(self.client.commands), command_count)
        self.assertTrue(checkpoint.exists())

    def test_resume_download_skips_valid_parts_and_completed_bert_jobs(self):
        original = self.client.command
        downloaded = []

        def fail_second_result_part(action, *arguments):
            is_result_part = (action == "download" and "/artifacts/" in str(arguments[0])
                              and Path(arguments[0]).name.startswith("part-"))
            if is_result_part and len(downloaded) == 1:
                Path(arguments[1]).write_bytes(b"partial download")
                raise subprocess.TimeoutExpired(["colab", "download"], 180)
            original(action, *arguments)
            if is_result_part:
                downloaded.append(str(arguments[0]))

        with patch.object(pipeline, "TRANSFER_CHUNK_BYTES", 1024):
            with patch.object(self.client, "command", side_effect=fail_second_result_part), redirect_stderr(StringIO()):
                with self.assertRaises(subprocess.TimeoutExpired):
                    pipeline.run_pipeline()
            checkpoint, = (self.project / "artifacts").glob(".pipeline-*")
            self.assertEqual(self.client.reference_builds, 1)
            self.assertEqual(self.client.discovery_runs, 1)
            output = pipeline.run_pipeline(resume=checkpoint)
        self.assertEqual(sum(c[0] == "download" and c[1] == downloaded[0] for c in self.client.commands), 1)
        self.assertEqual(self.client.reference_builds, 1)
        self.assertEqual(self.client.discovery_runs, 1)
        self.assertFalse(checkpoint.exists())
        load_run(output)

    def test_resume_after_lost_completion_response_does_not_repeat_discovery(self):
        original = self.client.execute

        def lose_discovery_response(code):
            original(code)
            if "subprocess.run(" in code and "src/run_reproduction.py" in code:
                raise RuntimeError("Connection lost after remote computation finished")

        with patch.object(self.client, "execute", side_effect=lose_discovery_response), redirect_stderr(StringIO()):
            with self.assertRaisesRegex(RuntimeError, "Connection lost"):
                pipeline.run_pipeline()
        checkpoint, = (self.project / "artifacts").glob(".pipeline-*")
        state = json.loads((checkpoint / "checkpoint.json").read_text())
        self.assertNotIn("discovery", state["completed_steps"])
        output = pipeline.run_pipeline(resume=checkpoint)
        self.assertEqual(self.client.discovery_runs, 1)
        load_run(output)

    def test_automatic_slices_precede_preparation_and_work_end_to_end(self):
        from test_slices import write_test_archive, ARCHIVE_MEMBER
        reference_bytes = (self.project / "reference.ndjson").read_bytes()
        discovery_bytes = (self.project / "discovery.ndjson").read_bytes()
        archive = self.project / "archive.tar.zst"
        write_test_archive(archive, reference_bytes + discovery_bytes)
        new_reference = self.project / "slices/reference.ndjson"
        new_discovery = self.project / "slices/discovery.ndjson"
        with patch.multiple(pipeline, PREPARE_SLICES_FROM_ARCHIVE=True,
                            RAW_ARCHIVE_PATH=str(archive), RAW_ARCHIVE_MEMBER=ARCHIVE_MEMBER,
                            REFERENCE_SOURCE=str(new_reference), DISCOVERY_SOURCE=str(new_discovery),
                            REFERENCE_SLICE_SIZE_GB=len(reference_bytes) / 1e9,
                            DISCOVERY_SLICE_SIZE_GB=len(discovery_bytes) / 1e9):
            pipeline.run_pipeline(check_only=True)
            self.assertFalse(new_reference.parent.exists())
            self.assertEqual(self.client.commands, [])
            output = pipeline.run_pipeline()
        self.assertEqual(new_reference.read_bytes(), reference_bytes)
        self.assertEqual(new_discovery.read_bytes(), discovery_bytes)
        settings = json.loads((output / "pipeline_settings.json").read_text())
        self.assertEqual(settings["slice_preparation"]["reference"]["threads"], 6)
        self.assertEqual(settings["slice_preparation"]["discovery"]["first_source_line"], 7)
        self.assertEqual(self.client.reference_builds, 1)
        self.assertEqual(self.client.discovery_runs, 1)

    def test_check_mode_has_no_side_effects(self):
        pipeline.run_pipeline(check_only=True)
        self.assertFalse((self.project / "data").exists())
        self.assertFalse((self.project / "artifacts").exists())
        self.assertEqual(self.client.commands, [])

    def test_zip_only_cache_and_corruption_rejection(self):
        pipeline.run_pipeline()
        bank = self.project / "artifacts/reference_banks/antisemitism_test"
        # Only delete a directory created inside this test's temporary project.
        shutil.rmtree(bank)
        with patch.object(pipeline, "RUN_NAME", "zip_cache"):
            pipeline.run_pipeline()
        self.assertEqual(self.client.reference_builds, 1)
        (bank / "reference_vectors.npy").write_bytes(b"corrupt")
        with patch.object(pipeline, "RUN_NAME", "bad_bank"):
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                pipeline.run_pipeline()

    def test_missing_corrupt_parts_and_unsafe_archives(self):
        source = self.directory / "source.bin"
        source.write_bytes(b"1234567890" * 50)
        parts = self.directory / "parts"
        manifest = pipeline.split_archive(source, parts, 64)
        (parts / "part-000000").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "Corrupt transfer"):
            pipeline.combine_archive_parts(parts, self.directory / "output.bin", manifest)
        (parts / "part-000000").unlink()
        with self.assertRaises(FileNotFoundError):
            pipeline.combine_archive_parts(parts, self.directory / "missing.bin", manifest)
        archive_path = self.directory / "bad.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("../escaped.txt", "bad")
        with self.assertRaisesRegex(ValueError, "Unsafe ZIP"):
            pipeline.extract_validated_archive(archive_path, self.directory / "bank", lambda path: None)
        self.assertFalse((self.directory / "escaped.txt").exists())


class RemoteErrorTests(unittest.TestCase):
    def test_transfers_retry_process_failures_and_timeouts_with_backoff(self):
        client = pipeline.ColabClient("unused-cli", "test", 30)
        for action in ("upload", "download"):
            with self.subTest(action=action), patch("subprocess.run", side_effect=[
                subprocess.CalledProcessError(1, ["colab", action]),
                subprocess.TimeoutExpired(["colab", action], 180),
                subprocess.CompletedProcess([], 0),
            ]) as command, patch("run_pipeline.time.sleep") as sleep, redirect_stdout(StringIO()):
                client.command(action, "local", "remote")
                self.assertEqual(command.call_count, 3)
                self.assertEqual([c.args[0] for c in sleep.call_args_list], [5, 10])
                self.assertTrue(all(c.kwargs["timeout"] == 180 for c in command.call_args_list))

    def test_transfer_retries_are_bounded(self):
        client = pipeline.ColabClient("unused-cli", "test", 30)
        error = subprocess.CalledProcessError(1, ["colab", "upload"])
        with patch("subprocess.run", side_effect=error) as command, patch("run_pipeline.time.sleep") as sleep, \
             patch.object(pipeline, "TRANSFER_MAX_ATTEMPTS", 3), redirect_stdout(StringIO()):
            with self.assertRaises(subprocess.CalledProcessError):
                client.command("upload", "local", "remote")
        self.assertEqual(command.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_session_commands_are_not_retried(self):
        client = pipeline.ColabClient("unused-cli", "test", 30)
        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, ["colab", "new"])) as command, \
             patch("run_pipeline.time.sleep") as sleep:
            with self.assertRaises(subprocess.CalledProcessError):
                client.command("new", "--gpu", "T4")
        command.assert_called_once()
        sleep.assert_not_called()

    def test_cli_zero_exit_without_success_marker_is_failure(self):
        # Simulates the installed CLI returning zero despite a remote Python error.
        real_popen = subprocess.Popen
        def fake_cli(command, **kwargs):
            return real_popen([sys.executable, "-c", "import sys; sys.stdin.read(); print('remote error')"], **kwargs)
        client = pipeline.ColabClient("unused-cli", "test", 30)
        with patch("subprocess.Popen", side_effect=fake_cli), redirect_stdout(StringIO()):
            with self.assertRaisesRegex(RuntimeError, "did not complete"):
                client.execute("raise ValueError('remote failure')")

    def test_success_marker_is_reached_after_execution(self):
        real_popen = subprocess.Popen
        def fake_cli(command, **kwargs):
            return real_popen([sys.executable, "-c", "import sys; exec(sys.stdin.read())"], **kwargs)
        client = pipeline.ColabClient("unused-cli", "test", 30)
        with patch("subprocess.Popen", side_effect=fake_cli), redirect_stdout(StringIO()):
            client.execute("print('completed step')")


if __name__ == "__main__":
    unittest.main()
