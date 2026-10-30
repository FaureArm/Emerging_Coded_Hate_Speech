"""Exercise real streaming tar/zstd extraction with tiny synthetic archives."""

from compression import zstd
from contextlib import redirect_stdout
from io import BytesIO, StringIO
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utils.split_raiders import byte_limit, prepare_slices


ARCHIVE_MEMBER = "posts.ndjson"


def write_test_archive(path, contents, member_name=ARCHIVE_MEMBER):
    with zstd.open(path, "wb") as compressed:
        with tarfile.open(fileobj=compressed, mode="w|") as archive:
            member = tarfile.TarInfo(member_name)
            member.size = len(contents)
            archive.addfile(member, BytesIO(contents))


class SliceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="hmapp-slices-")
        self.directory = Path(self.temporary.name)
        self.archive = self.directory / "archive.tar.zst"
        self.reference = self.directory / "reference.ndjson"
        self.discovery = self.directory / "discovery.ndjson"
        self.lines = [(json.dumps({"posts": [{"no": 100 + index, "resto": 0,
                                            "com": f"Example context {index}."}]}) + "\n").encode()
                      for index in range(8)]
        write_test_archive(self.archive, b"".join(self.lines))
        self.reference_gb = (len(b"".join(self.lines[:2])) + len(self.lines[2]) // 2) / 1e9
        self.discovery_gb = (len(b"".join(self.lines[2:4])) + len(self.lines[4]) // 2) / 1e9
        self.quiet = redirect_stdout(StringIO())
        self.quiet.__enter__()

    def tearDown(self):
        self.quiet.__exit__(None, None, None)
        self.temporary.cleanup()

    def prepare(self, **overrides):
        arguments = dict(archive_path=self.archive, archive_member=ARCHIVE_MEMBER,
                         reference_path=self.reference, discovery_path=self.discovery,
                         reference_size_gb=self.reference_gb, discovery_size_gb=self.discovery_gb)
        arguments.update(overrides)
        return prepare_slices(**arguments)

    def test_whole_record_boundaries_and_cached_reuse(self):
        manifest = self.prepare()
        self.assertEqual(self.reference.read_bytes(), b"".join(self.lines[:2]))
        self.assertEqual(self.discovery.read_bytes(), b"".join(self.lines[2:4]))
        self.assertEqual(manifest["reference"]["last_thread_id"], 101)
        self.assertEqual(manifest["discovery"]["first_thread_id"], 102)
        self.assertEqual(manifest["discovery"]["first_source_line"], 3)
        self.assertLessEqual(self.reference.stat().st_size, byte_limit(self.reference_gb))
        self.assertLessEqual(self.discovery.stat().st_size, byte_limit(self.discovery_gb))
        with patch("compression.zstd.open", side_effect=AssertionError("Cached slices must not decompress archive")):
            self.assertEqual(self.prepare(), manifest)

    def test_reuse_existing_prefix_and_resize_discovery_without_overwriting(self):
        self.reference.write_bytes(b"".join(self.lines[:2]))
        reference_modified_time = self.reference.stat().st_mtime_ns
        manifest = self.prepare()
        self.assertTrue(manifest["reference"]["reused_existing_prefix"])
        larger = self.directory / "larger.ndjson"
        self.prepare(discovery_path=larger, discovery_size_gb=len(b"".join(self.lines[2:6])) / 1e9)
        self.assertEqual(larger.read_bytes(), b"".join(self.lines[2:6]))
        self.assertEqual(self.discovery.read_bytes(), b"".join(self.lines[2:4]))
        self.assertEqual(self.reference.stat().st_mtime_ns, reference_modified_time)

    def test_wrong_existing_reference_is_preserved_and_rejected(self):
        wrong_prefix = b"".join(self.lines[1:3])
        self.reference.write_bytes(wrong_prefix)
        with self.assertRaisesRegex(ValueError, "differs from the archive prefix"):
            self.prepare()
        self.assertEqual(self.reference.read_bytes(), wrong_prefix)
        self.assertFalse(self.discovery.exists())
        self.assertFalse(self.discovery.with_suffix(".slices.json").exists())

    def test_cache_rejects_changed_sizes_source_or_content(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, "Slice settings or archive identity differ"):
            self.prepare(discovery_size_gb=self.discovery_gb * 2)
        original = self.discovery.read_bytes()
        self.discovery.write_bytes(original.replace(b"Example", b"Changed"))
        with self.assertRaisesRegex(ValueError, "checksum changed"):
            self.prepare()
        self.discovery.write_bytes(original)
        with self.archive.open("ab") as output:
            output.write(b"different")
        with self.assertRaisesRegex(ValueError, "archive identity differ"):
            self.prepare()

    def test_preexisting_discovery_without_manifest_is_not_overwritten(self):
        self.discovery.write_bytes(b"original data")
        with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
            self.prepare()
        self.assertEqual(self.discovery.read_bytes(), b"original data")

    def test_same_paths_invalid_budgets_and_missing_member_fail(self):
        with self.assertRaisesRegex(ValueError, "distinct reference and discovery"):
            self.prepare(discovery_path=self.reference)
        for size in [0, -1, float("nan"), float("inf"), "not a size"]:
            with self.subTest(size=size), self.assertRaises(ValueError):
                self.prepare(reference_size_gb=size)
        with self.assertRaisesRegex(ValueError, "NDJSON member not found"):
            self.prepare(archive_member="missing.ndjson")
        self.assertFalse(self.reference.exists())
        self.assertFalse(self.discovery.exists())

    def test_too_small_budget_does_not_publish_truncated_record(self):
        with self.assertRaisesRegex(ValueError, "No threads fit"):
            self.prepare(reference_size_gb=len(self.lines[0]) / 2e9)
        self.assertFalse(self.reference.exists())
        self.assertFalse(self.discovery.exists())
        self.assertFalse(self.discovery.with_suffix(".slices.json").exists())

    def test_end_of_source_saves_actual_smaller_size(self):
        manifest = self.prepare(discovery_size_gb=1)
        self.assertEqual(self.discovery.read_bytes(), b"".join(self.lines[2:]))
        self.assertTrue(manifest["discovery"]["source_exhausted"])


if __name__ == "__main__":
    unittest.main()
