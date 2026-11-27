# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Test the PyPI version guard against a local stand-in for the index."""

import contextlib
import http.server
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import threading
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_pypi_version.py"
spec = importlib.util.spec_from_file_location("check_pypi_version", SCRIPT)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
guard.time.sleep = lambda seconds: None

SOURCES = {
    "edem/CMakeLists.txt": b"set(CUDAQ_EDEM_VERSION 0.1.0)\n",
    "edem/lib/edem.cpp": b"int main() {}\n",
    "pyproject.toml": b"[project]\nname = 'cudaq-edem'\n",
}


def sdist(files, version="0.1.0", info=b"Metadata-Version: 2.4\n"):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in {**files, "PKG-INFO": info}.items():
            member = tarfile.TarInfo("cudaq_edem-%s/%s" % (version, name))
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return buffer.getvalue()


class Index(http.server.BaseHTTPRequestHandler):
    """Serves the release metadata and files in the `published` table."""
    published = {}

    def do_GET(self):
        base = "http://127.0.0.1:%d" % self.server.server_port
        if self.path.startswith("/files/"):
            body = self.published[self.path.split("/")[2]]["archive"]
        elif self.path.startswith("/pypi/cudaq-edem/"):
            version = self.path.split("/")[3]
            entry = self.published.get(version)
            if entry is None:
                self.send_error(404)
                return
            urls = [{
                "packagetype": kind,
                "url": "%s/files/%s/%s" % (base, version, kind)
            } for kind in entry["kinds"]]
            body = json.dumps({"urls": urls}).encode()
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *arguments):
        pass


class CheckPyPIVersion(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Index)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.url = "http://127.0.0.1:%d" % cls.server.server_port
        cls.directory = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.directory.cleanup()

    def run_check(self, local, published=None, url=None, offline=False):
        Index.published = {} if published is None else {"0.1.0": published}
        path = Path(self.directory.name) / "cudaq_edem-0.1.0.tar.gz"
        path.write_bytes(sdist(local))
        with contextlib.redirect_stdout(io.StringIO()) as output:
            code = guard.check(path, url or self.url, offline)
        return code, output.getvalue()

    def published(self, files, kinds=("sdist",), **arguments):
        return {"archive": sdist(files, **arguments), "kinds": kinds}

    def test_unpublished_version_passes(self):
        code, output = self.run_check(SOURCES)
        self.assertEqual(code, 0, output)
        self.assertIn("not on PyPI", output)

    def test_same_sources_pass(self):
        # Generated metadata and the archive layout may differ.
        published = self.published(SOURCES, info=b"Metadata-Version: 2.3\n")
        code, output = self.run_check(SOURCES, published)
        self.assertEqual(code, 0, output)

    def test_changed_file_fails(self):
        changed = {
            **SOURCES, "edem/lib/edem.cpp": b"int main() { return 1; }\n"
        }
        code, output = self.run_check(SOURCES, self.published(changed))
        self.assertEqual(code, 1)
        self.assertIn("changed: edem/lib/edem.cpp", output)
        self.assertIn("Bump CUDAQ_EDEM_VERSION", output)

    def test_added_and_removed_files_fail(self):
        published = self.published({**SOURCES, "edem/lib/old.cpp": b"\n"})
        code, output = self.run_check({
            **SOURCES, "edem/lib/new.cpp": b"\n"
        }, published)
        self.assertEqual(code, 1)
        self.assertIn("added: edem/lib/new.cpp", output)
        self.assertIn("removed: edem/lib/old.cpp", output)

    def test_published_without_sdist_fails(self):
        code, output = self.run_check(
            SOURCES, self.published(SOURCES, kinds=("bdist_wheel",)))
        self.assertEqual(code, 1)
        self.assertIn("without a source distribution", output)

    def test_unreachable_index(self):
        closed = "http://127.0.0.1:1"
        code, output = self.run_check(SOURCES, url=closed)
        self.assertEqual(code, 1)
        self.assertIn("Could not check", output)
        code, output = self.run_check(SOURCES, url=closed, offline=True)
        self.assertEqual(code, 0)
        self.assertIn("Skipping the PyPI version check", output)

    def test_rejects_other_archives(self):
        path = Path(self.directory.name) / "other-1.0.tar.gz"
        path.write_bytes(b"")
        with self.assertRaises(ValueError):
            guard.check(path, self.url, False)


if __name__ == "__main__":
    unittest.main()
