"""Pruebas de auto-indexación incremental: sync_new_books, lock de concurrencia y endpoint /api/books/sync."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import app as app_module
import ingest as ingest_module
from ingest import IngestResult, get_new_books, scan_pdf_entries, sync_new_books


class BooksSyncTests(unittest.TestCase):
    def test_scan_and_get_new_books_filters_indexed_and_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            books_dir = Path(tmp)
            (books_dir / "already.pdf").write_bytes(b"%PDF-1.4 content")
            (books_dir / "brand_new.pdf").write_bytes(b"%PDF-1.4 new book")
            (books_dir / "empty_downloading.pdf").write_bytes(b"")
            (books_dir / "notes.txt").write_text("ignore", encoding="utf-8")

            entries = scan_pdf_entries(books_dir)
            self.assertEqual(set(entries.keys()), {"already.pdf", "brand_new.pdf"})

            with patch(
                "ingest.get_indexed_metadata",
                return_value={"books": ["already.pdf"], "hashes_by_source": {}, "total_chunks": 10},
            ):
                new_books = get_new_books(books_dir)
            self.assertEqual([p.name for p in new_books], ["brand_new.pdf"])

    def test_sync_new_books_skips_when_lock_held(self) -> None:
        acquired = ingest_module._ingest_thread_lock.acquire(blocking=False)
        self.assertTrue(acquired)
        try:
            with patch(
                "ingest.get_indexed_metadata",
                return_value={"books": ["a.pdf"], "hashes_by_source": {}, "total_chunks": 5},
            ):
                res = sync_new_books()
            self.assertIn("en curso", res.message.lower())
        finally:
            ingest_module._ingest_thread_lock.release()

    def test_sync_new_books_processes_only_new_without_reset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            books_dir = Path(tmp)
            new_pdf = books_dir / "nuevo.pdf"
            new_pdf.write_bytes(b"%PDF-1.4 hello")

            fake_result = IngestResult(
                reset=False,
                processed_files=["nuevo.pdf"],
                total_chunks_added=12,
                total_chunks_in_db=112,
                indexed_books=["viejo.pdf", "nuevo.pdf"],
                message="OK",
            )
            with (
                patch("ingest.get_new_books", return_value=[new_pdf]),
                patch("ingest.ingest_books", return_value=fake_result) as mock_ingest,
            ):
                res = sync_new_books(books_dir=books_dir, verbose=False)

            mock_ingest.assert_called_once()
            _, kwargs = mock_ingest.call_args
            self.assertFalse(kwargs["reset"])
            self.assertEqual(kwargs["target_files"], [new_pdf])
            self.assertEqual(res.processed_files, ["nuevo.pdf"])

    def test_api_books_sync_up_to_date_and_started(self) -> None:
        app_module._sync_state["running"] = False
        with TestClient(app_module.app) as client:
            # Caso 1: sin libros nuevos
            with patch("app.get_new_books", return_value=[]):
                resp = client.post("/api/books/sync")
            self.assertEqual(resp.status_code, 200)
            body = resp.json()
            self.assertEqual(body["status"], "up_to_date")
            self.assertEqual(body["new_books"], [])
            self.assertFalse(body["running"])

            # Caso 2: detecta un libro nuevo y dispara sync en background
            fake_path = Path("C:/fake/nuevo_libro.pdf")
            fake_result = IngestResult(
                reset=False,
                processed_files=["nuevo_libro.pdf"],
                total_chunks_added=5,
                total_chunks_in_db=105,
                indexed_books=["nuevo_libro.pdf"],
                message="Ingesta completada: 1 procesados.",
            )
            with (
                patch("app.get_new_books", return_value=[fake_path]),
                patch("app.sync_new_books", return_value=fake_result),
            ):
                resp2 = client.post("/api/books/sync")
                self.assertEqual(resp2.status_code, 200)
                body2 = resp2.json()
                self.assertEqual(body2["status"], "started")
                self.assertEqual(body2["new_books"], ["nuevo_libro.pdf"])

            status_resp = client.get("/api/books/sync/status")
            self.assertEqual(status_resp.status_code, 200)
            self.assertIn("running", status_resp.json())


if __name__ == "__main__":
    unittest.main()
