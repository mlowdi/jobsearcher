from __future__ import annotations

import re
import tempfile
import unittest
from datetime import date
from pathlib import Path

from db import append_triage_event, connect, migrate
from web import create_app

TODAY = date(2026, 8, 23)


def build_web_db(path: Path) -> None:
    migrate(path)
    connection = connect(path)
    try:
        with connection:
            for index in range(1, 29):
                ad_id = str(100 + index)
                headline = "<img src=x onerror=alert(1)>" if ad_id == "101" else f"Role {index:02d}"
                url = "javascript:alert(1)" if ad_id == "101" else f"https://example.test/{ad_id}"
                connection.execute(
                    """INSERT INTO ads(
                       id,headline,employer,publication_date,application_deadline,
                       webpage_url,description_text,municipality,region,first_seen,last_seen
                       ) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (ad_id, headline, f"Employer {index}", f"2026-08-{(index % 20)+1:02d}",
                     "2026-09-30", url, f"Description {index}", "Stockholm", "Stockholm",
                     "2026-08-01", "2026-08-23"),
                )
                lane = "low" if ad_id == "128" else "high"
                connection.execute(
                    "INSERT INTO lane_snapshots(ad_id,lane,lane_rank,snapshot_at) VALUES (?,?,?,'snapshot')",
                    (ad_id, lane, 1 if lane == "low" else index),
                )
            connection.execute(
                """INSERT INTO ads(id,headline,employer,publication_date,application_deadline,
                   webpage_url,description_text,first_seen,last_seen)
                   VALUES ('999','Expired saved','Old Employer','2026-01-01','2026-08-01',
                           'https://example.test/999','Expired description','2026-01-01','2026-01-01')"""
            )
            connection.executemany(
                """INSERT INTO duplicate_groups(
                   ad_id,canonical_ad_id,normalized_key,method,similarity,updated_at)
                   VALUES (?, '101', 'key', 'exact', 1.0, 'now')""",
                [("101",), ("102",)],
            )
    finally:
        connection.close()
    append_triage_event("103", "dismissed", source="web", db_path=path)
    append_triage_event("999", "saved", source="web", db_path=path)


def csrf(html: bytes) -> str:
    match = re.search(rb'name="csrf_token" value="([^"]+)"', html)
    if match is None:
        raise AssertionError("CSRF token not rendered")
    return match.group(1).decode()


class WebWorkspaceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "workspace.db"
        build_web_db(self.path)
        self.app = create_app(self.path, secret_key="web-test", today=TODAY)
        self.app.testing = True
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_dashboard_all_lanes_pagination_filters_sorts_and_expired_state(self) -> None:
        dashboard = self.client.get("/")
        self.assertEqual(dashboard.status_code, 200)
        self.assertIn(b"Low", dashboard.data)
        first = self.client.get("/inbox/high")
        second = self.client.get("/inbox/high?page=2")
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertIn(b"Page 1 of 2", first.data)
        self.assertIn(b"Page 2 of 2", second.data)
        self.assertNotIn(b"Role 03", first.data)
        all_states = self.client.get("/inbox/high?state=all&sort=newest")
        self.assertIn(b"Role 03", all_states.data)
        self.assertIn(b"Role 28", self.client.get("/inbox/low").data)
        self.assertIn(b"Expired saved", self.client.get("/state/saved").data)
        self.assertEqual(self.client.get("/ads/999").status_code, 200)
        self.assertNotIn(b"Expired saved", all_states.data)
        self.assertIn(b"Role", self.client.get("/inbox/high?state=all&sort=deadline").data)

    def test_validation_csrf_cookie_and_redirect_safety(self) -> None:
        for url, status in (
            ("/inbox/not-a-lane", 404), ("/inbox/high?state=bogus", 400),
            ("/inbox/high?sort=bogus", 400), ("/inbox/high?page=0", 400),
            ("/inbox/high?page=nope", 400),
            ("/inbox/high?page=999999999999999999999", 400),
            ("/state/unread", 404),
            ("/ads/%21bad", 404),
        ):
            with self.subTest(url=url):
                self.assertEqual(self.client.get(url).status_code, status)
        detail = self.client.get("/ads/101")
        self.assertIn("SameSite=Strict", detail.headers.get("Set-Cookie", ""))
        self.assertEqual(self.client.post("/ads/101/triage", data={"state": "saved"}).status_code, 400)
        self.assertEqual(self.client.post("/ads/101/triage", data={"state": "saved", "csrf_token": "wrong"}).status_code, 400)
        token = csrf(detail.data)
        external = self.client.post("/ads/101/triage", data={
            "state": "saved", "csrf_token": token, "next": "https://evil.test/steal"})
        self.assertEqual(external.status_code, 303)
        self.assertEqual(external.headers["Location"], "/ads/101")
        malformed = self.client.post("/ads/101/triage", data={
            "state": "unread", "csrf_token": token, "next": "http://[::1"})
        self.assertEqual(malformed.status_code, 303)
        self.assertEqual(malformed.headers["Location"], "/ads/101")
        relative = self.client.post("/ads/101/triage", data={
            "state": "unread", "csrf_token": token, "next": "/inbox/high?state=all"})
        self.assertEqual(relative.headers["Location"], "/inbox/high?state=all")
        connection = connect(self.path, writable=False)
        try:
            self.assertEqual([tuple(row) for row in connection.execute(
                "SELECT state FROM triage_events WHERE ad_id='101' ORDER BY id")],
                [("saved",), ("unread",), ("unread",)])
        finally:
            connection.close()

    def test_latest_state_and_history_follow_append_id_not_timestamp_text(self) -> None:
        append_triage_event(
            "104", "saved", source="web", db_path=self.path,
            created_at="2026-08-23T10:00:00+01:00",
        )
        append_triage_event(
            "104", "dismissed", source="web", db_path=self.path,
            created_at="2026-08-23T09:30:00+00:00",
        )
        detail = self.client.get("/ads/104")
        self.assertIn(b'<option value="dismissed" selected>', detail.data)
        self.assertLess(
            detail.data.index(b"2026-08-23T09:30:00+00:00"),
            detail.data.index(b"2026-08-23T10:00:00+01:00"),
        )

    def test_escaping_safe_source_links_history_and_duplicate_independence(self) -> None:
        connection = connect(self.path)
        try:
            with connection:
                connection.execute(
                    "UPDATE ads SET webpage_url='https://[::1' WHERE id='101'"
                )
        finally:
            connection.close()
        detail = self.client.get("/ads/101")
        self.assertIn(b"&lt;img src=x onerror=alert(1)&gt;", detail.data)
        self.assertNotIn(b"javascript:alert(1)", detail.data)
        self.assertNotIn(b"https://[::1", detail.data)
        self.assertIn(b"Role 02", detail.data)
        self.assertIn(b"https://example.test/102", self.client.get("/ads/102").data)
        token = csrf(detail.data)
        response = self.client.post("/ads/101/triage", data={
            "state": "saved", "csrf_token": token,
            "note": "<script>alert('x')</script>", "next": "/ads/101"})
        self.assertEqual(response.status_code, 303)
        escaped = self.client.get("/ads/101")
        self.assertIn(b"&lt;script&gt;alert(&#39;x&#39;)&lt;/script&gt;", escaped.data)
        self.assertIn(b"unread", self.client.get("/ads/102").data)
        connection = connect(self.path, writable=False)
        try:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM triage_events WHERE ad_id='102'").fetchone()[0], 0)
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()
