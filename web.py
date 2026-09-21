#!/usr/bin/env python3
"""Private server-rendered web workspace for persistent job triage."""

from __future__ import annotations

import argparse
import hmac
import os
import re
import secrets
from datetime import date
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import unquote, urlsplit

from flask import (
    Flask,
    abort,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from db import DB_PATH, LANES, TRIAGE_STATES, append_triage_event, connect, migrate

PAGE_SIZE = 25
INBOX_STATES = set(TRIAGE_STATES) | {"all"}
STATE_VIEWS = {"saved", "dismissed", "applied", "interview", "offer", "closed"}
SORTS = {"frozen-rank", "newest", "deadline"}
AD_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

_CURRENT_STATE_CTE = """
WITH latest_state AS (
    SELECT ad_id, state,
           ROW_NUMBER() OVER (PARTITION BY ad_id ORDER BY id DESC) AS n
    FROM triage_events
), current_ads AS (
    SELECT a.*, COALESCE(ls.state, 'unread') AS current_state
    FROM ads a
    LEFT JOIN latest_state ls ON ls.ad_id=a.id AND ls.n=1
)
"""


def _page_number() -> int:
    raw = request.args.get("page", "1")
    if not raw.isascii() or not raw.isdigit() or len(raw) > 7:
        abort(400, "page must be a positive integer no greater than 1000000")
    page = int(raw)
    if page < 1 or page > 1_000_000:
        abort(400, "page must be a positive integer no greater than 1000000")
    return page


def _validated_ad_id(ad_id: str) -> str:
    if AD_ID_RE.fullmatch(ad_id) is None:
        abort(404)
    return ad_id


def _safe_return_target(value: str | None) -> str | None:
    if not value:
        return None
    decoded = unquote(value)
    try:
        parsed = urlsplit(value)
    except ValueError:
        return None
    if (
        parsed.scheme
        or parsed.netloc
        or not parsed.path.startswith("/")
        or decoded.startswith("//")
        or "\\" in decoded
        or any(ord(character) < 32 for character in decoded)
    ):
        return None
    return parsed.path + (f"?{parsed.query}" if parsed.query else "")


def safe_source_url(value: str | None) -> str | None:
    if not value:
        return None
    try:
        parsed = urlsplit(value)
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return value


def create_app(
    db_path: str | Path = DB_PATH,
    *,
    secret_key: str | None = None,
    today: date | None = None,
) -> Flask:
    path = Path(db_path).expanduser().resolve()
    migrate(path)
    app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))
    app.config.update(
        DATABASE=str(path),
        SECRET_KEY=secret_key or os.environ.get("JOBSEARCHER_SECRET_KEY") or secrets.token_hex(32),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Strict",
    )
    app.jinja_env.globals["safe_source_url"] = safe_source_url
    app.jinja_env.globals["triage_states"] = TRIAGE_STATES

    def today_iso() -> str:
        return (today or date.today()).isoformat()

    def csrf_token() -> str:
        token = session.get("csrf_token")
        if not isinstance(token, str):
            token = secrets.token_urlsafe(32)
            session["csrf_token"] = token
        return token

    @app.context_processor
    def workspace_context() -> dict[str, Any]:
        return {"csrf_token": csrf_token, "lanes": LANES}

    @app.get("/")
    def dashboard() -> str:
        connection = connect(app.config["DATABASE"], writable=False)
        try:
            last_refresh = connection.execute(
                """SELECT * FROM refresh_runs
                   WHERE status <> 'running' ORDER BY started_at DESC,id DESC LIMIT 1"""
            ).fetchone()
            active_total = int(
                connection.execute(
                    "SELECT COUNT(*) n FROM ads WHERE application_deadline>=?",
                    (today_iso(),),
                ).fetchone()["n"]
            )
            unread_total = int(
                connection.execute(
                    _CURRENT_STATE_CTE
                    + "SELECT COUNT(*) n FROM current_ads WHERE application_deadline>=? AND current_state='unread'",
                    (today_iso(),),
                ).fetchone()["n"]
            )
            lane_counts = {lane: 0 for lane in LANES}
            for row in connection.execute(
                """SELECT l.lane,COUNT(*) n FROM lane_snapshots l
                   JOIN ads a ON a.id=l.ad_id WHERE a.application_deadline>=?
                   GROUP BY l.lane""",
                (today_iso(),),
            ):
                lane_counts[str(row["lane"])] = int(row["n"])
            state_counts = {state: 0 for state in TRIAGE_STATES}
            for row in connection.execute(
                _CURRENT_STATE_CTE
                + "SELECT current_state,COUNT(*) n FROM current_ads GROUP BY current_state"
            ):
                state_counts[str(row["current_state"])] = int(row["n"])
        finally:
            connection.close()
        return render_template(
            "workspace_dashboard.html",
            last_refresh=last_refresh,
            active_total=active_total,
            unread_total=unread_total,
            lane_counts=lane_counts,
            state_counts=state_counts,
        )

    @app.get("/inbox/<lane>")
    def inbox(lane: str) -> str:
        if lane not in LANES:
            abort(404)
        selected_state = request.args.get("state", "unread")
        selected_sort = request.args.get("sort", "frozen-rank")
        if selected_state not in INBOX_STATES or selected_sort not in SORTS:
            abort(400, "invalid state or sort")
        page = _page_number()
        order = {
            "frozen-rank": "l.lane_rank ASC, a.id ASC",
            "newest": "a.publication_date DESC, a.id ASC",
            "deadline": "a.application_deadline ASC, a.id ASC",
        }[selected_sort]
        state_clause = "" if selected_state == "all" else " AND a.current_state=?"
        params: list[Any] = [lane, today_iso()]
        if selected_state != "all":
            params.append(selected_state)
        connection = connect(app.config["DATABASE"], writable=False)
        try:
            total = int(
                connection.execute(
                    _CURRENT_STATE_CTE
                    + f"""SELECT COUNT(*) n FROM lane_snapshots l
                            JOIN current_ads a ON a.id=l.ad_id
                            WHERE l.lane=? AND a.application_deadline>=? {state_clause}""",
                    params,
                ).fetchone()["n"]
            )
            rows = connection.execute(
                _CURRENT_STATE_CTE
                + f"""
                    SELECT a.*,l.lane,l.lane_rank,p.ensemble_score,p.scoring_mode,
                           d.canonical_ad_id,
                           (SELECT COUNT(*) FROM duplicate_groups ds
                            WHERE ds.canonical_ad_id=d.canonical_ad_id) duplicate_count
                    FROM lane_snapshots l
                    JOIN current_ads a ON a.id=l.ad_id
                    LEFT JOIN preference_scores p ON p.ad_id=a.id
                    LEFT JOIN duplicate_groups d ON d.ad_id=a.id
                    WHERE l.lane=? AND a.application_deadline>=? {state_clause}
                    ORDER BY {order} LIMIT ? OFFSET ?
                """,
                [*params, PAGE_SIZE, (page - 1) * PAGE_SIZE],
            ).fetchall()
        finally:
            connection.close()
        return render_template(
            "workspace_list.html",
            heading=f"{lane.title()} lane",
            rows=rows,
            page=page,
            pages=max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE),
            total=total,
            selected_state=selected_state,
            selected_sort=selected_sort,
            lane=lane,
            state_view=None,
        )

    @app.get("/state/<state>")
    def state_view(state: str) -> str:
        if state not in STATE_VIEWS:
            abort(404)
        selected_sort = request.args.get("sort", "newest")
        if selected_sort not in SORTS:
            abort(400, "invalid sort")
        page = _page_number()
        order = {
            "frozen-rank": "CASE WHEN l.lane_rank IS NULL THEN 1 ELSE 0 END,l.lane,l.lane_rank,a.id",
            "newest": "a.publication_date DESC,a.id ASC",
            "deadline": "a.application_deadline ASC,a.id ASC",
        }[selected_sort]
        connection = connect(app.config["DATABASE"], writable=False)
        try:
            total = int(
                connection.execute(
                    _CURRENT_STATE_CTE + "SELECT COUNT(*) n FROM current_ads WHERE current_state=?",
                    (state,),
                ).fetchone()["n"]
            )
            rows = connection.execute(
                _CURRENT_STATE_CTE
                + f"""
                    SELECT a.*,l.lane,l.lane_rank,p.ensemble_score,p.scoring_mode,
                           d.canonical_ad_id,
                           (SELECT COUNT(*) FROM duplicate_groups ds
                            WHERE ds.canonical_ad_id=d.canonical_ad_id) duplicate_count
                    FROM current_ads a
                    LEFT JOIN lane_snapshots l ON l.ad_id=a.id
                    LEFT JOIN preference_scores p ON p.ad_id=a.id
                    LEFT JOIN duplicate_groups d ON d.ad_id=a.id
                    WHERE a.current_state=? ORDER BY {order} LIMIT ? OFFSET ?
                """,
                (state, PAGE_SIZE, (page - 1) * PAGE_SIZE),
            ).fetchall()
        finally:
            connection.close()
        return render_template(
            "workspace_list.html",
            heading=f"{state.title()} jobs",
            rows=rows,
            page=page,
            pages=max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE),
            total=total,
            selected_state=state,
            selected_sort=selected_sort,
            lane=None,
            state_view=state,
        )

    @app.get("/ads/<ad_id>")
    def ad_detail(ad_id: str) -> str:
        ad_id = _validated_ad_id(ad_id)
        connection = connect(app.config["DATABASE"], writable=False)
        try:
            ad = connection.execute(
                _CURRENT_STATE_CTE
                + """
                    SELECT a.*,l.lane,l.lane_rank,l.snapshot_at,
                           p.model_version,p.scoring_mode,p.title_tfidf,
                           p.full_embedding,p.title_embedding,p.source_score,
                           p.ensemble_score,p.component_spread,p.scored_at,
                           p.embedding_model_name,p.embedding_provenance,
                           d.canonical_ad_id,d.normalized_key
                    FROM current_ads a
                    LEFT JOIN lane_snapshots l ON l.ad_id=a.id
                    LEFT JOIN preference_scores p ON p.ad_id=a.id
                    LEFT JOIN duplicate_groups d ON d.ad_id=a.id
                    WHERE a.id=?
                """,
                (ad_id,),
            ).fetchone()
            if ad is None:
                abort(404)
            history = connection.execute(
                """SELECT * FROM triage_events WHERE ad_id=?
                   ORDER BY id DESC""",
                (ad_id,),
            ).fetchall()
            siblings = []
            if ad["canonical_ad_id"]:
                siblings = connection.execute(
                    _CURRENT_STATE_CTE
                    + """
                        SELECT a.id,a.headline,a.employer,a.current_state,l.lane,l.lane_rank
                        FROM duplicate_groups d JOIN current_ads a ON a.id=d.ad_id
                        LEFT JOIN lane_snapshots l ON l.ad_id=a.id
                        WHERE d.canonical_ad_id=? AND a.id<>? ORDER BY a.id
                    """,
                    (ad["canonical_ad_id"], ad_id),
                ).fetchall()
        finally:
            connection.close()
        return render_template("workspace_detail.html", ad=ad, history=history, siblings=siblings)

    @app.post("/ads/<ad_id>/triage")
    def triage(ad_id: str):
        ad_id = _validated_ad_id(ad_id)
        supplied = request.form.get("csrf_token", "")
        expected = session.get("csrf_token", "")
        if not supplied or not isinstance(expected, str) or not hmac.compare_digest(supplied, expected):
            abort(400, "invalid CSRF token")
        state = request.form.get("state", "")
        if state not in TRIAGE_STATES:
            abort(400, "invalid state")
        reason = request.form.get("reason", "").strip()
        note = request.form.get("note", "").strip()
        if len(reason) > 100 or len(note) > 5000:
            abort(400, "reason or note is too long")
        try:
            append_triage_event(
                ad_id,
                state,
                reason_code=reason or None,
                note=note or None,
                source="web",
                db_path=app.config["DATABASE"],
            )
        except KeyError:
            abort(404)
        target = _safe_return_target(request.form.get("next"))
        return redirect(target or url_for("ad_detail", ad_id=ad_id), code=303)

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("port must be between 1 and 65535")
    from waitress import serve

    serve(create_app(args.db), host=args.host, port=args.port, threads=4)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
