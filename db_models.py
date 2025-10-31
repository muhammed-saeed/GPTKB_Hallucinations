# import sqlite3
# from typing import Iterable, Tuple
# from settings import QUEUE_DDL, FACTS_DDL

# def _open_sqlite(path: str) -> sqlite3.Connection:
#     conn = sqlite3.connect(path, check_same_thread=False)
#     conn.execute("PRAGMA journal_mode=WAL;")
#     conn.execute("PRAGMA synchronous=NORMAL;")
#     conn.execute("PRAGMA temp_store=MEMORY;")
#     conn.execute("PRAGMA mmap_size=30000000000;")
#     conn.commit()
#     return conn

# def open_queue_db(path: str) -> sqlite3.Connection:
#     conn = _open_sqlite(path); conn.executescript(QUEUE_DDL); return conn

# def open_facts_db(path: str) -> sqlite3.Connection:
#     conn = _open_sqlite(path); conn.executescript(FACTS_DDL); return conn

# def enqueue_subjects(db: sqlite3.Connection, items: Iterable[Tuple[str, int]]):
#     cur = db.cursor()
#     for subject, hop in items:
#         cur.execute("""INSERT OR IGNORE INTO queue(subject, hop, status, retries)
#                        VALUES (?, ?, 'pending', 0)""", (subject, hop))
#     db.commit()

# def reset_working_to_pending(conn: sqlite3.Connection) -> int:
#     cur = conn.cursor()
#     cur.execute("UPDATE queue SET status='pending' WHERE status='working'")
#     conn.commit()
#     return cur.rowcount

# def queue_has_rows(conn: sqlite3.Connection) -> bool:
#     cur = conn.cursor(); cur.execute("SELECT 1 FROM queue LIMIT 1"); return cur.fetchone() is not None

# def count_queue(conn: sqlite3.Connection):
#     cur = conn.cursor()
#     cur.execute("SELECT COUNT(1) FROM queue WHERE status='pending'"); pending = cur.fetchone()[0]
#     cur.execute("SELECT COUNT(1) FROM queue WHERE status='working'"); working = cur.fetchone()[0]
#     cur.execute("SELECT COUNT(1) FROM queue WHERE status='done'");    done    = cur.fetchone()[0]
#     return done, working, pending, done + working + pending

# def write_triples_accepted(db: sqlite3.Connection, rows: Iterable[Tuple[str, str, str, int, str, str, float | None]]):
#     if not rows: return
#     cur = db.cursor()
#     cur.executemany("""INSERT OR IGNORE INTO triples_accepted
#                        (subject, predicate, object, hop, model_name, strategy, confidence)
#                        VALUES (?, ?, ?, ?, ?, ?, ?)""", rows)
#     db.commit()

# def write_triples_sink(db: sqlite3.Connection, rows: Iterable[Tuple[str, str, str, int, str, str, float | None, str]]):
#     if not rows: return
#     cur = db.cursor()
#     cur.executemany("""INSERT INTO triples_sink
#                        (subject, predicate, object, hop, model_name, strategy, confidence, reason)
#                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", rows)
#     db.commit()
# db_models.py



# db_models.py
import sqlite3
from typing import Iterable, Tuple
from settings import QUEUE_DDL, FACTS_DDL

def _open_sqlite(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA mmap_size=30000000000;")
    conn.commit()
    return conn

def _ensure_queue_indexes(conn: sqlite3.Connection):
    """Indexes that belong to the queue database."""
    cur = conn.cursor()
    # Queue table must ensure (subject, hop) uniqueness
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_queue_subject_hop ON queue(subject, hop)")
    conn.commit()

def _ensure_facts_indexes(conn: sqlite3.Connection):
    """Indexes that belong to the facts database."""
    cur = conn.cursor()
    # Facts table must ensure unique triples per hop
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_triples ON triples_accepted(subject, predicate, object, hop)")
    conn.commit()

def open_queue_db(path: str) -> sqlite3.Connection:
    conn = _open_sqlite(path)
    # Ensure tables exist, then indexes for QUEUE DB only
    conn.executescript(QUEUE_DDL)
    _ensure_queue_indexes(conn)
    return conn

def open_facts_db(path: str) -> sqlite3.Connection:
    conn = _open_sqlite(path)
    # Ensure tables exist, then indexes for FACTS DB only
    conn.executescript(FACTS_DDL)
    _ensure_facts_indexes(conn)
    return conn

def enqueue_subjects(db: sqlite3.Connection, items: Iterable[Tuple[str, int]]):
    cur = db.cursor()
    for subject, hop in items:
        cur.execute(
            """INSERT OR IGNORE INTO queue(subject, hop, status, retries)
               VALUES (?, ?, 'pending', 0)""",
            (subject, hop),
        )
    db.commit()

def reset_working_to_pending(conn: sqlite3.Connection) -> int:
    cur = conn.cursor()
    cur.execute("UPDATE queue SET status='pending' WHERE status='working'")
    conn.commit()
    return cur.rowcount

def queue_has_rows(conn: sqlite3.Connection) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM queue LIMIT 1")
    return cur.fetchone() is not None

def count_queue(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(1) FROM queue WHERE status='pending'"); pending = cur.fetchone()[0]
    cur.execute("SELECT COUNT(1) FROM queue WHERE status='working'"); working = cur.fetchone()[0]
    cur.execute("SELECT COUNT(1) FROM queue WHERE status='done'");    done    = cur.fetchone()[0]
    return done, working, pending, done + working + pending

def write_triples_accepted(db: sqlite3.Connection, rows: Iterable[Tuple[str, str, str, int, str, str, float | None]]):
    if not rows:
        return
    cur = db.cursor()
    cur.executemany(
        """INSERT OR IGNORE INTO triples_accepted
           (subject, predicate, object, hop, model_name, strategy, confidence)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    db.commit()

def write_triples_sink(db: sqlite3.Connection, rows: Iterable[Tuple[str, str, str, int, str, str, float | None, str]]):
    if not rows:
        return
    cur = db.cursor()
    cur.executemany(
        """INSERT INTO triples_sink
           (subject, predicate, object, hop, model_name, strategy, confidence, reason)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    db.commit()
