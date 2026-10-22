"""LMDB shared memory cache for entity store.

Node-level L2 cache between per-process in-memory dicts (L1) and
CouchbaseKV (L3).  All entity workers on the same node share one LMDB
instance via hostPath volume (tmpfs = RAM-backed).

Read: ~1-2 us per key (mmap lookup, no syscall).
Write: sole writer (lmdb-sync pod), serialized via flock.

Note: LMDB's internal lock=True uses POSIX process-shared mutexes
(pthread_mutex in a shared mmap) which do NOT work across Docker
containers sharing a tmpfs volume (separate PID namespaces).
We use lock=False and serialize writes via fcntl.flock() instead.
"""
import fcntl
import logging
import os
import random
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_MISS_SENTINEL = b"__LMDB_MISS__"

# How long readonly openers wait for the DB file to be created (by lmdb-sync)
_READONLY_WAIT_TIMEOUT = int(os.environ.get("NAM_LMDB_WAIT_TIMEOUT", "60"))


def _storable_rows(env, items) -> Tuple[List[Tuple[bytes, bytes]], List[bytes]]:
    """(rows LMDB can store as (key, value) bytes, keys it cannot store).

    LMDB rejects an empty key or one longer than ``env.max_key_size()``
    with MDB_BAD_VALSIZE, and a failed put aborts the write transaction.
    (Kept here, not imported from the attribute cache: the supervisor image
    ships this package without attribute/storage.)"""
    max_key = env.max_key_size()
    rows: List[Tuple[bytes, bytes]] = []
    skipped: List[bytes] = []
    for key, value in items:
        key_b = key if isinstance(key, bytes) else key.encode("utf-8")
        if not key_b or len(key_b) > max_key:
            skipped.append(key_b)
            continue
        rows.append((key_b, value if isinstance(value, bytes) else value.encode("utf-8")))
    return rows, skipped


class LmdbEntityCache:
    """Thin wrapper around an LMDB environment for surf::/cand:: keys.

    Parameters
    ----------
    path : str
        Directory for the LMDB data file.
    map_size : int
        Maximum database size in bytes (default 512 MB).
    readonly : bool
        Open in read-only mode (no write txns allowed).
        When True, waits up to NAM_LMDB_WAIT_TIMEOUT seconds for the
        DB file to be created by the lmdb-sync writer.
    """

    def __init__(
        self,
        path: str,
        map_size: int = 512 * 1024 * 1024,
        readonly: bool = False,
    ):
        self._path = path
        self._env = None
        self._available = False
        self._lock_fd = None
        self._readonly = readonly
        try:
            import lmdb

            os.makedirs(path, exist_ok=True)

            if readonly:
                # Wait for lmdb-sync to create the DB file before opening readonly
                # Uses jittered exponential backoff (base 0.25s, cap 5s)
                db_file = os.path.join(path, "data.mdb")
                deadline = time.monotonic() + _READONLY_WAIT_TIMEOUT
                waited = False
                delay = 0.25
                while not os.path.exists(db_file):
                    if time.monotonic() > deadline:
                        raise FileNotFoundError(
                            f"LMDB data file not found after {_READONLY_WAIT_TIMEOUT}s: {db_file} "
                            f"(lmdb-sync may not have started yet)")
                    if not waited:
                        logger.info("LmdbEntityCache: waiting for DB file %s (lmdb-sync creates it)...", db_file)
                        waited = True
                    time.sleep(delay + random.uniform(0, delay * 0.5))
                    delay = min(delay * 2, 5.0)
                if waited:
                    logger.info("LmdbEntityCache: DB file found after wait")

            # Acquire flock for initial open (serializes env creation across containers)
            lock_path = os.path.join(path, ".flock")
            self._lock_fd = open(lock_path, "w")
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
            try:
                self._env = lmdb.open(
                    path,
                    map_size=map_size,
                    readonly=readonly,
                    readahead=False,
                    meminit=False,
                    max_readers=64,
                    max_dbs=0,
                    lock=False,
                    writemap=not readonly,
                    create=not readonly,
                )
            finally:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)

            self._available = True
            stat = self._env.stat()
            logger.info(
                "LmdbEntityCache.open path=%s map_size=%dMB entries=%d readonly=%s",
                path, map_size // (1024 * 1024), stat["entries"], readonly,
            )
        except Exception as e:
            logger.warning("LmdbEntityCache.open FAILED path=%s: %s", path, e)
            self._env = None
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def get(self, key: str) -> Optional[bytes]:
        """Single key lookup.  Returns None on miss, _MISS_SENTINEL for
        negative cache entries, or the stored bytes on hit."""
        if not self._available:
            return None
        try:
            with self._env.begin(write=False) as txn:
                val = txn.get(key.encode("utf-8"))
            return val  # None if absent
        except Exception as e:
            logger.debug("lmdb.get error key=%s: %s", key, e)
            return None

    def put(self, key: str, value: bytes) -> bool:
        """Single key write, serialized via flock. Returns True on success."""
        if not self._available or self._readonly:
            return False
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
            try:
                with self._env.begin(write=True) as txn:
                    txn.put(key.encode("utf-8"), value)
            finally:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            return True
        except Exception as e:
            logger.debug("lmdb.put error key=%s: %s", key, e)
            return False

    def batch_get(self, keys: List[str]) -> Dict[str, bytes]:
        """Batch lookup in a single read transaction.

        Returns a dict mapping key -> value for keys that exist.
        Missing keys are omitted (caller should interpret absence as miss).
        Keys whose value is _MISS_SENTINEL indicate a negative cache entry.
        """
        if not self._available or not keys:
            return {}
        result: Dict[str, bytes] = {}
        try:
            with self._env.begin(write=False) as txn:
                for key in keys:
                    val = txn.get(key.encode("utf-8"))
                    if val is not None:
                        result[key] = val
        except Exception as e:
            logger.debug("lmdb.batch_get error (%d keys): %s", len(keys), e)
        return result

    def batch_put(self, items: List[Tuple[str, bytes]]) -> int:
        """Batch write in a single write transaction, serialized via flock.

        Returns the number of keys committed. A key LMDB cannot store (empty
        or over the max key size) is skipped and logged rather than aborting
        the transaction for the whole batch; any other failure aborts the
        transaction, so the batch counts 0.
        """
        if not self._available or not items or self._readonly:
            return 0
        rows, skipped = _storable_rows(self._env, items)
        if skipped:
            logger.warning("lmdb.batch_put skipped %d of %d keys LMDB cannot store "
                           "(empty or over %d bytes), first %r",
                           len(skipped), len(items), self._env.max_key_size(), skipped[0][:80])
        if not rows:
            return 0
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
            try:
                with self._env.begin(write=True) as txn:
                    for key_b, val_b in rows:
                        txn.put(key_b, val_b)
            finally:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        except Exception as e:
            logger.warning("lmdb.batch_put error (%d items, none written): %s", len(rows), e)
            return 0
        return len(rows)

    def stat(self) -> dict:
        """Return LMDB environment stats."""
        if not self._available:
            return {}
        try:
            return dict(self._env.stat())
        except Exception:
            return {}

    def close(self):
        """Close the LMDB environment."""
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None
            self._available = False
        if self._lock_fd is not None:
            try:
                self._lock_fd.close()
            except Exception:
                pass
            self._lock_fd = None
