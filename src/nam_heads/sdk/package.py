"""Plugin packages: one implementation of one slot as one versioned archive.

A host (NAM) ships with no heads, only a factory; every head -- the defaults included -- is installed into a
running cluster as a package and loaded at runtime. A package carries everything the implementation needs:

    <slot>.<name>-<version>.nhp       a gzip'd tar, byte-for-byte reproducible from the same inputs
      plugin.json                     the manifest: format, id, IMPL_META as data, every file with its sha256
      pkg/...                         the implementation's package directory: head.py, optional learner.py,
                                      its modules, data and models (tests and caches left out)
      deps/*.whl                      its own Python dependencies, as wheels for the host's platform

The host installs ``deps/`` into the plugin's own environment and imports ``pkg/`` under a name of its
choosing, so a package must not import itself by an absolute path (relative imports only), and reaches the
host through ``nam_heads.sdk`` and the host's documented modules.

    nam-heads pack <dir> [-o OUT_DIR] [--wheel W ...] [--wheel-dir D]
    nam-heads verify <archive>
    nam-heads info <archive>
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .impl_meta import (FACET_CONTRACT, IMPL_NAME_RE, SLOT_CONTRACTS, SLOT_NAME_RE, learner_problems,
                        plugin_meta_problems, read_module_literal)

FORMAT = 1
SUFFIX = ".nhp"
MANIFEST = "plugin.json"
EXCLUDE_DIRS = {"tests", "__pycache__", ".pytest_cache", ".mypy_cache"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


class PackageError(ValueError):
    """A package cannot be built, or an archive is malformed or does not match its manifest."""


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def plugin_id(meta: Dict[str, Any]) -> str:
    return f"{meta['slot']}.{meta['name']}@{meta['version']}"


def _source_files(src: Path) -> List[Tuple[str, Path]]:
    out = []
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if any(part in EXCLUDE_DIRS for part in rel.parts) or p.suffix in EXCLUDE_SUFFIXES:
            continue
        if p.is_symlink():
            raise PackageError(f"{rel}: symlinks cannot be packed")
        if p.is_file():
            out.append((rel.as_posix(), p))
    return out


def meta_problems(meta: Any, head: Path) -> List[str]:
    """Everything that keeps ``meta`` (head.py's IMPL_META) from being packed."""
    if not isinstance(meta, dict):
        return ["head.py has no literal IMPL_META dict"]
    probs = [f"IMPL_META lacks {k!r}" for k in ("slot", "name", "version", "contract") if not meta.get(k)]
    if probs:
        return probs
    if not SLOT_NAME_RE.match(str(meta["slot"])) or not IMPL_NAME_RE.match(str(meta["name"])):
        probs.append("IMPL_META slot and name must match [a-z][a-z0-9_]{1,31}")
    want = SLOT_CONTRACTS.get(meta["slot"], FACET_CONTRACT)
    if meta["contract"] != want:
        probs.append(f"contract {meta['contract']!r} != slot {meta['slot']!r} contract {want!r}")
    probs += plugin_meta_problems(meta, require_output=True)
    probs += learner_problems(meta, head)
    return probs


def pack(src_dir, out_dir=".", wheels: Iterable = (), wheel_dir=None) -> Path:
    """Build ``<slot>.<name>-<version>.nhp`` from an implementation's package directory (the one holding
    head.py). ``wheels`` / ``wheel_dir``: its dependency wheels. Returns the archive's path."""
    src = Path(src_dir).resolve()
    head = src / "head.py"
    if not head.is_file():
        raise PackageError(f"{src}: no head.py")
    meta = read_module_literal(head, ("IMPL_META",)).get("IMPL_META")
    probs = meta_problems(meta, head)
    if probs:
        raise PackageError(f"{src}: " + "; ".join(probs))
    wheel_paths = [Path(w) for w in wheels]
    if wheel_dir:
        wheel_paths += sorted(Path(wheel_dir).glob("*.whl"))
    for w in wheel_paths:
        if w.suffix != ".whl" or not w.is_file():
            raise PackageError(f"{w}: not a wheel file")
    entries: List[Tuple[str, Path]] = [(f"pkg/{rel}", p) for rel, p in _source_files(src)]
    entries += [(f"deps/{w.name}", w) for w in sorted(set(wheel_paths), key=lambda x: x.name)]
    names = [n for n, _ in entries]
    if len(set(names)) != len(names):
        raise PackageError("two files would land at the same path in the package")
    manifest = {"format": FORMAT, "id": plugin_id(meta), "slot": meta["slot"], "name": meta["name"],
                "version": meta["version"], "package": src.name, "meta": meta,
                "files": {n: sha256_file(p) for n, p in entries}}
    out = Path(out_dir) / f"{meta['slot']}.{meta['name']}-{meta['version']}{SUFFIX}"
    out.parent.mkdir(parents=True, exist_ok=True)
    blob = io.BytesIO()
    with gzip.GzipFile(fileobj=blob, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
            data = json.dumps(manifest, sort_keys=True, indent=1).encode()
            _add(tar, MANIFEST, data)
            for name, p in entries:
                _add(tar, name, p.read_bytes())
    out.write_bytes(blob.getvalue())
    return out


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size, info.mtime, info.mode = len(data), 0, 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    tar.addfile(info, io.BytesIO(data))


def _safe_name(name: str) -> bool:
    p = PurePosixPath(name)
    return bool(name) and not p.is_absolute() and ".." not in p.parts and "\\" not in name


def read_manifest(archive) -> Dict[str, Any]:
    """plugin.json of an archive (not verified: use ``verify``)."""
    with tarfile.open(archive, mode="r:gz") as tar:
        m = tar.extractfile(MANIFEST) if MANIFEST in tar.getnames() else None
        if m is None:
            raise PackageError(f"{archive}: no {MANIFEST}")
        return json.loads(m.read())


def verify(archive) -> Dict[str, Any]:
    """The manifest of ``archive`` once every member is a regular file at a safe path, every file the
    manifest names is present with its sha256, and nothing else is in it. Raises PackageError."""
    try:
        tar = tarfile.open(archive, mode="r:gz")
    except (tarfile.TarError, OSError, EOFError) as e:
        raise PackageError(f"{archive}: not a package ({e})") from e
    with tar:
        manifest, seen = None, {}
        for m in tar.getmembers():
            if not _safe_name(m.name):
                raise PackageError(f"{archive}: unsafe path {m.name!r}")
            if not m.isfile():
                raise PackageError(f"{archive}: {m.name!r} is not a regular file")
            data = tar.extractfile(m).read()
            if m.name == MANIFEST:
                manifest = json.loads(data)
            elif m.name in seen:
                raise PackageError(f"{archive}: {m.name!r} appears twice")
            else:
                seen[m.name] = _sha256_bytes(data)
    if manifest is None:
        raise PackageError(f"{archive}: no {MANIFEST}")
    if manifest.get("format") != FORMAT:
        raise PackageError(f"{archive}: format {manifest.get('format')!r} != {FORMAT}")
    files = manifest.get("files") or {}
    missing = sorted(set(files) - set(seen))
    extra = sorted(set(seen) - set(files))
    bad = sorted(n for n in files if n in seen and seen[n] != files[n])
    if missing or extra or bad:
        raise PackageError(f"{archive}: manifest mismatch (missing {missing[:3]}, extra {extra[:3]}, "
                           f"checksum {bad[:3]})")
    if not all(n.startswith(("pkg/", "deps/")) for n in files) or "pkg/head.py" not in files:
        raise PackageError(f"{archive}: files must be under pkg/ or deps/, with pkg/head.py")
    meta = manifest.get("meta") or {}
    if plugin_id(meta) != manifest.get("id"):
        raise PackageError(f"{archive}: id {manifest.get('id')!r} does not match its IMPL_META")
    return manifest


def unpack(archive, dest) -> Path:
    """Verify ``archive`` and extract it into ``dest`` (plugin.json, pkg/, deps/). Returns ``dest``."""
    manifest = verify(archive)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, mode="r:gz") as tar:
        for m in tar.getmembers():
            target = dest / m.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(tar.extractfile(m).read())
    (dest / MANIFEST).write_text(json.dumps(manifest, sort_keys=True, indent=1))
    return dest


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="nam-heads", description="Build and check NAM head plugin packages.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack", help="build a package from an implementation directory")
    p.add_argument("src")
    p.add_argument("-o", "--out-dir", default=".")
    p.add_argument("--wheel", action="append", default=[])
    p.add_argument("--wheel-dir")
    for name in ("verify", "info"):
        q = sub.add_parser(name)
        q.add_argument("archive")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "pack":
            out = pack(a.src, a.out_dir, a.wheel, a.wheel_dir)
            print(f"{out}  sha256={sha256_file(out)}")
        elif a.cmd == "verify":
            m = verify(a.archive)
            print(f"ok {m['id']}  files={len(m['files'])}  sha256={sha256_file(Path(a.archive))}")
        else:
            m = read_manifest(a.archive)
            print(json.dumps({k: m.get(k) for k in ("id", "package", "meta")}, indent=1, sort_keys=True))
    except PackageError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
