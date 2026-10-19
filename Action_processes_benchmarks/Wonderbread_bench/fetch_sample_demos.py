"""Fetch ONLY the demos the judge-pilot sample needs, out of Wonderbread's Zenodo archive.

The full gold set is 33 GB (`gold_demos.zip`, with videos) — the scope question on
`.claude/references/benchmarks/action/wonderbread.md`. The pilot needs neither the videos nor most
demos: QA and SOP Generation read each demo's SOP, trace .json and screenshots, which Zenodo also
ships without videos as `gold_demos_sop_and_trace_and_screenshots_only.zip` (5.7 GB). Zenodo serves
byte ranges, so this reads the zip's central directory (a few requests) and then only the members
under the sampled demo folders — ~31 demos at a median 6.4 MB. zipfile checks every member's CRC.

Writes upstream's own layout, `data/demos/<demo folder>/{SOP - ….txt, ….json, screenshots/}`
(gitignored), and is idempotent: a file already present at the right size is skipped.

    python fetch_sample_demos.py            # what the sample needs (wb_eval_core.demos_needed)
    python fetch_sample_demos.py --check    # verify what is on disk, fetch nothing
"""

import argparse
import os
import sys
import time
import urllib.request
import zipfile

WB_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WB_ROOT)
import wb_eval_core as wb  # noqa: E402

RECORD = "https://zenodo.org/api/records/14094162/files"
ARCHIVE = "gold_demos_sop_and_trace_and_screenshots_only.zip"
PREFIX = "gold_demos_sop_and_trace_and_screenshots_only/"


class HTTPRangeFile:
    """A read-only, seekable file over HTTP range requests, with a small block cache — enough for
    zipfile, which seeks to the central directory and then to each member's local header."""
    BLOCK = 1 << 20

    def __init__(self, url, retries=5):
        self.url, self.pos, self.requests, self.retries, self.cache = url, 0, 0, retries, {}
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def _block(self, b):
        if b not in self.cache:
            start = b * self.BLOCK
            end = min(start + self.BLOCK, self.size) - 1
            for attempt in range(self.retries):
                try:
                    req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
                    with urllib.request.urlopen(req, timeout=120) as r:
                        if r.status != 206:
                            raise IOError(f"expected 206 Partial Content, got {r.status}")
                        data = r.read()
                    break
                except Exception as error:
                    print(f"  range {start}-{end}: {type(error).__name__}: {error} "
                          f"({attempt + 1}/{self.retries})", flush=True)
                    if attempt + 1 == self.retries:
                        raise
                    time.sleep(5 * (attempt + 1))
            self.requests += 1
            self.cache[b] = data
            if len(self.cache) > 64:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[b]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        out = bytearray()
        while n > 0 and self.pos < self.size:
            b, off = divmod(self.pos, self.BLOCK)
            chunk = self._block(b)[off:off + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="verify what is on disk; fetch nothing")
    args = ap.parse_args()
    need = wb.demos_needed()
    if not args.check:
        remote = HTTPRangeFile(f"{RECORD}/{ARCHIVE}/content")
        archive = zipfile.ZipFile(remote)
        members = [m for m in archive.infolist() if m.filename.startswith(PREFIX)
                   and m.filename[len(PREFIX):].split("/")[0] in need and not m.is_dir()
                   and not os.path.basename(m.filename).startswith("._")]
        found = {m.filename[len(PREFIX):].split("/")[0] for m in members}
        absent = sorted(set(need) - found)
        if absent:
            raise SystemExit(f"{len(absent)} needed demos are not in {ARCHIVE}: {absent[:5]}")
        total = sum(m.file_size for m in members)
        print(f"{len(need)} demos, {len(members)} files, {total / 1e6:.0f} MB "
              f"(central directory: {remote.requests} requests)", flush=True)
        written = 0
        for i, m in enumerate(members, 1):
            target = os.path.join(wb.DEMOS_DIR, m.filename[len(PREFIX):])
            if os.path.exists(target) and os.path.getsize(target) == m.file_size:
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            data = archive.read(m)                 # raises BadZipFile on a CRC mismatch
            with open(target + ".part", "wb") as fh:
                fh.write(data)
            os.replace(target + ".part", target)
            written += len(data)
            if i % 200 == 0:
                print(f"  {i}/{len(members)} files, {written / 1e6:.0f} MB written", flush=True)
        print(f"wrote {written / 1e6:.0f} MB in {remote.requests} HTTP requests", flush=True)
    problems = [p for demo in need for p in wb.check_demo(demo)]
    if problems:
        print("\n".join(problems))
        raise SystemExit(f"{len(problems)} problem(s) with the demos on disk")
    print(f"OK: all {len(need)} needed demos present under {wb.DEMOS_DIR}")


if __name__ == "__main__":
    main()
