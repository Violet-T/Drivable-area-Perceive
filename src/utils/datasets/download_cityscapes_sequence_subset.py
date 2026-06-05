"""按场景分块下载 Cityscapes leftImg8bit_sequence 子集。

该脚本使用 Cityscapes 官方登录会话，并通过 HTTP Range 读取 ZIP 中央目录
和目标文件的压缩数据块。它不会绕过 Cityscapes 账号与许可限制。
"""

from __future__ import annotations

import argparse
import csv
import getpass
import http.cookiejar
import os
import re
import ssl
import struct
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib import parse, request
from urllib.error import HTTPError, URLError

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


LOGIN_URL = "https://www.cityscapes-dataset.com/login/"
PACKAGE_URL_TEMPLATE = "https://www.cityscapes-dataset.com/file-handling/?packageID={package_id}"
DEFAULT_PACKAGE_ID = 14
MAX_EOCD_SEARCH_BYTES = 66_000


@dataclass(frozen=True)
class ZipEntry:
    name: str
    compress_type: int
    flag_bits: int
    crc32: int
    compressed_size: int
    file_size: int
    local_header_offset: int


@dataclass(frozen=True)
class SceneInfo:
    key: str
    short_key: str
    split: str
    city: str
    scene: str
    frame: int


class CityscapesRangeClient:
    def __init__(
        self,
        cookie_file: Path,
        timeout: int,
        insecure: bool = False,
        retries: int = 5,
        retry_sleep: float = 3.0,
    ) -> None:
        self.cookie_file = cookie_file
        self.timeout = timeout
        self.retries = max(1, retries)
        self.retry_sleep = max(0.0, retry_sleep)
        self.context = ssl._create_unverified_context() if insecure else None
        self.cookie_jar = http.cookiejar.MozillaCookieJar(str(cookie_file))
        if cookie_file.exists():
            self.cookie_jar.load(ignore_discard=True, ignore_expires=True)
        handlers = [request.HTTPCookieProcessor(self.cookie_jar)]
        if self.context is not None:
            handlers.append(request.HTTPSHandler(context=self.context))
        self.opener = request.build_opener(*handlers)

    def login(self, username: str | None, password: str | None) -> None:
        if not username:
            return
        if password is None:
            password = getpass.getpass("Cityscapes password: ")
        form = parse.urlencode(
            {"username": username, "password": password, "submit": "Login"}
        ).encode("utf-8")
        req = request.Request(LOGIN_URL, data=form, method="POST")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        with self.open(req) as resp:
            resp.read()
            if resp.status >= 400:
                raise RuntimeError(f"Cityscapes login failed with HTTP {resp.status}")
        self.cookie_file.parent.mkdir(parents=True, exist_ok=True)
        self.cookie_jar.save(ignore_discard=True, ignore_expires=True)

    def open(self, req: request.Request):
        return self.opener.open(req, timeout=self.timeout)

    def get_size(self, url: str) -> int:
        try:
            req = request.Request(url, method="HEAD")
            with self.open(req) as resp:
                length = resp.headers.get("Content-Length")
                if length and int(length) > 0:
                    return int(length)
        except (HTTPError, URLError, ValueError):
            pass

        req = request.Request(url, method="GET")
        req.add_header("Range", "bytes=0-0")
        with self.open(req) as resp:
            content_range = resp.headers.get("Content-Range", "")
            match = re.search(r"/(\d+)$", content_range)
            if resp.status != 206 or not match:
                raise RuntimeError(
                    "服务器没有返回 HTTP 206 Range 响应。无法只下载 ZIP 内部子场景，"
                    "请先确认登录有效，或改用官方 csDownload 整包下载。"
                )
            resp.read()
            return int(match.group(1))

    def fetch_range(self, url: str, start: int, end: int) -> bytes:
        if end < start:
            return b""
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                req = request.Request(url, method="GET")
                req.add_header("Range", f"bytes={start}-{end}")
                with self.open(req) as resp:
                    data = resp.read()
                    if resp.status != 206:
                        raise RuntimeError(
                            f"Range 请求未返回 HTTP 206：status={resp.status}, "
                            f"range={start}-{end}"
                        )
                    expected = end - start + 1
                    if len(data) != expected:
                        raise RuntimeError(
                            f"Range 数据长度不匹配：expected={expected}, got={len(data)}"
                        )
                    return data
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == self.retries:
                    break
                print(
                    f"Range retry {attempt}/{self.retries}: bytes={start}-{end}, error={exc}",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(self.retry_sleep * attempt)
        assert last_error is not None
        raise last_error

    def fetch_range_chunked(self, url: str, start: int, end: int, chunk_size: int) -> bytes:
        chunks: list[bytes] = []
        pos = start
        while pos <= end:
            chunk_end = min(pos + chunk_size - 1, end)
            chunks.append(self.fetch_range(url, pos, chunk_end))
            pos = chunk_end + 1
        return b"".join(chunks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="只下载 Cityscapes leftImg8bit_sequence 中的一两个连续场景"
    )
    parser.add_argument("--package-id", type=int, default=DEFAULT_PACKAGE_ID)
    parser.add_argument("--package-url", default=None)
    parser.add_argument("--output-root", default="/workspace/data/cityscapes")
    parser.add_argument("--cookie-file", default="/workspace/.cityscapes_cookies.txt")
    parser.add_argument("--username", default=os.environ.get("CITYSCAPES_USERNAME"))
    parser.add_argument("--password", default=os.environ.get("CITYSCAPES_PASSWORD"))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    parser.add_argument("--range-chunk-mib", type=float, default=4.0)
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--split", default=None)
    parser.add_argument("--city", default=None)
    parser.add_argument("--seq", default=None)
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-download-gib", type=float, default=10.0)
    parser.add_argument("--batch-range", action="store_true")
    parser.add_argument("--batch-max-mib", type=float, default=128.0)
    parser.add_argument("--batch-gap-kib", type=float, default=256.0)
    parser.add_argument("--list-scenes", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def package_url(args: argparse.Namespace) -> str:
    if args.package_url:
        return args.package_url
    return PACKAGE_URL_TEMPLATE.format(package_id=args.package_id)


def find_eocd(client: CityscapesRangeClient, url: str, zip_size: int) -> tuple[int, bytes, int]:
    tail_size = min(MAX_EOCD_SEARCH_BYTES, zip_size)
    tail_start = zip_size - tail_size
    tail = client.fetch_range(url, tail_start, zip_size - 1)
    signature = b"PK\x05\x06"
    pos = tail.rfind(signature)
    if pos < 0:
        raise RuntimeError("未找到 ZIP EOCD，文件可能不是 ZIP 或下载会话返回了 HTML。")
    return tail_start + pos, tail[pos:], tail_start


def read_central_directory_info(
    client: CityscapesRangeClient, url: str, zip_size: int
) -> tuple[int, int]:
    eocd_offset, eocd, _ = find_eocd(client, url, zip_size)
    if len(eocd) < 22:
        raise RuntimeError("ZIP EOCD 长度不足。")
    fields = struct.unpack_from("<4s4H2LH", eocd, 0)
    _, _, _, _, _, cd_size_32, cd_offset_32, _ = fields
    if cd_size_32 != 0xFFFFFFFF and cd_offset_32 != 0xFFFFFFFF:
        return int(cd_offset_32), int(cd_size_32)

    locator_offset = eocd_offset - 20
    locator = client.fetch_range(url, locator_offset, locator_offset + 19)
    if locator[:4] != b"PK\x06\x07":
        raise RuntimeError("需要 ZIP64，但未找到 ZIP64 EOCD locator。")
    _, _, zip64_eocd_offset, _ = struct.unpack_from("<4sLQL", locator, 0)
    zip64_eocd = client.fetch_range(url, zip64_eocd_offset, zip64_eocd_offset + 55)
    if zip64_eocd[:4] != b"PK\x06\x06":
        raise RuntimeError("ZIP64 EOCD signature 不匹配。")
    fields64 = struct.unpack_from("<4sQ2H2L4Q", zip64_eocd, 0)
    cd_size = fields64[-2]
    cd_offset = fields64[-1]
    return int(cd_offset), int(cd_size)


def parse_zip64_extra(extra: bytes, entry: ZipEntry) -> ZipEntry:
    values: list[int] = []
    pos = 0
    while pos + 4 <= len(extra):
        header_id, data_size = struct.unpack_from("<HH", extra, pos)
        pos += 4
        data = extra[pos : pos + data_size]
        pos += data_size
        if header_id == 0x0001:
            for idx in range(0, len(data), 8):
                if idx + 8 <= len(data):
                    values.append(struct.unpack_from("<Q", data, idx)[0])
            break

    value_index = 0
    file_size = entry.file_size
    compressed_size = entry.compressed_size
    local_header_offset = entry.local_header_offset
    if file_size == 0xFFFFFFFF:
        file_size = values[value_index]
        value_index += 1
    if compressed_size == 0xFFFFFFFF:
        compressed_size = values[value_index]
        value_index += 1
    if local_header_offset == 0xFFFFFFFF:
        local_header_offset = values[value_index]
    return ZipEntry(
        name=entry.name,
        compress_type=entry.compress_type,
        flag_bits=entry.flag_bits,
        crc32=entry.crc32,
        compressed_size=int(compressed_size),
        file_size=int(file_size),
        local_header_offset=int(local_header_offset),
    )


def read_entries(client: CityscapesRangeClient, url: str, range_chunk_bytes: int) -> list[ZipEntry]:
    zip_size = client.get_size(url)
    print(f"ZIP size: {zip_size / (1024 ** 3):.2f} GiB", flush=True)
    cd_offset, cd_size = read_central_directory_info(client, url, zip_size)
    print(f"central directory: offset={cd_offset}, size={cd_size / (1024 ** 2):.2f} MiB")
    cd = client.fetch_range_chunked(url, cd_offset, cd_offset + cd_size - 1, range_chunk_bytes)

    entries: list[ZipEntry] = []
    pos = 0
    while pos + 46 <= len(cd):
        if cd[pos : pos + 4] != b"PK\x01\x02":
            raise RuntimeError(f"central directory signature mismatch at {pos}")
        fields = struct.unpack_from("<4s6H3L5H2L", cd, pos)
        (
            _,
            _ver_made,
            _ver_needed,
            flag_bits,
            compress_type,
            _mod_time,
            _mod_date,
            crc32,
            compressed_size,
            file_size,
            name_len,
            extra_len,
            comment_len,
            _disk_start,
            _internal_attr,
            _external_attr,
            local_header_offset,
        ) = fields
        name_start = pos + 46
        extra_start = name_start + name_len
        comment_start = extra_start + extra_len
        name_bytes = cd[name_start:extra_start]
        encoding = "utf-8" if flag_bits & 0x800 else "cp437"
        name = name_bytes.decode(encoding)
        extra = cd[extra_start:comment_start]
        entry = ZipEntry(
            name=name,
            compress_type=compress_type,
            flag_bits=flag_bits,
            crc32=crc32,
            compressed_size=compressed_size,
            file_size=file_size,
            local_header_offset=local_header_offset,
        )
        if (
            file_size == 0xFFFFFFFF
            or compressed_size == 0xFFFFFFFF
            or local_header_offset == 0xFFFFFFFF
        ):
            entry = parse_zip64_extra(extra, entry)
        entries.append(entry)
        pos = comment_start + comment_len
    return entries


def parse_scene(entry_name: str) -> SceneInfo | None:
    match = re.search(
        r"(?:^|/)leftImg8bit_sequence/([^/]+)/([^/]+)/(.+?)_(\d{6})_leftImg8bit\.png$",
        entry_name,
    )
    if not match:
        return None
    split, city, prefix, frame = match.groups()
    scene = prefix
    key = f"{split}/{city}/{scene}"
    short_key = scene
    return SceneInfo(
        key=key,
        short_key=short_key,
        split=split,
        city=city,
        scene=scene,
        frame=int(frame),
    )


def normalized_targets(args: argparse.Namespace) -> set[str]:
    targets = set(args.scene)
    if args.city and args.seq:
        seq = f"{int(args.seq):06d}" if args.seq.isdigit() else args.seq
        short = f"{args.city}_{seq}"
        targets.add(short)
        if args.split:
            targets.add(f"{args.split}/{args.city}/{short}")
    return targets


def select_entries(entries: Iterable[ZipEntry], args: argparse.Namespace) -> list[tuple[ZipEntry, SceneInfo]]:
    targets = normalized_targets(args)
    selected: list[tuple[ZipEntry, SceneInfo]] = []
    for entry in entries:
        info = parse_scene(entry.name)
        if info is None:
            continue
        if args.split and info.split != args.split:
            continue
        if args.city and info.city != args.city:
            continue
        if targets and info.key not in targets and info.short_key not in targets:
            continue
        if args.frame_start is not None and info.frame < args.frame_start:
            continue
        if args.frame_end is not None and info.frame > args.frame_end:
            continue
        selected.append((entry, info))
    selected.sort(key=lambda item: (item[1].key, item[1].frame))
    if args.max_files is not None:
        selected = selected[: args.max_files]
    return selected


def list_scenes(entries: Iterable[ZipEntry], args: argparse.Namespace) -> None:
    stats: dict[str, dict[str, object]] = {}
    for entry in entries:
        info = parse_scene(entry.name)
        if info is None:
            continue
        if args.split and info.split != args.split:
            continue
        if args.city and info.city != args.city:
            continue
        scene_stats = stats.setdefault(
            info.key,
            {"frames": [], "compressed_size": 0, "file_size": 0},
        )
        scene_stats["frames"].append(info.frame)
        scene_stats["compressed_size"] += entry.compressed_size
        scene_stats["file_size"] += entry.file_size
    for key in sorted(stats):
        frames = stats[key]["frames"]
        compressed_gib = stats[key]["compressed_size"] / (1024**3)
        file_gib = stats[key]["file_size"] / (1024**3)
        print(
            f"{key}, frames={len(frames)}, range={min(frames):06d}-{max(frames):06d}, "
            f"zip={compressed_gib:.3f} GiB, png={file_gib:.3f} GiB"
        )


def summarize_selection(selected: list[tuple[ZipEntry, SceneInfo]]) -> None:
    compressed_size = sum(entry.compressed_size for entry, _ in selected)
    file_size = sum(entry.file_size for entry, _ in selected)
    scenes = sorted({info.key for _, info in selected})
    print(f"selected scenes: {len(scenes)}")
    for scene in scenes:
        scene_rows = [(entry, info) for entry, info in selected if info.key == scene]
        frames = [info.frame for _, info in scene_rows]
        scene_compressed = sum(entry.compressed_size for entry, _ in scene_rows)
        scene_file_size = sum(entry.file_size for entry, _ in scene_rows)
        print(
            f"  {scene}: files={len(scene_rows)}, "
            f"range={min(frames):06d}-{max(frames):06d}, "
            f"zip={scene_compressed / (1024**3):.3f} GiB, "
            f"png={scene_file_size / (1024**3):.3f} GiB"
        )
    print(
        f"selected size: zip={compressed_size / (1024**3):.3f} GiB, "
        f"png={file_size / (1024**3):.3f} GiB"
    )


def safe_output_path(output_root: Path, entry_name: str) -> Path:
    relative = Path(entry_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"不安全的 ZIP 路径：{entry_name}")
    return output_root / relative


def read_local_payload(client: CityscapesRangeClient, url: str, entry: ZipEntry) -> bytes:
    header = client.fetch_range(url, entry.local_header_offset, entry.local_header_offset + 29)
    if header[:4] != b"PK\x03\x04":
        raise RuntimeError(f"local header signature mismatch: {entry.name}")
    fields = struct.unpack_from("<4s5H3L2H", header, 0)
    _, _, flag_bits, compress_type, _, _, _, _, _, name_len, extra_len = fields
    if flag_bits & 0x1:
        raise RuntimeError(f"不支持加密 ZIP entry：{entry.name}")
    if compress_type != entry.compress_type:
        raise RuntimeError(f"compression method mismatch: {entry.name}")
    data_start = entry.local_header_offset + 30 + name_len + extra_len
    data_end = data_start + entry.compressed_size - 1
    return client.fetch_range(url, data_start, data_end)


def extract_entry(client: CityscapesRangeClient, url: str, entry: ZipEntry, output_path: Path) -> None:
    compressed = read_local_payload(client, url, entry)
    if entry.compress_type == 0:
        payload = compressed
    elif entry.compress_type == 8:
        payload = zlib.decompress(compressed, -15)
    else:
        raise RuntimeError(f"不支持的 ZIP 压缩方法 {entry.compress_type}: {entry.name}")
    if len(payload) != entry.file_size:
        raise RuntimeError(
            f"解压大小不匹配：{entry.name}, expected={entry.file_size}, got={len(payload)}"
        )
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    if crc != entry.crc32:
        raise RuntimeError(f"CRC 校验失败：{entry.name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)


def estimate_local_entry_end(entry: ZipEntry) -> int:
    name_bytes = entry.name.encode("utf-8" if entry.flag_bits & 0x800 else "cp437")
    # local extra field 通常很小；这里多取 4 KiB，后续会用 local header 精确定位。
    return entry.local_header_offset + 30 + len(name_bytes) + 4096 + entry.compressed_size - 1


def group_entries_by_local_offset(
    rows: list[tuple[ZipEntry, SceneInfo]],
    max_group_bytes: int,
    max_gap_bytes: int,
) -> list[list[tuple[ZipEntry, SceneInfo]]]:
    sorted_rows = sorted(rows, key=lambda item: item[0].local_header_offset)
    groups: list[list[tuple[ZipEntry, SceneInfo]]] = []
    current: list[tuple[ZipEntry, SceneInfo]] = []
    current_start = 0
    current_end = 0
    for row in sorted_rows:
        entry = row[0]
        entry_start = entry.local_header_offset
        entry_end = estimate_local_entry_end(entry)
        if not current:
            current = [row]
            current_start = entry_start
            current_end = entry_end
            continue
        gap = entry_start - current_end - 1
        merged_end = max(current_end, entry_end)
        merged_size = merged_end - current_start + 1
        if gap <= max_gap_bytes and merged_size <= max_group_bytes:
            current.append(row)
            current_end = merged_end
        else:
            groups.append(current)
            current = [row]
            current_start = entry_start
            current_end = entry_end
    if current:
        groups.append(current)
    return groups


def payload_from_group_block(block: bytes, base_offset: int, entry: ZipEntry) -> bytes:
    local_pos = entry.local_header_offset - base_offset
    if local_pos < 0 or local_pos + 30 > len(block):
        raise RuntimeError(f"batch block 未覆盖 local header：{entry.name}")
    header = block[local_pos : local_pos + 30]
    if header[:4] != b"PK\x03\x04":
        raise RuntimeError(f"local header signature mismatch: {entry.name}")
    fields = struct.unpack_from("<4s5H3L2H", header, 0)
    _, _, flag_bits, compress_type, _, _, _, _, _, name_len, extra_len = fields
    if flag_bits & 0x1:
        raise RuntimeError(f"不支持加密 ZIP entry：{entry.name}")
    if compress_type != entry.compress_type:
        raise RuntimeError(f"compression method mismatch: {entry.name}")
    data_start = local_pos + 30 + name_len + extra_len
    data_end = data_start + entry.compressed_size
    if data_end > len(block):
        raise RuntimeError(f"batch block 未覆盖压缩数据：{entry.name}")
    return block[data_start:data_end]


def write_decompressed_payload(entry: ZipEntry, compressed: bytes, output_path: Path) -> None:
    if entry.compress_type == 0:
        payload = compressed
    elif entry.compress_type == 8:
        payload = zlib.decompress(compressed, -15)
    else:
        raise RuntimeError(f"不支持的 ZIP 压缩方法 {entry.compress_type}: {entry.name}")
    if len(payload) != entry.file_size:
        raise RuntimeError(
            f"解压大小不匹配：{entry.name}, expected={entry.file_size}, got={len(payload)}"
        )
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    if crc != entry.crc32:
        raise RuntimeError(f"CRC 校验失败：{entry.name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)


def extract_entries_batched(
    client: CityscapesRangeClient,
    url: str,
    selected: list[tuple[ZipEntry, SceneInfo]],
    output_root: Path,
    overwrite: bool,
    max_group_bytes: int,
    max_gap_bytes: int,
) -> list[tuple[ZipEntry, SceneInfo]]:
    pending = [
        row
        for row in selected
        if overwrite or not safe_output_path(output_root, row[0].name).exists()
    ]
    skipped = [
        row
        for row in selected
        if not overwrite and safe_output_path(output_root, row[0].name).exists()
    ]
    groups = group_entries_by_local_offset(
        pending,
        max_group_bytes=max_group_bytes,
        max_gap_bytes=max_gap_bytes,
    )
    print(f"batch groups: {len(groups)}, skipped existing files: {len(skipped)}")

    iterator = groups
    if tqdm is not None:
        iterator = tqdm(groups, desc="extract sequence batches", unit="batch")

    written = list(skipped)
    for group in iterator:
        group_start = min(entry.local_header_offset for entry, _ in group)
        group_end = max(estimate_local_entry_end(entry) for entry, _ in group)
        block = client.fetch_range(url, group_start, group_end)
        for entry, info in group:
            output_path = safe_output_path(output_root, entry.name)
            compressed = payload_from_group_block(block, group_start, entry)
            write_decompressed_payload(entry, compressed, output_path)
            written.append((entry, info))
    return written


def write_manifest(output_root: Path, rows: list[tuple[ZipEntry, SceneInfo]]) -> None:
    manifest_path = output_root / "cityscapes_sequence_subset_manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "relative_path",
                "split",
                "city",
                "scene",
                "frame",
                "file_size",
                "compressed_size",
            ]
        )
        for entry, info in rows:
            writer.writerow(
                [
                    entry.name,
                    info.split,
                    info.city,
                    info.scene,
                    f"{info.frame:06d}",
                    entry.file_size,
                    entry.compressed_size,
                ]
            )
    print(f"manifest: {manifest_path}")


def main() -> None:
    args = parse_args()
    url = package_url(args)
    output_root = Path(args.output_root)
    cookie_file = Path(args.cookie_file)

    client = CityscapesRangeClient(
        cookie_file=cookie_file,
        timeout=args.timeout,
        insecure=args.insecure,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
    )
    client.login(args.username, args.password)

    start_time = time.time()
    range_chunk_bytes = max(1, int(args.range_chunk_mib * 1024 * 1024))
    entries = read_entries(client, url, range_chunk_bytes=range_chunk_bytes)
    print(f"ZIP entries: {len(entries)}")

    if args.list_scenes:
        list_scenes(entries, args)
        return

    selected = select_entries(entries, args)
    if not selected:
        raise RuntimeError("没有匹配到任何 sequence 文件，请先使用 --list-scenes 查看可用场景。")
    print(f"selected files: {len(selected)}")
    summarize_selection(selected)
    selected_compressed_gib = sum(entry.compressed_size for entry, _ in selected) / (1024**3)
    if args.max_download_gib is not None and selected_compressed_gib > args.max_download_gib:
        raise RuntimeError(
            f"选中的 ZIP 压缩数据约 {selected_compressed_gib:.3f} GiB，"
            f"超过 --max-download-gib={args.max_download_gib:.3f}。"
            "请减少 --scene、缩小 --frame-start/--frame-end，或显式调大限制。"
        )
    if args.dry_run:
        for entry, info in selected:
            print(f"{info.key}/{info.frame:06d}: {entry.name}")
        return

    if args.batch_range:
        written = extract_entries_batched(
            client=client,
            url=url,
            selected=selected,
            output_root=output_root,
            overwrite=args.overwrite,
            max_group_bytes=max(1, int(args.batch_max_mib * 1024 * 1024)),
            max_gap_bytes=max(0, int(args.batch_gap_kib * 1024)),
        )
    else:
        iterator = selected
        if tqdm is not None:
            iterator = tqdm(selected, desc="extract sequence files", unit="file")

        written: list[tuple[ZipEntry, SceneInfo]] = []
        for entry, info in iterator:
            output_path = safe_output_path(output_root, entry.name)
            if output_path.exists() and not args.overwrite:
                written.append((entry, info))
                continue
            extract_entry(client, url, entry, output_path)
            written.append((entry, info))

    write_manifest(output_root, written)
    elapsed = time.time() - start_time
    print(f"done: {len(written)} files, elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise
