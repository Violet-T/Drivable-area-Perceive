#!/usr/bin/env python3
"""Download a size-limited BDD100K video-scene subset.

The official BDD100K download portal may require a logged-in browser session.
This script does not bypass access control; pass an exported cookie file or
direct URLs after you have legitimate access to the dataset.
"""

from __future__ import annotations

import argparse
import contextlib
import html.parser
import json
import logging
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_DOWNLOAD_PAGE = "http://bdd-data.berkeley.edu/download.html"
DEFAULT_BDD_VIDEO_URL = "http://128.32.162.150/bdd100k/bdd100k_videos.zip"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
ARCHIVE_EXTENSIONS = {".zip"}
LABEL_EXTENSIONS = {".json", ".png", ".jpg", ".jpeg", ".txt", ".csv"}
EOCD_SIGNATURE = b"PK\x05\x06"
ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"
ZIP64_EOCD_SIGNATURE = b"PK\x06\x06"
CENTRAL_DIR_SIGNATURE = b"PK\x01\x02"
LOCAL_FILE_SIGNATURE = b"PK\x03\x04"


class DownloadLimitExceeded(RuntimeError):
    """Raised when a network transfer or final output would exceed the limit."""


class AnchorParser(html.parser.HTMLParser):
    """Small stdlib-only anchor parser for the BDD download page."""

    def __init__(self) -> None:
        super().__init__()
        self._current_href: str | None = None
        self._current_text: list[str] = []
        self.links: list[tuple[str, str]] = []
        self.raw_links: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        if tag.lower() == "a" and attr_map.get("href"):
            href = attr_map["href"]
            self._current_href = href
            self._current_text = []
        for value in attr_map.values():
            if value:
                self.raw_links.extend(extract_urls(value))

    def handle_data(self, data: str) -> None:
        if self._current_href is not None:
            self._current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or self._current_href is None:
            return
        self.links.append((self._current_href, " ".join(self._current_text).strip()))
        self._current_href = None
        self._current_text = []


@dataclass
class UrlInfo:
    content_length: int | None
    accepts_ranges: bool


@dataclass
class RemoteZipMember:
    filename: str
    compress_type: int
    flag_bits: int
    compress_size: int
    file_size: int
    header_offset: int
    crc32: int


@dataclass
class Budget:
    max_bytes: int
    root: Path
    network_bytes: int = 0

    def final_bytes(self) -> int:
        return directory_size(self.root)

    def ensure_network(self, incoming_bytes: int) -> None:
        if self.network_bytes + incoming_bytes > self.max_bytes:
            raise DownloadLimitExceeded(
                f"下载流量将超过限制: "
                f"{human_bytes(self.network_bytes + incoming_bytes)} > {human_bytes(self.max_bytes)}"
            )

    def add_network(self, incoming_bytes: int) -> None:
        self.ensure_network(incoming_bytes)
        self.network_bytes += incoming_bytes

    def ensure_final_file(self, file_bytes: int) -> None:
        projected = self.final_bytes() + file_bytes
        if projected > self.max_bytes:
            raise DownloadLimitExceeded(
                f"最终数据目录将超过限制: {human_bytes(projected)} > {human_bytes(self.max_bytes)}"
            )


@dataclass
class DownloadContext:
    output_dir: Path
    cache_dir: Path
    scenes_dir: Path
    manifest_dir: Path
    headers: dict[str, str]
    budget: Budget
    keep_archives: bool
    target_stems: set[str] = field(default_factory=set)
    selected_stems: list[str] = field(default_factory=list)
    selection_mode: str = "first"
    random_seed: int = 42

    @property
    def selected_set(self) -> set[str]:
        return set(self.selected_stems)

    def allows_scene(self, stem: str) -> bool:
        return not self.target_stems or stem in self.target_stems


@dataclass
class BDDSceneCandidate:
    scene_id: str
    attributes: dict[str, str]


def human_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024
    return f"{value} B"


def parse_size(value: str) -> int:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kmgt]?i?b?|[kmgt])?\s*", value, re.I)
    if not match:
        raise argparse.ArgumentTypeError(f"无法解析大小限制: {value}")
    number = float(match.group(1))
    unit = (match.group(2) or "b").lower()
    multipliers = {
        "b": 1,
        "": 1,
        "k": 1024,
        "kb": 1024,
        "kib": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "mib": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
        "gib": 1024**3,
        "t": 1024**4,
        "tb": 1024**4,
        "tib": 1024**4,
    }
    return int(number * multipliers[unit])


def parse_int_set(value: str) -> set[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        return set()
    try:
        return {int(item) for item in items}
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"无法解析整数集合: {value}") from exc


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file() and not item.is_symlink():
            total += item.stat().st_size
    return total


def scene_stem(path_or_name: str) -> str:
    name = Path(urllib.parse.urlparse(path_or_name).path).name
    stem = Path(name).stem
    if stem.endswith(".part"):
        stem = Path(stem).stem
    return stem


def label_scene_stem(path_or_name: str) -> str:
    """从 BDD100K label/mask 文件名中还原 scene id。"""
    stem = scene_stem(path_or_name)
    suffixes = (
        "_drivable_color",
        "_drivable_id",
        "_train_color",
        "_train_id",
        "_val_color",
        "_val_id",
        "_color",
        "_id",
    )
    for suffix in suffixes:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def safe_filename(value: str) -> str:
    name = Path(urllib.parse.urlparse(value).path).name or "download.bin"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def build_headers(cookie: str | None, cookie_file: Path | None) -> dict[str, str]:
    headers = {
        "User-Agent": "Perceive-BDD100K-downloader/1.0",
        "Accept": "*/*",
    }
    cookie_value = cookie or read_cookie_file(cookie_file)
    if cookie_value:
        headers["Cookie"] = cookie_value
    return headers


def read_cookie_file(cookie_file: Path | None) -> str | None:
    if cookie_file is None:
        return None
    raw = cookie_file.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    if raw.lower().startswith("cookie:"):
        return raw.split(":", 1)[1].strip()
    if "=" in raw and "\n" not in raw:
        return raw

    pairs: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Netscape cookie format: domain, flag, path, secure, expiry, name, value
        parts = line.split("\t")
        if len(parts) >= 7:
            pairs.append(f"{parts[-2]}={parts[-1]}")
    return "; ".join(pairs) if pairs else None


def request_with_headers(url: str, headers: dict[str, str], method: str = "GET") -> urllib.request.Request:
    return urllib.request.Request(url, headers=headers, method=method)


def extract_urls(text: str) -> list[tuple[str, str]]:
    """从 HTML 属性、onclick 和普通文本中提取 URL。"""
    result: list[tuple[str, str]] = []
    patterns = [
        r"""window\.open\(\s*['"]([^'"]+)['"]""",
        r"""https?://[^\s"'<>);]+""",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I):
            url = match.group(1) if match.lastindex else match.group(0)
            result.append((url, ""))
    return result


def get_url_info(url: str, headers: dict[str, str]) -> UrlInfo:
    try:
        with urllib.request.urlopen(request_with_headers(url, headers, "HEAD"), timeout=30) as response:
            length = response.headers.get("Content-Length")
            ranges = response.headers.get("Accept-Ranges", "")
            return UrlInfo(
                content_length=int(length) if length else None,
                accepts_ranges="bytes" in ranges.lower(),
            )
    except (urllib.error.URLError, ValueError, TimeoutError):
        return UrlInfo(content_length=None, accepts_ranges=False)


def get_content_length(url: str, headers: dict[str, str]) -> int | None:
    return get_url_info(url, headers).content_length


def request_range(
    url: str,
    start: int,
    end: int,
    context: DownloadContext,
    timeout: int = 60,
) -> urllib.response.addinfourl:
    headers = dict(context.headers)
    headers["Range"] = f"bytes={start}-{end}"
    request = urllib.request.Request(url, headers=headers)
    response = urllib.request.urlopen(request, timeout=timeout)
    if getattr(response, "status", None) != 206:
        response.close()
        raise RuntimeError(f"服务器未返回 206 Partial Content，无法分块读取: {url}")
    return response


def read_range(url: str, start: int, end: int, context: DownloadContext) -> bytes:
    with request_range(url, start, end, context) as response:
        data = response.read()
    context.budget.add_network(len(data))
    return data


def discover_download_links(page_url: str, headers: dict[str, str]) -> dict[str, list[str]]:
    logging.info("抓取下载页: %s", page_url)
    with urllib.request.urlopen(request_with_headers(page_url, headers), timeout=60) as response:
        body = response.read().decode("utf-8", errors="replace")

    parser = AnchorParser()
    parser.feed(body)

    video_urls: list[str] = []
    label_urls: list[str] = []
    all_urls: list[str] = []
    link_candidates = parser.links + parser.raw_links + extract_urls(body)
    for href, text in link_candidates:
        absolute = urllib.parse.urljoin(page_url, href)
        lowered = f"{absolute} {text}".lower()
        suffix = Path(urllib.parse.urlparse(absolute).path).suffix.lower()
        all_urls.append(absolute)
        if ("video" in lowered or "videos" in lowered) and (
            suffix in ARCHIVE_EXTENSIONS or suffix in VIDEO_EXTENSIONS
        ):
            video_urls.append(absolute)
        if any(token in lowered for token in ("label", "labels", "drivable", "track", "seg")) and (
            suffix in ARCHIVE_EXTENSIONS or suffix in LABEL_EXTENSIONS
        ):
            label_urls.append(absolute)

    return {
        "video_urls": unique_keep_order(video_urls),
        "label_urls": unique_keep_order(label_urls),
        "all_urls": unique_keep_order(all_urls),
    }


def unique_keep_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def prepare_member_order(
    members: list[Any],
    context: DownloadContext,
    key_func,
) -> list[Any]:
    if context.target_stems:
        return members
    ordered = list(members)
    if context.selection_mode in {"random", "stratified"}:
        rng = random.Random(context.random_seed)
        rng.shuffle(ordered)
    return ordered


def load_manifest(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return {"video_urls": [], "label_urls": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "video_urls": list(payload.get("video_urls", [])),
        "label_urls": list(payload.get("label_urls", [])),
    }


def filter_label_urls(urls: list[str], patterns: list[str]) -> list[str]:
    if any(pattern.lower() == "all" for pattern in patterns):
        return urls
    lowered_patterns = [pattern.lower() for pattern in patterns]
    return [
        url
        for url in urls
        if any(pattern in url.lower() for pattern in lowered_patterns)
    ]


def bdd_record_scene_id(record: dict[str, Any]) -> str | None:
    name = record.get("name") or record.get("file_name") or record.get("filename")
    if not isinstance(name, str) or not name:
        return None
    return scene_stem(name)


def bdd_record_attributes(record: dict[str, Any]) -> dict[str, str]:
    raw = record.get("attributes")
    if not isinstance(raw, dict):
        return {}
    result: dict[str, str] = {}
    for key in ("weather", "scene", "timeofday"):
        value = raw.get(key)
        if isinstance(value, str) and value:
            result[key] = value
    return result


def parse_bdd_label_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("frames", "annotations", "labels", "images"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def load_bdd_image_label_candidates(
    label_urls: list[str],
    context: DownloadContext,
    split: str | None,
) -> dict[str, BDDSceneCandidate]:
    candidates: dict[str, BDDSceneCandidate] = {}
    for url in label_urls:
        lowered_url = url.lower()
        if is_archive(url) and remote_zip_is_usable(url, context):
            members = list_remote_zip_members(url, context)
            for member in members:
                lowered = member.filename.lower()
                if not lowered.endswith(".json"):
                    continue
                if "labels_images" not in lowered and "image" not in lowered:
                    continue
                if split and split.lower() not in lowered:
                    continue
                try:
                    payload = json.loads(read_remote_zip_member_bytes(url, member, context).decode("utf-8"))
                except Exception as exc:  # noqa: BLE001
                    logging.debug("跳过无法解析的 BDD label JSON: %s (%s)", member.filename, exc)
                    continue
                add_bdd_candidates_from_records(candidates, parse_bdd_label_records(payload))
        elif is_archive(url):
            local = download_or_reuse_archive(url, context)
            with zipfile.ZipFile(local) as archive:
                for info in archive.infolist():
                    lowered = info.filename.lower()
                    if info.is_dir() or not lowered.endswith(".json"):
                        continue
                    if "labels_images" not in lowered and "image" not in lowered:
                        continue
                    if split and split.lower() not in lowered:
                        continue
                    try:
                        payload = json.loads(archive.read(info).decode("utf-8"))
                    except Exception as exc:  # noqa: BLE001
                        logging.debug("跳过无法解析的 BDD label JSON: %s (%s)", info.filename, exc)
                        continue
                    add_bdd_candidates_from_records(candidates, parse_bdd_label_records(payload))
            if not context.keep_archives:
                local.unlink(missing_ok=True)
        elif lowered_url.endswith(".json"):
            local = download_file(url, context.cache_dir / safe_filename(url), context)
            payload = json.loads(local.read_text(encoding="utf-8"))
            add_bdd_candidates_from_records(candidates, parse_bdd_label_records(payload))
    return candidates


def add_bdd_candidates_from_records(
    candidates: dict[str, BDDSceneCandidate],
    records: list[dict[str, Any]],
) -> None:
    for record in records:
        scene_id = bdd_record_scene_id(record)
        if not scene_id:
            continue
        candidates[scene_id] = BDDSceneCandidate(
            scene_id=scene_id,
            attributes=bdd_record_attributes(record),
        )


def find_bdd_drivable_stems(
    label_urls: list[str],
    context: DownloadContext,
    split: str | None,
) -> set[str]:
    stems: set[str] = set()
    for url in label_urls:
        lowered_url = url.lower()
        if "drivable" not in lowered_url and "drive" not in lowered_url:
            continue
        if is_archive(url) and remote_zip_is_usable(url, context):
            for member in list_remote_zip_members(url, context):
                lowered = member.filename.lower()
                if member.filename.endswith("/") or Path(member.filename).suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue
                if split and not should_use_split(member.filename, split):
                    continue
                stems.add(label_scene_stem(member.filename))
        elif is_archive(url):
            local = download_or_reuse_archive(url, context)
            with zipfile.ZipFile(local) as archive:
                for info in archive.infolist():
                    if info.is_dir() or Path(info.filename).suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                        continue
                    if split and not should_use_split(info.filename, split):
                        continue
                    stems.add(label_scene_stem(info.filename))
            if not context.keep_archives:
                local.unlink(missing_ok=True)
    return stems


def stratified_select_bdd_scenes(
    candidates: dict[str, BDDSceneCandidate],
    drivable_stems: set[str],
    num_scenes: int,
    seed: int,
) -> list[str]:
    usable = [candidate for scene_id, candidate in candidates.items() if scene_id in drivable_stems]
    if not usable:
        return []
    groups: dict[tuple[str, str, str], list[BDDSceneCandidate]] = {}
    for candidate in usable:
        attrs = candidate.attributes
        key = (
            attrs.get("weather", "unknown"),
            attrs.get("scene", "unknown"),
            attrs.get("timeofday", "unknown"),
        )
        groups.setdefault(key, []).append(candidate)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    group_keys = list(groups)
    rng.shuffle(group_keys)

    selected: list[str] = []
    while len(selected) < num_scenes and group_keys:
        next_keys: list[tuple[str, str, str]] = []
        for key in group_keys:
            values = groups[key]
            if values and len(selected) < num_scenes:
                selected.append(values.pop().scene_id)
            if values:
                next_keys.append(key)
        group_keys = next_keys
    return selected


def save_manifest(path: Path, payload: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def download_file(url: str, destination: Path, context: DownloadContext) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = destination.stat().st_size if destination.exists() else 0
    if existing > 0:
        logging.info("复用已存在文件: %s (%s)", destination, human_bytes(existing))
        return destination

    length = get_content_length(url, context.headers)
    if length is not None:
        context.budget.ensure_network(length)

    part_path = destination.with_suffix(destination.suffix + ".part")
    if part_path.exists():
        part_path.unlink()

    logging.info("开始下载: %s", url)
    logging.info("目标文件: %s", destination)
    with urllib.request.urlopen(request_with_headers(url, context.headers), timeout=60) as response:
        with part_path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                context.budget.add_network(len(chunk))
                handle.write(chunk)
    part_path.replace(destination)
    logging.info("下载完成: %s (%s)", destination, human_bytes(destination.stat().st_size))
    return destination


def download_or_reuse_archive(url: str, context: DownloadContext) -> Path:
    filename = safe_filename(url)
    destination = context.cache_dir / filename
    return download_file(url, destination, context)


def is_archive(path_or_url: str) -> bool:
    return Path(urllib.parse.urlparse(path_or_url).path).suffix.lower() in ARCHIVE_EXTENSIONS


def is_video(path_or_url: str) -> bool:
    return Path(urllib.parse.urlparse(path_or_url).path).suffix.lower() in VIDEO_EXTENSIONS


def remote_zip_is_usable(url: str, context: DownloadContext) -> bool:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    info = get_url_info(url, context.headers)
    return bool(info.content_length and info.accepts_ranges)


def list_remote_zip_members(url: str, context: DownloadContext) -> list[RemoteZipMember]:
    url_info = get_url_info(url, context.headers)
    if not url_info.content_length:
        raise RuntimeError(f"无法获取远程 zip 大小: {url}")

    archive_size = url_info.content_length
    tail_size = min(archive_size, 66_000)
    tail_start = archive_size - tail_size
    tail = read_range(url, tail_start, archive_size - 1, context)
    eocd_pos = tail.rfind(EOCD_SIGNATURE)
    if eocd_pos < 0:
        raise RuntimeError("未找到 zip EOCD 记录，无法远程解析 zip")

    eocd_abs = tail_start + eocd_pos
    if len(tail) - eocd_pos < 22:
        raise RuntimeError("zip EOCD 记录不完整")

    (
        _signature,
        _disk_no,
        _cd_start_disk,
        _disk_entries,
        total_entries,
        cd_size,
        cd_offset,
        _comment_len,
    ) = struct.unpack_from("<4s4H2LH", tail, eocd_pos)

    if total_entries == 0xFFFF or cd_size == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF:
        cd_size, cd_offset, total_entries = read_zip64_directory_info(url, eocd_abs, context)

    logging.info(
        "远程 zip 目录: entries=%s central_dir=%s offset=%s",
        total_entries,
        human_bytes(cd_size),
        cd_offset,
    )
    central_dir = read_range(url, cd_offset, cd_offset + cd_size - 1, context)
    return parse_central_directory(central_dir)


def read_zip64_directory_info(
    url: str,
    eocd_abs: int,
    context: DownloadContext,
) -> tuple[int, int, int]:
    locator_offset = eocd_abs - 20
    if locator_offset < 0:
        raise RuntimeError("zip64 locator offset 非法")
    locator = read_range(url, locator_offset, eocd_abs - 1, context)
    signature, _disk_with_eocd, zip64_eocd_offset, _total_disks = struct.unpack("<4sLQL", locator)
    if signature != ZIP64_LOCATOR_SIGNATURE:
        raise RuntimeError("未找到 zip64 locator")

    header = read_range(url, zip64_eocd_offset, zip64_eocd_offset + 55, context)
    signature = header[:4]
    if signature != ZIP64_EOCD_SIGNATURE:
        raise RuntimeError("未找到 zip64 EOCD")
    record_size = struct.unpack_from("<Q", header, 4)[0]
    full_record = header
    expected_len = 12 + record_size
    if len(full_record) < expected_len:
        full_record = read_range(url, zip64_eocd_offset, zip64_eocd_offset + expected_len - 1, context)

    (
        _signature,
        _record_size,
        _version_made,
        _version_needed,
        _disk_no,
        _cd_start_disk,
        _entries_disk,
        total_entries,
        cd_size,
        cd_offset,
    ) = struct.unpack_from("<4sQ2H2L4Q", full_record, 0)
    return cd_size, cd_offset, total_entries


def parse_central_directory(data: bytes) -> list[RemoteZipMember]:
    members: list[RemoteZipMember] = []
    pos = 0
    while pos + 46 <= len(data):
        if data[pos : pos + 4] != CENTRAL_DIR_SIGNATURE:
            break
        (
            _signature,
            _version_made,
            _version_needed,
            flag_bits,
            compress_type,
            _mod_time,
            _mod_date,
            crc32,
            compress_size,
            file_size,
            filename_len,
            extra_len,
            comment_len,
            _disk_start,
            _internal_attrs,
            _external_attrs,
            header_offset,
        ) = struct.unpack_from("<4s6H3L5H2L", data, pos)
        name_start = pos + 46
        name_end = name_start + filename_len
        extra_end = name_end + extra_len
        comment_end = extra_end + comment_len
        filename_bytes = data[name_start:name_end]
        extra = data[name_end:extra_end]
        encoding = "utf-8" if flag_bits & 0x800 else "cp437"
        filename = filename_bytes.decode(encoding, errors="replace")
        file_size, compress_size, header_offset = apply_zip64_extra(
            extra,
            file_size,
            compress_size,
            header_offset,
        )
        members.append(
            RemoteZipMember(
                filename=filename,
                compress_type=compress_type,
                flag_bits=flag_bits,
                compress_size=compress_size,
                file_size=file_size,
                header_offset=header_offset,
                crc32=crc32,
            )
        )
        pos = comment_end
    return members


def apply_zip64_extra(
    extra: bytes,
    file_size: int,
    compress_size: int,
    header_offset: int,
) -> tuple[int, int, int]:
    if not any(value == 0xFFFFFFFF for value in (file_size, compress_size, header_offset)):
        return file_size, compress_size, header_offset
    pos = 0
    while pos + 4 <= len(extra):
        header_id, data_size = struct.unpack_from("<HH", extra, pos)
        pos += 4
        payload = extra[pos : pos + data_size]
        pos += data_size
        if header_id != 0x0001:
            continue
        payload_pos = 0
        if file_size == 0xFFFFFFFF:
            file_size = struct.unpack_from("<Q", payload, payload_pos)[0]
            payload_pos += 8
        if compress_size == 0xFFFFFFFF:
            compress_size = struct.unpack_from("<Q", payload, payload_pos)[0]
            payload_pos += 8
        if header_offset == 0xFFFFFFFF:
            header_offset = struct.unpack_from("<Q", payload, payload_pos)[0]
        break
    return file_size, compress_size, header_offset


def extract_remote_video_archive(
    url: str,
    context: DownloadContext,
    split: str | None,
    num_scenes: int,
) -> None:
    logging.info("远程解析视频压缩包: %s", url)
    members = sorted(
        (
            member
            for member in list_remote_zip_members(url, context)
            if not member.filename.endswith("/")
            and Path(member.filename).suffix.lower() in VIDEO_EXTENSIONS
            and should_use_split(member.filename, split)
        ),
        key=lambda item: item.filename,
    )
    members = prepare_member_order(members, context, lambda item: item.filename)
    logging.info("远程 zip 中匹配到视频成员: %d", len(members))
    for member in members:
        if len(context.selected_stems) >= num_scenes:
            break
        stem = scene_stem(member.filename)
        if not context.allows_scene(stem) or stem in context.selected_set:
            continue
        write_remote_video_member(url, member, stem, context)
        context.selected_stems.append(stem)


def write_remote_video_member(
    url: str,
    member: RemoteZipMember,
    stem: str,
    context: DownloadContext,
) -> None:
    if member.flag_bits & 0x1:
        logging.warning("跳过加密 zip 成员: %s", member.filename)
        return
    if member.compress_type not in {0, 8}:
        logging.warning("跳过不支持的压缩方式 %s: %s", member.compress_type, member.filename)
        return

    scene_dir = context.scenes_dir / stem
    video_dir = scene_dir / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    destination = video_dir / Path(member.filename).name
    if destination.exists():
        logging.info("跳过已存在视频: %s", destination)
        write_scene_meta(scene_dir, stem, destination)
        return

    context.budget.ensure_final_file(member.file_size)
    context.budget.ensure_network(member.compress_size)
    data_start = remote_zip_data_start(url, member, context)
    data_end = data_start + member.compress_size - 1
    part_path = destination.with_suffix(destination.suffix + ".part")
    part_path.unlink(missing_ok=True)

    logging.info(
        "Range 下载场景视频: %s -> %s (%s)",
        member.filename,
        destination,
        human_bytes(member.file_size),
    )
    crc_value = 0
    written = 0
    decompressor = zlib.decompressobj(-15) if member.compress_type == 8 else None
    try:
        with request_range(url, data_start, data_end, context, timeout=120) as response:
            with part_path.open("wb") as target:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    context.budget.add_network(len(chunk))
                    if decompressor is not None:
                        output = decompressor.decompress(chunk)
                    else:
                        output = chunk
                    if output:
                        target.write(output)
                        written += len(output)
                        crc_value = zlib.crc32(output, crc_value)
                if decompressor is not None:
                    output = decompressor.flush()
                    if output:
                        target.write(output)
                        written += len(output)
                        crc_value = zlib.crc32(output, crc_value)
        if written != member.file_size:
            raise RuntimeError(f"解压大小不匹配: {written} != {member.file_size}")
        if (crc_value & 0xFFFFFFFF) != member.crc32:
            raise RuntimeError("CRC 校验失败")
        part_path.replace(destination)
        write_scene_meta(scene_dir, stem, destination)
    except Exception:
        part_path.unlink(missing_ok=True)
        raise


def read_remote_zip_member_bytes(
    url: str,
    member: RemoteZipMember,
    context: DownloadContext,
) -> bytes:
    if member.flag_bits & 0x1:
        raise RuntimeError(f"不支持加密 zip 成员: {member.filename}")
    if member.compress_type not in {0, 8}:
        raise RuntimeError(f"不支持的压缩方式 {member.compress_type}: {member.filename}")

    context.budget.ensure_network(member.compress_size)
    data_start = remote_zip_data_start(url, member, context)
    data_end = data_start + member.compress_size - 1
    decompressor = zlib.decompressobj(-15) if member.compress_type == 8 else None
    parts: list[bytes] = []
    crc_value = 0
    written = 0
    with request_range(url, data_start, data_end, context, timeout=120) as response:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            context.budget.add_network(len(chunk))
            output = decompressor.decompress(chunk) if decompressor is not None else chunk
            if output:
                parts.append(output)
                written += len(output)
                crc_value = zlib.crc32(output, crc_value)
        if decompressor is not None:
            output = decompressor.flush()
            if output:
                parts.append(output)
                written += len(output)
                crc_value = zlib.crc32(output, crc_value)
    if written != member.file_size:
        raise RuntimeError(f"解压大小不匹配: {written} != {member.file_size}")
    if (crc_value & 0xFFFFFFFF) != member.crc32:
        raise RuntimeError("CRC 校验失败")
    return b"".join(parts)


def remote_zip_data_start(url: str, member: RemoteZipMember, context: DownloadContext) -> int:
    local_header = read_range(url, member.header_offset, member.header_offset + 29, context)
    (
        signature,
        _version_needed,
        _flag_bits,
        _compress_type,
        _mod_time,
        _mod_date,
        _crc32,
        _compress_size,
        _file_size,
        filename_len,
        extra_len,
    ) = struct.unpack("<4s5H3L2H", local_header)
    if signature != LOCAL_FILE_SIGNATURE:
        raise RuntimeError(f"zip local header 非法: {member.filename}")
    return member.header_offset + 30 + filename_len + extra_len


def should_use_split(path: str, split: str | None) -> bool:
    if not split:
        return True
    lowered = path.lower().replace("\\", "/")
    split = split.lower()
    return f"/{split}/" in lowered or f"_{split}" in lowered or f"-{split}" in lowered or split in lowered


def extract_video_archive(archive_path: Path, context: DownloadContext, split: str | None, num_scenes: int) -> None:
    logging.info("扫描视频压缩包: %s", archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        members = sorted(
            (
                info
                for info in archive.infolist()
                if not info.is_dir()
                and Path(info.filename).suffix.lower() in VIDEO_EXTENSIONS
                and should_use_split(info.filename, split)
            ),
            key=lambda item: item.filename,
        )
        members = prepare_member_order(members, context, lambda item: item.filename)
        for info in members:
            if len(context.selected_stems) >= num_scenes:
                break
            stem = scene_stem(info.filename)
            if not context.allows_scene(stem):
                continue
            if stem in context.selected_set:
                continue
            write_video_member(archive, info, stem, context)
            context.selected_stems.append(stem)
    if not context.keep_archives:
        archive_path.unlink(missing_ok=True)


def write_video_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    stem: str,
    context: DownloadContext,
) -> None:
    scene_dir = context.scenes_dir / stem
    video_dir = scene_dir / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    destination = video_dir / Path(info.filename).name
    if destination.exists():
        logging.info("跳过已存在视频: %s", destination)
        return

    context.budget.ensure_final_file(info.file_size)
    logging.info("提取场景视频: %s -> %s", info.filename, destination)
    with archive.open(info) as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    write_scene_meta(scene_dir, stem, destination)


def download_single_video(url: str, context: DownloadContext, num_scenes: int) -> None:
    if len(context.selected_stems) >= num_scenes:
        return
    stem = scene_stem(url)
    if not context.allows_scene(stem):
        return
    if stem in context.selected_set:
        return
    scene_dir = context.scenes_dir / stem
    destination = scene_dir / "video" / safe_filename(url)
    length = get_content_length(url, context.headers)
    if length is not None:
        context.budget.ensure_final_file(length)
    download_file(url, destination, context)
    context.selected_stems.append(stem)
    write_scene_meta(scene_dir, stem, destination)


def write_scene_meta(scene_dir: Path, stem: str, video_path: Path) -> None:
    scene_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "scene_id": stem,
        "video": str(video_path.relative_to(scene_dir)),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (scene_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def process_label_url(url: str, context: DownloadContext, split: str | None) -> None:
    if not context.selected_stems:
        return
    if is_archive(url) and remote_zip_is_usable(url, context):
        extract_remote_label_archive(url, context, split)
        return
    local = download_or_reuse_archive(url, context) if is_archive(url) else download_file(
        url, context.cache_dir / safe_filename(url), context
    )
    suffix = local.suffix.lower()
    if suffix == ".zip":
        extract_label_archive(local, context, split)
        if not context.keep_archives:
            local.unlink(missing_ok=True)
    elif suffix == ".json":
        filter_json_label_file(local, context)
    else:
        copy_label_file_if_matched(local, context)


def extract_remote_label_archive(url: str, context: DownloadContext, split: str | None) -> None:
    logging.info("远程解析标签压缩包: %s", url)
    members = sorted(
        (
            member
            for member in list_remote_zip_members(url, context)
            if not member.filename.endswith("/")
            and Path(member.filename).suffix.lower() in LABEL_EXTENSIONS
            and should_use_split(member.filename, split)
        ),
        key=lambda item: item.filename,
    )
    matched = 0
    for member in members:
        stem = label_scene_stem(member.filename)
        if stem not in context.selected_set:
            continue
        scene_label_dir = context.scenes_dir / stem / "labels"
        scene_label_dir.mkdir(parents=True, exist_ok=True)
        destination = scene_label_dir / Path(member.filename).name
        if destination.exists():
            continue
        context.budget.ensure_final_file(member.file_size)
        data = read_remote_zip_member_bytes(url, member, context)
        destination.write_bytes(data)
        matched += 1
        logging.info("Range 下载场景标签: %s -> %s", member.filename, destination)
    if matched == 0:
        logging.warning("标签压缩包中没有匹配已选 scene 的文件: %s", url)


def extract_label_archive(archive_path: Path, context: DownloadContext, split: str | None) -> None:
    logging.info("扫描标签压缩包: %s", archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            suffix = Path(info.filename).suffix.lower()
            if suffix not in LABEL_EXTENSIONS:
                continue
            if not should_use_split(info.filename, split):
                continue
            if suffix == ".json":
                with archive.open(info) as source:
                    data = source.read()
                with tempfile.NamedTemporaryFile("wb", delete=False) as temp:
                    temp.write(data)
                    temp_path = Path(temp.name)
                try:
                    filter_json_label_file(temp_path, context, source_name=Path(info.filename).name)
                finally:
                    temp_path.unlink(missing_ok=True)
            else:
                copy_label_member_if_matched(archive, info, context)


def filter_json_label_file(
    json_path: Path,
    context: DownloadContext,
    source_name: str | None = None,
) -> None:
    name = source_name or json_path.name
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logging.warning("标签 JSON 无法解析，跳过: %s", name)
        return

    per_scene = split_json_labels_by_scene(payload, context.selected_set)
    for stem, records in per_scene.items():
        if not records:
            continue
        scene_label_dir = context.scenes_dir / stem / "labels"
        scene_label_dir.mkdir(parents=True, exist_ok=True)
        destination = scene_label_dir / name
        encoded = json.dumps(records, ensure_ascii=False, indent=2).encode("utf-8")
        context.budget.ensure_final_file(len(encoded))
        destination.write_bytes(encoded)
        logging.info("写入场景标签: %s", destination)


def split_json_labels_by_scene(payload: Any, stems: set[str]) -> dict[str, Any]:
    result: dict[str, list[Any]] = {stem: [] for stem in stems}
    if isinstance(payload, list):
        for record in payload:
            stem = record_scene_stem(record)
            if stem in result:
                result[stem].append(record)
        return {key: value for key, value in result.items() if value}

    if isinstance(payload, dict):
        # 常见情况：{"frames": [...]} 或 {"annotations": [...]}。
        for key in ("frames", "annotations", "labels", "videos", "images"):
            value = payload.get(key)
            if isinstance(value, list):
                partial = split_json_labels_by_scene(value, stems)
                if partial:
                    return partial
        stem = record_scene_stem(payload)
        if stem in result:
            result[stem].append(payload)
    return {key: value for key, value in result.items() if value}


def record_scene_stem(record: Any) -> str | None:
    if not isinstance(record, dict):
        return None
    candidate_keys = ("name", "videoName", "video_name", "image", "file_name", "filename", "url")
    for key in candidate_keys:
        value = record.get(key)
        if isinstance(value, str) and value:
            return scene_stem(value)
    return None


def copy_label_member_if_matched(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    context: DownloadContext,
) -> None:
    stem = label_scene_stem(info.filename)
    if stem not in context.selected_set:
        return
    scene_label_dir = context.scenes_dir / stem / "labels"
    scene_label_dir.mkdir(parents=True, exist_ok=True)
    destination = scene_label_dir / Path(info.filename).name
    context.budget.ensure_final_file(info.file_size)
    with archive.open(info) as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    logging.info("复制场景标签文件: %s", destination)


def copy_label_file_if_matched(local: Path, context: DownloadContext) -> None:
    stem = label_scene_stem(local.name)
    if stem not in context.selected_set:
        return
    scene_label_dir = context.scenes_dir / stem / "labels"
    scene_label_dir.mkdir(parents=True, exist_ok=True)
    destination = scene_label_dir / local.name
    context.budget.ensure_final_file(local.stat().st_size)
    shutil.copy2(local, destination)
    logging.info("复制场景标签文件: %s", destination)


def read_scene_list(path: Path | None) -> list[str]:
    if path is None:
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    return float(value)


def run_command(command: list[str]) -> None:
    logging.debug("运行命令: %s", " ".join(command))
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"命令执行失败: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def find_scene_video(scene_dir: Path) -> Path | None:
    video_dir = scene_dir / "video"
    if not video_dir.exists():
        return None
    for path in sorted(video_dir.iterdir()):
        if path.suffix.lower() in VIDEO_EXTENSIONS and path.is_file():
            return path
    return None


def format_seconds(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def postprocess_scenes(
    context: DownloadContext,
    clip_start: float | None,
    clip_end: float | None,
    clip_duration: float | None,
    extract_frames: bool,
    frame_fps: float | None,
    discard_full_video: bool,
) -> None:
    if clip_start is None and clip_end is None and clip_duration is None and not extract_frames:
        return
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("未找到 ffmpeg，无法切视频或导出图片帧")

    start = clip_start if clip_start is not None else 0.0
    if start < 0:
        raise ValueError("--clip-start 不能小于 0")
    if clip_end is not None and clip_duration is not None:
        raise ValueError("--clip-end 和 --clip-duration 只能二选一")
    if clip_end is not None and clip_end <= start:
        raise ValueError("--clip-end 必须大于 --clip-start")
    if clip_duration is not None and clip_duration <= 0:
        raise ValueError("--clip-duration 必须大于 0")

    for stem in context.selected_stems:
        scene_dir = context.scenes_dir / stem
        source_video = find_scene_video(scene_dir)
        if source_video is None:
            logging.warning("未找到 scene 视频，跳过后处理: %s", scene_dir)
            continue

        clip_video = source_video
        if clip_start is not None or clip_end is not None or clip_duration is not None:
            clip_dir = scene_dir / "clips"
            clip_dir.mkdir(parents=True, exist_ok=True)
            end_label = format_seconds(clip_end) if clip_end is not None else (
                f"{format_seconds(start)}+{format_seconds(clip_duration or 0.0)}"
            )
            clip_video = clip_dir / f"{stem}_{format_seconds(start)}s_{end_label}s.mp4"
            command = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                format_seconds(start),
                "-i",
                str(source_video),
            ]
            if clip_end is not None:
                command.extend(["-t", format_seconds(clip_end - start)])
            elif clip_duration is not None:
                command.extend(["-t", format_seconds(clip_duration)])
            # 重新编码，避免 MOV/H.264 关键帧导致的截取不精确。
            command.extend(["-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip_video)])
            run_command(command)
            logging.info("导出时间片段: %s", clip_video)
            if context.budget.final_bytes() > context.budget.max_bytes:
                clip_video.unlink(missing_ok=True)
                raise DownloadLimitExceeded("截取视频后最终目录超过大小限制")

        if extract_frames:
            frames_dir = scene_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            frame_pattern = frames_dir / "frame_%06d.jpg"
            command = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(clip_video),
            ]
            if frame_fps is not None and frame_fps > 0:
                command.extend(["-vf", f"fps={frame_fps}"])
            command.extend(["-q:v", "2", str(frame_pattern)])
            run_command(command)
            logging.info("导出图片帧目录: %s", frames_dir)
            if context.budget.final_bytes() > context.budget.max_bytes:
                shutil.rmtree(frames_dir, ignore_errors=True)
                raise DownloadLimitExceeded("导出图片帧后最终目录超过大小限制")

        if discard_full_video and clip_video != source_video and clip_video.exists():
            source_video.unlink(missing_ok=True)
            logging.info("已删除完整原视频，仅保留截取片段: %s", clip_video)


def find_scene_drivable_label(scene_dir: Path, stem: str) -> Path | None:
    label_dir = scene_dir / "labels"
    if not label_dir.exists():
        return None
    candidates = [
        path
        for path in label_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        and label_scene_stem(path.name) == stem
    ]
    if not candidates:
        return None
    drivable = [path for path in candidates if "drivable" in str(path).lower()]
    return sorted(drivable or candidates)[0]


def local_drivable_label_priority(path: Path, preferred_split: str | None) -> tuple[int, int, int, str]:
    parts = set(path.parts)
    split_rank = 0 if preferred_split and preferred_split in parts else 1
    source_rank = 0 if "labels" in parts else 1
    kind_rank = 0 if path.name.endswith("_drivable_id.png") else 1
    return split_rank, source_rank, kind_rank, str(path)


def build_local_drivable_label_index(root: Path | None, preferred_split: str | None) -> dict[str, Path]:
    """建立 BDD100K drivable maps 本地索引，优先使用 labels/*_drivable_id.png。"""
    if root is None:
        return {}
    root = root.resolve()
    if not root.exists():
        logging.warning("本地 BDD drivable root 不存在，将跳过: %s", root)
        return {}

    bdd_splits = {"train", "val", "test"}
    candidates = []
    for path in root.rglob("*"):
        if not (
            path.is_file()
            and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
            and "drivable" in path.name.lower()
        ):
            continue
        parts = set(path.parts)
        if preferred_split and parts & bdd_splits and preferred_split not in parts:
            continue
        candidates.append(path)
    index: dict[str, Path] = {}
    priority: dict[str, tuple[int, int, int, str]] = {}
    for path in candidates:
        stem = label_scene_stem(path.name)
        rank = local_drivable_label_priority(path, preferred_split)
        if stem not in index or rank < priority[stem]:
            index[stem] = path
            priority[stem] = rank
    logging.info("本地 BDD drivable 标签索引完成: %d 个场景，root=%s", len(index), root)
    return index


def find_drivable_label(
    scene_dir: Path,
    stem: str,
    local_drivable_index: dict[str, Path] | None = None,
) -> Path | None:
    if local_drivable_index and stem in local_drivable_index:
        return local_drivable_index[stem]
    return find_scene_drivable_label(scene_dir, stem)


def convert_bdd_drivable_mask(
    label_path: Path,
    output_npy: Path,
    output_png: Path,
    drivable_values: set[int] | None = None,
) -> None:
    image = None
    try:
        image = cv2_imread(str(label_path), unchanged=True)
    except Exception:
        image = None
    if image is None:
        raise FileNotFoundError(label_path)
    if image.ndim == 2:
        values = drivable_values or set()
        if values:
            mask = np.isin(image, list(values)).astype(np.float32)
        else:
            mask = (image > 0).astype(np.float32)
    else:
        # BDD drivable color maps 使用非黑色像素表示 direct/alternative drivable 区域。
        mask = (np.max(image[:, :, :3], axis=2) > 0).astype(np.float32)
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_npy, mask)
    cv2_imwrite(str(output_png), (mask * 255).astype(np.uint8))

def cv2_imread(path: str, unchanged: bool = False):
    import cv2

    flag = cv2.IMREAD_UNCHANGED if unchanged else cv2.IMREAD_COLOR
    return cv2.imread(path, flag)


def cv2_imwrite(path: str, image) -> None:
    import cv2

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(path, image):
        raise RuntimeError(f"写入图片失败: {path}")


def extract_bdd_stgru_scene_frames(
    scene_dir: Path,
    stem: str,
    fps: float,
    clip_start: float,
    clip_duration: float,
    center_second: float,
    context_frames: int,
    local_drivable_index: dict[str, Path] | None,
    drivable_values: set[int],
) -> tuple[Path, Path, Path, Path] | None:
    source_video = find_scene_video(scene_dir)
    label_path = find_drivable_label(scene_dir, stem, local_drivable_index)
    if source_video is None or label_path is None:
        logging.warning("跳过 STGRU 场景，缺少 video 或 drivable label: %s", stem)
        return None

    stgru_dir = scene_dir / "stgru"
    all_frames_dir = stgru_dir / "frames_9_12s"
    sequence_dir = stgru_dir / "sequence_pm10"
    target_npy = stgru_dir / "target_mask.npy"
    target_png = stgru_dir / "target_mask.png"
    all_frames_dir.mkdir(parents=True, exist_ok=True)
    sequence_dir.mkdir(parents=True, exist_ok=True)

    expected_frames = int(round(clip_duration * fps))
    existing = sorted(all_frames_dir.glob("frame_*.jpg"))
    if len(existing) < expected_frames:
        for path in existing:
            path.unlink(missing_ok=True)
        frame_pattern = all_frames_dir / "frame_%06d.jpg"
        command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            format_seconds(clip_start),
            "-i",
            str(source_video),
            "-t",
            format_seconds(clip_duration),
            "-vf",
            f"fps={fps}",
            "-q:v",
            "2",
            str(frame_pattern),
        ]
        run_command(command)

    center_index_zero_based = int(round((center_second - clip_start) * fps))
    for offset in range(-context_frames, context_frames + 1):
        source_index = center_index_zero_based + offset + 1
        source_frame = all_frames_dir / f"frame_{source_index:06d}.jpg"
        if not source_frame.exists():
            logging.warning("缺少 STGRU 上下文帧: %s", source_frame)
            return None
        destination = sequence_dir / f"frame_{offset:+04d}.jpg"
        shutil.copy2(source_frame, destination)

    convert_bdd_drivable_mask(label_path, target_npy, target_png, drivable_values)
    return sequence_dir / "frame_-001.jpg", sequence_dir / "frame_+000.jpg", target_npy, label_path


def write_bdd_stgru_manifests(
    context: DownloadContext,
    output_root: Path,
    train_count: int,
    val_count: int,
    test_count: int,
    seed: int,
    fps: float,
    clip_start: float,
    clip_duration: float,
    center_second: float,
    context_frames: int,
    local_drivable_root: Path | None,
    label_split: str | None,
    drivable_values: set[int],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    local_drivable_index = build_local_drivable_label_index(local_drivable_root, label_split)
    scene_ids = list(context.selected_stems)
    rng.shuffle(scene_ids)
    required = train_count + val_count + test_count
    if len(scene_ids) < required:
        logging.warning("STGRU split 需要 %d 个场景，但当前只有 %d 个。", required, len(scene_ids))
    split_by_scene: dict[str, str] = {}
    for scene_id in scene_ids[:train_count]:
        split_by_scene[scene_id] = "train"
    for scene_id in scene_ids[train_count : train_count + val_count]:
        split_by_scene[scene_id] = "val"
    for scene_id in scene_ids[train_count + val_count : train_count + val_count + test_count]:
        split_by_scene[scene_id] = "test"

    rows: list[dict[str, str]] = []
    for scene_id in scene_ids:
        split = split_by_scene.get(scene_id)
        if split is None:
            continue
        scene_dir = context.scenes_dir / scene_id
        result = extract_bdd_stgru_scene_frames(
            scene_dir=scene_dir,
            stem=scene_id,
            fps=fps,
            clip_start=clip_start,
            clip_duration=clip_duration,
            center_second=center_second,
            context_frames=context_frames,
            local_drivable_index=local_drivable_index,
            drivable_values=drivable_values,
        )
        if result is None:
            continue
        previous_image, current_image, target_mask, source_label = result
        rows.append(
            {
                "split": split,
                "sample_id": scene_id,
                "scene_id": scene_id,
                "previous_image": str(previous_image),
                "current_image": str(current_image),
                "target_mask": str(target_mask),
                "source_label": str(source_label),
                "sequence_dir": str((context.scenes_dir / scene_id / "stgru" / "sequence_pm10")),
                "all_frames_dir": str((context.scenes_dir / scene_id / "stgru" / "frames_9_12s")),
            }
        )

    fieldnames = [
        "split",
        "sample_id",
        "scene_id",
        "previous_image",
        "current_image",
        "target_mask",
        "source_label",
        "sequence_dir",
        "all_frames_dir",
    ]
    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        with (output_root / f"{split}_scenes.csv").open("w", newline="") as handle:
            writer = csv_dict_writer(handle, fieldnames)
            writer.writeheader()
            writer.writerows(split_rows)
    with (output_root / "bdd_stgru_scenes.csv").open("w", newline="") as handle:
        writer = csv_dict_writer(handle, fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logging.info("BDD STGRU scene manifests 已写入: %s", output_root)


def csv_dict_writer(handle, fieldnames: list[str]):
    import csv

    return csv.DictWriter(handle, fieldnames=fieldnames)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 BDD100K 下载页抓取视频/Drivable 标签，并整理为 STGRU 训练场景。",
    )
    parser.add_argument("--download-page", default=DEFAULT_DOWNLOAD_PAGE, help="BDD100K 下载页 URL")
    parser.add_argument("--manifest", type=Path, help="手工提供 video_urls/label_urls 的 JSON manifest")
    parser.add_argument("--video-url", action="append", default=[], help="手工追加单个视频或视频压缩包 URL")
    parser.add_argument("--label-url", action="append", default=[], help="手工追加标签 JSON/压缩包 URL")
    parser.add_argument(
        "--label-pattern",
        action="append",
        default=[],
        help="从下载页发现的 label 链接中过滤关键词，默认 bdd100k_labels.zip + bdd100k_drivable_maps.zip；传 all 表示全部",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/bdd100k_video_scenes"))
    parser.add_argument("--num-scenes", type=int, default=100, help="需要下载/整理的场景数量")
    parser.add_argument("--split", default="train", help="优先匹配的数据划分，如 train/val/test；留空则不过滤")
    parser.add_argument("--max-total-size", type=parse_size, default=parse_size("200G"), help="下载和落盘硬限制")
    parser.add_argument("--selection-mode", default="stratified", choices=["first", "random", "stratified"], help="场景选择方式")
    parser.add_argument("--random-seed", type=int, default=42, help="随机选择和 train/val/test 划分随机种子")
    parser.add_argument("--skip-remote-label-selection", action="store_true", help="跳过远程 label zip 解析，直接使用本地 drivable maps 随机场景")
    parser.add_argument("--cookie", help="直接传入 Cookie header 内容")
    parser.add_argument("--cookie-file", type=Path, help="浏览器导出的 Cookie 文件，支持 Netscape 格式")
    parser.add_argument("--scene-name", action="append", default=[], help="只下载指定场景 id，可重复传入")
    parser.add_argument("--scene-list", type=Path, help="场景 id 列表文件，每行一个")
    parser.add_argument("--keep-archives", action="store_true", help="保留下载的原始压缩包")
    parser.add_argument("--no-remote-zip", action="store_true", help="禁用 HTTP Range 远程 zip 成员下载")
    parser.add_argument("--clip-start", type=parse_optional_float, help="下载后截取片段起始秒，例如 7")
    parser.add_argument("--clip-end", type=parse_optional_float, help="下载后截取片段结束秒，例如 13")
    parser.add_argument("--clip-duration", type=parse_optional_float, help="下载后截取片段时长秒，例如 6")
    parser.add_argument("--extract-frames", action="store_true", help="将视频或截取片段导出为图片帧")
    parser.add_argument("--frame-fps", type=parse_optional_float, help="导出图片帧时的采样 fps；默认保留原 fps")
    parser.add_argument("--discard-full-video", action="store_true", help="截取片段成功后删除完整原视频，只保留 clip/frames")
    parser.add_argument("--prepare-stgru", action="store_true", help="下载后整理成 BDD STGRU 场景目录和 split manifest")
    parser.add_argument("--stgru-output-root", type=Path, default=Path("data/bdd100k_stgru"), help="BDD STGRU manifest 输出目录")
    parser.add_argument(
        "--local-drivable-root",
        type=Path,
        default=Path("data/bdd100k_drivable_maps"),
        help="本地 BDD100K drivable maps 根目录，优先用于 STGRU 监督标签匹配",
    )
    parser.add_argument(
        "--bdd-drivable-values",
        type=parse_int_set,
        default=parse_int_set("1,2"),
        help="BDD *_drivable_id.png 中视为 free-space 的像素值，默认 1,2；传空字符串表示所有 >0",
    )
    parser.add_argument("--stgru-train-count", type=int, default=80)
    parser.add_argument("--stgru-val-count", type=int, default=10)
    parser.add_argument("--stgru-test-count", type=int, default=10)
    parser.add_argument("--stgru-fps", type=float, default=30.0)
    parser.add_argument("--stgru-clip-start", type=float, default=9.0)
    parser.add_argument("--stgru-clip-duration", type=float, default=3.0)
    parser.add_argument("--stgru-center-second", type=float, default=10.0)
    parser.add_argument("--stgru-context-frames", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="只发现链接，不下载")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    output_dir = args.output_dir.resolve()
    cache_dir = output_dir / "_cache"
    scenes_dir = output_dir / "scenes"
    manifest_dir = output_dir / "manifests"
    for path in (cache_dir, scenes_dir, manifest_dir):
        path.mkdir(parents=True, exist_ok=True)

    headers = build_headers(args.cookie, args.cookie_file)
    context = DownloadContext(
        output_dir=output_dir,
        cache_dir=cache_dir,
        scenes_dir=scenes_dir,
        manifest_dir=manifest_dir,
        headers=headers,
        budget=Budget(max_bytes=args.max_total_size, root=output_dir),
        keep_archives=args.keep_archives,
        selection_mode=args.selection_mode,
        random_seed=args.random_seed,
    )

    wanted_scenes = unique_keep_order(args.scene_name + read_scene_list(args.scene_list))
    context.target_stems = set(wanted_scenes)

    manifest = load_manifest(args.manifest)
    discovered = {"video_urls": [], "label_urls": [], "all_urls": []}
    try:
        discovered = discover_download_links(args.download_page, headers)
    except (urllib.error.URLError, TimeoutError) as exc:
        logging.warning("下载页暂时无法抓取: %s", exc)
        logging.warning("将继续使用 --manifest/--video-url/--label-url 提供的链接。")
    save_manifest(manifest_dir / "discovered_links.json", discovered)

    discovered_label_patterns = args.label_pattern or ["bdd100k_labels.zip", "bdd100k_drivable_maps.zip"]
    discovered_label_urls = filter_label_urls(discovered["label_urls"], discovered_label_patterns)
    video_urls = unique_keep_order(
        manifest["video_urls"] + args.video_url + discovered["video_urls"] + [DEFAULT_BDD_VIDEO_URL]
    )
    label_urls = unique_keep_order(manifest["label_urls"] + args.label_url + discovered_label_urls)

    if (
        not context.target_stems
        and args.selection_mode == "stratified"
        and label_urls
        and not args.skip_remote_label_selection
    ):
        logging.info("尝试基于 BDD image attributes + drivable maps 分层随机选择场景。")
        selected: list[str] = []
        candidates: dict[str, BDDSceneCandidate] = {}
        drivable_stems: set[str] = set()
        try:
            candidates = load_bdd_image_label_candidates(label_urls, context, args.split or None)
            drivable_stems = find_bdd_drivable_stems(label_urls, context, args.split or None)
            selected = stratified_select_bdd_scenes(
                candidates=candidates,
                drivable_stems=drivable_stems,
                num_scenes=args.num_scenes,
                seed=args.random_seed,
            )
        except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
            logging.warning("远程 label zip 解析失败，将回退到本地 drivable maps 随机选择: %s", exc)
        if selected:
            context.target_stems = set(selected)
            save_manifest(
                manifest_dir / "selected_scene_plan.json",
                {
                    "selection_mode": [args.selection_mode],
                    "scene_ids": selected,
                    "candidate_count": [str(len(candidates))],
                    "drivable_count": [str(len(drivable_stems))],
                },
            )
            logging.info("已分层随机选择 %d 个候选场景。", len(selected))
        else:
            logging.warning("无法从标签中生成分层候选，将回退为视频压缩包内随机选择。")

    if not context.target_stems and args.prepare_stgru:
        local_drivable_index = build_local_drivable_label_index(args.local_drivable_root, args.split or None)
        if local_drivable_index:
            selected = sorted(local_drivable_index)
            rng = random.Random(args.random_seed)
            rng.shuffle(selected)
            selected = selected[: args.num_scenes]
            context.target_stems = set(selected)
            save_manifest(
                manifest_dir / "selected_scene_plan.json",
                {
                    "selection_mode": ["local_drivable_random"],
                    "scene_ids": selected,
                    "candidate_count": [str(len(local_drivable_index))],
                    "local_drivable_root": [str(args.local_drivable_root.resolve())],
                },
            )
            logging.info("已从本地 drivable maps 随机选择 %d 个有监督标签的候选场景。", len(selected))

    logging.info("发现 video 链接数量: %d", len(video_urls))
    logging.info("发现 label 链接数量: %d", len(label_urls))
    logging.info("发现清单已写入: %s", manifest_dir / "discovered_links.json")

    if args.dry_run:
        return 0
    if not video_urls:
        logging.error("未发现 video 链接。请登录 BDD100K 后传入 --cookie-file，或用 --manifest/--video-url 提供直链。")
        return 2

    for url in video_urls:
        if len(context.selected_stems) >= args.num_scenes:
            break
        if context.target_stems and context.target_stems.issubset(context.selected_set):
            break
        if is_archive(url):
            if not args.no_remote_zip and remote_zip_is_usable(url, context):
                extract_remote_video_archive(url, context, args.split or None, args.num_scenes)
            else:
                archive_path = download_or_reuse_archive(url, context)
                extract_video_archive(archive_path, context, args.split or None, args.num_scenes)
        elif is_video(url):
            download_single_video(url, context, args.num_scenes)

    if len(context.selected_stems) < args.num_scenes:
        logging.warning("只整理到 %d/%d 个场景。", len(context.selected_stems), args.num_scenes)

    for url in label_urls:
        process_label_url(url, context, args.split or None)

    postprocess_scenes(
        context=context,
        clip_start=args.clip_start,
        clip_end=args.clip_end,
        clip_duration=args.clip_duration,
        extract_frames=args.extract_frames,
        frame_fps=args.frame_fps,
        discard_full_video=args.discard_full_video,
    )

    if args.prepare_stgru:
        write_bdd_stgru_manifests(
            context=context,
            output_root=args.stgru_output_root.resolve(),
            train_count=args.stgru_train_count,
            val_count=args.stgru_val_count,
            test_count=args.stgru_test_count,
            seed=args.random_seed,
            fps=args.stgru_fps,
            clip_start=args.stgru_clip_start,
            clip_duration=args.stgru_clip_duration,
            center_second=args.stgru_center_second,
            context_frames=args.stgru_context_frames,
            local_drivable_root=args.local_drivable_root,
            label_split=args.split or None,
            drivable_values=args.bdd_drivable_values,
        )

    summary = {
        "download_page": args.download_page,
        "num_scenes_requested": args.num_scenes,
        "num_scenes_collected": len(context.selected_stems),
        "scene_ids": context.selected_stems,
        "network_bytes": context.budget.network_bytes,
        "final_bytes": context.budget.final_bytes(),
        "max_total_size": args.max_total_size,
    }
    save_manifest(manifest_dir / "download_summary.json", summary)  # type: ignore[arg-type]
    logging.info("完成。输出目录: %s", output_dir)
    logging.info("最终大小: %s / %s", human_bytes(context.budget.final_bytes()), human_bytes(args.max_total_size))
    return 0


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        try:
            raise SystemExit(main(sys.argv[1:]))
        except DownloadLimitExceeded as exc:
            logging.error("%s", exc)
            logging.error("已按限制停止下载。可降低 --num-scenes，或提供更小的单场景直链/manifest。")
            raise SystemExit(3)
    raise SystemExit(130)
