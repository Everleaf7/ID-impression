from __future__ import annotations

import hmac
import ipaddress
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = ROOT / "web"
MAX_BODY_BYTES = 4096
MAX_ID_LENGTH = 40
MAX_SEED = 2**32 - 1
JOB_ID = re.compile(r"^[0-9a-f]{32}$")
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/assets/autumn-outfit-front.png": ("assets/autumn-outfit-front.png", "image/png"),
    "/assets/autumn-outfit-pose.png": ("assets/autumn-outfit-pose.png", "image/png"),
}


class PublicError(ValueError):
    """An input error whose message is safe to return to the browser."""


def validate_id(value: Any) -> str:
    if not isinstance(value, str):
        raise PublicError("ID 必须是字符串")
    cleaned = value.strip()
    if not cleaned:
        raise PublicError("请输入 ID")
    if len(cleaned) > MAX_ID_LENGTH:
        raise PublicError(f"ID 最多 {MAX_ID_LENGTH} 个字符")
    if any(unicodedata.category(char).startswith("C") for char in cleaned):
        raise PublicError("ID 不能包含控制字符或不可见字符")
    return cleaned


def validate_seed(value: Any) -> int:
    if isinstance(value, bool):
        raise PublicError("seed 必须是整数")
    try:
        seed = int(value)
    except (TypeError, ValueError) as exc:
        raise PublicError("seed 必须是整数") from exc
    if not 0 <= seed <= MAX_SEED:
        raise PublicError(f"seed 必须在 0 到 {MAX_SEED} 之间")
    return seed


def is_loopback_host(host: str) -> bool:
    return host in {"127.0.0.1", "::1", "localhost"}


def valid_host_header(value: str, allowed_hosts: frozenset[str] | None) -> bool:
    if not value or any(char in value for char in ("\r", "\n", "\x00")):
        return False
    return allowed_hosts is None or value.lower() in allowed_hosts


def validate_exposure(host: str, token: str, allow_remote: bool) -> None:
    if is_loopback_host(host):
        return
    if not allow_remote:
        raise PublicError("非本机监听必须显式添加 --allow-remote")
    if len(token) < 24:
        raise PublicError("远程监听必须设置至少 24 字符的 ID_AVATAR_DEMO_TOKEN")


def validate_proxy_mode(host: str, public_behind_cloudflare: bool) -> None:
    if public_behind_cloudflare and not is_loopback_host(host):
        raise PublicError("Cloudflare 代理模式必须只监听本机回环地址")


def client_identity(peer: str, forwarded: str | None, trust_cloudflare: bool) -> str:
    """Return a non-logged rate-limit key, trusting Cloudflare only over loopback."""
    if not trust_cloudflare or not is_loopback_host(peer) or not forwarded:
        return peer
    candidate = forwarded.strip()
    try:
        return str(ipaddress.ip_address(candidate))
    except ValueError:
        return peer


class RateLimiter:
    def __init__(self, limit: int = 4, window_seconds: int = 600):
        self.limit = limit
        self.window_seconds = window_seconds
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, client: str, now: float | None = None) -> bool:
        moment = time.monotonic() if now is None else now
        cutoff = moment - self.window_seconds
        with self._lock:
            entries = self._requests[client]
            while entries and entries[0] <= cutoff:
                entries.popleft()
            if len(entries) >= self.limit:
                return False
            entries.append(moment)
            return True


@dataclass(frozen=True)
class GeneratorConfig:
    backend: str
    gpu: str
    output_root: Path
    model: Path | None = None
    lora: Path | None = None
    font: Path | None = None
    timeout_seconds: int = 300
    retention_hours: int = 24


class AvatarGenerator:
    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.busy = threading.Lock()

    def _command(self, user_id: str, seed: int, output_dir: Path) -> list[str]:
        if self.config.backend == "placeholder":
            return [
                sys.executable,
                str(ROOT / "scripts/generate.py"),
                "--id", user_id,
                "--seed", str(seed),
                "--output-dir", str(output_dir),
            ]
        command = [
            sys.executable,
            str(ROOT / "scripts/generate_avatar.py"),
            "--id", user_id,
            "--seed", str(seed),
            "--gpu", self.config.gpu,
            "--output-dir", str(output_dir),
        ]
        for option, path in (
            ("--model", self.config.model),
            ("--lora", self.config.lora),
            ("--font", self.config.font),
        ):
            if path is not None:
                command.extend((option, str(path)))
        return command

    def _cleanup(self) -> None:
        root = self.config.output_root.resolve()
        if not root.is_dir():
            return
        cutoff = time.time() - self.config.retention_hours * 3600
        for candidate in root.iterdir():
            if (
                candidate.is_dir()
                and not candidate.is_symlink()
                and JOB_ID.fullmatch(candidate.name)
                and candidate.stat().st_mtime < cutoff
            ):
                shutil.rmtree(candidate)

    def generate(self, user_id: str, seed: int) -> str:
        if not self.busy.acquire(blocking=False):
            raise RuntimeError("busy")
        try:
            self._cleanup()
            job_id = uuid.uuid4().hex
            output_dir = (self.config.output_root / job_id).resolve()
            root = self.config.output_root.resolve()
            if output_dir.parent != root:
                raise RuntimeError("invalid output directory")
            output_dir.mkdir(parents=True, exist_ok=False)
            environment = os.environ.copy()
            environment.pop("ID_AVATAR_DEMO_TOKEN", None)
            try:
                subprocess.run(
                    self._command(user_id, seed, output_dir),
                    cwd=ROOT,
                    env=environment,
                    check=True,
                    capture_output=True,
                    timeout=self.config.timeout_seconds,
                )
            except BaseException:
                shutil.rmtree(output_dir, ignore_errors=True)
                raise
            filename = "final_with_to.png" if self.config.backend == "placeholder" else "avatar_with_id.png"
            image = output_dir / filename
            if not image.is_file() or image.stat().st_size == 0:
                shutil.rmtree(output_dir, ignore_errors=True)
                raise RuntimeError("generator did not create an image")
            return job_id
        finally:
            self.busy.release()

    def image_path(self, job_id: str) -> Path | None:
        if not JOB_ID.fullmatch(job_id):
            return None
        filename = "final_with_to.png" if self.config.backend == "placeholder" else "avatar_with_id.png"
        candidate = (self.config.output_root / job_id / filename).resolve()
        root = self.config.output_root.resolve()
        if candidate.parent.parent != root or not candidate.is_file():
            return None
        return candidate


@dataclass(frozen=True)
class ServerContext:
    generator: AvatarGenerator
    token: str
    rate_limiter: RateLimiter
    global_rate_limiter: RateLimiter
    trust_cloudflare: bool = False


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 16

    def __init__(self, address: tuple[str, int], context: ServerContext):
        self.context = context
        host, port = address
        self.allowed_hosts = (
            frozenset({
                "127.0.0.1", f"127.0.0.1:{port}",
                "localhost", f"localhost:{port}",
                "[::1]", f"[::1]:{port}",
            })
            if is_loopback_host(host) and not context.trust_cloudflare
            else None
        )
        super().__init__(address, DemoHandler)


class DemoHandler(BaseHTTPRequestHandler):
    server: DemoServer
    server_version = "IDAvatar"
    sys_version = ""

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(15)

    def log_message(self, format: str, *args: object) -> None:
        # Do not place submitted IDs, tokens, or query strings in access logs.
        return

    def _security_headers(self, cache: bool = False) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; script-src 'self'; "
            "style-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'",
        )
        self.send_header("Cache-Control", "public, max-age=300" if cache else "no-store")

    def _send_bytes(self, status: HTTPStatus, content_type: str, payload: bytes, cache: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self._security_headers(cache=cache)
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._send_bytes(status, "application/json; charset=utf-8", data)

    def _authorized(self) -> bool:
        expected = self.server.context.token
        if not expected:
            return True
        supplied = self.headers.get("Authorization", "")
        prefix = "Bearer "
        return supplied.startswith(prefix) and hmac.compare_digest(supplied[len(prefix):], expected)

    def _valid_host(self) -> bool:
        return valid_host_header(self.headers.get("Host", ""), self.server.allowed_hosts)

    def _same_origin(self) -> bool:
        origin = self.headers.get("Origin")
        if not origin:
            return True
        parsed = urlsplit(origin)
        return parsed.scheme in {"http", "https"} and parsed.netloc == self.headers.get("Host", "")

    def do_GET(self) -> None:
        if not self._valid_host():
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Host 无效"})
            return
        path = urlsplit(self.path).path
        if path in STATIC_FILES:
            filename, content_type = STATIC_FILES[path]
            try:
                payload = (WEB_ROOT / filename).read_bytes()
            except OSError:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "页面资源不可用"})
                return
            self._send_bytes(HTTPStatus.OK, content_type, payload, cache=True)
            return
        if path == "/api/health":
            self._send_json(HTTPStatus.OK, {
                "status": "ok",
                "busy": self.server.context.generator.busy.locked(),
                "backend": self.server.context.generator.config.backend,
                "token_required": bool(self.server.context.token),
            })
            return
        match = re.fullmatch(r"/api/result/([0-9a-f]{32})", path)
        if match:
            if not self._authorized():
                self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "访问令牌无效"})
                return
            image = self.server.context.generator.image_path(match.group(1))
            if image is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "结果不存在或已过期"})
                return
            self._send_bytes(HTTPStatus.OK, "image/png", image.read_bytes())
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "页面不存在"})

    def do_POST(self) -> None:
        if not self._valid_host():
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Host 无效"})
            return
        if urlsplit(self.path).path != "/api/generate":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "接口不存在"})
            return
        if not self._same_origin():
            self._send_json(HTTPStatus.FORBIDDEN, {"error": "请求来源无效"})
            return
        client = client_identity(
            self.client_address[0],
            self.headers.get("CF-Connecting-IP"),
            self.server.context.trust_cloudflare,
        )
        if not self.server.context.rate_limiter.allow(client):
            self._send_json(HTTPStatus.TOO_MANY_REQUESTS, {"error": "请求过于频繁，请稍后再试"})
            return
        if not self.server.context.global_rate_limiter.allow("all"):
            self._send_json(HTTPStatus.TOO_MANY_REQUESTS, {"error": "当前生成额度已用完，请稍后再试"})
            return
        if not self._authorized():
            self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "访问令牌无效"})
            return
        if self.headers.get_content_type() != "application/json":
            self._send_json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "请求必须使用 JSON"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if not 0 < length <= MAX_BODY_BYTES:
            self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "请求体大小无效"})
            return
        try:
            raw = json.loads(self.rfile.read(length))
            if not isinstance(raw, dict):
                raise PublicError("请求内容必须是对象")
            user_id = validate_id(raw.get("id"))
            seed = validate_seed(raw.get("seed", 42))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "JSON 格式无效"})
            return
        except PublicError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        try:
            job_id = self.server.context.generator.generate(user_id, seed)
        except RuntimeError as exc:
            if str(exc) == "busy":
                self._send_json(HTTPStatus.TOO_MANY_REQUESTS, {"error": "GPU 正在生成另一张头像，请稍后再试"})
            else:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "生成失败，请稍后重试"})
            return
        except (OSError, subprocess.SubprocessError):
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "生成失败，请稍后重试"})
            return
        self._send_json(HTTPStatus.OK, {
            "job_id": job_id,
            "image_url": f"/api/result/{job_id}",
        })


def serve(host: str, port: int, context: ServerContext) -> None:
    server = DemoServer((host, port), context)
    print(f"ID Avatar service listening on http://{host}:{port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
