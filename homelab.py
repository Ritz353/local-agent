"""Homelab tools: check whether my server's services are up, and read its backup log.

Server addresses live in services.json (gitignored, since this repo is public).
services.example.json shows the format.
"""
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    # Verify HTTPS certificates with the operating system's trust store (like a browser does).
    # Python's own certificate list can contain expired entries that break the check.
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

HERE = Path(__file__).parent


def load_config() -> dict:
    path = HERE / "services.json"
    if not path.exists():
        path = HERE / "services.example.json"
    return json.loads(path.read_text(encoding="utf-8"))


def probe(url: str, timeout: float = 5) -> tuple[bool, str, int | None]:
    """Request a URL. Any HTTP answer below 500 counts as up, including 401 "login required"."""
    start = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "local-agent-healthcheck"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code = resp.status
    except urllib.error.HTTPError as e:
        code = e.code  # the server answered, it just said no (e.g. 401), so it's running
    except Exception as e:  # DNS failure, connection refused, timeout, certificate problem...
        reason = getattr(e, "reason", e)  # URLError wraps the real cause in .reason
        return False, f"unreachable: {type(reason).__name__}: {str(reason)[:120]}", None
    ms = round((time.time() - start) * 1000)
    return code < 500, f"HTTP {code}", ms


def check_homelab() -> str:
    """Check whether each service on Wilfredo's homelab server (ratlab) and his website is up.

    Returns:
        A summary line, then one line per service with UP or DOWN and the response time.
    """
    services = load_config()["services"]
    lines, up = [], 0
    for name, url in services.items():
        ok, detail, ms = probe(url)
        up += ok
        timing = f", {ms} ms" if ms is not None else ""
        lines.append(f"{'UP' if ok else 'DOWN'}: {name} ({detail}{timing})")
    return f"{up}/{len(services)} services up\n" + "\n".join(lines)


def get_backup_status() -> str:
    """Read the latest entries of the nightly backup log on Wilfredo's homelab server.

    Returns:
        The last few log lines, showing when backups were saved and uploaded to Google Drive.
    """
    host = load_config()["ssh_host"]
    # Fixed command: the model cannot choose what runs on the server.
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host,
           "tail -n 6 /var/log/homelab-backup.log"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except FileNotFoundError:
        return "Error: ssh is not installed on this computer."
    except subprocess.TimeoutExpired:
        return "Error: the server did not respond in time."
    if out.returncode != 0:
        return f"Error reading the backup log: {out.stderr.strip()[:200] or 'ssh failed'}"
    return out.stdout.strip() or "The backup log is empty (no backups have run yet)."
