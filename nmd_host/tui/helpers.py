"""Low-level process / port / PID helpers for the NebulonMind terminal UI.

These are the helper functions previously living in ``nebulonmind.py``;
they are shared by the start/stop/restart commands and may be used by any
terminal UI frontend.
"""

from __future__ import annotations

import os
import time
import socket

import platform
import subprocess


def is_port_open(host: str, port: int) -> bool:
    """Return True if a TCP port is open/listening on the given host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0


def is_pid_alive(pid: int) -> bool:
    """Return True if the process with ``pid`` is still alive."""
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        if platform.system() != "Windows":
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False
        else:
            try:
                output = subprocess.check_output(
                    ["tasklist", "/FI", f"PID eq {pid}"],
                    stderr=subprocess.DEVNULL,
                ).decode()
                return str(pid) in output
            except subprocess.CalledProcessError:
                return False


def find_process_tree_root(pid: int) -> int:
    """Walk up the parent chain to find the top-most uvicorn process."""
    try:
        import psutil
    except ImportError:
        return pid
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return pid
    root = proc
    while True:
        try:
            parent = root.parent()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
        if parent is None:
            break
        try:
            cmd = " ".join(parent.cmdline()).lower()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
        if "uvicorn" in cmd or "nmd_host" in cmd:
            root = parent
        else:
            break
    return root.pid


def kill_process_tree(pid: int) -> None:
    """Kill a process and its whole uvicorn tree."""
    root_pid = find_process_tree_root(pid)
    try:
        import psutil
        root = psutil.Process(root_pid)
        for child in reversed(root.children(recursive=True)):
            try:
                child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        root.kill()
        root.wait(timeout=5)
    except (ImportError, psutil.NoSuchProcess, psutil.TimeoutExpired):
        kill_process(root_pid, force=True)


def kill_process(pid: int, force: bool = False) -> None:
    try:
        import psutil
        proc = psutil.Process(pid)
        if force:
            proc.kill()
        else:
            proc.terminate()
        proc.wait(timeout=5)
    except ImportError:
        if platform.system() == "Windows":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/F", "/T"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            try:
                if force:
                    os.kill(pid, 9)
                else:
                    os.kill(pid, 15)
                    time.sleep(2)
                    if is_pid_alive(pid):
                        os.kill(pid, 9)
            except ProcessLookupError:
                pass
    except (psutil.NoSuchProcess, ProcessLookupError):
        pass
    except psutil.TimeoutExpired:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass


def find_pid_on_port(port: int) -> int | None:
    """Return the PID of the process listening on the given port (or None)."""
    try:
        import psutil
        for conn in psutil.net_connections(kind="inet"):
            laddr = conn.laddr
            if laddr and laddr.port == port and conn.pid:
                return conn.pid
    except (ImportError, psutil.AccessDenied, psutil.Error):
        pass

    try:
        if platform.system() == "Windows":
            output = subprocess.check_output(
                ["netstat", "-ano"],
                stderr=subprocess.DEVNULL,
            ).decode(errors="ignore")
            for line in output.splitlines():
                parts = line.split()
                if len(parts) >= 5 and f":{port}" in parts[1] and "LISTENING" in line:
                    pid = parts[-1]
                    if pid.isdigit():
                        return int(pid)
        else:
            output = subprocess.check_output(
                ["lsof", "-ti", f"tcp:{port}"],
                stderr=subprocess.DEVNULL,
            ).decode(errors="ignore").strip().splitlines()
            if output:
                pid = output[0].strip()
                if pid.isdigit():
                    return int(pid)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass
    return None


__all__ = [
    "is_port_open",
    "is_pid_alive",
    "find_process_tree_root",
    "kill_process_tree",
    "kill_process",
    "find_pid_on_port",
]