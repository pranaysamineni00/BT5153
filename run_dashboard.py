"""Kill any existing server on port 5001 and launch the LexScan dashboard."""
import os
import signal
import socket
import subprocess
import sys
import time
import webbrowser
from collections import deque
from pathlib import Path


PORT = 5001
URL = f"http://127.0.0.1:{PORT}"
APP_DIR = Path(__file__).resolve().parent
LOG_PATH = APP_DIR / "dashboard_server.log"
MAX_RESTARTS = 3
RESTART_WINDOW_SECONDS = 300
RESTART_DELAY_SECONDS = 2.0


def kill_port(port: int) -> None:
    result = subprocess.run(
        ["lsof", "-ti", f"tcp:{port}"],
        capture_output=True, text=True
    )
    pids = result.stdout.strip().split()
    for pid in filter(None, pids):
        subprocess.run(["kill", "-TERM", pid], check=False)
        print(f"Stopped process {pid} on port {port}.")

    if not pids:
        return

    deadline = time.time() + 3.0
    while time.time() < deadline:
        remaining = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}"],
            capture_output=True,
            text=True,
        ).stdout.strip().split()
        if not list(filter(None, remaining)):
            return
        time.sleep(0.2)

    for pid in filter(None, subprocess.run(
        ["lsof", "-ti", f"tcp:{port}"],
        capture_output=True,
        text=True,
    ).stdout.strip().split()):
        subprocess.run(["kill", "-KILL", pid], check=False)
        print(f"Force-stopped process {pid} on port {port}.")


def launch_server() -> subprocess.Popen:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONFAULTHANDLER", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    return subprocess.Popen(
        [sys.executable, "-u", "app.py"],
        cwd=str(APP_DIR),
        env=env,
        start_new_session=True,
    )


def stop_server(server: subprocess.Popen) -> None:
    if server.poll() is not None:
        return

    server.terminate()
    try:
        server.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait(timeout=5)


def wait_for_server(port: int, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.25)
    return False


def format_exit(code: int) -> str:
    if code >= 0:
        return f"exit code {code}"

    sig_num = abs(code)
    try:
        sig_name = signal.Signals(sig_num).name
    except ValueError:
        sig_name = "UNKNOWN"
    return f"signal {sig_name} ({code})"


def should_restart(restarts: deque[float]) -> bool:
    now = time.time()
    while restarts and now - restarts[0] > RESTART_WINDOW_SECONDS:
        restarts.popleft()
    return len(restarts) < MAX_RESTARTS


def print_kill_hint(code: int) -> None:
    if code == -9:
        print("SIGKILL usually means the OS or another process killed the server.")
        print("If this keeps happening, check memory pressure and the log file below.")


def main() -> None:
    kill_port(PORT)
    time.sleep(0.5)

    server = launch_server()
    restart_times: deque[float] = deque()

    print(f"Server starting (PID {server.pid})...")
    print(f"Server log: {LOG_PATH}")
    if not wait_for_server(PORT):
        print("Server did not become ready in time.")
        stop_server(server)
        sys.exit(1)

    try:
        webbrowser.open(URL)
        print(f"Opened {URL}")
    except Exception as exc:
        print(f"Server is running at {URL}, but the browser could not be opened automatically: {exc}")

    try:
        while True:
            code = server.poll()
            if code is not None:
                if code == 0:
                    print("Server stopped.")
                    sys.exit(0)

                exit_text = format_exit(code)
                print(f"Server exited unexpectedly with {exit_text}.")
                print_kill_hint(code)

                restart_times.append(time.time())
                if not should_restart(restart_times):
                    print("Restart limit reached. Leaving the dashboard stopped.")
                    sys.exit(1)

                print(f"Restarting server in {RESTART_DELAY_SECONDS:.0f}s...")
                time.sleep(RESTART_DELAY_SECONDS)
                server = launch_server()
                print(f"Server restarting (PID {server.pid})...")
                print(f"Server log: {LOG_PATH}")
                if not wait_for_server(PORT):
                    print("Server did not become ready after restart.")
                    stop_server(server)
                    sys.exit(1)
            time.sleep(1)
    except KeyboardInterrupt:
        stop_server(server)
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
