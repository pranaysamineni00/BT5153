"""Kill any existing server on port 5001 and launch the LexScan dashboard."""
import subprocess
import sys
import time
import webbrowser


PORT = 5001
URL = f"http://127.0.0.1:{PORT}"


def kill_port(port: int) -> None:
    result = subprocess.run(
        ["lsof", "-ti", f"tcp:{port}"],
        capture_output=True, text=True
    )
    pids = result.stdout.strip().split()
    for pid in filter(None, pids):
        subprocess.run(["kill", "-9", pid], check=False)
        print(f"Killed process {pid} on port {port}.")


def main() -> None:
    kill_port(PORT)
    time.sleep(0.5)

    server = subprocess.Popen(
        [sys.executable, "app.py"],
        cwd=str(__file__).rsplit("/", 1)[0],
    )

    print(f"Server starting (PID {server.pid})…")
    time.sleep(3)

    webbrowser.open(URL)
    print(f"Opened {URL}")

    try:
        server.wait()
    except KeyboardInterrupt:
        server.terminate()
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
