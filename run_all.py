import subprocess
import time
import sys
import webbrowser

BACKEND_PORT = 8000
FRONTEND_PORT = 8501


def kill_port(port):
    try:
        result = subprocess.check_output(f'netstat -ano | findstr :{port}', shell=True).decode()
    except:
        return

    for line in result.split("\n"):
        parts = line.split()
        if len(parts) >= 5:
            pid = parts[-1]
            if pid.isdigit():
                subprocess.call(f"taskkill /PID {pid} /F", shell=True)


def run_backend():
    return subprocess.Popen([
        "python", "-m", "uvicorn",
        "src.api.main:app",
        "--host", "0.0.0.0",
        "--port", str(BACKEND_PORT),
        "--reload"
    ])


def run_frontend():
    return subprocess.Popen([
        "python", "-m", "streamlit",
        "run", "app/streamlit_app.py",
        "--server.port", str(FRONTEND_PORT),
        "--server.address=0.0.0.0"
    ])


if __name__ == "__main__":
    print("🔍 Killing old processes...")
    kill_port(BACKEND_PORT)
    kill_port(FRONTEND_PORT)

    print("🚀 Starting backend...")
    backend = run_backend()
    time.sleep(2)

    print("🚀 Starting frontend...")
    frontend = run_frontend()

    # WAIT a little so frontend actually starts
    time.sleep(3)

    # 🚀 ONLY OPEN FRONTEND
    print("🌍 Opening frontend page...")
    webbrowser.open(f"http://localhost:{FRONTEND_PORT}")

    print(f"✔ Backend running on:   http://localhost:{BACKEND_PORT}")
    print(f"✔ Frontend running on:  http://localhost:{FRONTEND_PORT}")
    print("Press CTRL + C to stop both.")

    try:
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        backend.terminate()
        frontend.terminate()
