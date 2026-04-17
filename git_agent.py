"""
git_agent.py — Auto-commit agent for mlb_ai
Watches the project folder for changes and commits + pushes automatically.

Usage:
    pip install watchdog gitpython
    python git_agent.py

Stop with Ctrl+C
"""

import time
import logging
import subprocess
import threading
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ── Config ────────────────────────────────────────────────────────────────────
REPO_PATH = Path(__file__).parent.resolve()
DEBOUNCE_SECONDS = 30          # wait this long after last change before committing
LOG_FILE = REPO_PATH / "git_agent.log"
IGNORED_NAMES = {".git", "__pycache__", "git_agent.log", ".idea"}
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("git_agent")


def run_git(*args) -> tuple[int, str, str]:
    """Run a git command in the repo directory. Returns (returncode, stdout, stderr)."""
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_PATH,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def has_changes() -> bool:
    """Return True if there are staged or unstaged changes."""
    code, out, _ = run_git("status", "--porcelain")
    return bool(out)


def commit_and_push():
    """Stage all changes, commit with a timestamp message, and push."""
    if not has_changes():
        log.info("No changes detected — skipping commit.")
        return

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    commit_msg = f"Auto-commit: changes detected at {timestamp}"

    log.info("Changes found — staging all files...")
    code, out, err = run_git("add", ".")
    if code != 0:
        log.error(f"git add failed: {err}")
        return

    log.info(f"Committing: {commit_msg}")
    code, out, err = run_git("commit", "-m", commit_msg)
    if code != 0:
        log.error(f"git commit failed: {err}")
        return
    log.info(f"Committed: {out}")

    log.info("Pushing to origin...")
    code, out, err = run_git("push", "origin", "main")
    if code != 0:
        log.error(f"git push failed: {err}")
    else:
        log.info(f"Pushed successfully. {out}")


class ChangeHandler(FileSystemEventHandler):
    """Watches for file system events and triggers a debounced commit."""

    def __init__(self):
        super().__init__()
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def _is_ignored(self, path: str) -> bool:
        parts = Path(path).parts
        return any(part in IGNORED_NAMES for part in parts)

    def _schedule_commit(self):
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(DEBOUNCE_SECONDS, commit_and_push)
            self._timer.daemon = True
            self._timer.start()
            log.info(
                f"Change detected — commit scheduled in {DEBOUNCE_SECONDS}s "
                f"(timer resets on further changes)."
            )

    def on_modified(self, event):
        if not event.is_directory and not self._is_ignored(event.src_path):
            log.debug(f"Modified: {event.src_path}")
            self._schedule_commit()

    def on_created(self, event):
        if not event.is_directory and not self._is_ignored(event.src_path):
            log.debug(f"Created: {event.src_path}")
            self._schedule_commit()

    def on_deleted(self, event):
        if not event.is_directory and not self._is_ignored(event.src_path):
            log.debug(f"Deleted: {event.src_path}")
            self._schedule_commit()

    def on_moved(self, event):
        if not event.is_directory and not self._is_ignored(event.dest_path):
            log.debug(f"Moved: {event.src_path} -> {event.dest_path}")
            self._schedule_commit()


def main():
    log.info(f"git_agent starting — watching: {REPO_PATH}")
    log.info(f"Debounce: {DEBOUNCE_SECONDS}s after last change")

    # Sanity check — make sure this is actually a git repo
    code, _, err = run_git("rev-parse", "--git-dir")
    if code != 0:
        log.error("Not a git repository! Run 'git init' first.")
        return

    handler = ChangeHandler()
    observer = Observer()
    observer.schedule(handler, str(REPO_PATH), recursive=True)
    observer.start()

    log.info("Watching for changes... (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Stopping git_agent...")
        observer.stop()
    observer.join()
    log.info("git_agent stopped.")


if __name__ == "__main__":
    main()