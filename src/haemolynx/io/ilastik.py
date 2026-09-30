"""Utilities for running ilastik segmentation in headless mode."""
from __future__ import annotations

import logging
import os
import queue
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

#: How often (seconds) the wait loop wakes to check for cancellation, the
#: timeout and new output while ilastik runs.
POLL_INTERVAL_SECONDS = 0.5

#: How many of ilastik's last output lines a failure message quotes.
_TAIL_LINES = 40


def ilastik_output_is_up_to_date(
    output_path: str | Path, *sources: str | Path
) -> bool:
    """Whether *output_path* exists and is newer than every one of *sources*.

    The same rule ``make`` uses: an output is current when nothing it was made
    from has changed since. A source that does not exist cannot be compared,
    so the answer is then False and ilastik runs (and reports the missing file).
    """
    output_path = Path(output_path)
    if not output_path.is_file():
        return False
    made = output_path.stat().st_mtime
    for source in sources:
        source = Path(source)
        if not source.exists() or source.stat().st_mtime > made:
            return False
    return True


def _kill_process_tree(process: subprocess.Popen) -> None:
    """Stop *process* and anything it started.

    ``ilastik.exe`` (and a ``.bat``/shell wrapper round it) is a launcher that
    starts the real Python process; killing only the launcher leaves ilastik
    running and holding the output file open.
    """
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                capture_output=True,
                check=False,
            )
        else:
            import signal

            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass
    if process.poll() is None:
        process.kill()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover - the OS refused
        logger.warning("ilastik (pid %s) did not exit after being killed.", process.pid)


def _read_lines(stream, lines: "queue.Queue[Optional[str]]") -> None:
    """Hand every line of *stream* to *lines*, then ``None`` at its end."""
    try:
        for line in iter(stream.readline, ""):
            lines.put(line.rstrip("\r\n"))
    finally:
        lines.put(None)


def run_ilastik_headless_segmentation(
    input_image_path: str | Path,
    classifier_path: str | Path,
    output_path: str | Path,
    ilastik_executable: str | Path = "ilastik.exe",
    timeout: float | None = None,
    *,
    reuse_existing: bool = False,
    poll: Optional[Callable[[], None]] = None,
) -> Path:
    """Run ilastik headless segmentation for a single image.

    Parameters
    ----------
    input_image_path:
        Path to the image file to segment.
    classifier_path:
        Path to the ilastik project/classifier file (.ilp).
    output_path:
        Path to write the segmented output image.
    ilastik_executable:
        ilastik executable. Defaults to ``ilastik.exe`` and can also be a full path.
    timeout:
        Kill the subprocess and raise if it runs longer than this many
        seconds. ``None`` (this function's own default) waits forever; the
        pipeline itself always passes ``ilastik_timeout_seconds`` so a hung
        or misconfigured ilastik process cannot block a run indefinitely.
    reuse_existing:
        Return *output_path* without running ilastik when it already exists
        and is newer than both the image and the project -- a segmentation of
        this image with this classifier has been made since either changed.
    poll:
        Called about every :data:`POLL_INTERVAL_SECONDS` while ilastik runs.
        Whatever it raises stops the run: ilastik (and anything it started) is
        killed, and the exception propagates. This is how the napari panel's
        stop reaches a segmentation that can take hours.

    ilastik's own output is logged line by line as it arrives (``INFO``), so a
    long segmentation shows its progress in the run log rather than nothing
    until it ends.
    """
    input_image_path = Path(input_image_path)
    classifier_path = Path(classifier_path)
    output_path = Path(output_path)

    if not input_image_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_image_path}")
    if not classifier_path.exists():
        raise FileNotFoundError(f"Classifier/project file not found: {classifier_path}")

    output_suffix = output_path.suffix.lower()
    if output_suffix not in {".tif", ".tiff", ".h5"}:
        raise ValueError(
            "Unsupported ilastik output extension. "
            "Use one of: .tif, .tiff, .h5"
        )

    if reuse_existing and ilastik_output_is_up_to_date(
        output_path, input_image_path, classifier_path
    ):
        logger.info(
            "Reusing ilastik output %s: it is newer than both %s and %s. "
            "Delete it, or turn off ilastik_reuse_existing_output, to segment again.",
            output_path,
            input_image_path.name,
            classifier_path.name,
        )
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # An old output left in place would be returned as this run's if ilastik
    # exited cleanly without writing one.
    output_path.unlink(missing_ok=True)
    # ilastik expects output placeholders in the pattern string.
    output_pattern = str(output_path.parent / f"{{nickname}}{output_suffix}")
    command = [
        str(ilastik_executable),
        "--headless",
        f"--project={classifier_path}",
        "--export_source=Simple Segmentation",
        f"--output_filename_format={output_pattern}",
        str(input_image_path),
    ]
    popen_kwargs: dict = {}
    if os.name != "nt":
        # Its own process group, so a kill reaches whatever the launcher starts.
        popen_kwargs["start_new_session"] = True
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            **popen_kwargs,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Could not find ilastik executable '{ilastik_executable}'. "
            "Set the ilastik_executable setting to the full ilastik path."
        ) from exc

    lines: "queue.Queue[Optional[str]]" = queue.Queue()
    reader = threading.Thread(
        target=_read_lines, args=(process.stdout, lines), daemon=True
    )
    reader.start()
    tail: deque[str] = deque(maxlen=_TAIL_LINES)
    started = time.monotonic()
    stream_open = True

    def drain(wait: float) -> None:
        nonlocal stream_open
        deadline = time.monotonic() + wait
        while stream_open:
            remaining = deadline - time.monotonic()
            try:
                if remaining > 0:
                    line = lines.get(timeout=remaining)
                else:
                    line = lines.get_nowait()
            except queue.Empty:
                return
            if line is None:
                stream_open = False
                return
            tail.append(line)
            if line.strip():
                logger.info("[ilastik] %s", line)

    try:
        while True:
            drain(POLL_INTERVAL_SECONDS)
            if process.poll() is not None and not stream_open:
                break
            if process.poll() is not None:
                # Exited; collect whatever it wrote last, then stop.
                drain(2.0)
                break
            if poll is not None:
                poll()
            if timeout is not None and time.monotonic() - started > float(timeout):
                _kill_process_tree(process)
                raise RuntimeError(
                    f"ilastik segmentation did not finish within {timeout} seconds "
                    "and was killed. Raise ilastik_timeout_seconds if this is a "
                    "genuinely large volume, or check the project file and input "
                    "for what is making ilastik hang.\n"
                    f"Command: {' '.join(command)}"
                )
    except BaseException:
        _kill_process_tree(process)
        raise
    finally:
        if process.stdout is not None:
            process.stdout.close()

    returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(
            "ilastik segmentation failed "
            f"(exit code {returncode}).\n"
            f"Command: {' '.join(command)}\n"
            "Last output:\n" + "\n".join(tail)
        )

    generated_path = output_path.parent / f"{input_image_path.stem}{output_suffix}"
    if generated_path.exists():
        if generated_path != output_path:
            generated_path.replace(output_path)
    if not output_path.exists():
        raise RuntimeError(
            "ilastik finished, but no output file was found at "
            f"'{output_path}'. Expected generated path: '{generated_path}'."
        )
    return output_path
