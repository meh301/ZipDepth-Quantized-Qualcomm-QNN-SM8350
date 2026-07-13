#!/usr/bin/env python3
"""Display the ZipDepth 10x7-inner-corner calibration target fullscreen."""

from __future__ import annotations

import argparse
import ctypes
import sys
import tkinter as tk
from dataclasses import dataclass
from ctypes import wintypes


BOARD_COLUMNS = 11  # 10 inner corners
BOARD_ROWS = 8      # 7 inner corners
BOARD_SCREEN_FRACTION = 0.88


@dataclass(frozen=True)
class Monitor:
    left: int
    top: int
    width: int
    height: int
    primary: bool
    name: str


def enable_dpi_awareness() -> None:
    """Keep monitor coordinates and Tk geometry in physical pixels on Windows."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-monitor DPI aware.
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def windows_monitors() -> list[Monitor]:
    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", wintypes.LONG),
            ("top", wintypes.LONG),
            ("right", wintypes.LONG),
            ("bottom", wintypes.LONG),
        ]

    class MONITORINFOEXW(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("rcMonitor", RECT),
            ("rcWork", RECT),
            ("dwFlags", wintypes.DWORD),
            ("szDevice", wintypes.WCHAR * 32),
        ]

    monitors: list[Monitor] = []
    callback_type = ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HMONITOR,
        wintypes.HDC,
        ctypes.POINTER(RECT),
        wintypes.LPARAM,
    )

    def collect(handle, _device_context, _rect, _data):
        info = MONITORINFOEXW()
        info.cbSize = ctypes.sizeof(MONITORINFOEXW)
        if ctypes.windll.user32.GetMonitorInfoW(handle, ctypes.byref(info)):
            bounds = info.rcMonitor
            monitors.append(
                Monitor(
                    left=bounds.left,
                    top=bounds.top,
                    width=bounds.right - bounds.left,
                    height=bounds.bottom - bounds.top,
                    primary=bool(info.dwFlags & 1),
                    name=info.szDevice,
                )
            )
        return True

    callback = callback_type(collect)
    ctypes.windll.user32.EnumDisplayMonitors(None, None, callback, 0)
    return sorted(monitors, key=lambda item: (not item.primary, item.left, item.top))


def available_monitors(root: tk.Tk) -> list[Monitor]:
    if sys.platform == "win32":
        found = windows_monitors()
        if found:
            return found
    return [
        Monitor(
            left=0,
            top=0,
            width=root.winfo_screenwidth(),
            height=root.winfo_screenheight(),
            primary=True,
            name="primary",
        )
    ]


def draw_checkerboard(canvas: tk.Canvas, width: int, height: int) -> None:
    canvas.delete("all")
    square = max(
        1,
        int(
            min(
                width * BOARD_SCREEN_FRACTION / BOARD_COLUMNS,
                height * BOARD_SCREEN_FRACTION / BOARD_ROWS,
            )
        ),
    )
    board_width = square * BOARD_COLUMNS
    board_height = square * BOARD_ROWS
    left = (width - board_width) // 2
    top = (height - board_height) // 2

    for row in range(BOARD_ROWS):
        for column in range(BOARD_COLUMNS):
            if (row + column) % 2 == 0:
                x0 = left + column * square
                y0 = top + row * square
                canvas.create_rectangle(
                    x0,
                    y0,
                    x0 + square,
                    y0 + square,
                    fill="black",
                    outline="black",
                    width=0,
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show an 11x8-square (10x7 inner-corner) camera calibration target."
    )
    parser.add_argument(
        "--monitor",
        "-m",
        type=int,
        default=0,
        help="monitor index to use; the primary monitor is 0 (default: 0)",
    )
    parser.add_argument(
        "--list-monitors",
        action="store_true",
        help="print detected monitors and exit",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    enable_dpi_awareness()

    root = tk.Tk()
    root.withdraw()
    monitors = available_monitors(root)

    if args.list_monitors:
        for index, monitor in enumerate(monitors):
            primary = " primary" if monitor.primary else ""
            print(
                f"{index}: {monitor.name} {monitor.width}x{monitor.height} "
                f"at ({monitor.left},{monitor.top}){primary}"
            )
        root.destroy()
        return 0

    if args.monitor < 0 or args.monitor >= len(monitors):
        root.destroy()
        print(
            f"error: monitor {args.monitor} does not exist; "
            f"choose 0 through {len(monitors) - 1}",
            file=sys.stderr,
        )
        return 2

    monitor = monitors[args.monitor]
    root.deiconify()
    root.title("ZipDepth camera calibration target")
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.configure(background="white", cursor="none")
    root.geometry(
        f"{monitor.width}x{monitor.height}{monitor.left:+d}{monitor.top:+d}"
    )

    canvas = tk.Canvas(
        root,
        background="white",
        borderwidth=0,
        highlightthickness=0,
        cursor="none",
    )
    canvas.pack(fill=tk.BOTH, expand=True)

    def redraw(event: tk.Event) -> None:
        draw_checkerboard(canvas, event.width, event.height)

    def close(_event: tk.Event | None = None) -> None:
        root.destroy()

    canvas.bind("<Configure>", redraw)
    root.bind("<Escape>", close)
    root.bind("q", close)
    root.bind("Q", close)
    root.after(100, root.focus_force)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
