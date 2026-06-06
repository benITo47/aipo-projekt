"""Pull a match clip from YouTube via yt-dlp for demo / pseudo-labeling."""
from __future__ import annotations

from pathlib import Path

from rich.console import Console

console = Console()


def download(url: str, out_dir: Path, resolution: str = "1080") -> Path:
    try:
        from yt_dlp import YoutubeDL
    except ImportError as e:
        raise SystemExit("Install yt-dlp first: pip install yt-dlp") from e

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    opts = {
        "format": f"bestvideo[height<={resolution}][ext=mp4]+bestaudio[ext=m4a]/best[height<={resolution}]",
        "outtmpl": str(out_dir / "%(title).80s-%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": False,
        "noprogress": False,
    }

    console.log(f"[cyan]yt-dlp[/] {url} → {out_dir}")
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    path = Path(ydl.prepare_filename(info)).with_suffix(".mp4")
    console.log(f"[green]Saved[/] {path}")
    return path
