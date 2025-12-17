"""
Simple HTTP server for 3D visualization.
"""

from __future__ import annotations

import http.server
import socketserver
import webbrowser
from pathlib import Path
from threading import Thread

from rich.console import Console

console = Console()


def start_viewer_server(
    html_path: str | Path,
    port: int = 8080,
    open_browser: bool = True,
) -> None:
    """
    Start a simple HTTP server to serve the 3D viewer.

    Args:
        html_path: Path to the HTML file to serve
        port: Port number
        open_browser: Whether to open in default browser
    """
    html_path = Path(html_path)
    if not html_path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_path}")

    directory = html_path.parent

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def log_message(self, format, *args):
            # Suppress default logging
            pass

    with socketserver.TCPServer(("", port), Handler) as httpd:
        url = f"http://localhost:{port}/{html_path.name}"
        console.print(f"[green]3D Viewer running at:[/green] {url}")
        console.print("[dim]Press Ctrl+C to stop[/dim]")

        if open_browser:
            webbrowser.open(url)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            console.print("\n[yellow]Server stopped[/yellow]")


def serve_in_background(
    html_path: str | Path,
    port: int = 8080,
) -> Thread:
    """
    Start viewer server in background thread.

    Returns the thread (can be joined to wait for completion).
    """
    html_path = Path(html_path)
    directory = html_path.parent

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def log_message(self, format, *args):
            pass

    server = socketserver.TCPServer(("", port), Handler)

    def run():
        server.serve_forever()

    thread = Thread(target=run, daemon=True)
    thread.start()

    console.print(f"[green]Background server started:[/green] http://localhost:{port}/{html_path.name}")

    return thread
