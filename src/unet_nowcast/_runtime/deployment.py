"""Render user systemd units into this workspace; does not install or activate them."""

from pathlib import Path


def quote_path(path: Path, *, executable: bool = False) -> str:
    value = str(path).replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    if "\n" in value or "\r" in value:
        raise ValueError("Newlines are not supported in installation paths")
    if executable:
        value = value.replace("$", "$$")
    return f'"{value}"'


def render_units(project: Path, module: str, model_name: str, service_name: str):
    project = project.resolve()
    deploy = project / "deploy"
    executable = project / ".venv/bin/python"
    if not executable.exists():
        raise SystemExit("Run uv sync --locked before rendering the units")
    generated = deploy / "generated"
    generated.mkdir(exist_ok=True)
    command = (
        f"{quote_path(executable, executable=True)} -m {module} "
        "--config config/latam.json --source ftp --scheduled"
    )
    service = f"""[Unit]
Description=Acquire GSMaP NOW and publish new {model_name} forecasts

[Service]
Type=oneshot
WorkingDirectory={str(project).replace("%", "%%")}
EnvironmentFile={str(project / ".env").replace("%", "%%")}
ExecStart={command}
TimeoutStartSec=20min
UMask=0077
"""
    (generated / f"{service_name}.service").write_text(service, encoding="utf-8")
    (generated / f"{service_name}.timer").write_text(
        (deploy / f"{service_name}.timer").read_text(encoding="utf-8"), encoding="utf-8"
    )
    print(f"Units prepared in {generated}; nothing has been activated")
