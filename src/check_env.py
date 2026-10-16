"""Environment check module for validating the BankChurners MLOps setup.

Adheres to PEP 8 standards, providing dynamic package inspection from
requirements.txt, environment verification, and markdown report generation.
"""

from datetime import datetime
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import sys
from typing import Any, Dict, List

from src.logger import get_logger

# A központi strukturált logger használata
logger = get_logger("churn.env_runner")


def parse_requirements(req_path: str = "requirements.txt") -> List[str]:
    """Parse requirement package names from requirements.txt."""
    packages = []
    path = Path(req_path)
    if not path.exists():
        return packages

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Handle editable install (-e . or similar paths)
            if line.startswith("-e") or line.startswith("--editable"):
                packages.append("bankchurners")
                continue

            # Strip version specifiers (e.g., mlflow==3.1.0 -> mlflow)
            for char in ["==", ">=", "<=", "~=", ">", "<"]:
                if char in line:
                    line = line.split(char)[0].strip()
                    break
            if line:
                packages.append(line)
    return packages


def collect() -> Dict[str, Any]:
    """Collect environment, python, requirements,

    and directory details.
    """
    checks: Dict[str, Any] = {}

    # Python and system information
    checks["python_version"] = sys.version
    checks["python_executable"] = sys.executable
    checks["platform"] = platform.platform()

    # Virtual environment verification
    virtual_env = os.environ.get("VIRTUAL_ENV")
    checks["virtual_env"] = (
        virtual_env if virtual_env else "No active virtual environment found!"
    )

    # Dynamic package version checking based on requirements.txt
    required_packages = parse_requirements()
    pkg_versions = {}
    for pkg in required_packages:
        try:
            pkg_versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pkg_versions[pkg] = "Not installed ❌"
    checks["packages"] = pkg_versions

    # External MLOps tools verification
    tools = ["git", "dvc", "docker", "pre-commit"]
    checks["tools"] = {tool: shutil.which(tool) is not None for tool in tools}

    # Mandatory directory structure integrity
    required_dirs = [
        "data",
        "data/raw",
        "data/processed",
        "configs",
        "src",
        "api",
        "streamlit_app",
        "tests",
        "logs",
        "docs",
        "models",
    ]
    checks["directories"] = {d: Path(d).is_dir() for d in required_dirs}

    return checks


def render(checks: Dict[str, Any]) -> str:
    """Render the collected checks into a formatted Markdown report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# BankChurners - Environment Check Report
**Generated:** {timestamp}

## 1. System & Python Information
- **Platform:**
{checks['platform']}
- **Python Executable:** `
{checks['python_executable']}`
- **Python Version:** `
{checks['python_version']}`
- **Virtual Environment (VIRTUAL_ENV):** `
{checks['virtual_env']}`

## 2. Installed Packages (from requirements.txt)
"""
    for pkg, ver in checks["packages"].items():
        report += f"- **{pkg}:** {ver}\n"

    report += "\n## 3. MLOps Tools Availability\n"
    for tool, available in checks.get("tools", {}).items():
        status = "✅ Available" if available else "❌ NOT FOUND"
        report += f"- **{tool}:** {status}\n"

    report += "\n## 4. Directory Structure Integrity\n"
    for d, exists in checks["directories"].items():
        status = "✅ Exists" if exists else "❌ Missing"
        report += f"- `{d}/`: {status}\n"

    return report


def save_report(report_content: str, docs_dir: str = "docs") -> Path:
    """Save the rendered markdown report with a

    timestamp into the docs directory.
    """
    Path(docs_dir).mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(docs_dir) / f"env_report_{timestamp_str}.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    return report_path


if __name__ == "__main__":
    logger.info("Környezeti validáció indítása...")
    print("Running environment validation...")

    checks = collect()
    logger.info(
        "Környezeti adatok és csomagok összegyűjtve",
        extra={"package_count": len(checks.get("packages", {}))},
    )

    report = render(checks)
    logger.info("Környezeti jelentés (markdown) generálva.")

    print("\n" + "=" * 40)
    print(report)
    print("=" * 40 + "\n")

    saved_path = save_report(report)
    logger.info(
        "Környezeti jelentés sikeresen elmentve",
        extra={"saved_path": str(saved_path)},
    )
    print(f"[OK] Report successfully saved to: {saved_path}")
