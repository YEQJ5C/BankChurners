"""Projekt inicializáló modul a tiszta MLOps mappaszerkezet
és a Python csomagok felépítéséhez.
"""

from pathlib import Path
from typing import List


class ProjectInitializer:
    """A BankChurners projekt mappaszerkezetének platformfüggetlen,
    objektumorientált inicializálásáért felelős osztály.
    """

    def __init__(self, base_path: str = ".") -> None:
        self.base_path: Path = Path(base_path)
        self.directories: List[str] = [
            "data",
            "data/raw",
            "data/processed",
            "configs",
            ".github",          # GitHub mappa
            ".github/workflows",  # GitHub Actions workflows mappa
            "src",
            "api",
            "streamlit_app",
            "tests",
            "logs",
            "models",
            "docs",
        ]
        self.packages: List[str] = ["src", "api", "tests", "configs"]

    def create_project_structure(self) -> None:
        """Biztonságosan létrehozza a könyvtárakat és a csomag-fájlokat."""
        for dir_name in self.directories:
            dir_path = self.base_path / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"[OK] Könyvtár létrehozva vagy létezik: {dir_path}")

            # Python csomag mappákba __init__.py elhelyezése
            if dir_name in self.packages:
                init_file = dir_path / "__init__.py"
                if not init_file.exists():
                    init_file.touch()
                    print(f"[OK] Fájl létrehozva: {init_file}")


if __name__ == "__main__":
    initializer = ProjectInitializer()
    initializer.create_project_structure()
