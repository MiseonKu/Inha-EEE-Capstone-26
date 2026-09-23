from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final


PACKAGE_ROOT: Final = Path("09_submission_reproduction_package")


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


REQUIRED_PATHS: Final[tuple[str, ...]] = (
    "README.md",
    "MANIFEST.md",
    "environment/README.md",
    "data/processed/clean_dataset_1link_v1000/dataset.csv",
    "data/processed/clean_dataset_1link_v2/dataset.csv",
    "data/processed/clean_dataset_2link_v1/dataset.csv",
    "data/schema/data_dictionary.md",
    "ml/final_model_code/evaluate_v2_gpu_density_aware_local_krr.py",
    "ml/final_model_code/evaluate_2link_density_aware_local_krr.py",
    "ml/final_model_artifacts/1link_density_aware_local_krr_fulltrain.joblib",
    "ml/final_model_artifacts/2link_density_aware_local_krr_fulltrain.joblib",
    "ml/final_results/final_runs/1link_v1000_density_aware_local_krr_50seed/run_manifest.json",
    "ml/final_results/final_runs/2link_density_aware_local_krr_50seed/run_manifest.json",
    "optimizer/source/motor_angle_optimizer.py",
    "optimizer/source/optimizer_linkage_2link_smoke.py",
    "experiment_preparation/fkik_workspace_point_algorithm/README.md",
    "experiment_preparation/p1_one_link_experiment/README.md",
    "experiment_preparation/p2_two_link_experiment/README.md",
    "hardware/p1_one_link/SO101 Assembly-link11.step",
    "hardware/p2_two_link/2_LINK_SO101_ASSEMBLY.step",
)

README_TERMS: Final[tuple[str, ...]] = (
    "data/",
    "ml/",
    "optimizer/",
    "experiment_preparation/",
    "hardware/",
    "prediction",
    "not closed-loop control",
    "Vive Tracker",
)

BANNED_DIR_NAMES: Final[tuple[str, ...]] = (
    ".git",
    ".omx",
    ".hermes",
    ".codegraph",
    "__pycache__",
    ".pytest_cache",
)

BANNED_TEXT_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile("ml" + "flow", re.IGNORECASE),
    re.compile(r"api[_-]?key", re.IGNORECASE),
    re.compile(r"password", re.IGNORECASE),
    re.compile(r"secret", re.IGNORECASE),
    re.compile(r"C:\\Users\\", re.IGNORECASE),
    re.compile(re.escape("E" + ":" + "\\"), re.IGNORECASE),
)

TEXT_EXTENSIONS: Final[set[str]] = {
    ".csv",
    ".json",
    ".md",
    ".py",
    ".txt",
    ".yaml",
    ".yml",
}


def check_required_paths(root: Path) -> CheckResult:
    missing = [relative for relative in REQUIRED_PATHS if not (root / relative).exists()]
    return CheckResult(
        name="required_paths",
        passed=not missing,
        detail="all required paths exist" if not missing else "missing: " + ", ".join(missing),
    )


def check_readme_coverage(root: Path) -> CheckResult:
    readme = root / "README.md"
    if not readme.exists():
        return CheckResult("readme_coverage", False, "README.md is missing")
    text = readme.read_text(encoding="utf-8", errors="replace")
    missing = [term for term in README_TERMS if term not in text]
    return CheckResult(
        name="readme_coverage",
        passed=not missing,
        detail="README covers required terms" if not missing else "missing terms: " + ", ".join(missing),
    )


def check_banned_dirs(root: Path) -> CheckResult:
    found = sorted({path.name for path in root.rglob("*") if path.is_dir() and path.name in BANNED_DIR_NAMES})
    return CheckResult(
        name="banned_dirs",
        passed=not found,
        detail="no banned directories found" if not found else "found: " + ", ".join(found),
    )


def check_sensitive_text(root: Path) -> CheckResult:
    matches: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        if path.name == "verify_submission_package.py":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in BANNED_TEXT_PATTERNS:
            if pattern.search(text):
                matches.append(f"{path.relative_to(root)}::{pattern.pattern}")
                break
    return CheckResult(
        name="sensitive_text",
        passed=not matches,
        detail="no sensitive text patterns found" if not matches else "matches: " + ", ".join(matches[:20]),
    )


def run_checks(root: Path) -> list[CheckResult]:
    if not root.exists():
        return [CheckResult("package_root", False, f"{root} does not exist")]
    return [
        CheckResult("package_root", True, f"{root} exists"),
        check_required_paths(root),
        check_readme_coverage(root),
        check_banned_dirs(root),
        check_sensitive_text(root),
    ]


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else PACKAGE_ROOT
    results = run_checks(root)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status} {result.name}: {result.detail}")
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
