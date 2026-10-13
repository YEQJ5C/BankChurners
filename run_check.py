"""Runner script for executing the environment validation."""

from src.logger import get_logger
import check_env

# A központi strukturált logger használata
logger = get_logger("churn.env_runner")

if __name__ == "__main__":
    logger.info("Környezeti validáció indítása...")
    print("Running environment validation...")
    checks = check_env.collect()
    logger.info(
        "Környezeti adatok és csomagok összegyűjtve",
        extra={"package_count": len(checks.get("packages", {}))}
    )

    report = check_env.render(checks)
    logger.info("Környezeti jelentés (markdown) generálva.")

    print("\n" + "=" * 40)
    print(report)
    print("=" * 40 + "\n")

    saved_path = check_env.save_report(report)
    logger.info(
        "Környezeti jelentés sikeresen elmentve",
        extra={"saved_path": str(saved_path)}
    )
    print(f"[OK] Report successfully saved to: {saved_path}")
