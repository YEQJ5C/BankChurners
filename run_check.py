"""Runner script for executing the environment validation."""

import check_env

if __name__ == "__main__":
    print("Running environment validation...")
    checks = check_env.collect()
    report = check_env.render(checks)

    print("\n" + "=" * 40)
    print(report)
    print("=" * 40 + "\n")

    saved_path = check_env.save_report(report)
    print(f"[OK] Report successfully saved to: {saved_path}")
