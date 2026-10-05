"""依序產生、複製並發送天氣報告。"""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = (
    "weather.py",
    "copy_weather_report.py",
    "send_weather_report.py",
)


def main() -> int:
    for index, name in enumerate(SCRIPTS, start=1):
        print(f"\n[{index}/{len(SCRIPTS)}] 執行 {name}", flush=True)
        try:
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / name)],
                cwd=ROOT,
            )
        except OSError as error:
            print(f"無法執行 {name}：{error}", file=sys.stderr)
            return 1
        if result.returncode != 0:
            print(f"{name} 執行失敗，停止後續步驟。", file=sys.stderr)
            return result.returncode
    print("\n天氣報告產生、複製與發送流程已完成。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n已取消執行。", file=sys.stderr)
        raise SystemExit(130)
