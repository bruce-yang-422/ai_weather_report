"""將天氣圖片複製到 LINE media 目錄，使用台灣日期、時間與當日版本號。"""

import argparse
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DESTINATION = Path(r"D:\Tools\LINE_Automation\line-oa-archive\media\weather")


def copy_report(source: Path, destination: Path) -> Path:
    if not source.is_file():
        raise FileNotFoundError(f"找不到天氣圖片：{source}")

    destination.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone(timedelta(hours=8)))
    day = now.strftime("%Y%m%d")
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    pattern = re.compile(rf"{day}_\d{{6}}_v(\d+)\.png")
    versions = []
    for entry in destination.iterdir():
        match = pattern.fullmatch(entry.name)
        if match:
            versions.append(int(match.group(1)))
    version = max(versions, default=0) + 1

    while True:
        target = destination / f"{timestamp}_v{version:03d}.png"
        try:
            output = target.open("xb")
        except FileExistsError:
            version += 1
            continue
        try:
            with output, source.open("rb") as input_file:
                shutil.copyfileobj(input_file, output)
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "output" / "weather_report.png")
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    try:
        target = copy_report(args.source, args.destination)
    except OSError as error:
        parser.exit(1, f"複製失敗：{error}\n")
    print(f"已複製：{target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
