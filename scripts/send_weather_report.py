"""發送當天最新天氣圖片；重送既有收件人前要求 CLI 確認。"""

import argparse
import hashlib
import json
import re
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
LINE_ROOT = Path(r"D:\Tools\LINE_Automation\line-oa-archive")
TAIWAN = timezone(timedelta(hours=8))


def latest_image(directory, day):
    pattern = re.compile(rf"{day}_(\d{{6}})_v(\d+)\.png")
    candidates = []
    for path in directory.iterdir():
        match = pattern.fullmatch(path.name)
        if match and path.is_file():
            candidates.append((int(match[2]), match[1], path))
    if not candidates:
        raise ValueError(f"找不到 {day} 的天氣圖片：{directory}")
    return max(candidates)[2]


def read_env(path):
    values = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def save_state(path, state):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(args):
    day = datetime.now(TAIWAN).strftime("%Y%m%d")
    image = latest_image(args.media, day)
    url = args.base_url.rstrip("/") + "/weather/" + image.name
    data = json.loads(args.subscribers.read_text(encoding="utf-8-sig"))
    credentials = read_env(args.env)
    recipients = {}
    for subscriber in data["subscribers"]:
        if subscriber.get("active") is False or subscriber.get("status") in ("inactive", "unsubscribed", "disabled"):
            continue
        oa = subscriber["oa_basic_id"].lstrip("@").upper()
        recipient = subscriber["recipient_id"]
        token = credentials.get(f"LINE_CHANNEL_ACCESS_TOKEN_{oa}")
        if not token:
            raise ValueError(f"缺少官方帳號 {oa} 的 access token")
        recipients[(oa, recipient)] = token
    print(f"圖片：{image.name}\n網址：{url}\n收件人數：{len(recipients)}")
    if args.dry_run:
        print("預覽完成，未呼叫 LINE API。")
        return 0
    if not recipients:
        return 0
    args.history.parent.mkdir(parents=True, exist_ok=True)
    lock = args.history.with_suffix(".json.lock")
    try:
        lock_file = lock.open("x")
    except FileExistsError:
        raise ValueError(f"另一個發送程序執行中，或上次中斷留下鎖檔：{lock}")
    try:
        with lock_file:
            state = json.loads(args.history.read_text(encoding="utf-8-sig")) if args.history.exists() else {"schema_version": 1, "reports": {}}
            reports = state["reports"]
            report = reports.setdefault(image.name, {"url": url, "attempts": []})
            attempts = report["attempts"]
            previous = {(item["oa"], item["recipient_id"]) for item in attempts}
            repeated = previous.intersection(recipients)
            resend = False
            if repeated:
                print(f"此圖片已有 {len(repeated)} 位收件人的發送紀錄（包含失敗或結果不明）。")
                if sys.stdin.isatty():
                    resend = input("要對這些收件人重複發送嗎？[y/N]：").strip().lower() in ("y", "yes")
                else:
                    print("非互動模式：略過已有紀錄的收件人。")
            targets = {key: token for key, token in recipients.items() if resend or key not in previous}
            if not targets:
                print("沒有需要發送的收件人。")
                return 0
            # 發送前確認公開圖片正是本機選取的版本。
            response = requests.get(url, timeout=(10, 60))
            response.raise_for_status()
            if hashlib.sha256(response.content).digest() != hashlib.sha256(image.read_bytes()).digest():
                raise ValueError("公開圖片與本機圖片內容不一致，停止發送。")
            failures = 0
            for (oa, recipient), token in targets.items():
                attempt = {"oa": oa, "recipient_id": recipient, "time": datetime.now(TAIWAN).isoformat(), "status": "pending", "retry_key": str(uuid.uuid4())}
                attempts.append(attempt)
                save_state(args.history, state)
                try:
                    response = requests.post(
                        "https://api.line.me/v2/bot/message/push",
                        headers={"Authorization": f"Bearer {token}", "X-Line-Retry-Key": attempt["retry_key"]},
                        json={"to": recipient, "messages": [
                            {"type": "text", "text": args.message or f"{day[:4]}/{day[4:6]}/{day[6:]} 天氣報告"},
                            {"type": "image", "originalContentUrl": url, "previewImageUrl": url},
                        ]},
                        timeout=(10, 60),
                    )
                    attempt["http_status"] = response.status_code
                    attempt["request_id"] = response.headers.get("x-line-request-id")
                    attempt["status"] = "accepted" if response.status_code == 200 else ("unknown" if response.status_code >= 500 else "failed")
                except requests.RequestException:
                    attempt["status"] = "unknown"
                save_state(args.history, state)
                print(f"{oa} / …{recipient[-6:]}：{attempt['status']}")
                failures += attempt["status"] != "accepted"
            print(f"完成：LINE 已接受 {len(targets) - failures}，失敗或結果不明 {failures}。")
            return 1 if failures else 0
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media", type=Path, default=LINE_ROOT / "media" / "weather")
    parser.add_argument("--subscribers", type=Path, default=LINE_ROOT / "subscribers" / "weather.json")
    parser.add_argument("--env", type=Path, default=LINE_ROOT / ".env")
    parser.add_argument("--base-url", default="https://reports.stack-base.com/media")
    parser.add_argument("--history", type=Path, default=ROOT / "output" / "weather_send_history.json")
    parser.add_argument("--message", help="自訂報告文字")
    parser.add_argument("--dry-run", action="store_true", help="只檢查檔案、名冊及憑證，不發送")
    args = parser.parse_args()
    try:
        return run(args)
    except (OSError, ValueError, KeyError, TypeError, requests.RequestException) as error:
        # 不輸出 API 回應或憑證。
        print(f"發送停止：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
