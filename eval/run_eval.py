"""Прогон золотого набора. Метрики: Refusal Accuracy, Source Hit Rate."""
import json
import os
import sys

import httpx

API_URL = os.getenv("API_URL", "http://localhost:8000")


def run() -> int:
    with open("eval/golden_set.json", encoding="utf-8") as f:
        cases = json.load(f)

    correct_refusal = correct_source = 0
    for c in cases:
        try:
            r = httpx.post(
                f"{API_URL}/chat",
                json={"message": c["question"]},
                headers={"X-User-Id": c["expected_user"]},
                timeout=180.0,
            )
            r.raise_for_status()
            data = r.json()
        except (httpx.HTTPError, ValueError) as e:
            print(f"[ERR ] user={c['expected_user']:<9} q={c['question'][:45]:<45} {e}")
            continue

        refused = data["refused"]
        ok_refusal = refused == c["should_refuse"]
        correct_refusal += ok_refusal

        ok_source = True
        if not c["should_refuse"] and c["expected_source"]:
            sources = {cit["source"] for cit in data["citations"]}
            ok_source = c["expected_source"] in sources
            correct_source += ok_source

        status = "OK  " if ok_refusal and ok_source else "FAIL"
        print(f"[{status}] user={c['expected_user']:<9} q={c['question'][:45]:<45} "
              f"refused={refused} citations={[x['source'] for x in data['citations']]}")

    total = len(cases)
    need_source = sum(1 for c in cases if not c["should_refuse"] and c["expected_source"])
    print(f"\nRefusal Accuracy: {correct_refusal}/{total}")
    print(f"Source Hit Rate:  {correct_source}/{need_source}")
    return 0 if correct_refusal == total and correct_source == need_source else 1


if __name__ == "__main__":
    sys.exit(run())