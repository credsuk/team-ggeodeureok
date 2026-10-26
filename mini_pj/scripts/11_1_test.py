import requests
import time

url = "http://127.0.0.1:11434/api/generate"

model = "exaone3.5:2.4b"

prompts = {
    "A": "청년 일자리 정책의 향후 방향을 200자 이내로 설명하세요.",
    "B": "청년 일자리 정책의 향후 방향을 정확히 한 문장으로 설명하세요. 반드시 '향후 지방의회에서는'으로 시작하세요."
}

for name, prompt in prompts.items():
    start = time.perf_counter()

    response = requests.post(
        url,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 320
            }
        },
        timeout=120
    )

    response.raise_for_status()
    result = response.json()["response"].strip()

    print(f"\n[{name}]")
    print("응답:", result)
    print("글자 수:", len(result))
    print("소요 시간:", round(time.perf_counter() - start, 2), "초")