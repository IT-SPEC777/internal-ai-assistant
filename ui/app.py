"""Минимальный чат на Gradio. Переключение пользователей = проверка ACL."""
import os

import gradio as gr
import httpx

API_URL = os.getenv("API_URL", "http://localhost:8000")
USERS = ["ivanov", "petrova", "sidorov", "kuznetsov"]


def respond(message, history, user_id):
    if not message.strip():
        return ""
    try:
        r = httpx.post(
            f"{API_URL}/chat",
            json={"message": message},
            headers={"X-User-Id": user_id},
            timeout=180.0,
        )
        r.raise_for_status()
    except httpx.HTTPError as e:
        return f"Ошибка API: {e}"

    data = r.json()
    text = data["answer"]
    if data.get("citations"):
        text += "\n\nИсточники:\n" + "\n".join(
            f"- [{c['n']}] {c['source']}" for c in data["citations"]
        )
    return text


with gr.Blocks(title="ИИ-помощник (прототип)") as demo:
    gr.Markdown("## ИИ Помощник по внутренним процессам, прототип")
    user_dd = gr.Dropdown(choices=USERS, value="sidorov", label="Пользователь (ACL)")
    gr.ChatInterface(
        fn=respond,
        additional_inputs=[user_dd],
        # примеры = [вопрос, пользователь]: при клике подставляются оба значения
        examples=[
            ["Как оформить пропуск в офис?", "sidorov"],
            ["Какие суточные в командировке по России?", "petrova"],
            ["Какие суточные в командировке по России?", "sidorov"],
            ["Сколько дней отпуска по умолчанию?", "kuznetsov"],
            ["Как сбросить пароль от VPN?", "ivanov"],
        ],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)