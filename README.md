# ИИ-помощник по внутренним процессам, прототип

RAG-сервис: гибридный поиск (dense + BM25 + RRF), ACL-фильтрация до передачи в LLM,
детерминированный гейт релевантности, LangGraph-оркестрация с веткой отказа,
on-premise LLM через Ollama.

## 0. Офлайн-тесты логики (без зависимостей, только Python 3.10+)

    python -m unittest discover -s tests -t . -v

Проверяют токенизацию, ACL на seed-данных, чанкование, RRF и гейт релевантности.
Тесты не проверяют качество эмбеддингов и ответы LLM, для этого есть eval/run_eval.py.

## 1. Инфраструктура

    docker compose up -d
    docker compose exec ollama ollama pull qwen2.5:3b

Для слабых машин можно взять qwen2.5:1.5b (переменная OLLAMA_MODEL в .env).

## 2. Python-окружение (Python 3.10+)

    python -m venv .venv
    source .venv/bin/activate         # Linux/macOS
    .venv\Scripts\activate            # Windows
    pip install -r requirements.txt
    cp .env.example .env              # Windows: copy .env.example .env

sentence-transformers тянет PyTorch, первая установка скачивает несколько ГБ.

## 3. Индексация

    python -m ingestion.embed_and_upsert

## 4. Запуск API и UI (два терминала, из корня проекта)

    uvicorn api.main:app --port 8000
    python ui/app.py

UI на http://localhost:7860. API при старте загружает модель эмбеддингов и BM25-индекс.
Если коллекции нет в Qdrant, упадёт с подсказкой запустить индексацию.

## 5. Быстрая проверка без Docker

    pip install sentence-transformers rank-bm25
    python minimal_demo.py

Показывает гибридный поиск, ACL и гейт без Qdrant и LLM. Также печатает dense-скоры (см. п. 6).

## 6. Калибровка порога MIN_DENSE_SCORE

Гейт релевантности: контекст считается найденным, если есть лексическое совпадение
(BM25 > 0 после удаления стоп-слов и усечения слов до 5 символов) или лучший
dense-скор не ниже MIN_DENSE_SCORE. Иначе API отвечает отказом, не вызывая LLM.

У моделей e5 косинусная близость похожих и непохожих текстов лежит в узком диапазоне,
поэтому дефолт 0.82 стартовая оценка, а не проверенное значение. Как откалибровать:
запустите minimal_demo.py или смотрите в логах API строку retrieval: top_dense=...
для заведомо релевантных и заведомо нерелевантных вопросов и выберите порог между ними.

## 7. Проверка через curl

    # публичный документ, виден всем
    curl -s -H "X-User-Id: sidorov" -H "Content-Type: application/json" \
         -d '{"message":"Как оформить пропуск?"}' http://localhost:8000/chat

    # финансы: petrova получает ответ, sidorov отказ
    curl -s -H "X-User-Id: petrova" -H "Content-Type: application/json" \
         -d '{"message":"Какие суточные в командировке?"}' http://localhost:8000/chat
    curl -s -H "X-User-Id: sidorov" -H "Content-Type: application/json" \
         -d '{"message":"Какие суточные в командировке?"}' http://localhost:8000/chat

## 8. Eval

    python eval/run_eval.py

Метрики: Refusal Accuracy и Source Hit Rate на eval/golden_set.json (7 кейсов).
Результат зависит от модели LLM и калибровки порога.

## 9. Соответствие требованиям архитектуры

| Требование | Где реализовано |
|---|---|
| Гибридный поиск (dense + BM25 + RRF) | api/retrieval.py, api/utils.py |
| ACL-фильтрация до LLM | _dense (фильтр Qdrant + payload-индекс), _bm25_search (Python-фильтр) |
| Гейт релевантности (отказ без LLM) | api/utils.is_relevant, ветка refuse в api/rag.py |
| LangGraph с условной веткой отказа | api/rag.py |
| On-premise LLM | api/llm.py (Ollama) |
| Guardrails | SYSTEM_PROMPT, _sanitize_output (грубая маска 11-19 цифр) |
| Аудит и трассируемость | поле citations, лог audit |
| Eval | eval/run_eval.py |

## 10. Ограничения прототипа

- Аутентификация через заголовок X-User-Id (подделывается). В проде нужен OIDC/JWT и группы из AD.
- BM25-индекс в памяти процесса, без инкрементального обновления. Стемминг: усечение до 5 символов.
- Нет streaming, истории диалога и reranker (ENABLE_RERANKER зарезервирован).
- Аудит-лог пишется в stdout. В проде нужно структурированное хранилище.
- Источники: seed-файлы, без интеграции с Confluence/Jira.
- Малые модели (1.5-3B) могут игнорировать правила промпта. Гейт снижает, но не исключает риск галлюцинаций.

## 11. Железо (ориентировочно, не измерялось)

Эмбеддер multilingual-e5-small и qwen2.5:3b ориентировочно рассчитаны на машину с 8+ ГБ RAM
без GPU. Скорость ответа на CPU заметно ниже, чем на GPU. Замерьте на своём оборудовании.