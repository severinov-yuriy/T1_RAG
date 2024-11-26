rag/
├── app/
│   ├── main.py              # Запуск API
│   ├── config.py            # Конфигурация
│   └── routes/
│       ├── upload.py        # Эндпоинт загрузки архива
│       └── query.py         # Эндпоинт обработки запроса
├── core/
│   ├── document_processing.py  # Обработка документов
│   ├── vectorization.py        # Векторизация текста
│   ├── database.py             # Работа с БД
│   ├── vector_store.py         # Работа с векторным хранилищем
│   └── rag.py                  # Логика RAG
├── models/                 # Модели и эмбеддинги
├── data/                   # Папка для временных файлов
├── requirements.txt        # Зависимости
└── README.md               # Документация