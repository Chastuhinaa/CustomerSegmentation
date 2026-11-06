# Сегментація клієнтів

Лабораторна робота: кластеризація K-means власників кредитних карток (Credit Card Dataset for Clustering, Kaggle).

Виконала: Частухіна Юлія

## Запуск

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe src\segmentation.py
```

Програма виконує весь аналіз і зберігає рисунки, таблиці та `metrics.json` у папку `results`. Без аргументів k вибирається за найкращим silhouette (k = 4).

Короткі демонстрації окремих кроків, які нічого не перезаписують:

```bash
.venv\Scripts\python.exe src\demo.py data
```

Доступні команди: `data`, `missing`, `outliers`, `features`, `model`, `silhouette`, `zero`, `stability`, `k <число>`, `client <CUST_ID>`.

## Структура

- `data/CC_GENERAL.csv`: вхідні дані
- `src/segmentation.py`: увесь аналіз
- `src/demo.py`: демонстрації окремих кроків
- `results/`: рисунки, таблиці та метрики
- `report/Частухіна_Сегментація_клієнтів.docx`: звіт
