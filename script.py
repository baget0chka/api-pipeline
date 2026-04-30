import pandas as pd
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

chunk_size = 10 
input_file = './input.csv'
output_file = './output.csv'
load_dotenv()

if not os.path.exists(input_file):
    print(f"Ошибка: файл {input_file} не найден!")
    exit(1)

if not os.getenv('OPENAI_API_KEY'):
    print("Ошибка: OPENAI_API_KEY не найден в .env файле!")
    exit(1)

try:
    reader = pd.read_csv(input_file, chunksize=chunk_size)
except Exception as e:
    print(f"Ошибка при чтении файла {input_file}: {e}")
    exit(1)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key = os.getenv('OPENAI_API_KEY')
)

sys_promt = f'''Ты профессиональный аналитик отзывов.

Важное правило: Тема (theme) должна отвечать на вопрос "Что хотел сказать пользователь?", а не "О каком предмете он говорит?" и быть написана на русском языке.

НЕЛЬЗЯ использовать в теме название конкретного товара (пылесос, телефон, кофе, отель и т.д.).

Пример тем (интент пользователя):
- "качество товара" (если хвалит/ругает товар)
- "эффективность работы" (если описывает функционал)
- "удобство использования" (если говорит как легко/сложно пользоваться)
- "сервис и доставка" (если про доставку/сервис)
- "не отзыв" (если не является отзывом)

ВАЖНО: используй не только перечисленные темы.

Тональность (sentiment) представляет собой одно из трех значений positive/negative/neutral.

Верни JSON массив: [{{'id': '', 'sentiment': '', 'theme': ''}}]
Только plain JSON, без пояснений и какого либо форматирования.'''

result = []
chunk_number = 0

for reviewes in reader: 
    chunk_number += 1
    attempt = 0
    if reviewes.empty:
        continue

    while(attempt < 3):
        try:
            attempt += 1
            response = client.chat.completions.create(
            model="openai/gpt-oss-20b:free",
            messages=[
                {"role": "system", "content": sys_promt},
                {"role": "user", "content": reviewes.to_string(index=False, header=True)}
            ]
            )

            if not response.choices[0].message.content:
                print(f"Пустой ответ от API для чанка {chunk_number}")
                continue
                
            content = response.choices[0].message.content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            result += json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Попытка {attempt + 1} не удалась: {e}")

if result:
    df = pd.DataFrame(result)
    df.to_csv(output_file, index=False, encoding='utf-8')
else:
    print("Нет данных для сохранения. Проверьте входной файл и API.")