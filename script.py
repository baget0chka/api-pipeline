import pandas as pd
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

chunk_size = 10 
input_file = './input.csv'
output_file = './output.csv'
load_dotenv()


reader = pd.read_csv(input_file, chunksize=chunk_size)

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

for reviewes in reader:      
    response = client.chat.completions.create(
    model="openai/gpt-oss-20b:free",
    messages=[
        {"role": "system", "content": sys_promt},
        {"role": "user", "content": reviewes.to_string(index=False, header=True)}
    ]
    )
    print(response.choices[0].message.content)
    result += json.loads(response.choices[0].message.content)

df = pd.DataFrame(result)
df.to_csv(output_file, index=False, encoding='utf-8')