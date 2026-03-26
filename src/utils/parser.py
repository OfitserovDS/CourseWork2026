import requests
import re

doc_page_url = "https://www.hse.ru/docs/1026832917.html"
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

print(f"Загружаю страницу: {doc_page_url}")
response = requests.get(doc_page_url, headers=headers)
response.encoding = 'utf-8'
html = response.text

matches = re.findall(r'(/data/[^"\'\\s]+?\.docx)', html)

if matches:
    print(f"Найдено ссылок на DOCX: {len(matches)}")

    target_link = None
    for link in matches:
        if link.endswith('Положение.docx'):
            target_link = link
            print(f"Найдена нужная ссылка: {link}")
            break

    if target_link:
        file_url = 'https://www.hse.ru' + target_link
        print("Скачиваю файл...")
        file_response = requests.get(file_url, headers=headers)
        file_response.raise_for_status()

        with open('../../data/raw/popatkus.docx', 'wb') as f:
            f.write(file_response.content)

        print("Файл сохранён как 'popatkus.docx'")
    else:
        print("Среди найденных ссылок нет той, что заканчивается на 'Положение.docx'")
else:
    print("Ссылки на DOCX файлы не найдены.")