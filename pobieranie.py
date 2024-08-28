import os
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def fetch_images(query, path, number_of_images=100):
    # Tworzenie katalogu, jeśli nie istnieje
    create_directory(path)

    # Przygotowanie URL dla Google Images
    query_plus = '+'.join(query.split())
    url = f"https://www.google.com/search?hl=pl&q={query_plus}&tbm=isch"

    # Nagłówki do imitacji przeglądarki (może być potrzebne, aby uniknąć blokady)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    images = soup.find_all('img')

    count = 0
    for img in images:
        # Sprawdzanie, czy pobrano wymaganą liczbę obrazów
        if count >= number_of_images:
            break

        # Pobieranie obrazu
        img_url = img['src']
        try:
            img_response = requests.get(img_url)
            image = Image.open(BytesIO(img_response.content))
            image.save(os.path.join(path, f"{count}.jpg"))
            print(f"Pobrano {count+1}/{number_of_images} obrazów do {path}")
            count += 1
        except Exception as e:
            print(f"Nie można pobrać obrazu {img_url}: {e}")

def main():
    # Pobieranie obrazów kombinerek
    fetch_images("sword", "obrazy_do_zmiany/swords", 22)

    # Pobieranie obrazów niekombinerek
    fetch_images("random short spear or pike", "obrazy_do_zmiany/notSwords", 22)

if __name__ == "__main__":
    main()
