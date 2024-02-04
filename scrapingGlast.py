import requests
from bs4 import BeautifulSoup as bs

# Ürün sayfasının URL'si
product_url = "https://www.trendyol.com/sr?wc=106084,103108,103665"  

# Ürün sayfasının HTML içeriğini çek
product_html = requests.get(product_url).text
product_soup = bs(product_html, "html.parser")

# Yorumları içeren etiketleri bul 
reviews = product_soup.find_all("div", class_="reviews")

# Eğer en az bir yorum bulunuyorsa devam et
if reviews:
    for review in reviews:
        # Yorum içeriğini al
        comment_texts = review.find_all("div", class_="comment-text")
        
        # Eğer yorum içeriği bulunuyorsa devam et
        if comment_texts:
            for comment_text in comment_texts:
                # Yorum içeriğindeki paragraf etiketlerini bul
                paragraphs = comment_text.find_all('p')
                
                # Eğer en az bir paragraf bulunuyorsa yazdır
                if paragraphs:
                    for paragraph in paragraphs:
                        print(f"Yorum: {paragraph.text}")
                else:
                    print("Yorum içeriği bulunamadı.")
        else:
            print("Yorum içeriği bulunamadı.")
else:
    print("Ürünün yorumu bulunamadı.")