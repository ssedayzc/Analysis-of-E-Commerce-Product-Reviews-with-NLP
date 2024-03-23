import requests
from bs4 import BeautifulSoup
import pandas as pd

header = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 OPR/106.0.0.0"}

r = requests.get("https://www.trendyol.com/sr?wc=106084,103108,103665", headers=header)
print(f"Site Erişim Response Code :{r.status_code}")
soup = BeautifulSoup(r.content,"lxml")

urunler = soup.find_all("div",attrs={"class":"p-card-wrppr with-campaign-view"})
for i in urunler:
    urunlinkleri= i.find_all("div",attrs={"class":"p-card-chldrn-cntnr card-border"})
    for j in urunlinkleri:
        link = "https://www.trendyol.com" + j.a.get("href")
        print(link + "\n")

        detay = requests.get(link, headers=header)  # Detay sayfasına erişim
        print(f"Detay Erişim Response Code :{detay.status_code}")

        detay_soup = BeautifulSoup(detay.content, "lxml")  # Detay sayfasını parse et
        detail = requests.get(link, headers=header)  # Detay sayfasına erişim
        

        reviews_content = detay_soup.find("div", class_="pr-rnr-com")
        if reviews_content:
            reviews = []
            comment_texts = reviews_content.findAll("div", class_="rnr-com-tx")
            for comment_text in comment_texts:
                p_tags = comment_text.findAll("p")
                for p_tag in p_tags:
                    reviews.append(p_tag.text.strip())
            if reviews:
                all_reviews = '\n'.join(reviews)
                print("Reviews:", all_reviews)
            else:
                print("Yorum bulunamadı")
        else:
            print("Yorum bölümü bulunamadı")
