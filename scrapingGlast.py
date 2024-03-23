import requests
from bs4 import BeautifulSoup as bs

headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 OPR/106.0.0.0"}

# Ürün sayfasının URL'si
product_url = "https://www.trendyol.com/lenovo/ideapad-slim3-intel-core-i5-12450h-8gb-512gb-ssd-dos-15-6-fhd-arctic-grey-laptop-83er000wtr-p-772120068?boutiqueId=638145&merchantId=968"  

# Ürün sayfasının HTML içeriğini çek
def get_data(product_url):
    product_html = requests.get(product_url, headers=headers).text
    product_soup = bs(product_html, "lxml")
    return product_soup

product_soup=get_data(product_url)

# Ürün adını(title) çek
title_element = product_soup.find("h1" ,attrs={"class":"pr-new-br"})
title = title_element.text.strip() if title_element else "Ürün adı bulunamadı"
print("Title:", title)

# Fiyatı çek
price_element = product_soup.find("span", class_="prc-dsc")
price = price_element.text.strip() if price_element else "Fiyat bulunamadı"
print("Price:", price)

"""# Yorumları çek
reviews_elements = product_soup.find("div", class_="reviews-wrapper")
if reviews_elements:
    reviews_paragraphs = reviews_elements.findAll("p")
    reviews = '\n'.join([p.text.strip() for p in reviews_paragraphs])
    print("Reviews:", reviews)
else:
    print("Yorum bulunamadı")"""

# Yorumları çek

urunler = product_soup.find_all("div",attrs={"class":"p-card-wrppr with-campaign-view"})
for i in urunler:
    urunlinkleri= i.find_all("div",attrs={"class":"p-card-chldrn-cntnr card-border"})
    for j in urunlinkleri:
        link = "https://www.trendyol.com" + j.a.get("href")
        print(link + "\n")


detail = requests.get(link, headers=headers)  # Detay sayfasına erişim
print(f"Detay Erişim Response Code :{detail.status_code}")
detay_soup = bs(detail.content, "lxml")

reviews_content = detay_soup.find("div", class_="reviews-content")
if reviews_content:
    reviews = []
    comment_texts = reviews_content.findAll("div", class_="comment-text")
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

