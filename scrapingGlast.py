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
price_element = product_soup.find("div", class_="prc-dsc")
price = price_element.text.strip() if price_element else "Fiyat bulunamadı"
print("Price:", price)

# Yorumları çek
reviews_elements = product_soup.find("div", class_="rnr-com-tx")
if reviews_elements:
    reviews_paragraphs = reviews_elements.findAll('p')
    reviews = '\n'.join([p.text.strip() for p in reviews_paragraphs])
    print("Reviews:", reviews)
else:
    print("Yorum bulunamadı")

