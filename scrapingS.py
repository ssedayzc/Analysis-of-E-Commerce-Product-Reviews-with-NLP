import requests
from bs4 import BeautifulSoup

url="https://www.trendyol.com/sr?wc=106084,103108,103665"

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
}


html = requests.get(url, headers=headers).content
soup = BeautifulSoup(html, "html.parser")

liste = soup.find("div", class_="prdct-cntnr-wrppr").find_all("div", class_="p-card-chldrn-cntnr", limit=10)

for item in liste:
    title = item.find("div", class_="prdct-desc-cntnr-ttl").text
    name = item.find("div", class_="prdct-desc-cntnr-name").text
    price = item.find("div", class_="prc-box-dsnctd").text
    print(title,name, price)