import requests
from bs4 import BeautifulSoup



r = requests.get("https://www.trendyol.com/butik/liste/5/elektronik")
source = BeautifulSoup(r.content,"lxml")



#elektronik kategorisinin slider menüsü
slider = source.find_all("a",attrs={"class":"item"})

# İlk 6 elemanı bir listeye at
slider_list = [link.text for link in slider[3:9]]

# Listeyi yazdır
print("Kategoriler: ",slider_list)


laptop = requests.get("https://www.trendyol.com/sr?wc=106084,103108,103665")
source = BeautifulSoup(r.content,"lxml")

products = source.findAll("div", class_="prdct-cntnr-wrppr")

# Eğer en az bir eleman bulunuyorsa devam et
if products:
    # İlk elemanı al
    product = products[0]
    
   

else:
    print("Ürün bulunamadı.")



