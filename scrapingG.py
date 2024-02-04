#Veri İşlemleri
import time
import pandas as pd
import numpy as np

#Veri Kazıma
from bs4 import BeautifulSoup as bs
import requests 
import datetime

#Konsol Güzelleştirme
from progressbar import *
from IPython.display import clear_output


#Kategoriler
sections = ["https://www.trendyol.com/butik/liste/5/elektronik",      #elektronik   
            "https://www.trendyol.com/butik/liste/11/kozmetik",       #kozmetik
            "https://www.trendyol.com/butik/liste/12/ev-yasam",       #ev-yasam
            "https://www.trendyol.com/butik/liste/9/ayakkabi-canta",  #ayakkabı-çanta
            "https://www.trendyol.com/butik/liste/22/spor-outdoor",   #spor
            "https://www.trendyol.com/butik/liste/16/supermarket"]    #supermarket



reviews = []
#Öncelikle bir Kategori seçiyoruz.
for section in sections:
    #Kategorinin içerisinde sırayla 20 sayfa gezineceğiz.
    for i in range(1,20):
        try:
            #Öncelikle URL'imizi oluşturuyoruz.
            newurl = section+str(i)
            print(newurl)
            
            #Url'nin içerisindeki bütün html dosyasını indiriyoruz.
            html = requests.get(newurl).text
            soup = bs(html, "lxml")
            
            # Her bir ürünü içeren etiketleri bul
            products = soup.find_all("div", class_="p-card-chldrn-cntnr card-border")

            # Her bir ürünü yazdır
            for product in products:
                product_name = product.find("a", href=True).text
                print(f"Kategori: {section}, Ürün Adı: {product_name}")

            # Gerekirse, her bir alt sayfa arasında bir süre bekle (rate limiting önlemek için)
            time.sleep(1)

        except Exception as e:
            print(f"Hata oluştu: {e}")
            break



