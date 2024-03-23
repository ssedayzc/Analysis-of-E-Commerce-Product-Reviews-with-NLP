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
            products = soup.find_all("div", class_="p-card-chldrn-cntnr ")


            np.source.find("a",attrs={"class":"item"}).text



            # Her bir ürünü yazdır
            for product in products:
                product_name = product.find("a", href=True).text
                print(f"Kategori: {section}, Ürün Adı: {product_name}")

            # Gerekirse, her bir alt sayfa arasında bir süre bekle (rate limiting önlemek için)
            time.sleep(1)


        except Exception as e:
            print(f"Hata oluştu: {e}")
            break



#deneme2

# Yorumları içeren etiketleri bul 
#reviews = product_soup.find_all("div", class_="reviews")

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