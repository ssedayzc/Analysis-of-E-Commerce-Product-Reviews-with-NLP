from selenium import webdriver
from bs4 import BeautifulSoup
import time
from webdriver_manager.chrome import ChromeDriverManager
 
import pandas as pd
 
 
driver = webdriver.Chrome(ChromeDriverManager().install())
 
url = "https://www.trendyol.com/sr?wc=106084,103108,103665"
driver.get(url)
 
def slow_scroll_to_bottom(driver):
    scroll_pause_time = 0.05
    prev_height = 0
    range=10000
    while prev_height<range:
        driver.execute_script("window.scrollTo(0, {});".format(prev_height))
        time.sleep(scroll_pause_time)  
        new_height = driver.execute_script("return document.body.scrollHeight")
        prev_height=prev_height+100
        if new_height >= range:
            range = new_height
       
 
def scroll_to_bottom(driver):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)
 
def scroll_to_top(driver):
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(2)
 
num_iterations = 100 #Buraya scrollbarı kaç defa çalıştırmak istediğinizi yazıyorsunuz
#Sayfanın tamamının yüklenmesini istiyorsanız ki 55k ürün var. Ya çok yüksek bir iterasyon yazacaksınız
#Yada önceki ve sonraki scrollbar yüksekliğinin eşit olup olmadığını kontrol edeceksiniz. O zaman tüm sayfa yüklenmiş olur.
 
for _ in range(num_iterations):
    scroll_to_bottom(driver)
    scroll_to_top(driver)
    time.sleep(5)
 
links = []
html_content = driver.page_source
soup = BeautifulSoup(html_content, "html.parser")
div_tags = soup.find_all("div", class_="p-card-chldrn-cntnr card-border")
for div_tag in div_tags:
    a_tag = div_tag.find("a")
    if a_tag:
        href = a_tag.get("href")
        new_href = href.replace("?boutiqueId", "/yorumlar?boutiqueId")
        full_link = "https://www.trendyol.com" + new_href
        links.append(full_link)
 
driver.quit()
df = pd.DataFrame(links, columns=['Links'])
df.to_excel("trendyol_comments_links.xlsx", index=False)
#Yorum linklerini tekrar tekrar oluşturmamak için bu fazda tüm yorum linkleri bir dosyaya kaydediyoruz.
print("Excel dosyası oluşturuldu: trendyol_comments_links.xlsx")
 
######Yorum çekme bölümü burada başlıyor. Tüm yorumları çekebilmek için burada da scrollbar hareketi yapmak gerekiyor.
 
df = pd.read_excel("trendyol_comments_links.xlsx")
driver = webdriver.Chrome(ChromeDriverManager().install())
 
for link in df['Links']:
    driver.get(link)
    slow_scroll_to_bottom(driver)
    html_content = driver.page_source
    soup = BeautifulSoup(html_content, "html.parser")
    comments = []
    comment_divs = soup.find_all("div", class_="comment")
 
    for comment_div in comment_divs:
        comment_text = comment_div.find("div", class_="comment-text").find("p").text
        comments.append(comment_text)
 
    #Burada yorumları ekrana yazıyor onun yerine comments değişkenini istediğiniz şekilde excel'e basabilirsiniz.
    for idx, comment in enumerate(comments, start=1):
        print(f"Yorum {idx}: {comment}")
   
driver.quit()
 