import pandas as pd
from pyTrendyol import Kategori,Urun


"""
    Kategori : Trendyol'dan hedef kategori ürünlerini çevirir.

    Methodlar
    ----------
        .urunleri_ver(kategori_adi:str, sayfa_tara:int=1) -> list[dict] or None:
            ilgili kategori ürünlerini istenilen sayfa sayısı boyunca listeler 
        .mevcut_mu(kategori_adi:str) -> bool:
            ilgili kategori mevcut mu değil mi bilgisini verir
        .kategoriler -> dict[str, str]:
            mevcut kategorileri trendyol rss'inden ayrıştırıp çevirir
"""




kategori= Kategori()
urun = Urun()


#excel dosya yolunu ekle
#path=
kategoriler=["bilgisayar"]

for kategori_ad in kategoriler:
    urunler = kategori_ad.urunleri_ver(kategori_ad=kategori_ad, sayfa_tara=int(3))


urun_detay=[]


for urun_link in urunler:
    urun_detaylari= urun.detay_ver(urun_link.link)
    if urun_detaylari is not None:
        urun_detay_model=urun_detaylari.model_dump()
        urun_detay_model["Kategori"]= kategori_ad
        urun_detay.append(urun_detay_model)
   


urunler_df = pd.DataFrame(urun_detay)
#excele aktar
#urunler_df.to_excel()