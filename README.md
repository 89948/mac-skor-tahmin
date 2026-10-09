# Futbol Analiz & Tahmin Motoru v2

Replit zorunlu değildir. Bu Streamlit projesi yerel bilgisayarda veya uygun Python barındırma ortamında tarayıcıdan kullanılabilir.

## Kurulum (Windows)
1. ZIP dosyasını çıkarın.
2. Proje klasöründe terminal/PowerShell açın.
3. Komutları çalıştırın:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   streamlit run app.py
   ```

## macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Terminalde görünen yerel Streamlit adresini tarayıcıda açın.

## API anahtarları
Uygulama sol kenar çubuğunda oturum sırasında girilir; dosyaya yazılmaz.
- NVIDIA: https://build.nvidia.com/
- API-Football: https://www.api-football.com/

## Otomatik analiz nasıl çalışır?
1. Lig ID, sezon ve tarih seçilir.
2. API-Football fikstürü çekilir.
3. Bir maç seçilir.
4. İki takımın sezon iç saha/deplasman gol ortalamaları alınır.
5. Basit Poisson modeli 1X2, 2,5 Alt/Üst, KG Var/Yok ve olası skorları hesaplar.
6. NVIDIA anahtarı girilmişse model yorumlaması istenebilir.

## Önemli sınırlamalar
- API planı fikstür ve istatistik uç noktalarına erişimi kısıtlayabilir.
- Gol beklentisi (lambda) basit ortalamalardan türetilir; xG, sakatlık, kadro, rakip gücü ve oran hareketlerini henüz içermez.
- Bu bir başlangıç sürümüdür; geçmiş veriler üzerinde backtest ve olasılık kalibrasyonu yapılmadan güvenilirliği kanıtlanmış sayılmaz.
- API anahtarlarını paylaşmayın veya kaynak koda yazmayın.
