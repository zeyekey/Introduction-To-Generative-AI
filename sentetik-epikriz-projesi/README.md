# Sentetik Epikriz Projesi

Qwen tabanli fine-tuned model ile klinik karar destegi ve epikriz raporu uretimi.

## Klasor Yapisi

```
sentetik-epikriz-projesi/
├── src/                    # Python uygulamasi
│   └── tibbi_ajan.py
├── notebooks/              # Egitim ve rapor notebook'lari
├── data/
│   ├── veritabani/         # Hastalik JSON veritabani
│   └── dataset_gemini/     # Sentetik egitim verisi (100 ornek)
└── models/
    ├── qwen-epikriz-hazir/ # Final LoRA adapter
    └── checkpoints/        # Ara checkpoint'ler (yerel, git disi)
```

## Calistirma

```bash
cd sentetik-epikriz-projesi
pip install -r requirements.txt
python src/tibbi_ajan.py
```

Ilk calistirmada `Qwen/Qwen2.5-1.5B-Instruct` taban modeli Hugging Face uzerinden indirilir.

## Ortam Degiskenleri (istege bagli)

| Degisken | Aciklama |
|----------|----------|
| `EPIKRIZ_PROJE_YOLU` | Proje kok dizini (varsayilan: bu klasor) |
| `HF_HOME` | Hugging Face onbellek dizini |
