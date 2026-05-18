import os
import sys
import json
import re
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ==========================================
# 🛠️ SİSTEM VE DONANIM OPTİMİZASYONLARI
# ==========================================
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

PROJE_KOKU = Path(__file__).resolve().parent.parent


class KlinikAjan:
    def __init__(self):
        self.ANA_KLASOR = Path(os.environ.get("EPIKRIZ_PROJE_YOLU", PROJE_KOKU))
        self.MODEL_YOLU = self.ANA_KLASOR / "models" / "qwen-epikriz-hazir"
        self.DB_KLASORU = self.ANA_KLASOR / "data" / "veritabani"
        self.hf_cache = os.environ.get(
            "HF_HOME",
            str(Path.home() / ".cache" / "huggingface"),
        )
        
        self.model = None
        self.tokenizer = None
        self.hastalik_havuzu = []
        
        # Sistemi Başlat
        self._klasorleri_hazirla()
        self._veritabanini_kur_ve_yukle()
        self._modeli_yukle()

    def _klasorleri_hazirla(self):
        self.DB_KLASORU.mkdir(parents=True, exist_ok=True)

    def _veritabanini_kur_ve_yukle(self):
        nadir_dosya = self.DB_KLASORU / "nadir_hastaliklar.json"
        normal_dosya = self.DB_KLASORU / "normal_hastaliklar.json"

        if not nadir_dosya.exists():
            nadir_veriler = [
                {"hastalik_adi": "Behçet Hastalığı", "kategori": "NADİR HASTALIK", "belirtiler": ["aft", "üveit", "genital", "ülser", "hla-b51", "eritema nodozum"], "tedavi": "Sistemik kortikosteroidler (Prednol), Kolşisin, İmmünsüpresif ajanlar. Oftalmoloji acili."},
                {"hastalik_adi": "FMF (Ailevi Akdeniz Ateşi)", "kategori": "NADİR HASTALIK", "belirtiler": ["karın ağrısı", "tekrarlayan ateş", "göğüs ağrısı", "artrit", "mevf"], "tedavi": "Ömür boyu Kolşisin profilaksisi. Ataklarda NSAİİ. Nefroloji takibi."},
                {"hastalik_adi": "Kistik Fibrozis", "kategori": "NADİR HASTALIK", "belirtiler": ["tekrarlayan akciğer enfeksiyonu", "tuzlu ter", "büyüme geriliği", "kronik öksürük"], "tedavi": "Mukolitikler, solunum fizyoterapisi, pankreatik enzim replasmanı, inhale Tobramisin."}
            ]
            with open(nadir_dosya, "w", encoding="utf-8") as f:
                json.dump(nadir_veriler, f, ensure_ascii=False, indent=4)

        if not normal_dosya.exists():
            normal_veriler = [
                {"hastalik_adi": "Akut Gastroenterit", "kategori": "SIK GÖRÜLEN HASTALIK", "belirtiler": ["ishal", "kusma", "karın ağrısı", "bulantı", "ateş"], "tedavi": "IV hidrasyon, antiemetik, probiyotik desteği."},
                {"hastalik_adi": "Üst Solunum Yolu Enfeksiyonu (ÜSYE)", "kategori": "SIK GÖRÜLEN HASTALIK", "belirtiler": ["boğaz ağrısı", "burun akıntısı", "öksürük", "halsizlik"], "tedavi": "Semptomatik tedavi: Parasetamol, nazal dekonjestanlar, hidrasyon. Antibiyotik endikasyonu yoktur."},
                {"hastalik_adi": "Esansiyel Hipertansiyon Atağı", "kategori": "SIK GÖRÜLEN HASTALIK", "belirtiler": ["baş ağrısı", "çarpıntı", "ense ağrısı", "yüksek tansiyon"], "tedavi": "Acil serviste dilaltı kaptopril veya IV antihipertansif. Kardiyoloji takibi."}
            ]
            with open(normal_dosya, "w", encoding="utf-8") as f:
                json.dump(normal_veriler, f, ensure_ascii=False, indent=4)

        for dosya in self.DB_KLASORU.glob("*.json"):
            with open(dosya, "r", encoding="utf-8") as f:
                self.hastalik_havuzu.extend(json.load(f))
        
        print(f"📂 Veri tabanı hazır! Toplam {len(self.hastalik_havuzu)} hastalık profili yüklendi.")

    def _modeli_yukle(self):
        print("🧠 LLM Motoru Yükleniyor... (Bu işlem biraz sürebilir)")
        torch.cuda.empty_cache()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.MODEL_YOLU))
            base_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-1.5B-Instruct",
                device_map="auto",
                torch_dtype=torch.float16,
                cache_dir=self.hf_cache,
            )
            self.model = PeftModel.from_pretrained(base_model, str(self.MODEL_YOLU))
            self.model.eval()
            print("✅ LLM Motoru Aktif!")
        except Exception as e:
            print(f"❌ Kritik Hata - Model yüklenemedi: {e}")
            sys.exit()

    def _llm_ile_metin_uret(self, sistem_komutu, kullanici_komutu, max_token=100):
        mesajlar = [
            {"role": "system", "content": sistem_komutu},
            {"role": "user", "content": kullanici_komutu}
        ]
        prompt_metni = self.tokenizer.apply_chat_template(mesajlar, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt_metni, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=max_token, 
            do_sample=False, 
            repetition_penalty=1.05,
            pad_token_id=self.tokenizer.eos_token_id
        )
        uretilen_kisim = outputs[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(uretilen_kisim, skip_special_tokens=True).strip()

    def tani_koy(self, anamnez_metni):
        arama_metni = anamnez_metni.lower()
        en_iyi_eslesme = None
        en_yuksek_skor = 0

        for hastalik in self.hastalik_havuzu:
            skor = sum(1 for belirti in hastalik.get("belirtiler", []) if belirti.lower() in arama_metni)
            if skor > en_yuksek_skor:
                en_yuksek_skor = skor
                en_iyi_eslesme = hastalik

        if en_iyi_eslesme and en_yuksek_skor > 0:
            return en_iyi_eslesme["hastalik_adi"], en_iyi_eslesme["tedavi"], en_iyi_eslesme["kategori"]

        print("\n🔍 Uyarı: Veritabanında eşleşme bulunamadı. Yapay Zeka (LLM) otonom teşhis koyuyor...")
        
        sistem_prompt = "Sen uzman bir teşhis asistanısın. SADECE şu formatta cevap ver:\nTANI: [Hastalık Adı]\nTEDAVİ: [Kısa Tedavi Planı]"
        kullanici_prompt = f"Şikayetler: {anamnez_metni}\nLütfen olası tanıyı ve tedaviyi belirt."
        
        llm_cevabi = self._llm_ile_metin_uret(sistem_prompt, kullanici_prompt, max_token=100)
        
        tani_match = re.search(r'TANI:\s*(.*)', llm_cevabi)
        tedavi_match = re.search(r'TEDAVİ:\s*(.*)', llm_cevabi)
        
        llm_tani = tani_match.group(1).strip() if tani_match else "Otonom Tanı Konulamadı"
        llm_tedavi = tedavi_match.group(1).strip() if tedavi_match else "İleri tetkik gereklidir."
        
        return llm_tani, llm_tedavi, "YAPAY ZEKA OTONOM TEŞHİS"

    def epikriz_raporu_olustur(self, yas_cinsiyet, anamnez_metni, tani, tedavi, vaka_tipi):
        print("\n⚙️ Epikriz Raporu Derleniyor (Few-Shot Şablonlama devrede)...")

        sistem_prompt = "Sen bir tıbbi sekretersin. Verilen günlük dildeki girdiyi, resmi bir tıbbi özete çevir. Asla yabancı dil veya alakasız kelime (eposta, sipariş vb.) kullanma."
        kullanici_prompt = f"""Aşağıdaki örnekteki mantığı taklit ederek metni düzenle:

ÖRNEK GİRDİ: karnım çok ağrıyor ateşliyim. Ek Bulgular: mevf geni var
ÖRNEK ÇIKTI: Hastanın kliniğinde; karın ağrısı ve ateş bulguları saptanmıştır. Laboratuvar tetkiklerinde MEVF geni pozitifliği mevcuttur.

ÖRNEK GİRDİ: {anamnez_metni}
ÖRNEK ÇIKTI:"""
        
        tibbi_anamnez_tr = self._llm_ile_metin_uret(sistem_prompt, kullanici_prompt, max_token=60)
        tibbi_anamnez_tr = tibbi_anamnez_tr.split("ÖRNEK")[0].replace("\n", " ").strip()

        rapor = f"""
============================================================
                 T.C. SAĞLIK BAKANLIĞI
              HASTA ÇIKIŞ ÖZETİ (EPİKRİZ)
============================================================
1. HASTA DEMOGRAFİSİ       : {yas_cinsiyet}
2. KLİNİK KOD VE TANI      : {vaka_tipi} / {tani}
3. KLİNİK ANAMNEZ/BULGULAR : {tibbi_anamnez_tr}
4. TERAPÖTİK YAKLAŞIM      : {tedavi}
============================================================
"""
        return rapor

# ==========================================
# 🚀 MOTORU ÇALIŞTIRMA BLOĞU (BUNU SAKIN SİLME!)
# ==========================================
if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    print("="*60)
    print("   🏥 YZ DESTEKLİ KLİNİK KARAR VE EPİKRİZ SİSTEMİ")
    print("="*60)

    ajan = KlinikAjan()

    print("\nLütfen Hasta Bilgilerini Giriniz:")
    yas_cinsiyet = input("👉 Yaş ve Cinsiyet (Örn: 28 Yaş, Erkek): ")
    sikayetler = input("👉 Anamnez / Şikayetler: ")
    lab_sonuclari = input("👉 Laboratuvar / Ek Bulgular (Yoksa Enter'a bas): ")
    
    tam_anamnez = f"{sikayetler}. Ek Bulgular: {lab_sonuclari}"

    tani, tedavi, vaka_tipi = ajan.tani_koy(tam_anamnez)

    final_rapor = ajan.epikriz_raporu_olustur(yas_cinsiyet, tam_anamnez, tani, tedavi, vaka_tipi)
    
    print("\n" + "✅ İŞLEM TAMAMLANDI".center(60))
    print(final_rapor)