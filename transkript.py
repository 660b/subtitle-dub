import os
import sys
import importlib.util

# pip ile kurulan CUDA kütüphanelerinin (cublas, cudnn) bulunduğu klasörleri
# Windows'un DLL arama yoluna ekliyoruz, yoksa "cublas64_12.dll bulunamadı" hatası alınır.
if sys.platform == "win32":
    for paket_adi in ["nvidia.cublas", "nvidia.cudnn"]:
        try:
            spec = importlib.util.find_spec(paket_adi)
            if spec and spec.submodule_search_locations:
                paket_klasoru = list(spec.submodule_search_locations)[0]
                bin_klasoru = os.path.join(paket_klasoru, "bin")
                if os.path.isdir(bin_klasoru):
                    os.add_dll_directory(bin_klasoru)
                    os.environ["PATH"] = bin_klasoru + os.pathsep + os.environ["PATH"]
        except (ImportError, ModuleNotFoundError):
            pass

from faster_whisper import WhisperModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

VIDEO_DOSYASI = "video.mp4.webm"
HEDEF_DIL = "tr"

# Whisper'ın verdiği dil kodunu (ISO 639-1) NLLB'nin beklediği kodlara çeviriyoruz.
NLLB_DIL_KODLARI = {
    "en": "eng_Latn",
    "tr": "tur_Latn",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "ru": "rus_Cyrl",
    "ar": "arb_Arab",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "zh": "zho_Hans",
    "it": "ita_Latn",
    "pt": "por_Latn",
}

# ---------- 1) Konuşmayı metne dök ----------

# Üç modlu seçim: light | balanced | quality
MODEL_MODLARI = {
    "light": {"whisper": "small"},
    "balanced": {"whisper": "medium"},
    "quality": {"whisper": "large-v3"},
}
MODEL_MODU = "light"
WHISPER_MODEL_BOYUTU = MODEL_MODLARI[MODEL_MODU]["whisper"]

print("Whisper modeli yükleniyor...")
model = WhisperModel(WHISPER_MODEL_BOYUTU, device="cuda", compute_type="float16")

print("Çözümleme başlıyor...")
segments, info = model.transcribe(VIDEO_DOSYASI)  # language belirtmiyoruz, otomatik algılasın

kaynak_dil = info.language
print(f"Tespit edilen dil: {kaynak_dil} (güven: {info.language_probability:.2f})")
print("-" * 40)

satirlar = []
for segment in segments:
    satirlar.append((segment.start, segment.end, segment.text.strip()))
    print(f"[{segment.start:.1f}s -> {segment.end:.1f}s] {segment.text.strip()}")

print("-" * 40)
print(f"Toplam {len(satirlar)} satır bulundu.")

# ---------- 2) Türkçeye çevir (NLLB ile, yerel) ----------

if kaynak_dil == HEDEF_DIL:
    print("Video zaten Türkçe, çeviri atlanıyor.")
    cevrilmis_satirlar = satirlar
else:
    if kaynak_dil not in NLLB_DIL_KODLARI:
        raise SystemExit(f"'{kaynak_dil}' dili için NLLB kod eşlemesi tanımlı değil, listeye ekleyip tekrar dene.")

    print("Çeviri modeli (NLLB) yükleniyor, ilk seferde indirilecek (~2.4 GB)...")
    MODEL_ADI = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ADI)
    ceviri_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ADI).to("cuda")

    kaynak_kod = NLLB_DIL_KODLARI[kaynak_dil]
    hedef_kod = NLLB_DIL_KODLARI[HEDEF_DIL]
    tokenizer.src_lang = kaynak_kod

    print("Satırlar çevriliyor...")
    cevrilmis_satirlar = []
    for baslangic, bitis, metin in satirlar:
        girdi = tokenizer(metin, return_tensors="pt").to("cuda")
        hedef_id = tokenizer.convert_tokens_to_ids(hedef_kod)
        cikti = ceviri_model.generate(**girdi, forced_bos_token_id=hedef_id, max_length=256)
        ceviri = tokenizer.batch_decode(cikti, skip_special_tokens=True)[0]
        cevrilmis_satirlar.append((baslangic, bitis, ceviri))
        print(f"[{baslangic:.1f}s -> {bitis:.1f}s] {ceviri}")

# ---------- 3) .srt dosyası olarak kaydet ----------

def saniye_to_srt_zaman(saniye):
    saat = int(saniye // 3600)
    dakika = int((saniye % 3600) // 60)
    sn = int(saniye % 60)
    milisaniye = int((saniye - int(saniye)) * 1000)
    return f"{saat:02d}:{dakika:02d}:{sn:02d},{milisaniye:03d}"

with open("altyazi.srt", "w", encoding="utf-8") as f:
    for i, (baslangic, bitis, metin) in enumerate(cevrilmis_satirlar, start=1):
        f.write(f"{i}\n")
        f.write(f"{saniye_to_srt_zaman(baslangic)} --> {saniye_to_srt_zaman(bitis)}\n")
        f.write(f"{metin}\n\n")

print("altyazi.srt dosyası oluşturuldu (Türkçe).") 