import os
import re
import sys
import wave
import subprocess
import importlib.util

# ---------- Windows CUDA DLL düzeltmesi (whisper/transkript ile ortak, burada gerekmiyor ama zararı yok) ----------
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

from pydub import AudioSegment
from piper import PiperVoice

SRT_DOSYASI = "altyazi.srt"
VIDEO_DOSYASI = "video.mp4.webm"
CIKTI_VIDEO = "final_video.mp4"
GECICI_KLASOR = "gecici_ses"
ORIJINAL_SES_SEVIYESI_DB = -30  # orijinal sesi bu kadar kıs (daha negatif = daha kısık)
SES_MODELI = "tr_TR-dfki-medium.onnx"

# İngilizce/yabancı kelimelerin Piper tarafından Türkçe harf harf okunmasını
# engellemek için, bu kelimeleri seslendirmeden önce Türkçe fonetik yazımına
# çeviriyoruz. Solda orijinal kelime, sağda nasıl okunmasını istediğin yazım.
# Büyük/küçük harf önemli değil, kelimenin geçtiği her yerde eşleşir.
TELAFFUZ_SOZLUGU = {
    "minecraft": "Maynkraft",
    "bedrock": "Bedrok",
    "youtube": "Yutup",
    "subscribe": "Sabskrayb",
    # kendi kelimelerini buraya ekleyebilirsin, örnek:
    # "windows": "Vindovs",
}


def telaffuz_icin_hazirla(metin):
    def sozlukten_degistir(eslesme):
        orijinal_kelime = eslesme.group(0)
        yeni_kelime = TELAFFUZ_SOZLUGU[orijinal_kelime.lower()]
        if orijinal_kelime[0].isupper():
            yeni_kelime = yeni_kelime[0].upper() + yeni_kelime[1:]
        return yeni_kelime

    desen = r"\b(" + "|".join(re.escape(k) for k in TELAFFUZ_SOZLUGU.keys()) + r")\b"
    return re.sub(desen, sozlukten_degistir, metin, flags=re.IGNORECASE)


# ---------- Yabancı/özel isim kelimelerin otomatik tespiti ----------
# Cümle ortasında büyük harfle başlayan ve Türkçe'ye özgü harf (ç,ğ,ı,ö,ş,ü)
# içermeyen kelimeleri "muhtemelen özel isim/yabancı kelime" sayıyoruz.
TURKCEYE_OZGU_HARFLER = set("çğıöşüÇĞİÖŞÜ")

try:
    from g2p_en import G2p
    g2p_motoru = G2p()
except Exception:
    g2p_motoru = None

ARPABET_TO_TURKCE = {
    "AA": "a", "AE": "e", "AH": "a", "AO": "o", "AW": "au", "AY": "ay",
    "B": "b", "CH": "ç", "D": "d", "DH": "d", "EH": "e", "ER": "ır",
    "EY": "ey", "F": "f", "G": "g", "HH": "h", "IH": "i", "IY": "i",
    "JH": "c", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ng",
    "OW": "o", "OY": "oy", "P": "p", "R": "r", "S": "s", "SH": "ş",
    "T": "t", "TH": "t", "UH": "u", "UW": "u", "V": "v", "W": "v",
    "Y": "y", "Z": "z", "ZH": "j",
}


def kelimeyi_fonetik_yaz(kelime):
    """g2p_en ile kelimenin İngilizce okunuşunu tahmin edip Türkçe harflere çevirir."""
    if g2p_motoru is None:
        return kelime
    try:
        fonemler = g2p_motoru(kelime)
        turkce_yazim = ""
        for fonem in fonemler:
            fonem_temiz = re.sub(r"[0-9]", "", fonem)  # vurgu rakamlarını at (AY1 -> AY)
            turkce_yazim += ARPABET_TO_TURKCE.get(fonem_temiz, "")
        if not turkce_yazim:
            return kelime
        if kelime[0].isupper():
            turkce_yazim = turkce_yazim[0].upper() + turkce_yazim[1:]
        return turkce_yazim
    except Exception:
        return kelime


def ozel_isim_mi(kelime, cumledeki_konum):
    if cumledeki_konum == 0:
        return False  # cümlenin ilk kelimesi Türkçe olarak da büyük harfle başlar
    if not kelime.isalpha():
        return False
    if not kelime[0].isupper():
        return False
    if any(harf in TURKCEYE_OZGU_HARFLER for harf in kelime):
        return False  # Türkçe'ye özgü harf içeriyorsa muhtemelen zaten Türkçe bir isim
    return True


def yabanci_kelimeleri_otomatik_cevir(metin):
    kelimeler = metin.split(" ")
    yeni_kelimeler = []
    for i, kelime in enumerate(kelimeler):
        # kelimenin başındaki/sonundaki noktalama işaretlerini ayır
        on_ek = re.match(r"^[^\wçğıöşüÇĞİÖŞÜ]*", kelime).group(0)
        son_ek = re.search(r"[^\wçğıöşüÇĞİÖŞÜ]*$", kelime).group(0)
        govde = kelime[len(on_ek): len(kelime) - len(son_ek)] if son_ek else kelime[len(on_ek):]

        if govde.lower() in TELAFFUZ_SOZLUGU:
            yeni_kelimeler.append(kelime)  # sözlükteki halini sonra zaten değiştireceğiz
        elif ozel_isim_mi(govde, i):
            yeni_kelimeler.append(on_ek + kelimeyi_fonetik_yaz(govde) + son_ek)
        else:
            yeni_kelimeler.append(kelime)
    return " ".join(yeni_kelimeler)

os.makedirs(GECICI_KLASOR, exist_ok=True)


def srt_zaman_to_ms(zaman_str):
    saat, dakika, geri_kalan = zaman_str.split(":")
    saniye, milisaniye = geri_kalan.split(",")
    return int(saat) * 3600000 + int(dakika) * 60000 + int(saniye) * 1000 + int(milisaniye)


def srt_oku(dosya_yolu):
    with open(dosya_yolu, "r", encoding="utf-8") as f:
        icerik = f.read()
    bloklar = re.split(r"\n\s*\n", icerik.strip())
    satirlar = []
    for blok in bloklar:
        satirlar_blok = blok.strip().split("\n")
        if len(satirlar_blok) < 3:
            continue
        baslangic_str, bitis_str = [z.strip() for z in satirlar_blok[1].split("-->")]
        metin = " ".join(satirlar_blok[2:]).strip()
        satirlar.append((srt_zaman_to_ms(baslangic_str), srt_zaman_to_ms(bitis_str), metin))
    return satirlar


def sesi_sureye_sigdir(ses_dosyasi, hedef_sure_ms):
    mevcut_ses = AudioSegment.from_file(ses_dosyasi)
    mevcut_sure_ms = len(mevcut_ses)
    if mevcut_sure_ms == 0 or hedef_sure_ms <= 0:
        return mevcut_ses

    oran = mevcut_sure_ms / hedef_sure_ms
    oran = max(0.5, min(oran, 2.0))

    duzeltilmis_dosya = ses_dosyasi.replace(".wav", "_duzeltilmis.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", ses_dosyasi, "-filter:a", f"atempo={oran}", duzeltilmis_dosya],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return AudioSegment.from_file(duzeltilmis_dosya)


print("Altyazı satırları okunuyor...")
satirlar = srt_oku(SRT_DOSYASI)
print(f"{len(satirlar)} satır bulundu.")

print("Piper ses modeli yükleniyor...")
ses_modeli = PiperVoice.load(SES_MODELI)

try:
    from piper.config import SynthesisConfig
    # noise_scale: ses tonundaki doğal varyasyon (yüksek = daha canlı/az monoton)
    # noise_w_scale: hece sürelerindeki varyasyon (yüksek = daha doğal ritim)
    # length_scale: 1.0 normal hız, düşük = daha hızlı konuşma
    sentez_ayari = SynthesisConfig(noise_scale=1.0, noise_w_scale=1.0, length_scale=1.05)
except ImportError:
    sentez_ayari = None

video_suresi_ms = int(AudioSegment.from_file(VIDEO_DOSYASI).duration_seconds * 1000)
dublaj_parcasi = AudioSegment.silent(duration=video_suresi_ms)

for i, (baslangic_ms, bitis_ms, metin) in enumerate(satirlar):
    print(f"[{i + 1}/{len(satirlar)}] seslendiriliyor: {metin[:50]}...")
    gecici_wav = os.path.join(GECICI_KLASOR, f"satir_{i:04d}.wav")
    seslendirilecek_metin = yabanci_kelimeleri_otomatik_cevir(metin)
    seslendirilecek_metin = telaffuz_icin_hazirla(seslendirilecek_metin)

    with wave.open(gecici_wav, "wb") as wav_dosyasi:
        if sentez_ayari is not None:
            ses_modeli.synthesize_wav(seslendirilecek_metin, wav_dosyasi, syn_config=sentez_ayari)
        else:
            ses_modeli.synthesize_wav(seslendirilecek_metin, wav_dosyasi)

    hedef_sure_ms = bitis_ms - baslangic_ms
    duzeltilmis_ses = sesi_sureye_sigdir(gecici_wav, hedef_sure_ms)

    dublaj_parcasi = dublaj_parcasi.overlay(duzeltilmis_ses, position=baslangic_ms)

DUBLAJ_SES_DOSYASI = "dublaj_sesi.wav"
dublaj_parcasi.export(DUBLAJ_SES_DOSYASI, format="wav")
print(f"{DUBLAJ_SES_DOSYASI} oluşturuldu.")

print("Orijinal ses çıkarılıyor ve karıştırılıyor...")
orijinal_ses = AudioSegment.from_file(VIDEO_DOSYASI)
orijinal_ses_kisik = orijinal_ses + ORIJINAL_SES_SEVIYESI_DB

karisik_ses = orijinal_ses_kisik.overlay(dublaj_parcasi)
KARISIK_SES_DOSYASI = "karisik_ses.wav"
karisik_ses.export(KARISIK_SES_DOSYASI, format="wav")

print("Yeni ses videoya gömülüyor...")
subprocess.run(
    [
        "ffmpeg", "-y",
        "-i", VIDEO_DOSYASI,
        "-i", KARISIK_SES_DOSYASI,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac",
        "-shortest",
        CIKTI_VIDEO,
    ],
    check=True,
)

print(f"Bitti! Sonuç dosyası: {CIKTI_VIDEO}")