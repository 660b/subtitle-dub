import os
import re
import sys
import wave
import subprocess
import importlib.util
import threading
import traceback

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# ---------- Windows CUDA DLL düzeltmesi ----------
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

import webbrowser


def kaynak_yolu(goreli_yol):
    """Normalde ve PyInstaller ile paketlendiğinde dosya yollarının doğru bulunmasını sağlar."""
    taban_klasor = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(taban_klasor, goreli_yol)


app = Flask(
    __name__,
    template_folder=kaynak_yolu("templates"),
)


def calisma_cihazini_belirle():
    """GPU varsa cuda, yoksa cpu döner. Böylece uygulama sadece RTX kartı olanlarda değil, herkeste çalışır."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16"
    except ImportError:
        pass
    return "cpu", "int8"


# Hafif mod: daha düşük kaynak tüketimi ve daha yaygın uyumluluk.
# Büyük model istersen "medium" ya da "large-v3" olarak değiştirilebilir.
HAFIF_MOD = True
CIHAZ, HESAPLAMA_TIPI = calisma_cihazini_belirle()
WHISPER_MODEL_BOYUTU = "small" if HAFIF_MOD else "medium"
print(f"[Sesli] Kullanılan işlem birimi: {CIHAZ}")

def yazilabilir_klasor():
    """Paketlenmiş halde çalışırken exe'nin yanındaki klasörü, normalde proje klasörünü kullanır.
    static içeriği (indirilen video, üretilen ses) kalıcı ve yazılabilir olmalı,
    PyInstaller'ın geçici _MEIPASS klasörü her çalıştırmada silinir."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


STATIC_KLASOR = os.path.join(yazilabilir_klasor(), "static")
os.makedirs(STATIC_KLASOR, exist_ok=True)

VIDEO_YOLU = os.path.join(STATIC_KLASOR, "video.mp4")
VTT_YOLU = os.path.join(STATIC_KLASOR, "altyazi.vtt")
SRT_YOLU = os.path.join(STATIC_KLASOR, "altyazi.srt")
DUBLAJ_VIDEO_YOLU = os.path.join(STATIC_KLASOR, "final_video.mp4")
GECICI_SES_KLASOR = os.path.join(STATIC_KLASOR, "gecici_ses")
SES_MODELI = kaynak_yolu("tr_TR-dfki-medium.onnx")
ORIJINAL_SES_SEVIYESI_DB = -30

# Pakete gömülü ffmpeg.exe varsa onu kullan, yoksa sistemde kurulu ffmpeg'e güven.
_gomulu_ffmpeg = kaynak_yolu("ffmpeg.exe")
FFMPEG_YOLU = _gomulu_ffmpeg if os.path.exists(_gomulu_ffmpeg) else "ffmpeg"

MODEL_MODLARI = {
    "light": {"whisper": "small"},
    "balanced": {"whisper": "medium"},
    "quality": {"whisper": "large-v3"},
}


def onerilen_mod():
    try:
        import torch
        if torch.cuda.is_available():
            try:
                total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                if total_memory_gb >= 8:
                    return "balanced"
                return "light"
            except Exception:
                return "light"
    except ImportError:
        pass
    return "light"


MODEL_MODU = onerilen_mod()
CEVIRI_BATCH_BOYUTU = 8
CEVIRI_MODLARI = {"single": "tek tek", "batch": "batch"}
CEVIRI_MODU = "batch"


def dublaj_izinli_mi():
    return MODEL_MODU in ("balanced", "quality")


# Uygulamanın o anki durumunu tutan basit bir global nesne.
durum = {
    "asama": "bos",  # bos | indiriliyor | indirildi | cozumleniyor | hazir | dublaj_yapiliyor | dublaj_hazir | hata
    "mesaj": "",
    "yuzde": 0,
    "mod": MODEL_MODU,
    "ceviri_mod": CEVIRI_MODU,
}

# Son çözümlenen satırlar (başlangıç_sn, bitiş_sn, metin) - dublaj için tekrar kullanılıyor.
son_satirlar = []

NLLB_DIL_KODLARI = {
    "en": "eng_Latn", "tr": "tur_Latn", "de": "deu_Latn", "fr": "fra_Latn",
    "es": "spa_Latn", "ru": "rus_Cyrl", "ar": "arb_Arab", "ja": "jpn_Jpan",
    "ko": "kor_Hang", "zh": "zho_Hans", "it": "ita_Latn", "pt": "por_Latn",
}

TELAFFUZ_SOZLUGU = {
    "minecraft": "Maynkraft",
    "bedrock": "Bedrok",
    "youtube": "Yutup",
}


def saniye_to_vtt_zaman(saniye):
    saat = int(saniye // 3600)
    dakika = int((saniye % 3600) // 60)
    sn = int(saniye % 60)
    ms = int((saniye - int(saniye)) * 1000)
    return f"{saat:02d}:{dakika:02d}:{sn:02d}.{ms:03d}"


def saniye_to_srt_zaman(saniye):
    saat = int(saniye // 3600)
    dakika = int((saniye % 3600) // 60)
    sn = int(saniye % 60)
    ms = int((saniye - int(saniye)) * 1000)
    return f"{saat:02d}:{dakika:02d}:{sn:02d},{ms:03d}"


def eski_dosyalari_temizle():
    for eski_dosya in [VIDEO_YOLU, VTT_YOLU, SRT_YOLU, DUBLAJ_VIDEO_YOLU]:
        if os.path.exists(eski_dosya):
            try:
                os.remove(eski_dosya)
            except OSError:
                pass


def video_indir(url):
    import yt_dlp

    eski_dosyalari_temizle()
    durum.update(asama="indiriliyor", mesaj="Video indiriliyor...", yuzde=0)

    def ilerleme_kancasi(d):
        if d["status"] == "downloading":
            yuzde_metin = d.get("_percent_str", "0%").strip().replace("%", "")
            try:
                durum["yuzde"] = float(yuzde_metin)
            except ValueError:
                pass
        elif d["status"] == "finished":
            durum["yuzde"] = 100

    secenekler = {
        "outtmpl": VIDEO_YOLU,
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "progress_hooks": [ilerleme_kancasi],
        "quiet": True,
    }
    try:
        with yt_dlp.YoutubeDL(secenekler) as ydl:
            ydl.download([url])
        durum.update(asama="indirildi", mesaj="Video indirildi.", yuzde=100)
    except Exception as e:
        durum.update(asama="hata", mesaj=f"İndirme hatası: {e}")
        traceback.print_exc()


def cozumle_ve_cevir():
    from faster_whisper import WhisperModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    global son_satirlar
    try:
        whisper_model_adi = MODEL_MODLARI.get(MODEL_MODU, MODEL_MODLARI["light"])["whisper"]
        durum.update(asama="cozumleniyor", mesaj=f"Whisper ({whisper_model_adi}) modeli yükleniyor...", yuzde=0)
        model = WhisperModel(whisper_model_adi, device=CIHAZ, compute_type=HESAPLAMA_TIPI)

        durum.update(mesaj="Konuşma çözümleniyor...")
        segments, info = model.transcribe(VIDEO_YOLU)
        kaynak_dil = info.language

        satirlar = []
        for segment in segments:
            satirlar.append((segment.start, segment.end, segment.text.strip()))

        hedef_dil = "tr"
        if kaynak_dil == hedef_dil:
            cevrilmis = satirlar
        else:
            if kaynak_dil not in NLLB_DIL_KODLARI:
                raise ValueError(f"'{kaynak_dil}' dili için çeviri desteği yok.")

            durum.update(mesaj="Çeviri modeli yükleniyor...")
            MODEL_ADI = "facebook/nllb-200-distilled-600M"
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ADI)
            ceviri_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ADI).to(CIHAZ)
            tokenizer.src_lang = NLLB_DIL_KODLARI[kaynak_dil]
            hedef_id = tokenizer.convert_tokens_to_ids(NLLB_DIL_KODLARI[hedef_dil])

            tokenizer.src_lang = NLLB_DIL_KODLARI[kaynak_dil]
            hedef_id = tokenizer.convert_tokens_to_ids(NLLB_DIL_KODLARI[hedef_dil])

            if CEVIRI_MODU == "single":
                durum.update(mesaj="Satırlar tek tek çevriliyor...")
                cevrilmis = []
                toplam = max(1, len(satirlar))
                for i, (baslangic, bitis, metin) in enumerate(satirlar):
                    if not metin.strip():
                        cevrilmis.append((baslangic, bitis, ""))
                        continue
                    girdi = tokenizer(metin, return_tensors="pt").to(CIHAZ)
                    cikti = ceviri_model.generate(**girdi, forced_bos_token_id=hedef_id, max_length=256)
                    ceviri = tokenizer.batch_decode(cikti, skip_special_tokens=True)[0]
                    cevrilmis.append((baslangic, bitis, ceviri))
                    durum["yuzde"] = round((i + 1) / toplam * 100, 1)
            else:
                durum.update(mesaj="Satırlar batch olarak çevriliyor...")
                metinler = [metin for _, _, metin in satirlar if metin.strip()]
                cevrilmis = []
                toplam = max(1, len(metinler))

                for i in range(0, len(metinler), CEVIRI_BATCH_BOYUTU):
                    grup = metinler[i:i + CEVIRI_BATCH_BOYUTU]
                    girdi = tokenizer(grup, return_tensors="pt", padding=True, truncation=True).to(CIHAZ)
                    cikti = ceviri_model.generate(**girdi, forced_bos_token_id=hedef_id, max_length=256)
                    ceviri_sonuclar = tokenizer.batch_decode(cikti, skip_special_tokens=True)
                    cevrilmis.extend(ceviri_sonuclar)
                    durum["yuzde"] = round((i + len(grup)) / toplam * 100, 1)

                if len(cevrilmis) != len([metin for _, _, metin in satirlar if metin.strip()]):
                    raise ValueError("Çeviri adımındaki satır sayısı beklenenden farklı üretildi.")

                cevrilmis = [
                    (baslangic, bitis, ceviri_metin)
                    for (baslangic, bitis, _), ceviri_metin in zip(
                        [(baslangic, bitis, metin) for baslangic, bitis, metin in satirlar if metin.strip()],
                        cevrilmis,
                    )
                ]

                if len(cevrilmis) != len([metin for _, _, metin in satirlar if metin.strip()]):
                    raise ValueError("Batch çeviri sonrası eşleme bozuldu.")

                cevrilmis_map = {(baslangic, bitis): ceviri for (baslangic, bitis, _), ceviri in zip([(baslangic, bitis, metin) for baslangic, bitis, metin in satirlar if metin.strip()], cevrilmis)}
                cevrilmis = []
                for baslangic, bitis, metin in satirlar:
                    if not metin.strip():
                        cevrilmis.append((baslangic, bitis, ""))
                    else:
                        cevrilmis.append((baslangic, bitis, cevrilmis_map[(baslangic, bitis)]))

            if CEVIRI_MODU == "single":
                if len(cevrilmis) != len(satirlar):
                    raise ValueError("Tek tek çeviri sonrası satır sayısı eşleşmedi.")

            if CEVIRI_MODU == "batch":
                if len(cevrilmis) != len(satirlar):
                    raise ValueError("Batch çeviri sonrası satır sayısı eşleşmedi.")

        son_satirlar = cevrilmis

        with open(VTT_YOLU, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for baslangic, bitis, metin in cevrilmis:
                f.write(f"{saniye_to_vtt_zaman(baslangic)} --> {saniye_to_vtt_zaman(bitis)}\n")
                f.write(f"{metin}\n\n")

        with open(SRT_YOLU, "w", encoding="utf-8") as f:
            for i, (baslangic, bitis, metin) in enumerate(cevrilmis, start=1):
                f.write(f"{i}\n")
                f.write(f"{saniye_to_srt_zaman(baslangic)} --> {saniye_to_srt_zaman(bitis)}\n")
                f.write(f"{metin}\n\n")

        durum.update(asama="hazir", mesaj="Altyazı hazır.", yuzde=100)
    except Exception as e:
        durum.update(asama="hata", mesaj=f"Çözümleme hatası: {e}")
        traceback.print_exc()


def dublaj_yap():
    from pydub import AudioSegment
    from piper import PiperVoice

    try:
        from piper.config import SynthesisConfig
        sentez_ayari = SynthesisConfig(noise_scale=1.0, noise_w_scale=1.0, length_scale=1.05)
    except ImportError:
        sentez_ayari = None

    try:
        if not son_satirlar:
            raise ValueError("Önce çözümleme yapılmalı.")

        os.makedirs(GECICI_SES_KLASOR, exist_ok=True)
        durum.update(asama="dublaj_yapiliyor", mesaj="Ses modeli yükleniyor...", yuzde=0)
        ses_modeli = PiperVoice.load(SES_MODELI)

        video_suresi_ms = int(AudioSegment.from_file(VIDEO_YOLU).duration_seconds * 1000)
        dublaj_parcasi = AudioSegment.silent(duration=video_suresi_ms)

        for i, (baslangic_sn, bitis_sn, metin) in enumerate(son_satirlar):
            baslangic_ms, bitis_ms = int(baslangic_sn * 1000), int(bitis_sn * 1000)
            durum.update(mesaj=f"Seslendiriliyor {i + 1}/{len(son_satirlar)}...")
            durum["yuzde"] = round((i + 1) / len(son_satirlar) * 90, 1)

            metin_hazir = telaffuz_icin_hazirla(metin)
            gecici_wav = os.path.join(GECICI_SES_KLASOR, f"satir_{i:04d}.wav")
            with wave.open(gecici_wav, "wb") as wav_dosyasi:
                if sentez_ayari is not None:
                    ses_modeli.synthesize_wav(metin_hazir, wav_dosyasi, syn_config=sentez_ayari)
                else:
                    ses_modeli.synthesize_wav(metin_hazir, wav_dosyasi)

            hedef_sure_ms = bitis_ms - baslangic_ms
            duzeltilmis_ses = sesi_sureye_sigdir(gecici_wav, hedef_sure_ms)
            dublaj_parcasi = dublaj_parcasi.overlay(duzeltilmis_ses, position=baslangic_ms)

        durum.update(mesaj="Sesler karıştırılıyor...", yuzde=92)
        orijinal_ses = AudioSegment.from_file(VIDEO_YOLU)
        orijinal_ses_kisik = orijinal_ses + ORIJINAL_SES_SEVIYESI_DB
        karisik_ses = orijinal_ses_kisik.overlay(dublaj_parcasi)
        karisik_ses_yolu = os.path.join(GECICI_SES_KLASOR, "karisik.wav")
        karisik_ses.export(karisik_ses_yolu, format="wav")

        durum.update(mesaj="Videoya gömülüyor...", yuzde=96)
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", VIDEO_YOLU, "-i", karisik_ses_yolu,
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac", "-shortest",
                DUBLAJ_VIDEO_YOLU,
            ],
            check=True,
        )

        durum.update(asama="dublaj_hazir", mesaj="Dublaj hazır.", yuzde=100)
    except Exception as e:
        durum.update(asama="hata", mesaj=f"Dublaj hatası: {e}")
        traceback.print_exc()


def telaffuz_icin_hazirla(metin):
    if not TELAFFUZ_SOZLUGU:
        return metin
    desen = r"\b(" + "|".join(re.escape(k) for k in TELAFFUZ_SOZLUGU.keys()) + r")\b"

    def degistir(eslesme):
        orijinal = eslesme.group(0)
        yeni = TELAFFUZ_SOZLUGU[orijinal.lower()]
        if orijinal[0].isupper():
            yeni = yeni[0].upper() + yeni[1:]
        return yeni

    return re.sub(desen, degistir, metin, flags=re.IGNORECASE)


def sesi_sureye_sigdir(ses_dosyasi, hedef_sure_ms):
    from pydub import AudioSegment

    mevcut_ses = AudioSegment.from_file(ses_dosyasi)
    mevcut_sure_ms = len(mevcut_ses)
    if mevcut_sure_ms == 0 or hedef_sure_ms <= 0:
        return mevcut_ses

    oran = max(0.5, min(mevcut_sure_ms / hedef_sure_ms, 2.0))
    duzeltilmis_dosya = ses_dosyasi.replace(".wav", "_duzeltilmis.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", ses_dosyasi, "-filter:a", f"atempo={oran}", duzeltilmis_dosya],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return AudioSegment.from_file(duzeltilmis_dosya)


@app.route("/")
def anasayfa():
    return render_template("index.html")


@app.route("/api/indir", methods=["POST"])
def api_indir():
    url = request.json.get("url", "").strip()
    if not url:
        return jsonify({"hata": "Link boş olamaz."}), 400
    durum.update(asama="indiriliyor", mesaj="Başlatılıyor...", yuzde=0)
    threading.Thread(target=video_indir, args=(url,), daemon=True).start()
    return jsonify({"basladi": True})


@app.route("/api/yukle", methods=["POST"])
def api_yukle():
    dosya = request.files.get("video")
    if not dosya or dosya.filename == "":
        return jsonify({"hata": "Dosya seçilmedi."}), 400

    eski_dosyalari_temizle()
    durum.update(asama="indiriliyor", mesaj="Video yükleniyor...", yuzde=0)

    dosya.save(VIDEO_YOLU)
    durum.update(asama="indirildi", mesaj="Video yüklendi.", yuzde=100)
    return jsonify({"basladi": True})


@app.route("/api/cozumle", methods=["POST"])
def api_cozumle():
    if durum["asama"] in ("cozumleniyor", "dublaj_yapiliyor"):
        return jsonify({"hata": "Zaten bir işlem sürüyor, lütfen bitmesini bekle."}), 409
    if not os.path.exists(VIDEO_YOLU):
        return jsonify({"hata": "Önce bir video indirmeli ya da yüklemelisin."}), 400
    durum.update(asama="cozumleniyor", mesaj="Başlatılıyor...", yuzde=0)
    threading.Thread(target=cozumle_ve_cevir, daemon=True).start()
    return jsonify({"basladi": True})


@app.route("/api/dublaj", methods=["POST"])
def api_dublaj():
    if not dublaj_izinli_mi():
        return jsonify({"hata": "Dublaj için hafif mod kapalıdır; orta ya da kalite moduna geçip tekrar deneyin."}), 400
    if durum["asama"] in ("cozumleniyor", "dublaj_yapiliyor"):
        return jsonify({"hata": "Zaten bir işlem sürüyor, lütfen bitmesini bekle."}), 409
    if not son_satirlar:
        return jsonify({"hata": "Önce sesi çözümlemelisin."}), 400
    durum.update(asama="dublaj_yapiliyor", mesaj="Başlatılıyor...", yuzde=0)
    threading.Thread(target=dublaj_yap, daemon=True).start()
    return jsonify({"basladi": True})


@app.route("/api/mod", methods=["POST"])
def api_mod():
    global MODEL_MODU
    mode = request.json.get("mod", MODEL_MODU).strip().lower()
    if mode not in MODEL_MODLARI:
        return jsonify({"hata": "Geçersiz mod."}), 400
    MODEL_MODU = mode
    durum["mod"] = MODEL_MODU
    return jsonify({"mod": MODEL_MODU})


@app.route("/api/ceviri-mod", methods=["POST"])
def api_ceviri_mod():
    global CEVIRI_MODU
    mode = request.json.get("ceviri_mod", CEVIRI_MODU).strip().lower()
    if mode not in CEVIRI_MODLARI:
        return jsonify({"hata": "Geçersiz çeviri kipi."}), 400
    CEVIRI_MODU = mode
    durum["ceviri_mod"] = CEVIRI_MODU
    return jsonify({"ceviri_mod": CEVIRI_MODU})


@app.route("/api/system-info")
def api_system_info():
    gpu_var = False
    gpu_bellek_gb = 0
    try:
        import torch
        gpu_var = torch.cuda.is_available()
        if gpu_var:
            gpu_bellek_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except ImportError:
        pass

    onerilen = onerilen_mod()
    mesaj = "Hafif mod önerilir; dublajı açmak için daha güçlü bilgisayar gerekir."
    if gpu_var and gpu_bellek_gb >= 8:
        mesaj = "Orta mod uygun; dublaj için daha iyi sonuç alırsınız."
    elif gpu_var and gpu_bellek_gb < 8:
        mesaj = "Bilgisayarınız güçlü değil; hafif mod ve batch çeviri daha güvenli seçimdir."

    return jsonify({
        "gpu": gpu_var,
        "gpu_bellek_gb": round(gpu_bellek_gb, 1),
        "recommended_mode": onerilen,
        "recommended_ceviri_mode": "batch",
        "message": mesaj,
    })


@app.route("/api/durum")
def api_durum():
    durum["mod"] = MODEL_MODU
    durum["ceviri_mod"] = CEVIRI_MODU
    return jsonify(durum)


@app.route("/static/<path:dosya>")
def statik_dosya(dosya):
    return send_from_directory(STATIC_KLASOR, dosya)


if __name__ == "__main__":
    # Uygulama açılır açılmaz tarayıcıda otomatik göster, kullanıcı adres yazmak zorunda kalmasın.
    threading.Timer(1.2, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(debug=False, port=5000)
