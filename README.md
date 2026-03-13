# 🌍 TerraClimate Water Monitor

Interactive web platform for visualizing global water storage anomalies using Google Earth Engine and the TerraClimate dataset. Built with FastAPI + Leaflet.js.

![Python](https://img.shields.io/badge/Python-3.9+-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green) ![GEE](https://img.shields.io/badge/Google%20Earth%20Engine-API-yellow)

[🇬🇧 English](#english) | [🇹🇷 Türkçe](#türkçe)

---

<a name="english"></a>
## 🇬🇧 English

## Features

- 🗺️ **80+ country support** — click any country on the map to load data
- 🏙️ **Provincial choropleth maps** — FAO GAUL level-1 boundaries with hover tooltips
- 📅 **Monthly climatology** — multi-year averages per month (1990–2023)
- 📈 **Annual anomaly analysis** — deviation from 1990–2019 baseline
- 🔢 **Three climate variables** — Soil Moisture, Climate Water Deficit, Palmer Drought Index
- 🤖 **AI analysis assistant** — optional OpenAI GPT-4o-mini integration
- ⚡ **File-based caching** — GEE queries cached locally for fast repeat access

---

## Data Source

[TerraClimate](https://www.climatologylab.org/terraclimate.html) — University of Idaho  
4 km monthly global climate dataset (1958–2023), available on Google Earth Engine as `IDAHO_EPSCOR/TERRACLIMATE`

| Variable | Description | Unit |
|----------|-------------|------|
| `soil`   | Soil Moisture | mm |
| `def`    | Climate Water Deficit | mm |
| `pdsi`   | Palmer Drought Severity Index | — |

---

## Requirements

- Python 3.9+
- Google Earth Engine account with a **Service Account** and key file
- (Optional) ngrok account for public URL
- (Optional) OpenAI API key for AI assistant

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/terraclimate-water-monitor.git
cd terraclimate-water-monitor
```

### 2. Install dependencies

```bash
pip install earthengine-api fastapi uvicorn scipy matplotlib numpy pyngrok
```

### 3. Set up Google Earth Engine

- Go to [Google Cloud Console](https://console.cloud.google.com/) and create a project
- Enable the **Earth Engine API**
- Create a **Service Account**, grant it Earth Engine access
- Download the JSON key and save it as `gee-key.json` in the project folder

### 4. Configure `main.py`

Open `main.py` and fill in your credentials:

```python
KEY_FILE    = "gee-key.json"       # path to your service account key
GEE_PROJECT = "your-project-id"    # Google Cloud project ID
NGROK_TOKEN = "your-ngrok-token"   # optional, for public URL
```

### 5. Run

```bash
# Local access
python main.py

# Public URL via ngrok
python main.py --ngrok
```

Open your browser at `http://localhost:8000`

---

## Project Structure

```
terraclimate-water-monitor/
├── main.py          # Full application (backend + frontend)
├── gee-key.json     # GEE service account key (not included, see Setup)
├── gee_cache/       # Auto-created cache directory
├── .gitignore
└── README.md
```

---

## Background

This project evolved from a Google Colab notebook analyzing NASA GRACE satellite data over Australia. The web platform migrated to TerraClimate for its higher spatial resolution (4 km vs 55 km) and longer temporal coverage (1958–present), enabling provincial-level visualization across 80+ countries.

---

<a name="türkçe"></a>
## 🇹🇷 Türkçe

# 🌍 TerraClimate Su Depolama İzleme Platformu

Google Earth Engine ve TerraClimate veri seti kullanılarak küresel su depolama anomalilerini görselleştiren interaktif web platformu. FastAPI + Leaflet.js ile geliştirilmiştir.
---

## Özellikler

- 🗺️ **80+ ülke desteği** — haritada herhangi bir ülkeye tıklayarak veri yükleyin
- 🏙️ **İl bazlı choropleth haritaları** — FAO GAUL level-1 sınırları, hover tooltip desteğiyle
- 📅 **Aylık klimatoloji** — aylık bazda çok yıllık ortalamalar (1990–2023)
- 📈 **Yıllık anomali analizi** — 1990–2019 baseline'dan sapma
- 🔢 **Üç iklim değişkeni** — Toprak Nemi, Su Açığı, Palmer Kuraklık İndeksi
- 🤖 **AI analiz asistanı** — isteğe bağlı OpenAI GPT-4o-mini entegrasyonu
- ⚡ **Dosya tabanlı önbellekleme** — GEE sorguları yerel olarak cache'lenerek tekrar erişimlerde hız sağlanır

---

## Veri Kaynağı

[TerraClimate](https://www.climatologylab.org/terraclimate.html) — Idaho Üniversitesi  
4 km çözünürlüklü aylık küresel iklim veri seti (1958–2023), Google Earth Engine üzerinde `IDAHO_EPSCOR/TERRACLIMATE` olarak erişilebilir

| Değişken | Açıklama | Birim |
|----------|----------|-------|
| `soil`   | Toprak Nemi | mm |
| `def`    | Su Açığı (Climate Water Deficit) | mm |
| `pdsi`   | Palmer Kuraklık Şiddet İndeksi | — |

---

## Gereksinimler

- Python 3.9+
- **Servis Hesabı** ve anahtar dosyasına sahip Google Earth Engine hesabı
- (İsteğe bağlı) Dış erişim için ngrok hesabı
- (İsteğe bağlı) AI asistan için OpenAI API anahtarı

---

## Kurulum

### 1. Repoyu klonla

```bash
git clone https://github.com/kullaniciadin/terraclimate-water-monitor.git
cd terraclimate-water-monitor
```

### 2. Bağımlılıkları yükle

```bash
pip install earthengine-api fastapi uvicorn scipy matplotlib numpy pyngrok
```

### 3. Google Earth Engine kurulumu

- [Google Cloud Console](https://console.cloud.google.com/) üzerinden yeni bir proje oluştur
- **Earth Engine API**'yi etkinleştir
- Bir **Servis Hesabı** oluştur ve Earth Engine erişimi ver
- JSON anahtarını indirip proje klasörüne `gee-key.json` olarak kaydet

### 4. `main.py` yapılandırması

`main.py` dosyasını açıp kimlik bilgilerini gir:

```python
KEY_FILE    = "gee-key.json"       # servis hesabı anahtar dosyası
GEE_PROJECT = "proje-id-niz"       # Google Cloud proje ID'si
NGROK_TOKEN = "ngrok-tokeniniz"    # isteğe bağlı, dış erişim için
```

### 5. Çalıştır

```bash
# Yerel erişim
python main.py

# ngrok ile dış erişim
python main.py --ngrok
```

Tarayıcıdan `http://localhost:8000` adresini aç

---

## Proje Yapısı

```
terraclimate-water-monitor/
├── main.py          # Uygulamanın tamamı (backend + frontend)
├── gee-key.json     # GEE servis hesabı anahtarı (dahil değil, kuruluma bakın)
├── gee_cache/       # Otomatik oluşturulan önbellek klasörü
├── .gitignore
└── README.md
```

---

## Arka Plan

Bu proje, NASA GRACE uydu verilerini kullanarak Avustralya üzerinde su depolama analizi yapan bir Google Colab not defterinden geliştirilmiştir. Web platformuna geçişte TerraClimate veri setine geçilmiş; bu sayede uzamsal çözünürlük 55 km'den 4 km'ye yükseltilmiş ve 1958'e uzanan tarihsel veri derinliğiyle 80'den fazla ülkede il bazlı görselleştirme mümkün hale getirilmiştir.
