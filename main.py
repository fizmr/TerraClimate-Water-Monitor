"""
Water Storage Monitor
TerraClimate + GRACE · GEE live · smooth scroll UI
"""

import os, json, io, base64, math, hashlib, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.path as mpath
from matplotlib.patches import Circle, Rectangle, PathPatch
from scipy.interpolate import griddata
import ee
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pyngrok import ngrok
import uvicorn, threading, time

# ── CONFIG ────────────────────────────────────────────────────────────
KEY_FILE    = "gee-key.json"
GEE_PROJECT = ""   # GEE proje ID'nizi buraya girin
NGROK_TOKEN = ""   # ngrok token'ınızı buraya girin
CACHE_DIR   = "gee_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ── GEE INIT ──────────────────────────────────────────────────────────
print("🔑 Connecting to GEE...")
_creds = ee.ServiceAccountCredentials(email=None, key_file=KEY_FILE)
ee.Initialize(_creds, project=GEE_PROJECT)
print("✅ GEE connected!")

app = FastAPI(title="Water Storage Monitor")

MONTH_NAMES = ['January','February','March','April','May','June',
               'July','August','September','October','November','December']
MN_SHORT    = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']

# TerraClimate: 1958-2023, 4km çözünürlük
TC_COL   = "IDAHO_EPSCOR/TERRACLIMATE"
TC_YEARS = list(range(1990, 2024))   # 1990-2023 arası (34 yıl)
LSIB     = "USDOS/LSIB_SIMPLE/2017"

# TerraClimate değişkenleri: band adı → {label, unit, description, scale}
# scale: GEE raw değeri → gerçek birim çarpanı
TC_VARS = {
    'soil': {
        'label':       'Toprak Nemi',
        'unit':        'mm',
        'description': 'Aylık ortalama toprak nemi',
        'scale':       0.1,    # raw * 0.1 = mm
        'cmap':        'YlGnBu',
    },
    'def': {
        'label':       'Su Açığı',
        'unit':        'mm',
        'description': 'Klimatik su açığı (kuraklık göstergesi)',
        'scale':       0.1,
        'cmap':        'YlOrRd',
    },
    'pdsi': {
        'label':       'Palmer Kuraklık İndeksi',
        'unit':        '',
        'description': 'PDSI: < −2 kuraklık, > +2 ıslak',
        'scale':       0.01,
        'cmap':        'RdYlBu',
    },
}
TC_VAR_DEFAULT = 'soil'

# Baseline: 1990-2019 klimatoloji (WMO standardı)
BASELINE_START = '1990-01-01'
BASELINE_END   = '2020-01-01'  # exclusive

# ISO2 → display name (for UI)
COUNTRY_NAMES = {
    'AF':'Afghanistan','AL':'Albania','DZ':'Algeria','AR':'Argentina',
    'AU':'Australia','AT':'Austria','AZ':'Azerbaijan','BD':'Bangladesh',
    'BE':'Belgium','BR':'Brazil','BG':'Bulgaria','KH':'Cambodia',
    'CM':'Cameroon','CA':'Canada','CL':'Chile','CN':'China',
    'CO':'Colombia','HR':'Croatia','CZ':'Czechia','DK':'Denmark',
    'EG':'Egypt','ET':'Ethiopia','FI':'Finland','FR':'France',
    'DE':'Germany','GH':'Ghana','GR':'Greece','GT':'Guatemala',
    'HU':'Hungary','IN':'India','ID':'Indonesia','IR':'Iran',
    'IQ':'Iraq','IE':'Ireland','IL':'Israel','IT':'Italy',
    'JP':'Japan','JO':'Jordan','KZ':'Kazakhstan','KE':'Kenya',
    'KR':'South Korea','KW':'Kuwait','LB':'Lebanon','MY':'Malaysia',
    'MX':'Mexico','MA':'Morocco','MZ':'Mozambique','MM':'Myanmar',
    'NP':'Nepal','NL':'Netherlands','NZ':'New Zealand','NG':'Nigeria',
    'NO':'Norway','PK':'Pakistan','PE':'Peru','PH':'Philippines',
    'PL':'Poland','PT':'Portugal','RO':'Romania','RU':'Russia',
    'SA':'Saudi Arabia','SN':'Senegal','ZA':'South Africa','ES':'Spain',
    'SD':'Sudan','SE':'Sweden','CH':'Switzerland','SY':'Syria',
    'TJ':'Tajikistan','TZ':'Tanzania','TH':'Thailand',
    'TR':'Turkey','TM':'Turkmenistan','UG':'Uganda','UA':'Ukraine',
    'AE':'United Arab Emirates','GB':'United Kingdom',
    'US':'United States','UZ':'Uzbekistan','VE':'Venezuela',
    'VN':'Vietnam','YE':'Yemen','ZM':'Zambia','ZW':'Zimbabwe',
}

# ISO2 → GEE LSIB "country_na" overrides
# Only entries that differ from COUNTRY_NAMES display names
GEE_NAME_MAP = {
    'KR': 'South Korea',
    'CZ': 'Czechia',
    'KH': 'Cambodia',
}

_border_cache = {}

# ── HELPERS ───────────────────────────────────────────────────────────

def cache_key(code, mode, period):
    return hashlib.md5(f"{code}_{mode}_{period}".encode()).hexdigest()

def cache_path(key):
    return os.path.join(CACHE_DIR, f"{key}.json")

def cache_get(key):
    p = cache_path(key)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None

def cache_set(key, data):
    with open(cache_path(key), 'w') as f:
        json.dump(data, f)

def get_gee_name(code):
    """GEE LSIB country_na field value for an ISO2 code."""
    return GEE_NAME_MAP.get(code) or COUNTRY_NAMES.get(code, code)

def get_border(code):
    """Fetch country border from GEE, cache locally."""
    if code in _border_cache:
        return _border_cache[code]
    bp = os.path.join(CACHE_DIR, f"border_{code}.json")
    if os.path.exists(bp):
        with open(bp) as f:
            _border_cache[code] = json.load(f)
            return _border_cache[code]
    name = get_gee_name(code)
    fc   = ee.FeatureCollection(LSIB).filter(ee.Filter.eq('country_na', name))
    geom = fc.geometry().simplify(maxError=8000)
    info = geom.getInfo()
    _border_cache[code] = info
    with open(bp, 'w') as f:
        json.dump(info, f)
    return info

def extract_polygons(geom):
    """Extract list of coordinate rings from any GEE geometry type."""
    polys = []
    if geom['type'] == 'Polygon':
        polys.append(geom['coordinates'][0])
    elif geom['type'] == 'MultiPolygon':
        for part in geom['coordinates']:
            polys.append(part[0])
    elif geom['type'] == 'GeometryCollection':
        for g in geom.get('geometries', []):
            polys.extend(extract_polygons(g))
    return polys

def _month_end(yr, mo):
    """Return first day of next month (filterDate end is exclusive in GEE)."""
    if mo == 12:
        return f"{yr+1}-01-01"
    return f"{yr}-{mo+1:02d}-01"

def fetch_tc_grid(code, year, var='soil'):
    """TerraClimate yıllık anomaly grid: seçilen yıl − 1990-2019 baseline."""
    ck = cache_key(code, f'tc_anomaly_{var}', str(year))
    cached = cache_get(ck)
    if cached:
        return cached

    scale    = TC_VARS[var]['scale']
    name     = get_gee_name(code)
    fc       = ee.FeatureCollection(LSIB).filter(ee.Filter.eq('country_na', name))
    region   = fc.geometry().bounds()

    col      = ee.ImageCollection(TC_COL).select(var)
    baseline = col.filterDate(BASELINE_START, BASELINE_END).mean().multiply(scale)
    annual   = col.filterDate(f"{year}-01-01", f"{year+1}-01-01").mean().multiply(scale)
    image    = annual.subtract(baseline)

    records  = _sample_region(image, region, var)
    cache_set(ck, records)
    return records


def _get_bounds(region):
    """Return (lon_min, lon_max, lat_min, lat_max) from a GEE region."""
    coords = region.bounds().getInfo()['coordinates'][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return min(lons), max(lons), min(lats), max(lats)


def _sample_region(image, region, var='soil'):
    """
    TerraClimate imgesini düzenli grid üzerinde örnekler.
    TerraClimate 4km çözünürlük → step 0.1°-0.5° arası yeterli.
    """
    lon_min, lon_max, lat_min, lat_max = _get_bounds(region)
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min
    area      = lon_range * lat_range

    # TerraClimate 4km → GRACE'den çok daha sık grid kullanabiliriz
    if   area > 3000: step = 1.0   # Rusya, Kanada
    elif area > 1000: step = 0.5   # ABD, Çin, Avustralya
    elif area > 300:  step = 0.3   # Brezilya, Hindistan
    elif area > 80:   step = 0.2   # Türkiye, Fransa, Almanya
    else:             step = 0.15  # Küçük ülkeler

    features = []
    lon = lon_min + step / 2.0
    while lon < lon_max:
        lat = lat_min + step / 2.0
        while lat < lat_max:
            features.append(
                ee.Feature(ee.Geometry.Point([float(round(lon, 4)),
                                              float(round(lat, 4))]))
            )
            lat += step
        lon += step

    if not features:
        return []

    grid_fc = ee.FeatureCollection(features)
    # TerraClimate native scale ~4638m
    sampled = image.sampleRegions(
        collection=grid_fc,
        scale=4638,
        geometries=True,
        tileScale=4
    )

    result = []
    for f in sampled.getInfo()['features']:
        coords = f['geometry']['coordinates']
        props  = f['properties']
        # Band adı direkt var adı (soil, def, pdsi)
        val = props.get(var)
        if val is None:
            val = props.get('constant')
        if val is not None:
            result.append({
                'lon': round(float(coords[0]), 4),
                'lat': round(float(coords[1]), 4),
                'lwe': round(float(val),        4)   # lwe key'i tutuyoruz — render değişmez
            })
    return result


def fetch_tc_months(code, var='soil'):
    """
    TerraClimate 12 aylık klimatoloji: her ay için 1990-2023 çok yıllık ortalama.
    4km çözünürlük — gerçek il bazlı detay.
    """
    ck = cache_key(code, f'tc_months_{var}', 'all')
    cached = cache_get(ck)
    if cached and isinstance(cached, dict) and len(cached) == 12:
        if all(len(cached.get(str(m), [])) > 0 for m in range(1, 13)):
            print(f"[fetch_tc_months] {code}/{var}: cache'den yüklendi")
            return cached

    scale  = TC_VARS[var]['scale']
    name   = get_gee_name(code)
    fc     = ee.FeatureCollection(LSIB).filter(ee.Filter.eq('country_na', name))
    region = fc.geometry().bounds()

    # Grid adımını hesapla
    lon_min, lon_max, lat_min, lat_max = _get_bounds(region)
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min
    area      = lon_range * lat_range
    if   area > 3000: step = 1.0
    elif area > 1000: step = 0.5
    elif area > 300:  step = 0.3
    elif area > 80:   step = 0.2
    else:             step = 0.15

    # Grid noktaları bir kez oluştur
    features = []
    lon = lon_min + step / 2.0
    while lon < lon_max:
        lat = lat_min + step / 2.0
        while lat < lat_max:
            features.append(
                ee.Feature(ee.Geometry.Point([float(round(lon, 4)),
                                              float(round(lat, 4))]))
            )
            lat += step
        lon += step

    grid_fc  = ee.FeatureCollection(features)
    full_col = ee.ImageCollection(TC_COL).select(var).filterDate('1990-01-01', '2024-01-01')
    print(f"[fetch_tc_months] {code}/{var}: {len(features)} grid noktası, step={step}°")

    result = {}
    for month in range(1, 13):
        mn_label = MN_SHORT[month - 1]
        month_col = full_col.filter(ee.Filter.calendarRange(month, month, 'month'))
        try:
            img     = month_col.mean().multiply(scale)
            sampled = img.sampleRegions(
                collection=grid_fc,
                scale=4638,
                geometries=True,
                tileScale=4
            )
            records = []
            for f in sampled.getInfo()['features']:
                coords = f['geometry']['coordinates']
                val    = f['properties'].get(var)
                if val is None:
                    val = f['properties'].get('constant')
                if val is not None:
                    records.append({
                        'lon': round(float(coords[0]), 4),
                        'lat': round(float(coords[1]), 4),
                        'lwe': round(float(val),        4)
                    })
            result[str(month)] = records
            vals = [r['lwe'] for r in records]
            rng  = f"[{min(vals):.1f} → {max(vals):.1f}]" if vals else "BOŞ"
            print(f"  Ay {month:2d} ({mn_label}): {len(records):4d} nokta  {rng}")
        except Exception as e:
            print(f"  Ay {month:2d} ({mn_label}): HATA — {e}")
            result[str(month)] = []

    cache_set(ck, result)
    return result


def fetch_annual_anomaly_ts(code, var='soil'):
    """TerraClimate yıllık anomaly time series: her nokta = 1 yıl, 1990-2019 baseline."""
    ck = cache_key(code, f'tc_ts_{var}', 'all')
    cached = cache_get(ck)
    if cached:
        return cached

    scale    = TC_VARS[var]['scale']
    name     = get_gee_name(code)
    fc       = ee.FeatureCollection(LSIB).filter(ee.Filter.eq('country_na', name))
    region   = fc.geometry()
    col      = ee.ImageCollection(TC_COL).select(var)
    baseline = col.filterDate(BASELINE_START, BASELINE_END).mean().multiply(scale)

    records = []
    for yr in TC_YEARS:
        try:
            annual = col.filterDate(f"{yr}-01-01", f"{yr+1}-01-01").mean().multiply(scale)
            anom   = annual.subtract(baseline)
            stats  = anom.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=4638,
                maxPixels=1e9,
                bestEffort=True
            )
            val = stats.getInfo().get(var)
            if val is not None:
                records.append({'yil': yr, 'lwe_anomaly': round(float(val), 4)})
                print(f"  [ts] {yr}: {val:+.3f} {TC_VARS[var]['unit']}")
        except Exception as e:
            print(f"  [ts] {yr}: HATA — {e}")

    cache_set(ck, records)
    return records


# ── PROVINCE DATA ─────────────────────────────────────────────────────

# FAO GAUL level-1 ISO2 → ADM0_NAME mapping overrides
# (default: COUNTRY_NAMES value is tried first)
GAUL_NAME_MAP = {
    'KR': 'Republic of Korea',
    'KP': "Democratic People's Republic of Korea",
    'IR': 'Iran  (Islamic Republic of)',
    'TZ': 'United Republic of Tanzania',
    'VE': 'Venezuela (Bolivarian Republic of)',
    'BO': 'Bolivia (Plurinational State of)',
    'MD': 'Republic of Moldova',
    'RU': 'Russian Federation',
    'SY': 'Syrian Arab Republic',
    'VN': 'Viet Nam',
    'GB': 'United Kingdom of Great Britain and Northern Ireland',
    'US': 'United States of America',
    'CD': 'Democratic Republic of the Congo',
    'CZ': 'Czech Republic',
    'MK': 'The former Yugoslav Republic of Macedonia',
    'TW': 'Taiwan',
}

_province_border_cache = {}

def get_gaul_name(code):
    return GAUL_NAME_MAP.get(code) or COUNTRY_NAMES.get(code, code)

def fetch_provinces(code, var='soil', mode='monthly', period='all'):
    """
    Her il/eyalet için TerraClimate ortalaması.
    FAO GAUL level-1 sınırları kullanır.
    Döner: [ {name, value, geometry_geojson}, ... ]
    """
    ck = cache_key(code, f'prov_{var}_{mode}', str(period))
    cached = cache_get(ck)
    if cached:
        print(f"[provinces] {code}/{var}/{mode}/{period}: cache'den yüklendi")
        return cached

    scale    = TC_VARS[var]['scale']
    gaul_name = get_gaul_name(code)
    gaul      = ee.FeatureCollection('FAO/GAUL/2015/level1')
    provinces = gaul.filter(ee.Filter.eq('ADM0_NAME', gaul_name)) \
        .select(['ADM1_NAME'])

    col = ee.ImageCollection(TC_COL).select(var)

    # İmgeyi seç
    if mode == 'monthly':
        # Tüm yılların tüm ayları — yıllık ortalama klimatoloji
        image = col.filterDate('1990-01-01', '2024-01-01').mean().multiply(scale)
    elif mode == 'monthly_single':
        # Belirli bir ay, tüm yıllar ortalaması
        mo    = int(period)
        image = col.filter(ee.Filter.calendarRange(mo, mo, 'month')).mean().multiply(scale)
    else:
        # Anomaly: seçilen yıl − baseline
        yr       = int(period)
        baseline = col.filterDate(BASELINE_START, BASELINE_END).mean().multiply(scale)
        annual   = col.filterDate(f"{yr}-01-01", f"{yr+1}-01-01").mean().multiply(scale)
        image    = annual.subtract(baseline)

    # Her il için reduceRegion
    def reduce_province(feat):
        stats = image.reduceRegion(
            reducer  = ee.Reducer.mean(),
            geometry = feat.geometry(),
            scale    = 4638,
            maxPixels= 1e8,
            bestEffort=True
        )
        return feat.set('tc_val', stats.get(var))

    result_fc = provinces.map(reduce_province) \
        .map(lambda f: f.simplify(maxError=1000))
    info      = result_fc.getInfo()

    records = []
    for feat in info['features']:
        props = feat['properties']
        val   = props.get('tc_val')
        name  = props.get('ADM1_NAME', 'Unknown')
        geom  = feat['geometry']
        if val is not None:
            records.append({
                'name':     name,
                'value':    round(float(val), 3),
                'geometry': geom
            })
        else:
            # il var ama veri yok — yine de sınırı gönder, değer null
            records.append({
                'name':     name,
                'value':    None,
                'geometry': geom
            })

    print(f"[provinces] {code}/{var}/{mode}/{period}: {len(records)} il")
    cache_set(ck, records)
    return records


# ── RENDERER ──────────────────────────────────────────────────────────
BG   = '#F4F1EC'
CMAP = 'turbo_r'

def _norm(vmax):
    return mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

def _b64(fig, dpi=140):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi,
                bbox_inches='tight', facecolor=BG, edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _draw_panel(ax, pts, vals, border_geom, norm, lon_lim, lat_lim, title, cmap=None):
    """
    Tek bir ay panelini çizer.
    lon_lim / lat_lim tüm 12 ay için sabittir — ortak extent.
    ax hiçbir zaman gizlenmez, veri yoksa sadece boş çerçeve kalır.
    """
    if cmap is None: cmap = CMAP
    ax.set_facecolor(BG)

    # ── aspect correction ────────────────────────────────────────────
    mean_lat   = (lat_lim[0] + lat_lim[1]) / 2
    aspect_cor = math.cos(math.radians(mean_lat))
    ax.set_aspect(1.0 / aspect_cor, adjustable='datalim')

    ax.set_xlim(lon_lim[0], lon_lim[1])
    ax.set_ylim(lat_lim[0], lat_lim[1])

    # ── interpolate & draw ───────────────────────────────────────────
    mask = np.isfinite(vals)
    pts_ok  = pts[mask]
    vals_ok = vals[mask]

    if len(vals_ok) >= 3:
        pad   = 0.3
        lon_r = lon_lim[1] - lon_lim[0]
        lat_r = lat_lim[1] - lat_lim[0]
        res   = max(0.15, min(lon_r, lat_r) / 80)

        lon_g = np.arange(lon_lim[0] - pad, lon_lim[1] + pad, res)
        lat_g = np.arange(lat_lim[0] - pad, lat_lim[1] + pad, res)
        LLon, LLat = np.meshgrid(lon_g, lat_g)
        grid_v = griddata(pts_ok, vals_ok, (LLon, LLat), method='linear')

        mesh = ax.pcolormesh(LLon, LLat, grid_v,
                             cmap=cmap, norm=norm,
                             shading='gouraud', zorder=2)

        # Ülke sınır clip + beyaz çizgi
        # Tüm polygon parçaları birleştirilir (Rusya adaları, Kanada adaları vb.)
        polys = extract_polygons(border_geom)
        if polys:
            # Tüm polygonları tek bir compound path olarak birleştir
            verts, codes = [], []
            for poly in polys:
                arr = np.array(poly)
                verts.extend(arr.tolist())
                c = [mpath.Path.LINETO] * len(arr)
                c[0] = mpath.Path.MOVETO
                codes.extend(c)
            compound = mpath.Path(np.array(verts), codes)
            clip_patch = PathPatch(compound, transform=ax.transData)
            mesh.set_clip_path(clip_patch)
            # Tüm sınır çizgilerini çiz
            for poly in polys:
                arr = np.array(poly)
                ax.plot(arr[:, 0], arr[:, 1],
                        color=BG, linewidth=0.6, zorder=5)

    ax.axis('off')
    ax.set_title(title, fontsize=7.5, color='#555',
                 fontfamily='monospace', pad=2)


def render_heatmap_12months(records_by_month, border_geom, var='soil'):
    """
    12 aylık ızgara: 3 satır × 4 sütun.
    add_axes ile mutlak koordinat — hiçbir panel kaybolmaz.
    Paylaşımlı renk skalası, sabit extent.
    """
    MN    = MN_SHORT
    vcmap = TC_VARS.get(var, TC_VARS['soil'])['cmap']

    # ── Global vmax ve extent ─────────────────────────────────────────
    all_lwe, all_lon, all_lat = [], [], []
    for mo in range(1, 13):
        for r in records_by_month.get(str(mo), []):
            if math.isfinite(r['lwe']):
                all_lwe.append(r['lwe'])
                all_lon.append(r['lon'])
                all_lat.append(r['lat'])

    if not all_lwe:
        raise ValueError("Hiç veri yok")

    vmax    = max(abs(min(all_lwe)), abs(max(all_lwe))) or 1.0
    norm    = _norm(vmax)
    lon_lim = (min(all_lon), max(all_lon))
    lat_lim = (min(all_lat), max(all_lat))

    # ── Figure: add_axes ile tam kontrol ─────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(BG)

    # Harita gridinin kaplama alanı (figure koordinatı 0..1)
    L, R = 0.00, 0.90   # sol / sağ
    B, T = 0.01, 0.99   # alt / üst
    COLS, ROWS = 4, 3
    cell_w = (R - L) / COLS
    cell_h = (T - B) / ROWS
    PAD_W  = 0.006
    PAD_H  = 0.018

    for idx in range(12):
        row = idx // COLS
        col = idx  % COLS
        left   = L + col * cell_w + PAD_W / 2
        bottom = T - (row + 1) * cell_h + PAD_H / 2
        w      = cell_w - PAD_W
        h      = cell_h - PAD_H

        ax = fig.add_axes([left, bottom, w, h])

        mo   = idx + 1
        recs = records_by_month.get(str(mo), [])
        if recs:
            pts  = np.array([(r['lon'], r['lat']) for r in recs], dtype=float)
            vals = np.array([r['lwe']             for r in recs], dtype=float)
        else:
            pts  = np.empty((0, 2))
            vals = np.array([])

        _draw_panel(ax, pts, vals, border_geom, norm,
                    lon_lim, lat_lim, title=MN[idx], cmap=vcmap)

    # ── Colorbar ─────────────────────────────────────────────────────
    cbar_ax = fig.add_axes([0.912, 0.10, 0.016, 0.80])
    vinfo = TC_VARS.get(var, TC_VARS['soil'])
    ulabel = f"{vinfo['label']} ({vinfo['unit']})" if vinfo['unit'] else vinfo['label']
    sm = plt.cm.ScalarMappable(cmap=vcmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(ulabel, color='#888', fontsize=8, fontfamily='monospace')
    cbar.ax.tick_params(labelsize=7, colors='#aaa')
    cbar.outline.set_edgecolor('#ddd')

    # bbox_inches=None — kesinlikle crop yok
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120,
                bbox_inches=None, facecolor=BG, edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def render_heatmap(code, records, border_geom, var='soil'):
    """Tekil heatmap (anomaly modu için)."""
    if not records:
        raise HTTPException(500, "No data points")

    pts  = np.array([(r['lon'], r['lat']) for r in records], dtype=float)
    vals = np.array([r['lwe']             for r in records], dtype=float)
    mask = np.isfinite(vals)
    pts, vals = pts[mask], vals[mask]
    if len(vals) < 3:
        raise HTTPException(500, "Not enough data points")

    vmax       = max(abs(vals.min()), abs(vals.max())) or 1.0
    norm       = _norm(vmax)
    lon_lim    = (pts[:,0].min(), pts[:,0].max())
    lat_lim    = (pts[:,1].min(), pts[:,1].max())
    mean_lat   = (lat_lim[0] + lat_lim[1]) / 2
    aspect_cor = math.cos(math.radians(mean_lat))

    lon_range = lon_lim[1] - lon_lim[0]
    lat_range = lat_lim[1] - lat_lim[0]
    pad       = max(0.5, min(3.0, max(lon_range, lat_range) * 0.08))
    fig_w     = 9.0
    fig_h     = max(4.0, min(10.0,
                   fig_w * ((lat_range + 2*pad) /
                             max(lon_range + 2*pad, 0.01)) / aspect_cor))

    fig = matplotlib.figure.Figure(figsize=(fig_w, fig_h), facecolor=BG)
    ax  = fig.add_subplot(111)
    ax.set_facecolor(BG)

    vinfo  = TC_VARS.get(var, TC_VARS['soil'])
    vcmap  = vinfo['cmap']
    ulabel = f"{vinfo['label']} ({vinfo['unit']})" if vinfo['unit'] else vinfo['label']

    _draw_panel(ax, pts, vals, border_geom, norm, lon_lim, lat_lim, title=None, cmap=vcmap)

    sm   = plt.cm.ScalarMappable(cmap=vcmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.022, pad=0.02, aspect=22)
    cbar.set_label(ulabel, color='#888', fontsize=7.5, fontfamily='monospace')
    cbar.ax.tick_params(labelsize=7, colors='#aaa')
    cbar.outline.set_edgecolor('#ddd')

    fig.tight_layout(pad=0.3)
    return _b64(fig)


def render_timeseries(records, code, var='soil'):
    """Yıllık anomaly dot+line — Colab stili."""
    if not records:
        raise HTTPException(500, "No data")

    years  = [r['yil']        for r in records]
    vals   = [r['lwe_anomaly'] for r in records]
    xs     = list(range(len(years)))
    vinfo  = TC_VARS.get(var, TC_VARS['soil'])
    ylabel = f"{vinfo['label']} ({vinfo['unit']})" if vinfo['unit'] else vinfo['label']

    fig = matplotlib.figure.Figure(figsize=(12, 4), facecolor=BG)
    ax  = fig.add_subplot(111)
    ax.set_facecolor(BG)
    ax.plot(xs, vals, color='#1f77b4', linewidth=1.2, zorder=3)
    ax.scatter(xs, vals, s=22, color='#1f77b4', zorder=4)
    ax.axhline(0, color='#ccc', linewidth=0.8, zorder=2)
    ax.grid(True, color='#eee', linewidth=0.6, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(y) for y in years], rotation=30, ha='right', fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel('Zaman', fontsize=9)
    ax.set_title(f'Zaman İçerisindeki Ortalama {ylabel} Anomalisi', fontsize=10)
    ax.tick_params(labelsize=8)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.8)
    return _b64(fig, dpi=130)


def render_point(records_point, lat, lon, mode):
    if not records_point:
        raise HTTPException(500, "No data")
    if mode == 'monthly':
        labels = MN_SHORT
        vals   = np.array([records_point.get(str(m), 0) for m in range(1,13)])
    else:
        years  = sorted(records_point.keys())
        labels = years
        vals   = np.array([records_point[y] for y in years])

    n    = len(vals)
    xs   = np.arange(n)
    vmax = max(abs(vals.min()), abs(vals.max())) or 1.0
    norm = _norm(vmax)
    cmap = plt.cm.get_cmap(CMAP)

    fig = matplotlib.figure.Figure(figsize=(8, 3.0), facecolor=BG)
    ax  = fig.add_subplot(111)
    ax.set_facecolor(BG)
    colors = [cmap(norm(float(v))) for v in vals]
    ax.bar(xs, vals, color=colors, width=0.7, zorder=3)
    ax.axhline(0, color='#C8C3BB', linewidth=0.7, zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30 if mode=='anomaly' else 45,
                       ha='right', fontsize=7.5, color='#888',
                       fontfamily='monospace')
    ax.set_yticks([])
    ax.set_xlim(-0.7, n-0.3)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(length=0)
    title = f"{abs(lat):.2f}°{'N' if lat>=0 else 'S'}  {abs(lon):.2f}°{'E' if lon>=0 else 'W'}"
    ax.set_title(title, fontsize=8.5, color='#888',
                 fontfamily='monospace', pad=5)
    fig.tight_layout(pad=0.4)
    return _b64(fig, dpi=130)


# ── API ───────────────────────────────────────────────────────────────


@app.get("/api/debug/{code}")
def api_debug(code: str, var: str = Query(default='soil')):
    """
    Veri test endpoint — ülke için tüm 12 ayın özet istatistiklerini döner.
    Tarayıcıda: http://localhost:8000/api/debug/TR?var=soil
    """
    c = code.upper()
    if var not in TC_VARS: var = 'soil'
    try:
        data = fetch_tc_months(c, var)
        summary = {}
        for mo in range(1, 13):
            mn   = MN_SHORT[mo-1]
            recs = data.get(str(mo), [])
            vals = [r['lwe'] for r in recs if math.isfinite(r['lwe'])]
            if vals:
                summary[mn] = {
                    'n_points': len(vals),
                    'min':  round(min(vals), 2),
                    'max':  round(max(vals), 2),
                    'mean': round(sum(vals)/len(vals), 2),
                    'status': 'OK'
                }
            else:
                summary[mn] = {'n_points': 0, 'status': 'BOŞ — veri yok!'}
        return {'country': c, 'months': summary}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/cache/clear/{code}")
def api_clear_cache(code: str):
    """Clear all cached data for a country (useful when data seems stale)."""
    c = code.upper()
    removed = []
    for fname in os.listdir(CACHE_DIR):
        fpath = os.path.join(CACHE_DIR, fname)
        # Remove hashed data caches (not border/bbox files)
        if fname.endswith('.json') and not fname.startswith('border_') and not fname.startswith('bbox_'):
            try:
                with open(fpath) as f:
                    content = f.read()
                # Check if this cache likely belongs to this country by re-computing keys
                for mode in ['all_months', 'tc_anomaly_soil', 'tc_anomaly_def', 'tc_anomaly_pdsi', 'tc_months_soil', 'tc_months_def', 'tc_months_pdsi', 'tc_ts_soil', 'tc_ts_def', 'tc_ts_pdsi']:
                    for period in ['all'] + [str(y) for y in range(2003, 2024)] + [str(m) for m in range(1, 13)]:
                        if cache_key(c, mode, period) + '.json' == fname:
                            os.remove(fpath)
                            removed.append(fname)
                            break
            except:
                pass
    return {"cleared": len(removed), "files": removed}

@app.get("/api/countries")
def api_countries():
    return [{"code":k,"name":v} for k,v in COUNTRY_NAMES.items()]

@app.get("/api/bbox/{code}")
def api_bbox(code: str):
    """Get country bounding box — cached to disk."""
    c  = code.upper()
    bp = os.path.join(CACHE_DIR, f"bbox_{c}.json")
    if os.path.exists(bp):
        with open(bp) as f:
            return json.load(f)
    try:
        name   = get_gee_name(c)
        fc     = ee.FeatureCollection(LSIB).filter(ee.Filter.eq('country_na', name))
        bounds = fc.geometry().bounds().getInfo()
        coords = bounds['coordinates'][0]
        lons   = [p[0] for p in coords]
        lats   = [p[1] for p in coords]
        result = {
            "code":   c,
            "name":   COUNTRY_NAMES.get(c, c),
            "bbox":   [min(lats), min(lons), max(lats), max(lons)],
            "center": [(min(lats)+max(lats))/2, (min(lons)+max(lons))/2]
        }
        with open(bp, 'w') as f:
            json.dump(result, f)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))

def _lwe_stats(records):
    """min/max/mean LWE for frontend context."""
    vals = [r['lwe'] for r in records if math.isfinite(r['lwe'])]
    if not vals: return None
    return {"min": round(min(vals),1), "max": round(max(vals),1),
            "mean": round(sum(vals)/len(vals),1)}

@app.get("/api/render/heatmap/{code}/{mode}/{period}")
def api_heatmap(code: str, mode: str, period: str,
                var: str = Query(default='soil')):
    c = code.upper()
    if var not in TC_VARS:
        var = 'soil'
    try:
        border = get_border(c)
        if mode == 'monthly':
            data = fetch_tc_months(c, var)
            if not data or not any(data.values()):
                raise ValueError("No monthly data returned from GEE")
            img = render_heatmap_12months(data, border, var)
            all_recs = [r for m in data.values() for r in m]
            stats = _lwe_stats(all_recs)
        else:
            records = fetch_tc_grid(c, int(period), var)
            if not records:
                raise ValueError(f"No data for {c}/{var} anomaly {period}")
            img   = render_heatmap(c, records, border, var)
            stats = _lwe_stats(records)
        return {"img": img, "stats": stats}
    except Exception as e:
        import traceback
        print(f"[heatmap error] {c}/{mode}/{period}/{var}: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, str(e))

@app.get("/api/render/timeseries/{code}")
def api_ts(code: str, var: str = Query(default='soil')):
    c = code.upper()
    if var not in TC_VARS:
        var = 'soil'
    try:
        records = fetch_annual_anomaly_ts(c, var)
        img     = render_timeseries(records, c, var)
        trend = None
        if len(records) >= 10:
            vals      = [r['lwe_anomaly'] for r in records]
            early     = sum(vals[:5])  / 5
            late      = sum(vals[-5:]) / 5
            diff      = round(late - early, 2)
            vinfo     = TC_VARS[var]
            direction = "artış" if diff > 0 else "azalış"
            trend     = f"{direction} ({diff:+.2f} {vinfo['unit']})"
        return {"img": img, "trend": trend}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/render/point/{code}/{mode}")
def api_point(code: str, mode: str,
              lat: float = Query(...), lon: float = Query(...),
              var: str   = Query(default='soil')):
    c = code.upper()
    if var not in TC_VARS:
        var = 'soil'
    scale = TC_VARS[var]['scale']
    try:
        col = ee.ImageCollection(TC_COL).select(var)
        pt  = ee.Geometry.Point([lon, lat])
        result = {}

        if mode == 'monthly':
            for mo in range(1, 13):
                try:
                    img = col.filter(ee.Filter.calendarRange(mo, mo, 'month')).mean().multiply(scale)
                    v   = img.sample(pt, 4638).first().get(var).getInfo()
                    if v is not None:
                        result[str(mo)] = round(float(v), 4)
                except: pass
        else:
            baseline = col.filterDate(BASELINE_START, BASELINE_END).mean().multiply(scale)
            for yr in TC_YEARS:
                try:
                    annual = col.filterDate(f"{yr}-01-01", f"{yr+1}-01-01").mean().multiply(scale)
                    v = annual.subtract(baseline).sample(pt, 4638).first().get(var).getInfo()
                    if v is not None:
                        result[str(yr)] = round(float(v), 4)
                except: pass

        img = render_point(result, lat, lon, mode)
        return {"img": img, "lat": lat, "lon": lon}
    except Exception as e:
        raise HTTPException(500, str(e))



@app.get("/api/provinces/{code}")
def api_provinces(code: str,
                  var:    str = Query(default='soil'),
                  mode:   str = Query(default='monthly'),
                  period: str = Query(default='all')):
    """
    İl/eyalet bazlı TerraClimate verisi — Leaflet choropleth için GeoJSON döner.
    mode: monthly | anomaly
    period: 'all' (tüm yıllar ortalaması) | ay numarası | yıl
    """
    c = code.upper()
    if var not in TC_VARS:
        var = 'soil'
    try:
        records = fetch_provinces(c, var, mode, period)
        vinfo   = TC_VARS[var]

        # Değerleri normalize et — frontend için min/max/colorscale
        vals = [r['value'] for r in records if r['value'] is not None]
        if not vals:
            raise ValueError("No province data returned")

        vmin = min(vals)
        vmax_abs = max(abs(vmin), abs(max(vals))) if mode == 'anomaly' else max(vals)

        return {
            "var":      var,
            "label":    vinfo['label'],
            "unit":     vinfo['unit'],
            "mode":     mode,
            "period":   period,
            "vmin":     round(vmin, 3),
            "vmax":     round(max(vals), 3),
            "vmax_abs": round(vmax_abs, 3),
            "cmap":     vinfo['cmap'],
            "provinces": records
        }
    except Exception as e:
        import traceback
        print(f"[provinces error] {c}/{var}/{mode}/{period}: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, str(e))


@app.get("/", response_class=HTMLResponse)
def index(): return HTML

# ── HTML ──────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Water Storage Monitor · TerraClimate</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@200;300;400&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
*,*::before,*::after{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#F4F1EC;--bg2:#EDE9E2;--bg3:#E4E0D8;
  --ink:#1A1814;--ink2:#3D3A35;--ink3:#7A756E;--ink4:#B5B0A8;
  --line:rgba(26,24,20,.07);--line2:rgba(26,24,20,.13);
  --mono:'DM Mono',monospace;--sans:'DM Sans',sans-serif;
}
html,body{height:100%;background:var(--bg);color:var(--ink);font-family:var(--sans);overflow:hidden}

/* HERO */
#hero{position:fixed;inset:0;z-index:10;transition:transform .9s cubic-bezier(.77,0,.18,1),opacity .5s ease;will-change:transform,opacity}
#hero.away{transform:translateY(-100vh) scale(.97);opacity:0;pointer-events:none}
#map{width:100%;height:100%}
.hero-grad{position:absolute;inset:0;pointer-events:none;background:linear-gradient(to bottom,rgba(244,241,236,0) 50%,rgba(244,241,236,.65) 100%)}
.hero-label{position:absolute;bottom:52px;left:56px;pointer-events:none}
.hero-label h1{font-family:var(--mono);font-size:clamp(1.6rem,2.6vw,2.5rem);font-weight:300;letter-spacing:-.02em;color:var(--ink);line-height:1.05}
.hero-label p{font-family:var(--mono);font-size:.6rem;color:var(--ink3);letter-spacing:.2em;text-transform:uppercase;margin-top:10px}
.hero-cue{position:absolute;bottom:52px;right:56px;font-family:var(--mono);font-size:.57rem;color:var(--ink3);letter-spacing:.18em;text-transform:uppercase;pointer-events:none;animation:pulse 2.6s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:.3}50%{opacity:1}}
.tip{position:absolute;pointer-events:none;z-index:300;background:var(--ink);color:var(--bg);font-family:var(--mono);font-size:.58rem;letter-spacing:.1em;text-transform:uppercase;padding:4px 11px;border-radius:2px;opacity:0;transition:opacity .12s;white-space:nowrap}
.tip.on{opacity:1}

/* PANEL */
#panel{position:fixed;inset:0;z-index:5;display:flex;flex-direction:column;background:var(--bg);overflow:hidden;transform:translateY(100vh);transition:transform .9s cubic-bezier(.77,0,.18,1);will-change:transform}
#panel.on{transform:translateY(0)}

/* TOP BAR */
.pbar{height:50px;flex-shrink:0;display:flex;align-items:center;padding:0 28px;border-bottom:1px solid var(--line2);background:var(--bg);z-index:20;gap:0}
.pbar-logo{font-family:var(--mono);font-size:.68rem;font-weight:500;letter-spacing:.22em;text-transform:uppercase;color:var(--ink);margin-right:24px}
.psep{width:1px;height:16px;background:var(--line2);margin:0 16px}
.pbar-country{font-family:var(--mono);font-size:.62rem;color:var(--ink3);letter-spacing:.1em;text-transform:uppercase;flex:1}
.pbar-country b{color:var(--ink);font-weight:500}
.back{font-family:var(--mono);font-size:.57rem;letter-spacing:.14em;text-transform:uppercase;color:var(--ink3);cursor:pointer;background:none;border:1px solid var(--line2);padding:5px 13px;border-radius:2px;transition:all .15s}
.back:hover{color:var(--ink);background:var(--bg2)}

/* BODY */
.pbody{flex:1;display:grid;grid-template-columns:1fr 290px;overflow:hidden;min-height:0}

/* LEFT */
.left{display:flex;flex-direction:column;border-right:1px solid var(--line2);overflow:hidden;min-height:0}
.tabs{display:flex;border-bottom:1px solid var(--line2);flex-shrink:0}
.tab{font-family:var(--mono);font-size:.56rem;letter-spacing:.16em;text-transform:uppercase;color:var(--ink4);cursor:pointer;padding:10px 22px;border:none;background:none;border-bottom:1.5px solid transparent;margin-bottom:-1px;transition:all .15s}
.tab:hover{color:var(--ink3)}.tab.on{color:var(--ink);border-bottom-color:var(--ink)}
.chart-box{flex:1;min-height:0;position:relative;overflow:hidden;display:flex;align-items:center;justify-content:center;background:var(--bg)}
.cimg{max-width:100%;max-height:100%;object-fit:contain;display:none;opacity:0;transition:opacity .3s}
.cimg.on{display:block;opacity:1}
.loader{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;background:var(--bg)}
.loader.off{display:none}
.ring{width:24px;height:24px;border:1.5px solid var(--line2);border-top-color:var(--ink);border-radius:50%;animation:spin .75s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.loader-lbl{font-family:var(--mono);font-size:.52rem;color:var(--ink4);letter-spacing:.2em;text-transform:uppercase}
.chart-empty{font-family:var(--mono);font-size:.58rem;color:var(--ink4);letter-spacing:.1em;text-align:center;line-height:2.2;display:none}

/* CONTROLS */
.ctrls{height:46px;flex-shrink:0;display:flex;align-items:center;gap:12px;padding:0 22px;border-top:1px solid var(--line2);background:var(--bg)}
.mode{display:flex}
.mbtn{font-family:var(--mono);font-size:.56rem;letter-spacing:.12em;text-transform:uppercase;color:var(--ink4);padding:4px 11px;border:1px solid var(--line2);background:none;cursor:pointer;transition:all .15s}
.mbtn:first-child{border-radius:2px 0 0 2px}.mbtn:last-child{border-radius:0 2px 2px 0;border-left:none}
.mbtn.on{color:var(--bg);background:var(--ink);border-color:var(--ink)}
.pdrop{position:relative}.vdrop{position:relative}
.ptrig{font-family:var(--mono);font-size:.59rem;letter-spacing:.1em;text-transform:uppercase;color:var(--ink3);padding:4px 10px;border:1px solid transparent;background:none;cursor:pointer;border-radius:2px;display:flex;align-items:center;gap:6px;transition:all .2s;white-space:nowrap}
.ptrig:hover,.ptrig.on{color:var(--ink);background:var(--bg2);border-color:var(--line2)}
.ptrig .arr{font-size:.5rem;transition:transform .2s}.ptrig.on .arr{transform:rotate(180deg)}
.plist{position:absolute;bottom:calc(100% + 6px);left:0;background:var(--bg);border:1px solid var(--line2);border-radius:3px;min-width:130px;max-height:220px;overflow-y:auto;z-index:500;box-shadow:0 -8px 24px rgba(26,24,20,.07);display:none;scrollbar-width:thin;scrollbar-color:var(--line2) transparent}
.plist.on{display:block}
.popt{font-family:var(--mono);font-size:.59rem;letter-spacing:.08em;padding:7px 14px;cursor:pointer;color:var(--ink3);transition:background .1s}
.popt:hover{background:var(--bg2);color:var(--ink)}.popt.sel{color:var(--ink);font-weight:500}
.cinfo{font-family:var(--mono);font-size:.52rem;color:var(--ink4);letter-spacing:.1em;margin-left:auto;text-align:right}

/* CHAT */
.chat{display:flex;flex-direction:column;overflow:hidden;min-height:0}
.chat-hd{padding:13px 16px 10px;border-bottom:1px solid var(--line);flex-shrink:0}
.chat-hd-t{font-family:var(--mono);font-size:.57rem;letter-spacing:.18em;text-transform:uppercase;color:var(--ink3)}
.apibox{margin:10px 14px;padding:10px 12px;border:1px solid var(--line2);border-radius:3px;flex-shrink:0}
.apilbl{font-family:var(--mono);font-size:.5rem;letter-spacing:.14em;text-transform:uppercase;color:var(--ink4);margin-bottom:5px}
.apirow{display:flex;gap:5px}
.apiin{flex:1;font-family:var(--mono);font-size:.59rem;border:1px solid var(--line2);background:var(--bg);color:var(--ink);padding:5px 8px;border-radius:2px;outline:none}
.apiin:focus{border-color:var(--ink4)}
.apisave{font-family:var(--mono);font-size:.55rem;letter-spacing:.1em;text-transform:uppercase;background:var(--ink);color:var(--bg);border:none;padding:0 10px;border-radius:2px;cursor:pointer}
.msgs{flex:1;overflow-y:auto;padding:10px 14px;display:flex;flex-direction:column;gap:8px;min-height:0;scrollbar-width:thin;scrollbar-color:var(--line2) transparent}
.msg{display:flex;flex-direction:column;gap:2px;animation:fu .22s ease}
@keyframes fu{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:translateY(0)}}
.mrole{font-family:var(--mono);font-size:.47rem;letter-spacing:.16em;text-transform:uppercase;color:var(--ink4)}
.msg.u .mrole{color:var(--ink3)}
.mbody{font-family:var(--sans);font-size:.68rem;line-height:1.55;color:var(--ink2);background:var(--bg2);border-radius:3px;padding:7px 10px}
.msg.u .mbody{background:var(--ink);color:var(--bg)}
.mbody.t::after{content:'▋';animation:blink .8s infinite;color:var(--ink4)}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}
.cinput-row{padding:10px 14px;border-top:1px solid var(--line);display:flex;gap:6px;flex-shrink:0}
.cin{flex:1;font-family:var(--sans);font-size:.66rem;border:1px solid var(--line2);background:var(--bg);color:var(--ink);padding:7px 10px;border-radius:3px;outline:none;resize:none;line-height:1.4;transition:border-color .15s}
.cin:focus{border-color:var(--ink4)}
.csend{font-family:var(--mono);font-size:.57rem;letter-spacing:.1em;text-transform:uppercase;background:var(--ink);color:var(--bg);border:none;padding:0 12px;border-radius:3px;cursor:pointer;align-self:flex-end;height:30px;transition:opacity .15s}
.csend:hover{opacity:.8}.csend:disabled{opacity:.3;cursor:not-allowed}

/* Leaflet */
.leaflet-container{background:var(--bg)!important}
.leaflet-tile{filter:saturate(.15) brightness(1.04) contrast(1.05)}
.leaflet-control-zoom a{background:var(--bg)!important;color:var(--ink3)!important;border-color:var(--line2)!important;font-family:var(--mono)!important}

/* PROVINCE MAP */
#provBox{flex:1;min-height:0;position:relative;display:none}
#provBox.on{display:block}
#provMap{width:100%;height:100%}
.prov-tooltip{background:var(--ink)!important;color:var(--bg)!important;font-family:var(--mono)!important;font-size:.58rem!important;letter-spacing:.06em!important;border:none!important;border-radius:2px!important;padding:4px 10px!important;box-shadow:none!important;white-space:nowrap}
.prov-tooltip::before{border-top-color:var(--ink)!important}
#provLegend{position:absolute;bottom:12px;left:12px;z-index:600;background:var(--bg);border:1px solid var(--line2);border-radius:3px;padding:8px 12px;font-family:var(--mono);font-size:.52rem;color:var(--ink3);pointer-events:none}
#provLegend canvas{display:block;width:120px;height:8px;border-radius:2px;margin-bottom:4px}
.prov-legend-row{display:flex;justify-content:space-between;gap:30px}
</style>
</head>
<body>
<div id="hero">
  <div id="map"></div>
  <div class="hero-grad"></div>
  <div class="hero-label">
    <h1>Water<br>Storage</h1>
    <p>TerraClimate · 1990–2023 · 4km</p>
  </div>
  <div class="hero-cue" id="heroCue">Click country · zoom in for provinces</div>
  <div class="tip" id="tip"></div>
</div>

<div id="panel">
  <div class="pbar">
    <div class="pbar-logo">Water</div>
    <div class="psep"></div>
    <div class="pbar-country"><b id="pcname">—</b></div>
    <button class="back" onclick="goBack()">← World Map</button>
  </div>
  <div class="pbody">
    <div class="left">
      <div class="tabs">
        <button class="tab on" data-t="heatmap"    onclick="switchTab('heatmap')">Heat Map</button>
        <button class="tab"    data-t="provinces"  onclick="switchTab('provinces')">Provinces</button>
        <button class="tab"    data-t="timeseries" onclick="switchTab('timeseries')">Time Series</button>
      </div>
      <div class="chart-box" id="chartBox">
        <div class="loader" id="ldr"><div class="ring"></div><div class="loader-lbl" id="ldrLbl">Loading</div></div>
        <div class="chart-empty" id="cempty">Select a period below</div>
        <img class="cimg" id="cimg" alt="">
      </div>
      <div id="provBox">
        <div id="provMap"></div>
        <div id="provLegend" style="display:none">
          <canvas id="provLegendCanvas"></canvas>
          <div class="prov-legend-row">
            <span id="provLegMin"></span>
            <span id="provLegUnit"></span>
            <span id="provLegMax"></span>
          </div>
        </div>
      </div>
      <div class="ctrls">
        <div class="mode">
          <button class="mbtn on" data-m="monthly" onclick="setMode('monthly')">Month</button>
          <button class="mbtn"    data-m="anomaly" onclick="setMode('anomaly')">Year</button>
        </div>
        <div class="psep"></div>
        <div class="pdrop" id="pdrop">
          <button class="ptrig" id="ptrig" onclick="toggleP()">
            <span id="plbl">January</span><span class="arr">▾</span>
          </button>
          <div class="plist" id="plist"></div>
        </div>
        <div class="psep"></div>
        <div class="vdrop" id="vdrop">
          <button class="ptrig" id="vtrig" onclick="toggleV()">
            <span id="vlbl">Toprak Nemi</span><span class="arr">▾</span>
          </button>
          <div class="plist" id="vlist"></div>
        </div>
        <div class="psep"></div>
        <div class="cinfo" id="cinfo">Click inside country to inspect a point</div>
      </div>
    </div>

    <div class="chat">
      <div class="chat-hd"><div class="chat-hd-t">AI Assistant</div></div>
      <div class="apibox" id="apibox">
        <div class="apilbl">OpenAI Key</div>
        <div class="apirow">
          <input class="apiin" id="apiin" type="password" placeholder="sk-...">
          <button class="apisave" onclick="saveKey()">Save</button>
        </div>
      </div>
      <div class="msgs" id="msgs">
        <div class="msg"><div class="mrole">AI</div>
          <div class="mbody">Click any country on the world map to load GRACE water storage data. I can help you interpret droughts, floods, La Niña cycles, and long-term trends.</div>
        </div>
      </div>
      <div class="cinput-row">
        <textarea class="cin" id="cin" rows="2" placeholder="Ask about the data..."
          onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();send()}"></textarea>
        <button class="csend" id="csend" onclick="send()">Send</button>
      </div>
    </div>
  </div>
</div>

<script>
const MNL=['January','February','March','April','May','June','July','August','September','October','November','December'];
const YEARS=Array.from({length:34},(_,i)=>String(1990+i));
const VARS={soil:{label:'Toprak Nemi',unit:'mm'},def:{label:'Su Açığı',unit:'mm'},pdsi:{label:'Palmer Kuraklık İndeksi',unit:''}};
const S={code:null,mode:'monthly',period:'1',tab:'heatmap',var:'soil',
         apiKey:localStorage.getItem('g_oai')||null,ctx:'',
         countries:[],map:null,marker:null,bboxCache:{},
         provMonth:'all'};  // provinces monthly ay seçimi

async function init(){
  S.map=L.map('map',{zoomControl:true,attributionControl:false});
  L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png',{maxZoom:18}).addTo(S.map);
  S.map.setView([20,10],2);

  S.countries=await fetch('/api/countries').then(r=>r.json());

  // Map click
  S.map.on('click', e=>{
    const{lat,lng}=e.latlng;
    let clickedCode=null, clickedName=null;
    for(const c of S.countries){
      const bb=S.bboxCache[c.code];
      if(bb&&lat>=bb[0]&&lat<=bb[2]&&lng>=bb[1]&&lng<=bb[3]){
        clickedCode=c.code; clickedName=c.name; break;
      }
    }
    if(clickedCode){
      if(S.code===clickedCode) analyzePoint(lat,lng);
      else loadCountry(clickedCode, clickedName);
    } else if(S.code){
      analyzePoint(lat,lng);
    }
  });

  S.map.on('mousemove',e=>{
    const{lat,lng}=e.latlng;
    for(const c of S.countries){
      const bb=S.bboxCache[c.code];
      if(bb&&lat>=bb[0]&&lat<=bb[2]&&lng>=bb[1]&&lng<=bb[3]){
        showTip(c.name,e.originalEvent); return;
      }
    }
    hideTip();
  });
  S.map.on('mouseout',hideTip);


    // Eager: fetch ALL country bboxes in background so every country is clickable
  prefetchAllBboxes();
  buildVars();

  if(S.apiKey) document.getElementById('apibox').style.display='none';
}

async function prefetchAllBboxes(){
  // Fetch in small parallel batches; priority: common countries first
  const priority=['TR','DE','AU','US','GB','FR','RU','CN','IN','BR',
                  'CA','JP','KR','MX','ID','SA','IR','EG','NG','ZA',
                  'AR','PK','UA','PL','IT','ES','KZ'];
  const rest=S.countries.map(c=>c.code).filter(c=>!priority.includes(c));
  const ordered=[...priority,...rest];
  const BATCH=6;
  for(let i=0;i<ordered.length;i+=BATCH){
    await Promise.all(ordered.slice(i,i+BATCH).map(fetchBbox));
  }
}

async function fetchBbox(code){
  if(S.bboxCache[code]) return;
  try{
    const r=await fetch(`/api/bbox/${code}`).then(r=>r.json());
    if(r&&r.bbox) S.bboxCache[code]=r.bbox;
  }catch(e){}
}

const tip=document.getElementById('tip');
function showTip(name,e){tip.textContent=name;tip.classList.add('on');tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY-30)+'px'}
function hideTip(){tip.classList.remove('on')}

function goBack(){
  document.getElementById('hero').classList.remove('away');
  document.getElementById('panel').classList.remove('on');
  S.code=null;
}

async function loadCountry(code,name){
  S.code=code;
  document.getElementById('hero').classList.add('away');
  document.getElementById('panel').classList.add('on');
  document.getElementById('pcname').textContent=name||code;

  // Fetch bbox if not cached
  if(!S.bboxCache[code]){
    showLoad(true,'Loading country...');
    const r=await fetch(`/api/bbox/${code}`).then(r=>r.json()).catch(()=>null);
    if(r){
      S.bboxCache[code]=r.bbox;
      setTimeout(()=>S.map.setView(r.center, getZoom(r.bbox)),300);
    }
  } else {
    const bb=S.bboxCache[code];
    const center=[(bb[0]+bb[2])/2,(bb[1]+bb[3])/2];
    setTimeout(()=>S.map.setView(center,getZoom(bb)),300);
  }

  buildPeriods();
  renderCurrent();
}

function getZoom(bb){
  const latR=bb[2]-bb[0], lonR=bb[3]-bb[1];
  const maxR=Math.max(latR,lonR);
  if(maxR>50) return 3;
  if(maxR>20) return 4;
  if(maxR>10) return 5;
  return 6;
}

function setMode(m){
  S.mode=m;
  document.querySelectorAll('.mbtn').forEach(b=>b.classList.toggle('on',b.dataset.m===m));
  updateDropVisibility();
  buildPeriods();
  renderCurrent();
}

// pdrop: anomaly'de her zaman, monthly+provinces'ta da göster
function updateDropVisibility(){
  const showDrop = S.mode==='anomaly' || S.tab==='provinces';
  document.getElementById('pdrop').style.display = showDrop ? '' : 'none';
}

function buildPeriods(){
  const list=document.getElementById('plist');
  list.innerHTML='';
  if(S.mode==='monthly'){
    // Provinces tabında: ay seçici
    if(S.tab==='provinces'){
      if(!S.provMonth) S.provMonth='all';
      const monthOpts=[{v:'all',l:'All Months'},...MNL.map((n,i)=>({v:String(i+1),l:n}))];
      monthOpts.forEach(o=>{
        const d=document.createElement('div');
        d.className='popt'+(o.v===S.provMonth?' sel':'');
        d.textContent=o.l;
        d.onclick=()=>{
          S.provMonth=o.v;
          document.querySelectorAll('#plist .popt').forEach(el=>el.classList.toggle('sel',el.textContent===o.l));
          document.getElementById('plbl').textContent=o.l;
          closeP();
          renderProvinces();
        };
        list.appendChild(d);
      });
      const cur=monthOpts.find(o=>o.v===S.provMonth);
      document.getElementById('plbl').textContent=cur?cur.l:'All Months';
      return;
    }
    // Diğer tablarda: dropdown yok, period sıfırla
    S.period='all';
    document.getElementById('plbl').textContent='All Months';
    return;
  }
  // anomaly: yıl listesi
  const opts=YEARS.map(y=>({v:y,l:y}));
  if(!S.period||S.period==='all') S.period=YEARS[YEARS.length-1];
  opts.forEach(o=>{
    const d=document.createElement('div');
    d.className='popt'+(o.v===S.period?' sel':'');
    d.textContent=o.l;
    d.onclick=()=>{
      S.period=o.v;
      document.querySelectorAll('#plist .popt').forEach(el=>el.classList.toggle('sel',el.textContent===o.l));
      document.getElementById('plbl').textContent=o.l;
      closeP();
      if(S.tab==='heatmap') renderHeatmap(); else if(S.tab==='provinces') renderProvinces();
    };
    list.appendChild(d);
  });
  document.getElementById('plbl').textContent=S.period;
}

function buildVars(){
  const list=document.getElementById('vlist');
  list.innerHTML='';
  Object.entries(VARS).forEach(([k,v])=>{
    const d=document.createElement('div');
    d.className='popt'+(k===S.var?' sel':'');
    d.textContent=v.label;
    d.onclick=()=>{
      S.var=k;
      document.querySelectorAll('#vlist .popt').forEach(el=>el.classList.toggle('sel',el.textContent===v.label));
      document.getElementById('vlbl').textContent=v.label;
      closeV();
      renderCurrent();
    };
    list.appendChild(d);
  });
  document.getElementById('vlbl').textContent=VARS[S.var].label;
}
function toggleV(){
  const t=document.getElementById('vtrig'),l=document.getElementById('vlist');
  const o=l.classList.toggle('on');t.classList.toggle('on',o);
}
function closeV(){
  document.getElementById('vlist').classList.remove('on');
  document.getElementById('vtrig').classList.remove('on');
}
document.addEventListener('click',e=>{
  const vd=document.getElementById('vdrop');
  if(vd&&!vd.contains(e.target))closeV();
});

function toggleP(){
  const t=document.getElementById('ptrig'),l=document.getElementById('plist');
  const o=l.classList.toggle('on');t.classList.toggle('on',o);
}
function closeP(){
  document.getElementById('plist').classList.remove('on');
  document.getElementById('ptrig').classList.remove('on');
}
document.addEventListener('click',e=>{
  if(!document.getElementById('pdrop').contains(e.target))closeP();
});

function switchTab(t){
  S.tab=t;
  document.querySelectorAll('.tab').forEach(b=>b.classList.toggle('on',b.dataset.t===t));
  const isProvince = t==='provinces';
  document.getElementById('chartBox').style.display = isProvince?'none':'';
  document.getElementById('provBox').classList.toggle('on', isProvince);
  updateDropVisibility();
  buildPeriods();   // provinces'ta ay, diğerlerinde yıl/gizli
  renderCurrent();
}
function renderCurrent(){
  if(S.tab==='heatmap') renderHeatmap();
  else if(S.tab==='provinces') renderProvinces();
  else renderTimeseries();
}

function showLoad(show,msg='Rendering'){
  document.getElementById('ldrLbl').textContent=msg;
  document.getElementById('ldr').classList.toggle('off',!show);
  if(show){
    document.getElementById('cimg').classList.remove('on');
    document.getElementById('cempty').style.display='none';
    // chartBox'ın görünür olduğundan emin ol
    document.getElementById('chartBox').style.display='flex';
  }
}
function setImg(b64){
  const img=document.getElementById('cimg');
  img.onload=()=>{img.classList.add('on');document.getElementById('ldr').classList.add('off');};
  img.src=`data:image/png;base64,${b64}`;
}

async function renderHeatmap(){
  if(!S.code) return;
  const cname=document.getElementById('pcname').textContent;
  if(S.mode==='monthly'){
    showLoad(true,`${VARS[S.var].label} yükleniyor...`);
    document.getElementById('cinfo').textContent=`${VARS[S.var].label} — aylık ortalama (1990–2023)`;
    const r=await fetch(`/api/render/heatmap/${S.code}/monthly/all?var=${S.var}`)
      .then(r=>r.json()).catch(()=>null);
    if(r?.img){
      setImg(r.img);
      S.ctx=`Country: ${cname}. Mode: monthly averages 2003-2023. `+
            (r.stats?`LWE range: ${r.stats.min} to ${r.stats.max} cm, mean: ${r.stats.mean} cm.`:'');
      document.getElementById('cinfo').textContent='Monthly averages (2003–2023)';
    } else{showLoad(false);document.getElementById('cinfo').textContent='Error loading data';}
  } else {
    if(!S.period||S.period==='all') S.period=YEARS[YEARS.length-1];
    showLoad(true,`${S.period} anomalisi yükleniyor...`);
    const r=await fetch(`/api/render/heatmap/${S.code}/anomaly/${S.period}?var=${S.var}`)
      .then(r=>r.json()).catch(()=>null);
    if(r?.img){
      setImg(r.img);
      S.ctx=`Country: ${cname}. Year: ${S.period} anomaly vs 2004-2013 baseline. `+
            (r.stats?`Anomaly range: ${r.stats.min} to ${r.stats.max} cm, mean: ${r.stats.mean} cm.`:'');
      document.getElementById('cinfo').textContent=`${VARS[S.var].label} — ${S.period} anomalisi (1990-2019 baseline)`;
    } else{showLoad(false);document.getElementById('cinfo').textContent='Error loading data';}
  }
}

async function renderTimeseries(){
  if(!S.code)return;
  showLoad(true,'Fetching time series...');
  const r=await fetch(`/api/render/timeseries/${S.code}?var=${S.var}`)
    .then(r=>r.json()).catch(()=>null);
  if(r?.img){
    setImg(r.img);
    S.ctx=`Country: ${document.getElementById('pcname').textContent}. `+
          `Time series: annual LWE anomaly vs 2004-2013 baseline.`+
          (r.trend ? ` Trend: ${r.trend}.` : '');
  } else showLoad(false);
}

// ── PROVINCE CHOROPLETH ──────────────────────────────────────────────
// Colourscale definitions matching Python TC_VARS cmaps
const CMAPS = {
  YlGnBu: ['#ffffd9','#edf8b1','#c7e9b4','#7fcdbb','#41b6c4','#1d91c0','#225ea8','#0c2c84'],
  YlOrRd: ['#ffffcc','#ffeda0','#fed976','#feb24c','#fd8d3c','#fc4e2a','#e31a1c','#800026'],
  RdYlBu: ['#d73027','#f46d43','#fdae61','#fee090','#e0f3f8','#abd9e9','#74add1','#4575b4'],
};

function cmapColor(t, name){
  // t in [0,1]
  const colors = CMAPS[name] || CMAPS.YlGnBu;
  const n = colors.length - 1;
  const i = Math.min(Math.floor(t * n), n - 1);
  const f = t * n - i;
  return lerpColor(colors[i], colors[i+1], f);
}

function lerpColor(a, b, t){
  const parse = h => [parseInt(h.slice(1,3),16), parseInt(h.slice(3,5),16), parseInt(h.slice(5,7),16)];
  const [ar,ag,ab] = parse(a), [br,bg,bb] = parse(b);
  const r = Math.round(ar + (br-ar)*t);
  const g = Math.round(ag + (bg-ag)*t);
  const bl= Math.round(ab + (bb-ab)*t);
  return `rgb(${r},${g},${bl})`;
}

function drawLegend(vmin, vmax, unit, cmap, isAnomaly){
  const canvas = document.getElementById('provLegendCanvas');
  const ctx    = canvas.getContext('2d');
  canvas.width = 240; canvas.height = 16;
  const grad   = ctx.createLinearGradient(0,0,240,0);
  for(let i=0;i<=10;i++) grad.addColorStop(i/10, cmapColor(i/10, cmap));
  ctx.fillStyle = grad;
  ctx.fillRect(0,0,240,16);
  document.getElementById('provLegMin').textContent = vmin.toFixed(1);
  document.getElementById('provLegMax').textContent = vmax.toFixed(1);
  document.getElementById('provLegUnit').textContent = unit || '';
  document.getElementById('provLegend').style.display = '';
}

let provLayer = null;
let provMapInst = null;

async function renderProvinces(){
  if(!S.code) return;
  document.getElementById('cinfo').textContent = 'İller yükleniyor...';

  // Init province Leaflet map once
  if(!provMapInst){
    provMapInst = L.map('provMap',{zoomControl:true,attributionControl:false});
    L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png',{maxZoom:18}).addTo(provMapInst);
  }

  const mode   = S.mode;
  // monthly modda: provMonth seçilmişse monthly_single, değilse monthly
  let apiMode = mode, apiPeriod = S.period || YEARS[YEARS.length-1];
  if(mode === 'monthly'){
    if(S.provMonth && S.provMonth !== 'all'){
      apiMode   = 'monthly_single';
      apiPeriod = S.provMonth;
    } else {
      apiMode   = 'monthly';
      apiPeriod = 'all';
    }
  }
  const url = `/api/provinces/${S.code}?var=${S.var}&mode=${apiMode}&period=${apiPeriod}`;

  const r = await fetch(url).then(r=>r.json()).catch(()=>null);
  if(!r || !r.provinces){
    document.getElementById('cinfo').textContent = 'Province data error';
    return;
  }

  // Remove old layer
  if(provLayer) provMapInst.removeLayer(provLayer);

  const {vmin, vmax, vmax_abs, cmap, provinces, unit, label} = r;
  const isAnomaly = apiMode !== 'monthly' && apiMode !== 'monthly_single';

  provLayer = L.geoJSON({type:'FeatureCollection', features: provinces.map(p=>({
    type:'Feature',
    geometry: p.geometry,
    properties: {name: p.name, value: p.value}
  }))}, {
    style: feat => {
      const v = feat.properties.value;
      if(v === null || v === undefined) return {fillColor:'#ccc',fillOpacity:.4,color:'#999',weight:.5};
      let t;
      if(isAnomaly){
        t = (v + vmax_abs) / (2 * vmax_abs);  // center 0 → 0.5
      } else {
        t = vmax > vmin ? (v - vmin) / (vmax - vmin) : 0.5;
      }
      t = Math.max(0, Math.min(1, t));
      // RdYlBu is reversed for anomaly (negative=blue=wet, positive=red=dry deficit)
      if(isAnomaly && cmap === 'RdYlBu') t = 1 - t;
      return {
        fillColor:   cmapColor(t, cmap),
        fillOpacity: 0.75,
        color:       'rgba(26,24,20,.35)',
        weight:      0.8
      };
    },
    onEachFeature: (feat, layer) => {
      const {name, value} = feat.properties;
      const valStr = value !== null ? `${value.toFixed(2)} ${unit}` : 'N/A';
      layer.bindTooltip(`<b>${name}</b><br>${label}: ${valStr}`,
        {className:'prov-tooltip', sticky:true, direction:'top'});
      layer.on('mouseover', function(){ this.setStyle({fillOpacity:.95, weight:1.5}); });
      layer.on('mouseout',  function(){ provLayer.resetStyle(this); });
    }
  }).addTo(provMapInst);

  // Zoom to country
  try{ provMapInst.fitBounds(provLayer.getBounds(), {padding:[20,20]}); }catch(e){}

  // Legend
  drawLegend(isAnomaly ? -vmax_abs : vmin, isAnomaly ? vmax_abs : vmax, unit, cmap, isAnomaly);

  const cname = document.getElementById('pcname').textContent;
  document.getElementById('cinfo').textContent = `${label} — ${provinces.length} bölge`;
  S.ctx = `Country: ${cname}. Province choropleth: ${label}. Min:${vmin} Max:${vmax} ${unit}.`;

  // Invalidate size (panel may have just appeared)
  setTimeout(()=>provMapInst.invalidateSize(), 100);
}

async function analyzePoint(lat,lng){
  if(!S.code)return;
  if(S.marker)S.map.removeLayer(S.marker);
  S.marker=L.circleMarker([lat,lng],{radius:5,color:'#1A1814',fillColor:'#1A1814',fillOpacity:.9,weight:1.5}).addTo(S.map);
  showLoad(true,'Analyzing point...');
  document.getElementById('cinfo').textContent=`${lat.toFixed(2)}°, ${lng.toFixed(2)}°`;
  const r=await fetch(`/api/render/point/${S.code}/${S.mode}?lat=${lat}&lon=${lng}&var=${S.var}`)
    .then(r=>r.json()).catch(()=>null);
  if(r?.img){
    setImg(r.img);
    document.getElementById('cinfo').textContent=`${lat.toFixed(2)}°, ${lng.toFixed(2)}°`;
    S.ctx+=` | Point:${lat.toFixed(2)}°N ${lng.toFixed(2)}°E`;
  }else{showLoad(false);document.getElementById('cinfo').textContent='No data here';}
}

function saveKey(){
  const k=document.getElementById('apiin').value.trim();
  if(!k.startsWith('sk-')){alert('Invalid key');return;}
  S.apiKey=k;localStorage.setItem('g_oai',k);
  document.getElementById('apibox').style.display='none';
  addMsg('ai','Key saved. Ask me about the water storage data.');
}
const hist=[];
async function send(){
  const inp=document.getElementById('cin'),msg=inp.value.trim();
  if(!msg)return;
  if(!S.apiKey){addMsg('ai','Please enter your OpenAI API key first.');return;}
  inp.value='';addMsg('user',msg);document.getElementById('csend').disabled=true;
  hist.push({role:'user',content:msg});
  const tid=addMsg('ai','',true);
  try{
    const r=await fetch('https://api.openai.com/v1/chat/completions',{
      method:'POST',headers:{'Content-Type':'application/json','Authorization':`Bearer ${S.apiKey}`},
      body:JSON.stringify({model:'gpt-4o-mini',max_tokens:260,messages:[
        {role:'system',content:`You are a hydrology expert. Context: ${S.ctx}. LWE=Liquid Water Equivalent cm. Answer 2-4 sentences. Reference: La Niña 2010-12/2020-23, El Niño, Black Summer AU 2019-20, Turkey 2021 drought.`},
        ...hist.slice(-8)
      ]})
    });
    const d=await r.json();const rep=d.choices?.[0]?.message?.content||'No response.';
    hist.push({role:'assistant',content:rep});updMsg(tid,rep);
  }catch(e){updMsg(tid,'Connection error.');}
  document.getElementById('csend').disabled=false;
}
let mc=0;
function addMsg(role,text,typing=false){
  const id='m'+(++mc),el=document.createElement('div');
  el.className='msg'+(role==='user'?' u':'');el.id=id;
  el.innerHTML=`<div class="mrole">${role==='user'?'You':'AI'}</div><div class="mbody${typing?' t':''}">${text}</div>`;
  const c=document.getElementById('msgs');c.appendChild(el);c.scrollTop=c.scrollHeight;return id;
}
function updMsg(id,t){const el=document.querySelector(`#${id} .mbody`);if(el){el.textContent=t;el.classList.remove('t');}}
init();
</script>
</body>
</html>
"""

# ── ENTRY ─────────────────────────────────────────────────────────────
if __name__=="__main__":
    import sys
    use_ngrok="--ngrok" in sys.argv
    if use_ngrok:
        if not NGROK_TOKEN: print("❌ Set NGROK_TOKEN"); sys.exit(1)
        ngrok.set_auth_token(NGROK_TOKEN)
        def _s(): uvicorn.run(app,host="0.0.0.0",port=8000,log_level="warning")
        threading.Thread(target=_s,daemon=True).start()
        time.sleep(2)
        tunnel = ngrok.connect(8000)
        url = tunnel.public_url  # ya da str(tunnel) da çalışır
        print(f"\n🌍  {url}\n📖  {url}/docs\n")
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt: ngrok.kill()
    else:
        print("\n🚀  http://localhost:8000\n")
        uvicorn.run(app,host="0.0.0.0",port=8000,log_level="info")
