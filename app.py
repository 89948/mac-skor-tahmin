import math
import requests
import streamlit as st
from openai import OpenAI

API_BASE = "https://v3.football.api-sports.io"
NVIDIA_BASE = "https://integrate.api.nvidia.com/v1"

st.set_page_config(page_title="Futbol Analiz & Tahmin", page_icon="⚽", layout="wide")
st.title("⚽ Futbol Analiz & Tahmin Motoru")
st.caption("Fikstür + takım gol istatistikleri + Poisson olasılıkları + isteğe bağlı NVIDIA AI yorumu")

with st.sidebar:
    st.header("API bağlantıları")
    football_key = st.text_input("API-Football anahtarı", type="password")
    nvidia_key = st.text_input("NVIDIA API anahtarı", type="password")
    nvidia_model = st.text_input("NVIDIA model adı", value="meta/llama-3.3-70b-instruct")
    st.caption("Anahtarlar yalnızca bu oturumda kullanılır; dosyaya kaydedilmez.")
    st.divider()
    st.markdown("**Resmî platformlar**")
    st.markdown("- [NVIDIA API Catalog](https://build.nvidia.com/)")
    st.markdown("- [API-Football](https://www.api-football.com/)")

def poisson_pmf(k, lam):
    return math.exp(-lam) * lam**k / math.factorial(k)

def calculate_probabilities(home_lambda, away_lambda, max_goals=12):
    home_probs = [poisson_pmf(i, home_lambda) for i in range(max_goals + 1)]
    away_probs = [poisson_pmf(i, away_lambda) for i in range(max_goals + 1)]
    p1 = px = p2 = under25 = btts = 0.0
    scores = []
    for h, ph in enumerate(home_probs):
        for a, pa in enumerate(away_probs):
            p = ph * pa
            scores.append((p, h, a))
            if h > a:
                p1 += p
            elif h == a:
                px += p
            else:
                p2 += p
            if h + a <= 2:
                under25 += p
            if h >= 1 and a >= 1:
                btts += p
    # Kalan Poisson kuyruğu 12 gol sınırında olağan lambda değerlerinde çok küçüktür.
    return {
        "MS 1": p1, "MS X": px, "MS 2": p2,
        "2,5 Alt": under25, "2,5 Üst": 1 - under25,
        "KG Var": btts, "KG Yok": 1 - btts,
        "scores": sorted(scores, reverse=True)[:6]
    }

def api_get(path, api_key, params):
    response = requests.get(
        f"{API_BASE}/{path}",
        headers={"x-apisports-key": api_key},
        params=params,
        timeout=30
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise RuntimeError(f"API hata yanıtı: {payload['errors']}")
    return payload.get("response", []), payload.get("results", 0)

def number_or_none(value):
    if value is None or value == "" or value == "-":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def get_avg(stats, section, side):
    try:
        return number_or_none(stats[section]["average"][side])
    except (KeyError, TypeError):
        return None

def team_goal_rates(team_id, league_id, season, api_key):
    response, _ = api_get(
        "teams/statistics", api_key,
        {"team": team_id, "league": league_id, "season": season}
    )
    if not response:
        return None
    data = response[0] if isinstance(response, list) else response
    goals = data.get("goals", {})
    scored = goals.get("for", {})
    conceded = goals.get("against", {})
    return {
        "scored_home": get_avg(scored, "total", "home"),
        "scored_away": get_avg(scored, "total", "away"),
        "conceded_home": get_avg(conceded, "total", "home"),
        "conceded_away": get_avg(conceded, "total", "away"),
        "played_home": data.get("fixtures", {}).get("played", {}).get("home"),
        "played_away": data.get("fixtures", {}).get("played", {}).get("away"),
    }

def average_valid(values):
    values = [v for v in values if v is not None and v >= 0]
    return sum(values) / len(values) if values else None

def estimate_lambdas(home_rates, away_rates):
    # İç saha hücumu ile deplasman savunması; deplasman hücumu ile iç saha savunmasının basit ortalaması.
    home_lambda = average_valid([home_rates.get("scored_home"), away_rates.get("conceded_away")])
    away_lambda = average_valid([away_rates.get("scored_away"), home_rates.get("conceded_home")])
    if home_lambda is None or away_lambda is None:
        raise ValueError("Takımlar için gerekli iç/dış saha gol ortalamaları API'den alınamadı.")
    return max(0.05, min(home_lambda, 4.5)), max(0.05, min(away_lambda, 4.5))

def nvidia_commentary(home, away, stats_text, probabilities, key, model):
    client = OpenAI(base_url=NVIDIA_BASE, api_key=key)
    prob_text = "\n".join(
        f"{name}: %{value*100:.1f}"
        for name, value in probabilities.items() if name != "scores"
    )
    prompt = f"""Maç: {home} - {away}
Veri özeti:
{stats_text}

Poisson modelinin hesapladığı olasılıklar:
{prob_text}
En olası skorlar: {probabilities['scores']}

Türkçe, temkinli ve teknik futbol analizi yaz. 2,5 Alt/Üst ile KG Var/Yok sinyallerini değerlendir.
Bu olasılıkların basit bağımsız Poisson yaklaşımından geldiğini, form/sakatlık/xG/kadro verilerini içermeyebileceğini açıkla.
Eksik verileri uydurma, kesinlik veya garanti iddiasında bulunma. Bir bahis tavsiyesi yerine model çıktısını ve riskleri açıkla."""
    result = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Sen veri kalitesi ve belirsizliği açıkça belirten futbol istatistik analistisin."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=1100
    )
    return result.choices[0].message.content

def show_probabilities(home, away, home_lam, away_lam, probs):
    st.subheader(f"{home} – {away}")
    st.caption(f"Model gol beklentisi: {home} λ={home_lam:.2f} | {away} λ={away_lam:.2f}")
    columns = st.columns(4)
    for col, key in zip(columns, ["2,5 Alt", "2,5 Üst", "KG Var", "KG Yok"]):
        col.metric(key, f"%{probs[key]*100:.1f}")
    st.subheader("Maç sonucu (1X2)")
    columns = st.columns(3)
    for col, key in zip(columns, ["MS 1", "MS X", "MS 2"]):
        col.metric(key, f"%{probs[key]*100:.1f}")
    st.write("**En olası skorlar:** " + " · ".join(
        f"{h}-{a} (%{p*100:.1f})" for p, h, a in probs["scores"]
    ))

tab_auto, tab_manual, tab_notes = st.tabs(["Otomatik maç analizi", "Manuel analiz", "Model ve sınırlar"])

with tab_auto:
    st.subheader("Fikstürü çek ve takım istatistiklerinden tahmin üret")
    c1, c2, c3 = st.columns(3)
    with c1:
        league_id = st.number_input("Lig ID", min_value=1, value=39, step=1)
    with c2:
        season = st.number_input("Sezon başlangıç yılı", min_value=2000, max_value=2035, value=2026, step=1)
    with c3:
        match_date = st.text_input("Tarih (YYYY-MM-DD)", "2026-10-10")
    if st.button("Fikstürü getir", type="primary"):
        if not football_key:
            st.error("Sol menüye API-Football anahtarını gir.")
        else:
            try:
                fixtures, count = api_get("fixtures", football_key, {
                    "league": int(league_id), "season": int(season), "date": match_date
                })
                if not fixtures:
                    st.warning("Bu lig/tarih için maç bulunamadı. Lig, sezon, tarih ve API planını kontrol et.")
                    st.session_state.pop("fixtures", None)
                else:
                    st.session_state["fixtures"] = fixtures
                    st.session_state["fixture_meta"] = (int(league_id), int(season))
                    st.success(f"{len(fixtures)} maç bulundu.")
            except Exception as exc:
                st.error(f"Fikstür alınamadı: {exc}")

    fixtures = st.session_state.get("fixtures", [])
    if fixtures:
        options = {}
        for item in fixtures:
            fx = item.get("fixture", {})
            teams = item.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})
            label = f"{fx.get('date','')} | {home.get('name','Ev')} – {away.get('name','Deplasman')} | {fx.get('status',{}).get('short','')}"
            options[label] = item
        selected_label = st.selectbox("Analiz edilecek maç", list(options.keys()))
        if st.button("Takım istatistiklerini çek ve tahmin et"):
            if not football_key:
                st.error("API-Football anahtarı gerekli.")
            else:
                try:
                    selected = options[selected_label]
                    league_id, season = st.session_state["fixture_meta"]
                    home_obj = selected["teams"]["home"]
                    away_obj = selected["teams"]["away"]
                    with st.spinner("Takım sezon istatistikleri alınıyor..."):
                        home_rates = team_goal_rates(home_obj["id"], league_id, season, football_key)
                        away_rates = team_goal_rates(away_obj["id"], league_id, season, football_key)
                    if not home_rates or not away_rates:
                        raise ValueError("Takım istatistikleri boş döndü. API planınız veya sezon kapsamınız sınırlı olabilir.")
                    home_lam, away_lam = estimate_lambdas(home_rates, away_rates)
                    probs = calculate_probabilities(home_lam, away_lam)
                    st.session_state["auto_result"] = {
                        "home": home_obj["name"], "away": away_obj["name"],
                        "home_lam": home_lam, "away_lam": away_lam,
                        "probs": probs, "home_rates": home_rates, "away_rates": away_rates
                    }
                except Exception as exc:
                    st.error(f"Analiz tamamlanamadı: {exc}")

    result = st.session_state.get("auto_result")
    if result:
        show_probabilities(result["home"], result["away"], result["home_lam"], result["away_lam"], result["probs"])
        with st.expander("Kullanılan takım gol ortalamaları"):
            st.write("**Ev sahibi:**", result["home_rates"])
            st.write("**Deplasman:**", result["away_rates"])
        if nvidia_key and st.button("NVIDIA AI yorumunu üret"):
            try:
                summary = (
                    f"Ev sahibi gol attı (iç saha): {result['home_rates'].get('scored_home')}; "
                    f"ev sahibi gol yedi (iç saha): {result['home_rates'].get('conceded_home')}; "
                    f"deplasman gol attı (deplasman): {result['away_rates'].get('scored_away')}; "
                    f"deplasman gol yedi (deplasman): {result['away_rates'].get('conceded_away')}."
                )
                with st.spinner("NVIDIA modelinden yorum alınıyor..."):
                    commentary = nvidia_commentary(
                        result["home"], result["away"], summary, result["probs"], nvidia_key, nvidia_model
                    )
                st.markdown(commentary)
            except Exception as exc:
                st.error(f"NVIDIA analizi başarısız: {exc}")

with tab_manual:
    st.subheader("İstatistikleri kendin girerek analiz et")
    c1, c2 = st.columns(2)
    with c1:
        manual_home = st.text_input("Ev sahibi takım", "Takım A")
        manual_home_lam = st.number_input("Ev sahibinin beklenen golü (λ)", 0.05, 6.0, 1.40, 0.05)
    with c2:
        manual_away = st.text_input("Deplasman takımı", "Takım B")
        manual_away_lam = st.number_input("Deplasmanın beklenen golü (λ)", 0.05, 6.0, 1.10, 0.05)
    if st.button("Manuel olasılıkları hesapla"):
        manual_probs = calculate_probabilities(manual_home_lam, manual_away_lam)
        show_probabilities(manual_home, manual_away, manual_home_lam, manual_away_lam, manual_probs)
        if nvidia_key and st.button("Manuel analiz için NVIDIA yorumu"):
            try:
                st.markdown(nvidia_commentary(
                    manual_home, manual_away, "Kullanıcı manuel λ değerleri girdi; takım verileri doğrulanmadı.",
                    manual_probs, nvidia_key, nvidia_model
                ))
            except Exception as exc:
                st.error(f"NVIDIA analizi başarısız: {exc}")

with tab_notes:
    st.subheader("Modelin çalışma şekli ve önemli sınırlar")
    st.markdown("""
**Otomatik akış:** API-Football fikstürü → takım sezon gol ortalamaları → basit gol beklentisi (λ) → bağımsız Poisson skor dağılımı → 1X2, 2,5 Alt/Üst ve KG Var/Yok olasılıkları.

**Dikkat edilmesi gerekenler**
- Gol beklentileri; ev sahibinin iç saha attığı gol ve rakibin deplasmanda yediği gol ortalamasının, ayrıca deplasman için ilgili iki ortalamanın basit ortalamasıdır. Bu başlangıç yaklaşımıdır, gelişmiş/kalibre edilmiş bir model değildir.
- xG/xGA, rakip gücü, son maç formu, sakatlıklar, ilk 11, kırmızı kartlar ve oran hareketleri bu sürümde otomatik olarak hesaba katılmaz.
- API-Football hesabının planı hangi ligleri, sezonları ve istatistik uç noktalarını kullanabileceğini belirler.
- NVIDIA yalnızca sağlanan özeti yorumlar; bağımsız bir olasılık hesaplayıcısı yerine geçmez.
- Tahminler kesin değildir. Gerçek performansı değerlendirmek için tarihsel backtest, kalibrasyon ve log-loss/Brier skoru gibi metrikler eklenmelidir.
    """)
