"""Generate ``places.json``, the sealed place gazetteer of the context axis.

Inputs: the GeoNames extracts shipped in the ``geonamescache`` package
(countries, continents, US states, cities with population >= 15,000; GeoNames
data is CC-BY 4.0), plus the curated tables below for what GeoNames extracts
lack: aliases, first-order regions outside the US, historical countries,
multi-country regions and demonyms.

Usage (from repo root):
    pip install --target /tmp/gnc geonamescache
    python -m nam_heads.impls.context.default.context_axis.data.generate_places \
        /tmp/gnc/geonamescache/data

Output schema:
    {"meta": {...},
     "places":   {surface_lower: [[path, tier, population], ...]},
     "demonyms": {demonym_lower: [path, tier]}}

A path is the place's containment chain, coarse to fine, "/"-joined:
continent code / ISO 3166-1 country code / ISO 3166-2 subdivision (US and GB
only, from GeoNames admin1 codes) / "gn<geonameid>" for a city. Places with
no GeoNames id get a minted, prefixed slug: "x-" historical country,
"r-" multi-country region, "s-" first-order region outside the US and GB.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

OUT = Path(__file__).parent / "places.json"

CONTINENT_NAMES = {
    "AF": "Africa", "AS": "Asia", "EU": "Europe", "NA": "North America",
    "OC": "Oceania", "SA": "South America", "AN": "Antarctica",
}

# Extra surface forms for GeoNames countries (ISO code -> aliases).
COUNTRY_ALIASES = {
    "US": ["United States of America", "USA", "U.S.", "U.S.A.", "US", "America",
           "the United States"],
    "GB": ["UK", "U.K.", "Britain", "Great Britain", "the United Kingdom"],
    "NL": ["Netherlands", "Holland", "the Netherlands"],
    "CZ": ["Czech Republic"],
    "CI": ["Côte d'Ivoire", "Cote d'Ivoire"],
    "TL": ["East Timor", "Timor-Leste"],
    "VA": ["Vatican City", "Holy See"],
    "PS": ["Palestine", "Palestinian territories"],
    "MO": ["Macau"],
    "MM": ["Burma"],
    "SZ": ["Swaziland"],
    "MK": ["Macedonia", "Republic of Macedonia"],
    "CV": ["Cape Verde"],
    "IE": ["Republic of Ireland"],
    "KR": ["Republic of Korea"],
    "KP": ["Democratic People's Republic of Korea"],
    "CD": ["DR Congo", "DRC"],
    "CG": ["Congo"],
    "RU": ["Russian Federation"],
    "CN": ["People's Republic of China", "PRC", "Mainland China"],
    "TW": ["Republic of China"],
    "AE": ["UAE"],
    "BA": ["Bosnia"],
    "BS": ["The Bahamas"],
    "GM": ["The Gambia"],
    "TR": ["Türkiye"],
    "IR": ["Persia"],
}

# Country codes whose GeoNames English name is a word that is far more often
# something else in running text; matched only with a locative cue.
CUE_ONLY_COUNTRIES = {"JE", "GG", "TD", "NE", "GN", "TO", "DM", "GE"}

# ISO 3166-2 codes for GB first-order regions (GeoNames admin1 codes).
GB_ADMIN1 = {"England": "GB-ENG", "Scotland": "GB-SCT", "Wales": "GB-WLS",
             "Northern Ireland": "GB-NIR"}

# First-order regions outside the US and GB: (name, country, tier, aliases).
OTHER_ADMIN1 = [
    # Canada: provinces and territories
    ("Ontario", "CA", "province", []), ("Quebec", "CA", "province", ["Québec"]),
    ("British Columbia", "CA", "province", []), ("Alberta", "CA", "province", []),
    ("Manitoba", "CA", "province", []), ("Saskatchewan", "CA", "province", []),
    ("Nova Scotia", "CA", "province", []), ("New Brunswick", "CA", "province", []),
    ("Newfoundland and Labrador", "CA", "province", ["Newfoundland"]),
    ("Prince Edward Island", "CA", "province", []), ("Yukon", "CA", "province", []),
    ("Northwest Territories", "CA", "province", []), ("Nunavut", "CA", "province", []),
    # Australia
    ("New South Wales", "AU", "state", []), ("Queensland", "AU", "state", []),
    ("Western Australia", "AU", "state", []), ("South Australia", "AU", "state", []),
    ("Tasmania", "AU", "state", []), ("Northern Territory", "AU", "state", []),
    ("Australian Capital Territory", "AU", "state", []),
    # Germany
    ("Bavaria", "DE", "state", []), ("Saxony", "DE", "state", []),
    ("Baden-Württemberg", "DE", "state", ["Baden-Wurttemberg"]), ("Hesse", "DE", "state", []),
    ("Lower Saxony", "DE", "state", []), ("North Rhine-Westphalia", "DE", "state", []),
    ("Rhineland-Palatinate", "DE", "state", []), ("Schleswig-Holstein", "DE", "state", []),
    ("Thuringia", "DE", "state", []), ("Brandenburg", "DE", "state", []),
    ("Saxony-Anhalt", "DE", "state", []), ("Mecklenburg-Vorpommern", "DE", "state", []),
    # India
    ("Maharashtra", "IN", "state", []), ("Kerala", "IN", "state", []),
    ("Tamil Nadu", "IN", "state", []), ("Karnataka", "IN", "state", []),
    ("West Bengal", "IN", "state", []), ("Uttar Pradesh", "IN", "state", []),
    ("Gujarat", "IN", "state", []), ("Rajasthan", "IN", "state", []),
    ("Bihar", "IN", "state", []), ("Andhra Pradesh", "IN", "state", []),
    ("Telangana", "IN", "state", []), ("Odisha", "IN", "state", ["Orissa"]),
    ("Assam", "IN", "state", []), ("Madhya Pradesh", "IN", "state", []),
    ("Himachal Pradesh", "IN", "state", []), ("Jharkhand", "IN", "state", []),
    # Others frequently named in encyclopedic text
    ("Baja California", "MX", "admin1", []), ("Baja California Sur", "MX", "admin1", []),
    ("Catalonia", "ES", "admin1", []), ("Andalusia", "ES", "admin1", []),
    ("Basque Country", "ES", "admin1", []), ("Galicia", "ES", "admin1", []),
    ("Sicily", "IT", "admin1", []), ("Sardinia", "IT", "admin1", []),
    ("Tuscany", "IT", "admin1", []), ("Lombardy", "IT", "admin1", []),
    ("Brittany", "FR", "admin1", []), ("Normandy", "FR", "admin1", []),
    ("Provence", "FR", "admin1", []), ("Flanders", "BE", "admin1", []),
    ("Wallonia", "BE", "admin1", []), ("Ulster", "IE", "admin1", []),
    ("Siberia", "RU", "admin1", []), ("Tibet", "CN", "admin1", []),
    ("Punjab", "AS", "region", []), ("Kashmir", "AS", "region", []),
]

# English counties (second-order; GB-ENG children).
ENGLISH_COUNTIES = """Bedfordshire Berkshire Buckinghamshire Cambridgeshire Cheshire Cornwall
Cumbria Derbyshire Devon Dorset Essex Gloucestershire Hampshire Herefordshire
Hertfordshire Kent Lancashire Leicestershire Lincolnshire Norfolk
Northamptonshire Northumberland Nottinghamshire Oxfordshire Rutland Shropshire
Somerset Staffordshire Suffolk Surrey Sussex Warwickshire Wiltshire
Worcestershire Yorkshire Merseyside""".split() + [
    "East Sussex", "West Sussex", "North Yorkshire", "West Yorkshire", "South Yorkshire",
    "East Riding of Yorkshire", "Greater Manchester", "Tyne and Wear", "West Midlands",
    "Isle of Wight", "Greater London", "County Durham"]

# Never places in running text: month and weekday names (GeoNames has a
# town "March" in Cambridgeshire and "Mayo"/"Sunday" namesakes).
BLOCKED = {"january", "february", "march", "april", "may", "june", "july", "august",
           "september", "october", "november", "december", "monday", "tuesday",
           "wednesday", "thursday", "friday", "saturday", "sunday"}

# Historical countries: (name, containing path or "", aliases).
HISTORICAL = [
    ("Soviet Union", "EU", ["USSR", "U.S.S.R."]),
    ("Yugoslavia", "EU", []),
    ("Czechoslovakia", "EU", []),
    ("East Germany", "EU/DE", ["German Democratic Republic", "GDR"]),
    ("West Germany", "EU/DE", ["Federal Republic of Germany"]),
    ("Prussia", "EU", ["Kingdom of Prussia"]),
    ("Austria-Hungary", "EU", ["Austro-Hungarian Empire"]),
    ("Holy Roman Empire", "EU", []),
    ("Ottoman Empire", "", []),
    ("Roman Empire", "", ["Ancient Rome"]),
    ("Byzantine Empire", "", []),
    ("British Empire", "", []),
    ("Russian Empire", "EU", []),
    ("Mughal Empire", "AS", []),
    ("British Raj", "AS/IN", []),
    ("Ceylon", "AS/LK", []),
    ("Rhodesia", "AF/ZW", []),
    ("Zaire", "AF/CD", []),
    ("Siam", "AS/TH", []),
    ("Kingdom of England", "EU/GB/GB-ENG", []),
    ("Kingdom of Scotland", "EU/GB/GB-SCT", []),
    ("Confederate States", "NA/US", ["Confederacy", "Confederate States of America"]),
]

# Multi-country regions: (name, containing path or "", tier, aliases).
REGIONS = [
    ("Middle East", "", "region", []), ("Near East", "", "region", []),
    ("Scandinavia", "EU", "region", []), ("Balkans", "EU", "region", []),
    ("Western Europe", "EU", "region", []), ("Eastern Europe", "EU", "region", []),
    ("Central Europe", "EU", "region", []), ("Southern Europe", "EU", "region", []),
    ("Northern Europe", "EU", "region", []), ("Mediterranean", "", "region", []),
    ("Caribbean", "NA", "region", ["West Indies"]), ("Central America", "NA", "region", []),
    ("Latin America", "", "region", []), ("New England", "NA/US", "region", []),
    ("Midwest", "NA/US", "region", []), ("Southeast Asia", "AS", "region", []),
    ("East Asia", "AS", "region", []), ("South Asia", "AS", "region", []),
    ("Central Asia", "AS", "region", []), ("West Africa", "AF", "region", []),
    ("East Africa", "AF", "region", []), ("North Africa", "AF", "region", []),
    ("Southern Africa", "AF", "region", []), ("Sub-Saharan Africa", "AF", "region", []),
    ("Indian subcontinent", "AS", "subcontinent", []), ("Pacific Islands", "OC", "region", []),
    ("Polynesia", "OC", "region", []), ("Melanesia", "OC", "region", []),
    ("Micronesia region", "OC", "region", []), ("Arctic", "", "region", []),
    ("Patagonia", "SA", "region", []), ("Anatolia", "AS/TR", "region", []),
    ("Mesopotamia", "AS", "region", []), ("Levant", "AS", "region", []),
    ("Korea", "AS", "region", []), ("Ireland island", "EU", "region", []),
]

# Demonym / nationality adjective -> place name it refers to (a key of the
# gazetteer). Covers the rule NER's nationalities.txt plus common others.
DEMONYMS = {
    "American": "United States", "British": "United Kingdom", "English": "England",
    "Scottish": "Scotland", "Scots": "Scotland", "Welsh": "Wales", "Irish": "Ireland",
    "Northern Irish": "Northern Ireland", "Canadian": "Canada", "Chinese": "China",
    "French": "France", "German": "Germany", "Indian": "India", "Italian": "Italy",
    "Japanese": "Japan", "Korean": "Korea", "South Korean": "South Korea",
    "North Korean": "North Korea", "Mexican": "Mexico", "Russian": "Russia",
    "Spanish": "Spain", "Australian": "Australia", "Brazilian": "Brazil", "Dutch": "Netherlands",
    "Swedish": "Sweden", "Norwegian": "Norway", "Danish": "Denmark", "Finnish": "Finland",
    "Polish": "Poland", "Portuguese": "Portugal", "Greek": "Greece", "Turkish": "Turkey",
    "Egyptian": "Egypt", "Iranian": "Iran", "Iraqi": "Iraq", "Israeli": "Israel",
    "Saudi": "Saudi Arabia", "Pakistani": "Pakistan", "Afghan": "Afghanistan",
    "Argentine": "Argentina", "Argentinian": "Argentina", "Austrian": "Austria",
    "Belgian": "Belgium", "Bolivian": "Bolivia", "Bulgarian": "Bulgaria", "Burmese": "Myanmar",
    "Cambodian": "Cambodia", "Chilean": "Chile", "Colombian": "Colombia", "Croatian": "Croatia",
    "Cuban": "Cuba", "Czech": "Czechia", "Dominican": "Dominican Republic",
    "Ecuadorian": "Ecuador", "Ethiopian": "Ethiopia", "Filipino": "Philippines",
    "Philippine": "Philippines", "Ghanaian": "Ghana", "Guatemalan": "Guatemala",
    "Haitian": "Haiti", "Hungarian": "Hungary", "Icelandic": "Iceland",
    "Indonesian": "Indonesia", "Jamaican": "Jamaica", "Jordanian": "Jordan", "Kenyan": "Kenya",
    "Kuwaiti": "Kuwait", "Latvian": "Latvia", "Lebanese": "Lebanon", "Libyan": "Libya",
    "Lithuanian": "Lithuania", "Malaysian": "Malaysia", "Moroccan": "Morocco",
    "Nepali": "Nepal", "Nepalese": "Nepal", "Nicaraguan": "Nicaragua", "Nigerian": "Nigeria",
    "Palestinian": "Palestine", "Panamanian": "Panama", "Paraguayan": "Paraguay",
    "Peruvian": "Peru", "Romanian": "Romania", "Rwandan": "Rwanda", "Senegalese": "Senegal",
    "Serbian": "Serbia", "Singaporean": "Singapore", "Slovak": "Slovakia",
    "Slovenian": "Slovenia", "Slovene": "Slovenia", "Somali": "Somalia",
    "South African": "South Africa", "Sri Lankan": "Sri Lanka", "Sudanese": "Sudan",
    "Swiss": "Switzerland", "Syrian": "Syria", "Taiwanese": "Taiwan", "Tanzanian": "Tanzania",
    "Thai": "Thailand", "Tunisian": "Tunisia", "Ukrainian": "Ukraine", "Uruguayan": "Uruguay",
    "Uzbek": "Uzbekistan", "Venezuelan": "Venezuela", "Vietnamese": "Vietnam",
    "Zambian": "Zambia", "Zimbabwean": "Zimbabwe", "Algerian": "Algeria", "Angolan": "Angola",
    "Bahraini": "Bahrain", "Bangladeshi": "Bangladesh", "Belarusian": "Belarus",
    "Bosnian": "Bosnia and Herzegovina", "Bruneian": "Brunei", "Cameroonian": "Cameroon",
    "Congolese": "Democratic Republic of the Congo", "Costa Rican": "Costa Rica",
    "Cypriot": "Cyprus", "Emirati": "United Arab Emirates", "Eritrean": "Eritrea",
    "Estonian": "Estonia", "Fijian": "Fiji", "Georgian": "Georgia", "Guyanese": "Guyana",
    "Honduran": "Honduras", "Ivorian": "Ivory Coast", "Kazakh": "Kazakhstan",
    "Kyrgyz": "Kyrgyzstan", "Lao": "Laos", "Laotian": "Laos", "Liberian": "Liberia",
    "Luxembourgish": "Luxembourg", "Malagasy": "Madagascar", "Malawian": "Malawi",
    "Maltese": "Malta", "Mongolian": "Mongolia", "Montenegrin": "Montenegro",
    "Mozambican": "Mozambique", "Namibian": "Namibia", "New Zealander": "New Zealand",
    "Omani": "Oman", "Qatari": "Qatar", "Salvadoran": "El Salvador", "Samoan": "Samoa",
    "Tajik": "Tajikistan", "Turkmen": "Turkmenistan", "Ugandan": "Uganda", "Yemeni": "Yemen",
    "Albanian": "Albania", "Armenian": "Armenia", "Azerbaijani": "Azerbaijan",
    "Macedonian": "North Macedonia", "Moldovan": "Moldova", "Puerto Rican": "Puerto Rico",
    "Trinidadian": "Trinidad and Tobago", "Hong Kong": "Hong Kong",
    "European": "Europe", "African": "Africa", "Asian": "Asia",
    "North American": "North America", "South American": "South America",
    "Latin American": "Latin America", "Oceanian": "Oceania",
    "Scandinavian": "Scandinavia", "Nordic": "Scandinavia", "Balkan": "Balkans",
    "Caribbean": "Caribbean", "Middle Eastern": "Middle East",
    "Soviet": "Soviet Union", "Yugoslav": "Yugoslavia", "Czechoslovak": "Czechoslovakia",
    "Prussian": "Prussia", "Ottoman": "Ottoman Empire", "Byzantine": "Byzantine Empire",
    "Roman": "Roman Empire", "Persian": "Iran", "Bavarian": "Bavaria",
    "Catalan": "Catalonia", "Basque": "Basque Country", "Sicilian": "Sicily",
    "Flemish": "Flanders", "Tibetan": "Tibet", "Quebecois": "Quebec",
    "Texan": "Texas", "Californian": "California",
}
# Demonyms that also name ethnic groups or languages more than a place, and
# whose place is ambiguous: deliberately absent -- Arab, Kurdish, Maori,
# Aboriginal, Indigenous, Slavic, Celtic, Saxon, Germanic, Anglo-Saxon,
# Hispanic, Latino, Latina, Baltic.


def fold(s: str) -> str:
    import unicodedata
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).lower()


def main(data_dir: str) -> None:
    d = Path(data_dir)
    files = ["countries.json", "continents.json", "us_states.json", "cities15000.json"]
    raw = {f: (d / f).read_bytes() for f in files}
    countries = json.loads(raw["countries.json"])
    continents = json.loads(raw["continents.json"])
    states = json.loads(raw["us_states.json"])
    cities = json.loads(raw["cities15000.json"])

    places = {}

    def add(name, path, tier, pop=0):
        places.setdefault(norm(name), []).append([path, tier, int(pop or 0)])

    for code, c in continents.items():
        add(c["name"], code, "continent", c.get("population", 0))
    country_path = {}
    for iso, c in countries.items():
        cont = c["continentcode"]
        path = f"{cont}/{iso}"
        country_path[iso] = path
        tier = "country" if iso not in CUE_ONLY_COUNTRIES else "country_cue"
        name = c["name"].strip()
        add(name, path, tier, c.get("population", 0))
        if name.lower().startswith("the "):
            add(name[4:], path, tier, c.get("population", 0))
        for a in COUNTRY_ALIASES.get(iso, []):
            add(a, path, "country", c.get("population", 0))
    for code, s in states.items():
        add(s["name"], f"NA/US/US-{code}", "state")
    for name, code in GB_ADMIN1.items():
        add(name, f"EU/GB/{code}", "admin1")
    for name, iso, tier, aliases in OTHER_ADMIN1:
        base = country_path.get(iso, iso)
        slug = "s-" + re.sub(r"[^a-z0-9]+", "_", norm(name)).strip("_")
        for n in [name] + aliases:
            add(n, f"{base}/{slug}", tier)
    for name in ENGLISH_COUNTIES:
        slug = "s-" + re.sub(r"[^a-z0-9]+", "_", norm(name)).strip("_")
        add(name, f"EU/GB/GB-ENG/{slug}", "county")
    for name, parent, aliases in HISTORICAL:
        slug = "x-" + re.sub(r"[^a-z0-9]+", "_", norm(name)).strip("_")
        path = f"{parent}/{slug}" if parent else slug
        for n in [name] + aliases:
            add(n, path, "historical_country")
    for name, parent, tier, aliases in REGIONS:
        nm = name.replace(" region", "").replace(" island", "")
        slug = "r-" + re.sub(r"[^a-z0-9]+", "_", norm(nm)).strip("_")
        path = f"{parent}/{slug}" if parent else slug
        for n in [nm] + aliases:
            if nm != name and n == nm and norm(nm) in places:
                continue  # never shadow a country/state of the same name
            add(n, path, tier)
    for c in cities.values():
        iso = c["countrycode"]
        base = country_path.get(iso)
        if not base:
            continue
        a1 = c.get("admin1code") or ""
        if iso == "US" and a1:
            base = f"{base}/US-{a1}"
        elif iso == "GB" and a1 in ("ENG", "SCT", "WLS", "NIR"):
            base = f"{base}/GB-{a1}"
        add(c["name"], f"{base}/gn{c['geonameid']}", "city", c.get("population", 0))
        # English text usually drops diacritics ("Montreal", "Sao Paulo"):
        # the ASCII-folded name is the same place, never a new one.
        folded = fold(c["name"])
        if folded != c["name"]:
            add(folded, f"{base}/gn{c['geonameid']}", "city", c.get("population", 0))

    # Keep each surface's candidates sorted: tier rank, then population.
    rank = {"continent": 0, "region": 1, "subcontinent": 1, "country": 2,
            "historical_country": 2, "country_cue": 3, "state": 3, "province": 3,
            "admin1": 3, "county": 3, "city": 4}
    for b in BLOCKED:
        places.pop(b, None)
    for k, v in places.items():
        uniq = {tuple(x[:2]): x for x in v}
        places[k] = sorted(uniq.values(), key=lambda x: (rank[x[1]], -x[2]))

    demonyms = {}
    for dem, place in DEMONYMS.items():
        cands = places.get(norm(place))
        if not cands:
            raise SystemExit(f"demonym {dem}: place {place!r} not in gazetteer")
        demonyms[norm(dem)] = cands[0][:2]

    meta = {
        "source": "GeoNames via geonamescache (CC-BY 4.0) + curated tables in generate_places.py",
        "inputs_sha256": {f: hashlib.sha256(b).hexdigest() for f, b in raw.items()},
        "surfaces": len(places), "demonyms": len(demonyms),
    }
    OUT.write_text(json.dumps({"meta": meta, "places": places, "demonyms": demonyms},
                              ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    print(f"wrote {OUT} surfaces={len(places)} demonyms={len(demonyms)} "
          f"bytes={OUT.stat().st_size}")


if __name__ == "__main__":
    main(sys.argv[1])
