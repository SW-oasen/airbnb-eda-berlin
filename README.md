# 🏠 Berlin Airbnb Marktanalyse 2025 (Business Intelligence Fokus)

Ein umfassendes Data-Science-Portfolio-Projekt zur Analyse des Berliner Airbnb-Marktes (Stand: September 2025). Dieses Projekt kombiniert Python-basierte Datenverarbeitung mit professioneller Business Intelligence in Power BI, um Umsatzpotenziale und regulatorische Trends aufzuzeigen.

## 🌐 Portfolio-Website
Besuchen Sie: [https://sw-oasen.github.io/yuchuan-portfolio/#projects](https://sw-oasen.github.io/yuchuan-portfolio/#projects)

## 🔄 Versions-Update: Was ist neu in v2?
Im Gegensatz zur Vorgängerversion (v1), die stark auf Preisvorhersagen mittels Machine Learning fokussiert war, konzentriert sich **v2** auf **Business Intelligence und Markt-Compliance**:
- **Umsatzmodellierung**: Implementierung des "San Francisco Modells" zur Schätzung von monatlichen Einnahmen (Revenue Proxies).
- **Active Market Filtering**: Fokus auf hochaktive Listings (Reviews > 2/Monat) statt auf den gesamten "Ghost-Bestand".
- **Regulatorische Analyse**: Einzigartiges Cleaning der Berliner Lizenzdaten (`Zweckentfremungsverbot`).
- **Power BI Integration**: Optimierter Workflow für deutsche Gebietsschemata und fortgeschrittene Visualisierungen (Pareto-Analyse, Compliance-Boxplots).

## 📊 Projekt-Übersicht

Dieses Projekt analysiert die wirtschaftliche Dynamik des Berliner Kurzzeitvermietungsmarktes. Es beantwortet Kernfragen zu Profitabilität, Professionalisierungsgrad der Hosts und der Effektivität städtischer Regulierungen.

### 🎯 Hauptziele
- **Umsatzschätzung**: Berechnung von Upper- und Lower-Revenue-Proxies basierend auf Review-Raten und Belegungsobergrenzen.
- **Markt-Segmentierung**: Identifizierung von "Power-Hosts" vs. Gelegenheitsvermietern.
- **Compliance-Check**: Analyse der Lizenzierungsmuster in den Berliner Bezirken.
- **Dashboarding**: Entwicklung eines interaktiven Dashboards für Stakeholder (Investoren, Stadtplaner).

## 📁 Projektstruktur

```
airbnb-eda-berlin/
├── notebooks/
│   ├── v1/ (Legacy: Price Prediction & ML Experiments)
│   └── v2/ 
│       └── AirBnB_Berlin_EDA.ipynb        # Haupt-Analyse & Daten-Pipeline
│       └── berlin_airbnb_20250923_revenue_powerbi_de.csv # Export für Power BI
│       └── listings_Berlin_20250923.csv
├── data/
│   └── AirBnB-Berlin/
│       └── 2025-09-23/                    # Aktueller Datensatz (Inside Airbnb)
│           └── listings_Berlin_20250923.csv
├── reports/
│   └── v2/
│       ├── airbnb_berlin_20250923_eda.pbix # Interaktives Power BI Dashboard
│       └── airbnb_berlin_20250923_eda.pdf  # PDF-Export des Berichts
└── README.md
```

## 🛠️ Technischer Stack & Methoden

### 1. Python Data Pipeline (Pandas & Seaborn)
- **Data Cleaning**: Konvertierung komplexer Währungsformate, Handling von Boolean-Werten und Datumsanpassungen.
- **Revenue Model**: 
  - *Formel*: `Preis * min(Buchungen basierend auf Review-Rate, Max. Kapazität bei 90% Auslastung)`.
  - Unterscheidung zwischen Upper Proxy (30% Review-Rate) und Lower Proxy (70% Review-Rate).
- **Lizenz-Sanitierung**: Regex-basierte Klassifizierung der Berliner Lizenznummern in `Gültig`, `Befreit`, `Ungültig/Dirty` und `Fehlt`.
- **Statistische Visualisierung**: Einsatz von Violin- und Strip-Plots zur Analyse der Umsatzverteilung pro Immobilientyp.

### 2. Power BI Business Intelligence
- **Daten-Lokalisierung**: Export mit deutschem Trennzeichen-Standard (Semikolon/Komma) für nahtlosen Import.
- **DAX-Measures**: 
  - Pareto-Analyse (80/20 Regel) für Host-Umsätze.
  - Dynamische Host-Kategorisierung (Commercial vs. Private).
- **Advanced Visuals**: Heatmaps für Stadtteil-Performance und Boxplots für die Risiko-Analyse (Umsatz vs. Compliance).

## 📈 Zentrale Erkenntnisse (Insights)

- **Das Compliance-Paradoxon**: Listings mit "Dirty/Invalid"-Lizenzen erzielen oft höhere Mediansätze als vollkommen konforme Anbieter.
- **Professionalisierung**: Ein kleiner Teil der kommerziellen Hosts (3+ Units) kontrolliert einen überproportionalen Anteil des Marktumsatzes (Pareto-Effekt).
- **Geografische Hotspots**: Mitte und Pankow dominieren preislich, weisen aber auch die strengste regulatorische Aktivität auf.
- **Filter-Effekt**: Nur ca. 35-40% der Listings in Berlin sind "echte" aktive Marktteilnehmer mit regelmäßigen Buchungen.

## 🚀 Installation & Nutzung

1. **Repository klonen**:
   ```bash
   git clone https://github.com/SW-oasen/airbnb-eda-berlin.git
   ```
2. **Abhängigkeiten installieren**:
   ```bash
   pip install pandas matplotlib seaborn numpy
   ```
3. **Notebook ausführen**:
   Öffne `notebooks/v2/AirBnB_Berlin_EDA.ipynb`, um die gesamte Pipeline von der Rohdatenverarbeitung bis zum Power-BI-Export nachzuvollziehen.

---
**Projekt von Yuchuan – Data Analyst & Pythonista**  
*Fokus: Real Estate Analytics & Business Intelligence*