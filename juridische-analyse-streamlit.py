#!/usr/bin/env python3
"""
Juridische Analyse Streamlit App met Gemini 2.0 Flash
Een interactieve webapplicatie voor het analyseren van juridische documenten.

Installatie:
pip install streamlit google-generativeai python-dotenv

Start de app:
streamlit run app.py
"""

import streamlit as st
import os
import sys
from pathlib import Path
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv
import time
import tempfile

# Laad environment variabelen
load_dotenv()

# Configuratie
st.set_page_config(
    page_title="Juridische Analyse Tool",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# System prompts (zelfde als in origineel script)
PROMPT_1_EXTRACT_LAWS = """**Rol:** Je bent een gespecialiseerde juridische assistent AI. Je taak is het analyseren van een juridisch document (een conclusie in een rechtszaak) en het diepgaand extraheren van informatie over de aangehaalde wetsartikelen.

**KRITIEKE INSTRUCTIE:** Je MOET ABSOLUUT ALLE wetsartikelen identificeren. Mis je er ook maar één, dan is je analyse onvolledig en onbruikbaar.

**Context:** Ik heb je een volledige conclusie verstrekt in een juridisch geschil. De partij die de conclusie neerlegt, haalt diverse wetsartikelen aan om haar vorderingen te ondersteunen.

**SCAN SPECIFIEK VOOR:**
- Artikelen met nummers (bv. artikel 1382, art. 1134, artikel 6 EVRM)
- Artikelen met paragrafen (§1, §2, lid 1, lid 2)
- Afkortingen: Art., art., Artikel, artikel
- Wetsboeken: BW, Ger.W., Sw., KB, Wet van [datum]
- Europese wetgeving: EVRM, EU-verordeningen, Richtlijnen
- In voetnoten vermelde artikelen
- In citaten genoemde artikelen
- Artikelen in jurisprudentieverwijzingen
- Procedurele artikelen (bv. art. 700 Ger.W.)

**Opdracht:** 
1. EERSTE SCAN: Lees het VOLLEDIGE document en markeer ELKE verwijzing naar een wetsartikel
2. TWEEDE SCAN: Controleer of je geen artikelen hebt gemist
3. ANALYSE: Voor elk gevonden artikel, geef de gevraagde analyse

Voor **elk** wetsartikel dat in het document wordt genoemd, presenteer je de volgende informatie:

### Artikel [NUMMER] [WETBOEK/WET]
**Locatie in document:** [Sectienaam]
**Analyse van de Toepassing:**
* **Wie en Waarom:** [Uitleg]
* **Argumentatie:** [Samenvatting met evt. citaat]
* **Relevantie voor de Feiten:** [Verbinding met de zaak]

**EINDINSTRUCTIE:** Tel aan het einde hoeveel wetsartikelen je hebt gevonden en vermeld dit aantal."""

PROMPT_2_EXTRACT_LIST = """**Rol:** Je bent een AI-assistent gespecialiseerd in het extraheren van specifieke data uit een gestructureerde tekst.
**Opdracht:** Je taak is om uit de onderstaande tekst een beknopte lijst te genereren van alle genoemde wetsartikelen.
**Instructies:**
1. Scan de volledige tekst.
2. Identificeer alle koppen die een wetsartikel aanduiden (deze beginnen doorgaans met "Artikel...").
3. Extraheer **enkel en alleen** de naam van het wetsartikel zelf.
4. Negeer alle verdere beschrijvingen, analyses, context, of bullet points die onder het wetsartikel staan.
5. Presenteer het resultaat als een eenvoudige lijst, met elk wetsartikel op een nieuwe regel.

**TE VERWERKEN TEKST:**"""

PROMPT_3_LOOKUP_LAWS = """**Rol:** Je bent een juridische informatieassistent. Je taak is het verstrekken van duidelijke, neutrale en feitelijke informatie over Belgische wetsartikelen.
**Opdracht:** Voor elk wetsartikel in de hieronder verstrekte lijst, zoek je de officiële betekenis en de algemene werking op. Je presenteert deze informatie op een heldere en gestructureerde manier.
**Invoer:** Een lijst van Belgische wetsartikelen, zonder enige context of toelichting.
**Instructies:** Voor **elk** artikel in de lijst, volg je deze stappen:
1. **Identificeer het artikel:** Gebruik de naam van het wetsartikel als een duidelijke kop.
2. **Zoek de informatie op:** Raadpleeg betrouwbare juridische bronnen om de inhoud en het doel van het artikel te achterhalen.
3. **Structureer de output:** Geef voor elk artikel de volgende informatie:
   * **Kernbetekenis:** Geef een beknopte en heldere samenvatting van wat dit wetsartikel in essentie regelt.
   * **Toepassingsgebied:** Leg uit in welke algemene situaties of contexten dit artikel doorgaans wordt gebruikt. Wat is het doel van de wetgever geweest?
   * **Belangrijkste Elementen (indien van toepassing):** Benoem de belangrijkste voorwaarden, concepten of criteria die het artikel bevat. Bijvoorbeeld, als het artikel gaat over aansprakelijkheid, wat zijn de voorwaarden om van die aansprakelijkheid te kunnen spreken?
**Cruciale Voorwaarde:** Bied een **algemene, neutrale uitleg**. De uitleg moet volledig losstaan van elke context en puur informatief zijn over de wet zelf. Je geeft geen juridisch advies.

**TE ONDERZOEKEN WETSARTIKELEN:**"""

PROMPT_4_COMPARE = """**Prompt voor Vergelijkende Juridische Analyse:**
**Rol:** Je bent een juridisch analytisch assistent. Je taak is om een kritische vergelijking te maken tussen de manier waarop een wet in een specifiek document wordt toegepast en de algemene, objectieve betekenis van diezelfde wet.
**Missie:** Voer een vergelijkende analyse uit tussen de twee hieronder verstrekte teksten.
* **Input 1** bevat de analyse van hoe wetsartikelen in een specifiek juridisch document worden geïnterpreteerd en toegepast.
* **Input 2** bevat de algemene, objectieve en contextloze uitleg van diezelfde wetsartikelen.
Je doel is om voor elk wetsartikel te evalueren of de interpretatie en toepassing in het document (Input 1) consistent en correct zijn in het licht van de algemene juridische betekenis (Input 2).
**Gedetailleerde Instructies:** Doorloop de volgende stappen voor **elk wetsartikel** dat in beide inputs voorkomt:
1. **Identificeer en Match:** Neem een wetsartikel uit Input 1 en vind de corresponderende objectieve uitleg in Input 2. Gebruik de naam van het wetsartikel als hoofding voor je analyse.
2. **Evalueer de Correctheid van de Interpretatie:**
   * Vergelijk de beschrijving en het doel van het artikel zoals geschetst in Input 1 met de neutrale definitie uit Input 2.
   * Beantwoord de vraag: "Komt de manier waarop het artikel in het document wordt voorgesteld overeen met de algemene, correcte betekenis? Worden de belangrijkste elementen correct weergegeven, of worden er nuances weggelaten of foutief geïnterpreteerd?"
3. **Beoordeel de Relevantie van de Toepassing:**
   * Analyseer de concrete situatie (de feiten van de zaak) zoals beschreven in Input 1.
   * Beantwoord de vraag: "Is de toepassing van het wetsartikel op deze specifieke feiten logisch en relevant? Wordt het artikel ingezet om een argument te ondersteunen waarvoor het volgens de algemene uitleg (Input 2) bedoeld is?"
4. **Formuleer een Conclusie:**
   * Vat je bevindingen uit stap 2 en 3 samen in een beknopte eindconclusie per artikel. Geef duidelijk aan of de argumentatie in het document steekhoudend lijkt, of dat er mogelijke zwaktes of onjuistheden zijn in de manier waarop de wet wordt gebruikt.
**Output Formaat:** Presenteer je analyse per wetsartikel met de volgende structuur:
**Analyse: [Naam van het Wetsartikel]**
* **Correctheid van de Interpretatie:** [Jouw evaluatie hier, waarin je de interpretatie uit het document vergelijkt met de algemene uitleg.]
* **Relevantie van de Toepassing:** [Jouw evaluatie hier, waarin je de toepassing op de feiten van de zaak beoordeelt.]
* **Eindconclusie:** [Een korte samenvatting van je bevindingen. Bijvoorbeeld: "De interpretatie is correct en de toepassing op de feiten is zeer relevant en logisch." of "Hoewel de wet correct wordt beschreven, lijkt de toepassing op deze specifieke feiten vergezocht, omdat..."]
**BELANGRIJKE OPMERKING:** Deze analyse is een logische evaluatie gebaseerd op de verstrekte teksten en vormt **geen** juridisch advies. Het is een hulpmiddel om de consistentie en logica van de argumentatie in het document te doorgronden."""

PROMPT_5_SUMMARY = """**Rol:** Je bent een juridisch analytisch assistent die beknopte samenvattingen maakt.
**Opdracht:** Analyseer de vergelijkende juridische analyse hieronder en maak een beknopte samenvatting.

Voor ELK wetsartikel dat in de analyse voorkomt, geef je ALLEEN:
- De naam van het wetsartikel
- Een indicator: ✅ (correct toegepast), ❌ (incorrect/problematisch), of ⚠️ (twijfelachtig)

**Bepaal de indicator op basis van:**
- Als de interpretatie correct is EN de toepassing relevant/logisch is → ✅
- Als de interpretatie foutief is OF de toepassing niet relevant/vergezocht is → ❌  
- Als er twijfel is, nuances zijn, of deels correct maar deels problematisch → ⚠️

**Output formaat:**
Artikel X BW ✅
Artikel Y Ger.W. ❌
Artikel Z KB ⚠️

Geef ALLEEN de lijst, geen uitleg of extra tekst."""

PROMPT_6_STRATEGIC_ADVICE = """**Rol:** Je bent een ervaren juridisch adviseur gespecialiseerd in Belgisch recht. Je taak is om strategisch advies te geven voor het versterken van juridische argumentatie.

**Context:** Je hebt toegang tot:
1. Het originele juridische document
2. Een analyse van de gebruikte wetsartikelen
3. Een vergelijkende analyse van de correctheid van de toegepaste wetsartikelen

**Opdracht:** Analyseer het document grondig en voorzie gedetailleerd advies volgens onderstaande structuur:

## 1. VERSTERKING VAN DE FACTUALE ONDERBOUWING

### Analyse van de feiten
- Beoordeel of de feiten duidelijk, chronologisch en logisch zijn opgebouwd
- Identificeer waar de presentatie helderder of overtuigender kan
- Geef concrete suggesties voor verbetering van de feitenpresentatie

### Identificatie van cruciale bewijsstukken
- Bepaal welke feiten essentieel zijn voor de zaak
- Specificeer welk bewijs (e-mails, foto's, contracten, getuigenverklaringen) noodzakelijk is
- Geef aan welk bewijs momenteel ontbreekt

### Suggesties voor bijkomend bewijs
- Identificeer welk bijkomend bewijsmateriaal de positie significant zou versterken
- Geef concrete suggesties waar en hoe dit bewijs verzameld kan worden
- Prioriteer het bewijs naar belangrijkheid

## 2. JURIDISCH-ARGUMENTATIEVE VERSTERKING

### Bijkomende argumenten
- Identificeer juridische argumenten die nog toegevoegd kunnen worden
- Geef per argument aan waarom het de zaak zou versterken
- Prioriteer de argumenten naar sterkte

### Wetsartikelen en decreten
- Lijst concrete wetsartikelen en decreten uit het Belgisch recht op die van toepassing zijn
- Geef per artikel/decreet aan hoe het de claim ondersteunt
- Identificeer eventuele gemiste relevante wetgeving

### Rechtsleer en rechtspraak
- Verwijs naar relevante rechtsleer (juridische publicaties)
- Citeer cruciale rechtspraak (vonnissen en arresten) die de argumenten onderbouwen
- Geef per verwijzing aan waarom deze relevant is

## 3. STRATEGISCHE ANALYSE

### Tegenargumenten en weerlegging
- Identificeer de te verwachten tegenargumenten van de andere partij (feitelijk en juridisch)
- Geef per tegenargument een proactieve weerlegging
- Suggereer hoe deze weerleggingen in het document verwerkt kunnen worden

### Structuur en formulering
- Analyseer de huidige opbouw van het document
- Geef concrete suggesties voor structurele verbeteringen
- Identificeer formuleringen die aangepast moeten worden voor meer overtuigingskracht
- Houd rekening met de doelgroep (rechter, ambtenaar, tegenpartij)

**BELANGRIJKE INSTRUCTIE:** Wees zeer concreet en praktisch in je advies. Vermijd algemene aanbevelingen en focus op specifieke, implementeerbare suggesties."""

PROMPT_7_PROBLEM_FOCUS = """**Rol:** Je bent een juridisch expert die zich focust op problematische wetsartikelen.

**Opdracht:** Op basis van de indicatoren (✅, ❌, ⚠️) en de vergelijkende analyse, maak een gefocuste eindconclusie die ALLEEN de artikelen bespreekt die NIET correct of volledig waren (❌ of ⚠️).

Voor elk problematisch artikel:
1. **Wat was er niet goed?** - Specifiek aangeven wat er mis was
2. **Waarom was het fout?** - De juridische reden waarom de toepassing incorrect is
3. **Wat is wel correct?** - Hoe het artikel correct toegepast had moeten worden

**Output formaat:**
## Eindconclusie: Problematische Wetsartikelen

### [Artikel X]
**Probleem:** [Wat was er mis]
**Reden:** [Waarom het fout is]
**Correcte toepassing:** [Hoe het wel moet]

### [Artikel Y]
[etc.]

Als ALLE artikelen correct zijn toegepast (alleen ✅), vermeld dit kort."""

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'api_key' not in st.session_state:
    st.session_state.api_key = os.getenv('GEMINI_API_KEY', '')

def call_gemini(prompt, content, model_name="gemini-2.0-flash-lite"):
    """Roep Gemini API aan met de gegeven prompt en content"""
    try:
        genai.configure(api_key=st.session_state.api_key)
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=prompt
        )
        
        response = model.generate_content(content)
        return response.text
    except Exception as e:
        st.error(f"Error bij Gemini API call: {e}")
        return None

def call_gemini_with_search(prompt, content):
    """Roep Gemini API aan met Google Search grounding voor actuele informatie"""
    try:
        genai.configure(api_key=st.session_state.api_key)
        model = genai.GenerativeModel(
            model_name='gemini-1.5-pro',
            system_instruction=prompt,
            tools=[{'google_search_retrieval': {}}]
        )
        
        response = model.generate_content(content)
        
        # Check voor search results in de response
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'grounding_metadata'):
                st.info("📚 Gebruikt Google Search voor actuele informatie")
        
        return response.text
        
    except Exception as e:
        st.warning(f"⚠️ Grounding niet beschikbaar: {e}")
        st.info("Gebruik standaard Gemini zonder live search...")
        return call_gemini(prompt, content, model_name="gemini-1.5-pro")

def save_report(results, filename="juridisch_rapport.md"):
    """Genereer en retourneer het complete rapport"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# Juridische Analyse Rapport
Gegenereerd op: {timestamp}

## 📋 Overzicht Wetsartikelen en Indicatoren

{results.get('summary', 'Geen samenvatting beschikbaar')}

**Legenda:**
- ✅ = Correct toegepast
- ❌ = Incorrect of problematisch toegepast  
- ⚠️ = Twijfelachtig of nadere analyse vereist

---

## 🎯 Eindconclusie: Focus op Problematische Artikelen

{results.get('problem_focus', 'Geen probleemanalyse beschikbaar')}

---

## 💡 Strategisch Juridisch Advies

{results.get('strategic_advice', 'Geen strategisch advies beschikbaar')}

---

*Dit rapport is automatisch gegenereerd met Gemini AI en dient enkel ter informatie. Het vormt geen juridisch advies.*
"""
    
    return report

def main():
    # Header
    st.markdown('<h1 class="main-header">⚖️ Juridische Analyse Tool</h1>', unsafe_allow_html=True)
    st.markdown("**Analyseer juridische documenten met behulp van Gemini AI**")
    
    # Sidebar voor configuratie
    with st.sidebar:
        st.header("⚙️ Configuratie")
        
        # API Key input
        api_key_input = st.text_input(
            "Gemini API Key", 
            value=st.session_state.api_key,
            type="password",
            help="Voer je Gemini API key in"
        )
        
        if api_key_input:
            st.session_state.api_key = api_key_input
            
        if not st.session_state.api_key:
            st.error("⚠️ Geen API key gevonden. Voer je Gemini API key in.")
            
        st.markdown("---")
        
        # Instructies
        st.header("📖 Instructies")
        st.markdown("""
        1. **Upload** je juridisch document
        2. **Specificeer** het doel (optioneel)
        3. **Start** de analyse
        4. **Download** het rapport
        
        De analyse bestaat uit 7 stappen:
        - Extractie van wetsartikelen
        - Opzoeken van wettelijke informatie
        - Vergelijkende analyse
        - Strategisch advies
        - Focus op problemen
        """)
        
        st.markdown("---")
        st.caption("Ontwikkeld met Gemini AI")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload
        st.markdown('<h2 class="step-header">📄 Upload Document</h2>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Kies een juridisch document",
            type=['txt', 'docx', 'pdf'],
            help="Upload een tekstbestand met je juridische conclusie"
        )
        
        # Specifiek doel
        st.markdown('<h2 class="step-header">🎯 Specifiek Doel</h2>', unsafe_allow_html=True)
        specific_goal = st.text_area(
            "Wat is het specifieke doel van uw document?",
            placeholder="bijv. de tegenpartij in gebreke stellen, een vergunning aanvragen, etc.",
            height=100
        )
    
    with col2:
        st.markdown('<h2 class="step-header">📊 Status</h2>', unsafe_allow_html=True)
        
        if uploaded_file:
            st.success(f"✅ Bestand geladen: {uploaded_file.name}")
            st.info(f"📏 Grootte: {uploaded_file.size:,} bytes")
        else:
            st.info("⏳ Wachten op bestand...")
            
        if specific_goal:
            st.success("✅ Doel gespecificeerd")
        else:
            st.info("ℹ️ Optioneel: specificeer doel")
    
    # Start analyse button
    if uploaded_file and st.session_state.api_key:
        if st.button("🚀 Start Analyse", type="primary", use_container_width=True):
            # Reset previous results
            st.session_state.analysis_complete = False
            st.session_state.results = {}
            
            # Read file content
            if uploaded_file.type == "text/plain":
                document_content = str(uploaded_file.read(), "utf-8")
            else:
                st.error("Momenteel worden alleen .txt bestanden ondersteund")
                return
            
            # Progress container
            progress_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Stap 1: Extraheer wetsartikelen
                status_text.text("🔍 Stap 1/7: Analyseren van wetsartikelen...")
                progress_bar.progress(14)
                
                with st.spinner("Analyseren van document..."):
                    output_1 = call_gemini(PROMPT_1_EXTRACT_LAWS, document_content, model_name="gemini-1.5-pro")
                
                if not output_1:
                    st.error("❌ Fout bij het analyseren van wetsartikelen")
                    return
                
                st.session_state.results['extraction'] = output_1
                
                # Stap 2: Extraheer lijst
                status_text.text("📋 Stap 2/7: Extraheren van lijst met wetsartikelen...")
                progress_bar.progress(28)
                time.sleep(1)
                
                with st.spinner("Lijst maken..."):
                    output_2 = call_gemini(PROMPT_2_EXTRACT_LIST, output_1)
                
                if not output_2:
                    st.error("❌ Fout bij het extraheren van lijst")
                    return
                
                st.session_state.results['article_list'] = output_2
                
                # Stap 3: Zoek algemene informatie
                status_text.text("📚 Stap 3/7: Opzoeken van wettelijke informatie...")
                progress_bar.progress(42)
                time.sleep(1)
                
                with st.spinner("Raadplegen van juridische bronnen..."):
                    output_3 = call_gemini_with_search(PROMPT_3_LOOKUP_LAWS, output_2)
                
                if not output_3:
                    st.error("❌ Fout bij het opzoeken van informatie")
                    return
                
                st.session_state.results['legal_info'] = output_3
                
                # Stap 4: Vergelijkende analyse
                status_text.text("⚖️ Stap 4/7: Uitvoeren van vergelijkende analyse...")
                progress_bar.progress(57)
                time.sleep(1)
                
                compare_content = f"""
**[INPUT 1: ANALYSE VAN HET DOCUMENT]**

{output_1}

**[INPUT 2: ALGEMENE WETSUITLEG]**

{output_3}
"""
                
                with st.spinner("Vergelijken van interpretaties..."):
                    output_4 = call_gemini(PROMPT_4_COMPARE, compare_content, model_name="gemini-1.5-pro")
                
                if not output_4:
                    st.error("❌ Fout bij vergelijkende analyse")
                    return
                
                st.session_state.results['comparison'] = output_4
                
                # Stap 5: Genereer samenvatting
                status_text.text("📊 Stap 5/7: Genereren van overzicht...")
                progress_bar.progress(71)
                time.sleep(1)
                
                with st.spinner("Samenvatting maken..."):
                    output_5 = call_gemini(PROMPT_5_SUMMARY, output_4, model_name="gemini-2.0-flash-lite")
                
                if not output_5:
                    articles = output_2.strip().split('\n')
                    output_5 = "\n".join([f"{article.strip()} ⚠️" for article in articles if article.strip()])
                
                st.session_state.results['summary'] = output_5
                
                # Stap 6: Strategisch advies
                status_text.text("💡 Stap 6/7: Genereren van strategisch advies...")
                progress_bar.progress(85)
                time.sleep(1)
                
                strategic_content = f"""
**ORIGINEEL DOCUMENT:**
{document_content}

**ANALYSE VAN WETSARTIKELEN:**
{output_1}

**VERGELIJKENDE ANALYSE:**
{output_4}

**SPECIFIEK DOEL:** {specific_goal if specific_goal else "Niet gespecificeerd"}
"""
                
                with st.spinner("Strategisch advies formuleren..."):
                    output_6 = call_gemini(PROMPT_6_STRATEGIC_ADVICE, strategic_content, model_name="gemini-1.5-pro")
                
                if not output_6:
                    output_6 = "Strategisch advies kon niet worden gegenereerd."
                
                st.session_state.results['strategic_advice'] = output_6
                
                # Stap 7: Focus op problemen
                status_text.text("🎯 Stap 7/7: Analyseren van problematische artikelen...")
                progress_bar.progress(95)
                time.sleep(1)
                
                problem_content = f"""
**INDICATOREN:**
{output_5}

**VERGELIJKENDE ANALYSE:**
{output_4}
"""
                
                with st.spinner("Probleemanalyse..."):
                    problem_focus = call_gemini(PROMPT_7_PROBLEM_FOCUS, problem_content, model_name="gemini-2.0-flash-lite")
                
                if not problem_focus:
                    problem_focus = "Analyse van problematische artikelen kon niet worden uitgevoerd."
                
                st.session_state.results['problem_focus'] = problem_focus
                
                # Complete
                progress_bar.progress(100)
                status_text.text("✅ Analyse voltooid!")
                time.sleep(1)
                
                st.session_state.analysis_complete = True
    
    # Display results
    if st.session_state.analysis_complete:
        st.markdown("---")
        st.markdown('<h2 class="step-header">📊 Analyseresultaten</h2>', unsafe_allow_html=True)
        
        # Tabs voor verschillende secties
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Overzicht", 
            "🎯 Problematische Artikelen",
            "💡 Strategisch Advies",
            "📄 Volledig Rapport",
            "💾 Download"
        ])
        
        with tab1:
            st.markdown("### Wetsartikelen en Indicatoren")
            st.markdown(st.session_state.results.get('summary', 'Geen samenvatting beschikbaar'))
            
            st.markdown("**Legenda:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.success("✅ = Correct toegepast")
            with col2:
                st.error("❌ = Incorrect toegepast")
            with col3:
                st.warning("⚠️ = Twijfelachtig")
        
        with tab2:
            st.markdown("### Focus op Problematische Artikelen")
            st.markdown(st.session_state.results.get('problem_focus', 'Geen probleemanalyse beschikbaar'))
        
        with tab3:
            st.markdown("### Strategisch Juridisch Advies")
            st.markdown(st.session_state.results.get('strategic_advice', 'Geen strategisch advies beschikbaar'))
        
        with tab4:
            report = save_report(st.session_state.results)
            st.markdown(report)
        
        with tab5:
            st.markdown("### Download Rapport")
            
            # Generate report
            report = save_report(st.session_state.results)
            
            # Download button
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"juridisch_rapport_{timestamp}.md"
            
            st.download_button(
                label="📥 Download Rapport (Markdown)",
                data=report,
                file_name=filename,
                mime="text/markdown",
                use_container_width=True
            )
            
            st.info("💡 Tip: Het rapport is in Markdown formaat en kan geopend worden met elke teksteditor.")

if __name__ == "__main__":
    main()