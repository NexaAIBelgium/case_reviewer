#!/usr/bin/env python3
"""
Juridisch Multi-Agent Analysesysteem - Streamlit Interface
Moderne web interface voor het juridische analyse systeem met Gemini AI
Inclusief OCR en beeldanalyse functionaliteit
"""

import streamlit as st
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Union
import os
from pathlib import Path
import google.generativeai as genai
import time
from io import StringIO, BytesIO
import base64
import re

# Try imports met fallbacks
try:
    import pandas as pd
except ImportError:
    pd = None
    st.warning("Pandas niet beschikbaar")

try:
    from PIL import Image
except ImportError:
    Image = None
    st.warning("PIL niet beschikbaar")

try:
    import pypdf2 as PyPDF2
except ImportError:
    try:
        import PyPDF2
    except ImportError:
        PyPDF2 = None
        st.warning("PyPDF2 niet beschikbaar - PDF verwerking beperkt")

# Pagina configuratie
st.set_page_config(
    page_title="Juridisch Multi-Agent Analysesysteem",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS voor moderne styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1e3d59;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-card {
        background-color: #f5f7fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #1e3d59;
    }
    .metric-card {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 5px;
        color: #155724;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        padding: 1rem;
        border-radius: 5px;
        color: #856404;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 1rem;
        border-radius: 5px;
        color: #721c24;
    }
    .ocr-preview {
        background-color: #f8f9fa;
        border: 2px dashed #dee2e6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .image-analysis {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialiseer session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'api_key_set' not in st.session_state:
    st.session_state.api_key_set = False
if 'extracted_texts' not in st.session_state:
    st.session_state.extracted_texts = {}

# Model configuratie
LITE_MODEL = "gemini-2.0-flash-lite"
ADVANCED_MODEL = "gemini-2.5-flash"
VISION_MODEL = "gemini-1.5-flash"  # Voor OCR en beeldanalyse

def setup_logging():
    """Configureer logging voor Streamlit"""
    log_string = StringIO()
    handler = logging.StreamHandler(log_string)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    return log_string

def extract_text_from_pdf(pdf_file) -> tuple[str, List[Dict]]:
    """Extract tekst en detecteer afbeeldingen in PDF"""
    text = ""
    images_info = []
    
    if PyPDF2 is None:
        return "PDF verwerking niet beschikbaar - upload als afbeelding voor OCR", images_info
    
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        for page_num, page in enumerate(pdf_reader.pages):
            # Extract tekst
            page_text = page.extract_text()
            text += f"\n--- Pagina {page_num + 1} ---\n{page_text}"
            
            # Detecteer of er afbeeldingen zijn (simplified check)
            if '/XObject' in page.get('/Resources', {}):
                xobject = page['/Resources']['/XObject'].get_object()
                for obj_name in xobject:
                    if xobject[obj_name]['/Subtype'] == '/Image':
                        images_info.append({
                            'page': page_num + 1,
                            'name': obj_name,
                            'detected': True
                        })
    except Exception as e:
        st.error(f"Fout bij PDF verwerking: {str(e)}")
        text = "Fout bij PDF verwerking - probeer het bestand als afbeelding te uploaden"
        
    return text, images_info

def extract_text_from_docx(docx_file) -> tuple[str, List[Dict]]:
    """Extract tekst en detecteer afbeeldingen in DOCX"""
    text = ""
    images_info = []
    
    if docx is None:
        return "DOCX verwerking niet beschikbaar - converteer naar PDF of upload als afbeelding", images_info
    
    try:
        doc = docx.Document(docx_file)
        
        for i, paragraph in enumerate(doc.paragraphs):
            text += paragraph.text + "\n"
        
        # Check voor afbeeldingen
        for rel in doc.part.rels.values():
            if "image" in rel.reltype:
                images_info.append({
                    'detected': True,
                    'relationship_id': rel.rId
                })
                
    except Exception as e:
        st.error(f"Fout bij DOCX verwerking: {str(e)}")
        text = "Fout bij DOCX verwerking - probeer het bestand als PDF of afbeelding"
        
    return text, images_info

def process_image_with_gemini(image_bytes: bytes, context: str = "") -> str:
    """Verwerk afbeelding met Gemini Vision voor OCR en beschrijving"""
    if Image is None:
        return "PIL niet beschikbaar voor beeldverwerking"
        
    try:
        model = genai.GenerativeModel(VISION_MODEL)
        
        # Maak PIL Image voor Gemini
        image = Image.open(BytesIO(image_bytes))
        
        prompt = f"""Analyseer deze afbeelding in een juridische context. 
        
        {f'Context: {context}' if context else ''}
        
        Geef een gedetailleerde analyse met:
        1. TEKST EXTRACTIE: Transcribeer ALLE zichtbare tekst exact
        2. VISUELE BESCHRIJVING: Beschrijf wat je ziet (grafieken, schema's, handtekeningen, etc.)
        3. JURIDISCHE RELEVANTIE: Identificeer mogelijke juridisch relevante elementen
        4. DOCUMENT TYPE: Identificeer het type document/afbeelding indien mogelijk
        
        Format je antwoord gestructureerd met duidelijke kopjes."""
        
        response = model.generate_content([prompt, image])
        return response.text
        
    except Exception as e:
        return f"Fout bij beeldanalyse: {str(e)}"

def extract_and_analyze_content(uploaded_file, file_key: str) -> str:
    """Extract tekst uit bestand en analyseer afbeeldingen"""
    full_content = ""
    images_found = []
    
    # Verwerk op basis van bestandstype
    if uploaded_file.type == "text/plain":
        full_content = uploaded_file.read().decode('utf-8')
        
    elif uploaded_file.type == "application/pdf":
        text, images_info = extract_text_from_pdf(uploaded_file)
        full_content = text
        images_found = images_info
        
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text, images_info = extract_text_from_docx(uploaded_file)
        full_content = text
        images_found = images_info
        
    elif uploaded_file.type.startswith('image/'):
        # Direct een afbeelding geüpload
        with st.spinner("🔍 Analyseer afbeelding met Gemini Vision..."):
            image_bytes = uploaded_file.read()
            analysis = process_image_with_gemini(image_bytes, f"Document: {uploaded_file.name}")
            full_content = f"[AFBEELDING ANALYSE]\n{analysis}"
            
            # Toon preview
            st.markdown('<div class="image-analysis">', unsafe_allow_html=True)
            st.write("**🖼️ Afbeelding Geanalyseerd:**")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(image_bytes, width=200)
            with col2:
                st.text(analysis[:300] + "..." if len(analysis) > 300 else analysis)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Informeer gebruiker over gevonden afbeeldingen
    if images_found:
        st.warning(f"⚠️ {len(images_found)} afbeelding(en) gedetecteerd in {uploaded_file.name}")
        st.info("💡 Voor volledige analyse: upload een PDF met geëxtraheerde afbeeldingen of upload afbeeldingen apart")
    
    # Sla geëxtraheerde content op
    st.session_state.extracted_texts[file_key] = full_content
    
    return full_content

def call_gemini(system_prompt: str, user_prompt: str, model_name: str = LITE_MODEL) -> str:
    """Roep Gemini model aan met gegeven prompts"""
    try:
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]
        
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            },
            safety_settings=safety_settings
        )
        
        full_prompt = f"""System: {system_prompt}

User: {user_prompt}"""
        
        response = model.generate_content(full_prompt)
        
        if not response.parts:
            return json.dumps({
                "error": "Gemini response geblokkeerd",
                "reason": "Mogelijk safety filter of te lange input",
                "fallback": True
            })
        
        return response.text
        
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "fallback": True
        })

def parse_json_response(response: str, agent_name: str) -> Dict:
    """Parse JSON uit LLM response met error handling"""
    try:
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            json_str = response[start:end].strip()
        else:
            start = response.find("{")
            end = response.rfind("}") + 1
            json_str = response[start:end]
        
        return json.loads(json_str)
    except Exception as e:
        return {"error": "JSON parsing failed", "raw_response": response}

# Agent Classes (zelfde als origineel maar met progress updates)
class Agent0_NieuwheidsDetector:
    def __init__(self):
        self.name = "Agent 0 - Nieuwheidsdetector"
        self.model = LITE_MODEL
        self.system_prompt = """Je bent een gespecialiseerde juridische analyst met één kerncompetentie: het detecteren van volledig nieuwe elementen in juridische conclusies. Je werkt uitsluitend differentieel - je identificeert alleen wat ECHT nieuw is.

KERNPRINCIPE: Als iets eerder is vermeld (zelfs in andere bewoordingen), is het NIET nieuw.

Je output is de enige input voor alle downstream analyses. Fouten in jouw detectie kunnen niet worden hersteld door latere agenten. Wees daarom uiterst nauwkeurig.

BELANGRIJK: Geef je antwoord ALLEEN in JSON format, zonder extra tekst."""
        
    def analyze(self, historiek: str, laatste_conclusie: str, progress_callback=None) -> Dict:
        """Analyseer en detecteer nieuwe elementen"""
        if progress_callback:
            progress_callback(f"🔍 {self.name} - Start analyse...")
        
        user_prompt = f"""TAAK: Vergelijk het DOELWIT document met de HISTORIEK database. Identificeer ALLEEN elementen die:
1. Nog NOOIT eerder zijn vermeld (zelfs niet impliciet)
2. Geen herformulering zijn van bestaande argumenten
3. Geen samenvatting zijn van eerdere punten
4. Geen logische afleiding zijn uit eerder gestelde feiten

LET OP: Als er [AFBEELDING ANALYSE] secties zijn, behandel deze als belangrijke nieuwe informatie die mogelijk juridisch relevante details bevat.

INPUT:
- HISTORIEK_TEGENPARTIJ: 
{historiek}

- LAATSTE_CONCLUSIE_TEGENPARTIJ: 
{laatste_conclusie}

OUTPUT FORMAT (JSON):
{{
  "nieuwe_elementen": [
    {{
      "id": "N001",
      "categorie": "[Nieuw Argument|Nieuwe Juridische Bron|Nieuw Stuk/Feit|Nieuw Procedureel Middel|Nieuw Visueel Bewijs]",
      "citaat": "[Exacte tekst uit document]",
      "locatie": "[Paragraaf/pagina referentie]",
      "waarom_nieuw": "[Korte verklaring waarom dit element niet in historiek voorkomt]",
      "confidence": [0.0-1.0],
      "twijfelgeval": [true/false],
      "twijfel_reden": "[Alleen indien twijfelgeval=true]",
      "bevat_visueel_element": [true/false]
    }}
  ],
  "samenvatting": {{
    "totaal_nieuwe_elementen": [aantal],
    "hoogste_impact_element": "[id van element met potentieel grootste impact]",
    "visuele_elementen_gevonden": [aantal]
  }}
}}

Geef ALLEEN het JSON object als antwoord, zonder extra uitleg."""
        
        response = call_gemini(self.system_prompt, user_prompt, self.model)
        result = parse_json_response(response, self.name)
        
        if progress_callback:
            if "nieuwe_elementen" in result:
                progress_callback(f"✅ {self.name} - {result['samenvatting']['totaal_nieuwe_elementen']} nieuwe elementen gevonden")
            else:
                progress_callback(f"⚠️ {self.name} - Geen valide output")
        
        return result

class Agent1_ProcedureleStrateeg:
    def __init__(self):
        self.name = "Agent 1 - Procedurele Strateeg"
        self.model = LITE_MODEL
        self.system_prompt = """Je bent een expert in Belgisch procesrecht, gespecialiseerd in procedurele "knock-out" argumenten. Je analyseert UITSLUITEND nieuwe procedurele elementen die door Agent 0 zijn geïdentificeerd.

Je focus ligt op argumenten die de zaak kunnen beëindigen zonder inhoudelijke behandeling: verjaring, bevoegdheid, ontvankelijkheid.

BELANGRIJK: Geef je antwoord ALLEEN in JSON format."""
    
    def analyze(self, agent0_output: Dict, progress_callback=None) -> Dict:
        """Analyseer procedurele elementen"""
        if progress_callback:
            progress_callback(f"⚖️ {self.name} - Start analyse...")
        
        procedurele_elementen = [
            elem for elem in agent0_output.get('nieuwe_elementen', [])
            if 'Procedureel' in elem.get('categorie', '')
        ]
        
        if not procedurele_elementen:
            if progress_callback:
                progress_callback(f"ℹ️ {self.name} - Geen nieuwe procedurele elementen")
            return {"procedurele_analyse": [], "prioriteit_volgorde": [], "onmiddellijke_actie_vereist": False}
        
        user_prompt = f"""INPUT: Nieuwe procedurele elementen:
{json.dumps(procedurele_elementen, indent=2)}

ANALYSEOPDRACHT:
Voor elk nieuw procedureel element, evalueer:

1. CLASSIFICATIE (verjaring/bevoegdheid/ontvankelijkheid)
2. JURIDISCHE STERKTE met artikelen en rechtspraak
3. TIMING ANALYSE: waarom nu pas?

OUTPUT FORMAT (JSON):
{{
  "procedurele_analyse": [
    {{
      "element_id": "[ref naar Agent 0 output]",
      "type_exceptie": "[classificatie]",
      "juridische_basis": {{
        "artikelen": ["Art. X BW", "Art. Y Ger.W."],
        "rechtspraak": ["Cass. datum, AR nr"]
      }},
      "sterkte_beoordeling": {{
        "score": "[Hoog/Medium/Laag]",
        "motivering": "[waarom deze score]"
      }},
      "timing_implicaties": {{
        "mogelijke_redenen": ["reden 1", "reden 2"],
        "rechtsverwerking_mogelijk": [true/false],
        "contra_argument": "[suggestie voor weerlegging]"
      }}
    }}
  ],
  "prioriteit_volgorde": ["element_id1", "element_id2"],
  "onmiddellijke_actie_vereist": [true/false]
}}"""
        
        response = call_gemini(self.system_prompt, user_prompt, self.model)
        result = parse_json_response(response, self.name)
        
        if progress_callback:
            if "procedurele_analyse" in result:
                progress_callback(f"✅ {self.name} - {len(result['procedurele_analyse'])} risico's geanalyseerd")
            else:
                progress_callback(f"⚠️ {self.name} - Analyse afgerond")
        
        return result

class Agent2_FeitelijkeOnderzoeker:
    def __init__(self):
        self.name = "Agent 2 - Feitelijke Onderzoeker"
        self.model = LITE_MODEL
        self.system_prompt = """Je bent een forensisch analist voor juridische procedures. Je specialiteit is het analyseren van nieuwe feitelijke beweringen en bewijsstukken. Je bent van nature sceptisch en zoekt altijd naar inconsistenties, contradicties en bewijsgaten.

Je werkt ALLEEN met nieuwe feiten/bewijzen geïdentificeerd door Agent 0.

EXTRA AANDACHT: Visuele bewijzen (foto's, schema's, grafieken) kunnen cruciale informatie bevatten. Analyseer [AFBEELDING ANALYSE] secties extra zorgvuldig.

BELANGRIJK: Geef je antwoord ALLEEN in JSON format."""
    
    def analyze(self, agent0_output: Dict, mijn_argumentatie: str, progress_callback=None) -> Dict:
        """Analyseer feitelijke elementen"""
        if progress_callback:
            progress_callback(f"🔬 {self.name} - Start analyse...")
        
        feitelijke_elementen = [
            elem for elem in agent0_output.get('nieuwe_elementen', [])
            if 'Feit' in elem.get('categorie', '') or 'Stuk' in elem.get('categorie', '') or 'Visueel' in elem.get('categorie', '')
        ]
        
        if not feitelijke_elementen:
            if progress_callback:
                progress_callback(f"ℹ️ {self.name} - Geen nieuwe feitelijke elementen")
            return {"feitelijke_analyse": [], "grootste_risico_feiten": []}
        
        user_prompt = f"""INPUT: 
- Nieuwe feiten/stukken: {json.dumps(feitelijke_elementen, indent=2)}
- Mijn argumentatie (voor contradictie-check): {mijn_argumentatie[:1000]}...

ANALYSEER:
1. Consistentie met eerdere beweringen
2. Bewijsgaten
3. Geloofwaardigheid en timing
4. Visuele bewijzen (indien aanwezig)

OUTPUT FORMAT (JSON):
{{
  "feitelijke_analyse": [
    {{
      "element_id": "[ref naar Agent 0]",
      "feit_samenvatting": "[kern van de bewering]",
      "type_bewijs": "[Documentair|Getuigenis|Visueel|Technisch|Anders]",
      "contradictie_analyse": {{
        "interne_contradicties": [],
        "conflict_met_mijn_standpunt": []
      }},
      "bewijs_analyse": {{
        "vereist_bewijs": [],
        "aangeleverd_bewijs": [],
        "bewijs_gaten": [],
        "bewijskracht": "[Sterk/Matig/Zwak]",
        "visuele_component": "[beschrijving indien aanwezig]"
      }},
      "geloofwaardigheid": {{
        "timing_verdacht": [true/false],
        "timing_verklaring": "",
        "bron_betrouwbaarheid": "[Hoog/Medium/Laag]",
        "selectieve_presentatie": [true/false],
        "manipulatie_risico": "[Laag/Medium/Hoog]"
      }}
    }}
  ],
  "grootste_risico_feiten": [],
  "aanvullend_onderzoek_nodig": [],
  "visueel_bewijs_impact": "[beschrijving van impact van visuele elementen]"
}}"""
        
        response = call_gemini(self.system_prompt, user_prompt, self.model)
        result = parse_json_response(response, self.name)
        
        if progress_callback:
            if "feitelijke_analyse" in result:
                progress_callback(f"✅ {self.name} - {len(result['feitelijke_analyse'])} elementen geanalyseerd")
            else:
                progress_callback(f"⚠️ {self.name} - Analyse afgerond")
        
        return result

class Agent3_JuridischeTacticus:
    def __init__(self):
        self.name = "Agent 3 - Juridische Tacticus"
        self.model = LITE_MODEL
        self.system_prompt = """Je bent een expert in Belgisch materieel recht. Je analyseert nieuwe juridische argumenten op hun impact op de constitutieve elementen van rechtsvorderingen. Je denkt in termen van aanval en verdediging op elk juridisch element.

Je werkt ALLEEN met nieuwe juridische argumenten geïdentificeerd door Agent 0.

BELANGRIJK: Geef je antwoord ALLEEN in JSON format."""
    
    def analyze(self, agent0_output: Dict, agent2_output: Dict, mijn_argumentatie: str, progress_callback=None) -> Dict:
        """Analyseer juridische argumenten"""
        if progress_callback:
            progress_callback(f"📚 {self.name} - Start analyse...")
        
        juridische_elementen = [
            elem for elem in agent0_output.get('nieuwe_elementen', [])
            if 'Argument' in elem.get('categorie', '') or 'Juridische' in elem.get('categorie', '')
        ]
        
        if not juridische_elementen:
            if progress_callback:
                progress_callback(f"ℹ️ {self.name} - Geen nieuwe juridische argumenten")
            return {"juridische_analyse": [], "prioritaire_verweren": []}
        
        user_prompt = f"""INPUT:
- Nieuwe juridische argumenten: {json.dumps(juridische_elementen, indent=2)}
- Feitelijke ondersteuning: {json.dumps(agent2_output.get('feitelijke_analyse', []), indent=2)}

ANALYSEER impact op constitutieve elementen en formuleer verweer.

OUTPUT FORMAT (JSON):
{{
  "juridische_analyse": [
    {{
      "element_id": "[ref naar Agent 0]",
      "aangevallen_element": "[Fout/Schade/Causaal Verband/Andere]",
      "argument_samenvatting": "",
      "juridische_sterkte": {{
        "score": "[Sterk/Matig/Zwak]",
        "onderbouwing": {{
          "sterke_punten": [],
          "zwakke_punten": []
        }}
      }},
      "impact_op_mijn_vordering": {{
        "ernst": "[Fataal/Ernstig/Beperkt/Minimaal]",
        "getroffen_elementen": []
      }},
      "verweer_opties": []
    }}
  ],
  "prioritaire_verweren": []
}}"""
        
        response = call_gemini(self.system_prompt, user_prompt, self.model)
        result = parse_json_response(response, self.name)
        
        if progress_callback:
            if "juridische_analyse" in result:
                progress_callback(f"✅ {self.name} - {len(result['juridische_analyse'])} argumenten geanalyseerd")
            else:
                progress_callback(f"⚠️ {self.name} - Analyse afgerond")
        
        return result

class Agent5_ImpactStrateeg:
    def __init__(self):
        self.name = "Agent 5 - Impact Strateeg"
        self.model = LITE_MODEL
        self.system_prompt = """Je bent een strategisch analist die de gecombineerde output van alle specialistische agenten synthetiseert. Je identificeert cascade-effecten, kruisverbanden en emergente patronen die individuele agenten mogelijk hebben gemist.

Je creëert een holistische impactanalyse en identificeert strategische opportuniteiten.

BELANGRIJK: Geef je antwoord ALLEEN in JSON format."""
    
    def analyze(self, agent1_output: Dict, agent2_output: Dict, agent3_output: Dict, 
                agent0_output: Dict, mijn_argumentatie: str, progress_callback=None) -> Dict:
        """Analyseer gecombineerde impact"""
        if progress_callback:
            progress_callback(f"🎯 {self.name} - Start synthese...")
        
        user_prompt = f"""SYNTHESISEER de analyses van alle agenten:
- Agent 0 (nieuwe elementen): {json.dumps(agent0_output.get('samenvatting', {}), indent=2)}
- Procedureel: {json.dumps(agent1_output, indent=2)}
- Feitelijk: {json.dumps(agent2_output, indent=2)}
- Juridisch: {json.dumps(agent3_output, indent=2)}

IDENTIFICEER:
1. Cascade-effecten
2. Synergiën tussen argumenten
3. Strategische opportuniteiten
4. Impact van visuele bewijzen

OUTPUT FORMAT (JSON):
{{
  "impact_matrix": {{
    "direct_aangevallen_argumenten": [],
    "indirecte_impacts": []
  }},
  "synergie_analyse": [],
  "nieuwe_opportuniteiten": [],
  "visueel_bewijs_strategie": "[hoe om te gaan met visuele elementen]",
  "structurele_integriteit": {{
    "hoofdargumentatie_intact": [true/false],
    "kritieke_pijlers_aangetast": [],
    "herstructurering_nodig": [true/false],
    "voorgestelde_aanpassingen": []
  }}
}}"""
        
        response = call_gemini(self.system_prompt, user_prompt, self.model)
        result = parse_json_response(response, self.name)
        
        if progress_callback:
            progress_callback(f"✅ {self.name} - Impact matrix opgesteld")
        
        return result

class Agent4_Synthesizer:
    def __init__(self):
        self.name = "Agent 4 - Eindverantwoordelijke Verdediger"
        self.model = ADVANCED_MODEL
        self.system_prompt = """Je bent de hoofdstrateeg die alle analyses integreert tot een coherent actieplan. Je formuleert concrete teksten voor de repliek, prioriteert acties, en ontwikkelt de optimale processtrategie.

Je output is direct bruikbaar voor de advocaat in de rechtszaal. Wees zeer precies in je juridische formuleringen en gebruik de Belgische juridische stijl."""
    
    def synthesize(self, all_outputs: Dict, progress_callback=None) -> Dict:
        """Creëer finale synthese en actieplan"""
        if progress_callback:
            progress_callback(f"🏆 {self.name} - Finale synthese (Advanced Model)...")
        
        summary = {
            "nieuwe_elementen": len(all_outputs.get('agent0', {}).get('nieuwe_elementen', [])),
            "procedurele_risicos": len(all_outputs.get('agent1', {}).get('procedurele_analyse', [])),
            "feitelijke_issues": len(all_outputs.get('agent2', {}).get('feitelijke_analyse', [])),
            "juridische_aanvallen": len(all_outputs.get('agent3', {}).get('juridische_analyse', [])),
            "visuele_elementen": all_outputs.get('agent0', {}).get('samenvatting', {}).get('visuele_elementen_gevonden', 0),
            "hoofdrisico": all_outputs.get('agent5', {}).get('structurele_integriteit', {})
        }
        
        user_prompt = f"""FINALE SYNTHESE - Samenvatting van analyses:
{json.dumps(summary, indent=2)}

BELANGRIJKSTE BEVINDINGEN:
- Agent 1 (Procedureel): {all_outputs.get('agent1', {}).get('prioriteit_volgorde', [])}
- Agent 2 (Feitelijk): {all_outputs.get('agent2', {}).get('grootste_risico_feiten', [])}
- Agent 3 (Juridisch): {all_outputs.get('agent3', {}).get('prioritaire_verweren', [])}
- Visuele bewijzen: {summary['visuele_elementen']} elementen gedetecteerd

CREËER:
1. Executive summary (3 zinnen)
2. Top 3 prioritaire acties
3. Hoofdverweer formulering
4. Strategie voor visuele bewijzen

OUTPUT in JSON format."""
        
        response = call_gemini(self.system_prompt, user_prompt, self.model)
        result = parse_json_response(response, self.name)
        
        if "error" in result or "fallback" in result:
            result = {
                "strategisch_memorandum": {
                    "executive_summary": "Synthese gefaald - check individuele agent outputs",
                    "prioritaire_acties": ["Check agent logs voor details"],
                    "verweer_hierarchie": {}
                }
            }
        
        if progress_callback:
            progress_callback(f"✅ {self.name} - Synthese compleet")
        
        return result

def main():
    st.markdown('<h1 class="main-header">⚖️ Juridisch Multi-Agent Analysesysteem</h1>', unsafe_allow_html=True)
    
    # Sidebar voor configuratie
    with st.sidebar:
        st.header("🔧 Configuratie")
        
        # API Key input - check eerst Streamlit secrets
        api_key = st.secrets.get("GOOGLE_API_KEY", "")
        
        if not api_key:
            api_key = st.text_input(
                "Google API Key",
                type="password",
                help="Voer je Google Gemini API key in"
            )
        
        if api_key:
            genai.configure(api_key=api_key)
            st.session_state.api_key_set = True
            st.success("✅ API Key geconfigureerd")
        else:
            st.error("⚠️ Voer eerst je API key in")
        
        st.divider()
        
        # OCR & Vision info
        st.subheader("🖼️ OCR & Beeldanalyse")
        st.info("""
        **Nieuwe Functionaliteit:**
        - 📷 Automatische OCR voor afbeeldingen
        - 📄 PDF/DOCX tekst extractie
        - 🔍 Beeldanalyse voor grafieken/schema's
        - 🖊️ Handtekening detectie
        
        **Ondersteunde formaten:**
        PDF, DOCX, TXT, JPG, PNG, GIF
        """)
        
        st.divider()
        
        # Model info
        st.subheader("🤖 Model Configuratie")
        st.info(f"""
        **Analyse Agents (0-3, 5):**
        {LITE_MODEL}
        
        **Synthesizer (Agent 4):**
        {ADVANCED_MODEL}
        
        **OCR & Vision:**
        {VISION_MODEL}
        """)
        
        st.divider()
        
        # Help sectie
        st.subheader("❓ Help")
        with st.expander("Hoe werkt het?"):
            st.write("""
            1. Upload de drie vereiste documenten
            2. Het systeem extraheert automatisch tekst
            3. Afbeeldingen worden geanalyseerd met AI
            4. Klik op 'Start Analyse'
            5. Download het rapport en de repliek
            
            **Tips:**
            - Upload PDFs met gescande documenten
            - Afbeeldingen worden automatisch verwerkt
            - Schema's en grafieken worden beschreven
            """)
    
    # Hoofdinterface
    if not st.session_state.api_key_set:
        st.warning("👈 Configureer eerst je API key in de sidebar")
        return
    
    # Document upload sectie
    st.header("📄 Document Upload & Verwerking")
    
    # Info box over OCR
    st.info("💡 **Tip:** Upload documenten met afbeeldingen, handtekeningen of gescande pagina's. Het systeem zal deze automatisch analyseren!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1️⃣ Historiek Tegenpartij")
        historiek_file = st.file_uploader(
            "Upload historiek",
            type=['txt', 'pdf', 'docx', 'jpg', 'png', 'gif'],
            key="historiek",
            help="Alle eerdere conclusies van de tegenpartij (inclusief scans)"
        )
        
        if historiek_file:
            with st.expander("📋 Preview & Extractie"):
                historiek_text = extract_and_analyze_content(historiek_file, "historiek")
                st.text_area("Geëxtraheerde tekst:", historiek_text[:500] + "...", height=150)
    
    with col2:
        st.subheader("2️⃣ Laatste Conclusie")
        laatste_conclusie_file = st.file_uploader(
            "Upload laatste conclusie",
            type=['txt', 'pdf', 'docx', 'jpg', 'png', 'gif'],
            key="laatste",
            help="De nieuwste conclusie van de tegenpartij"
        )
        
        if laatste_conclusie_file:
            with st.expander("📋 Preview & Extractie"):
                laatste_text = extract_and_analyze_content(laatste_conclusie_file, "laatste")
                st.text_area("Geëxtraheerde tekst:", laatste_text[:500] + "...", height=150)
    
    with col3:
        st.subheader("3️⃣ Mijn Argumentatie")
        mijn_argumentatie_file = st.file_uploader(
            "Upload eigen argumentatie",
            type=['txt', 'pdf', 'docx', 'jpg', 'png', 'gif'],
            key="mijn",
            help="Je eigen juridische argumentatie"
        )
        
        if mijn_argumentatie_file:
            with st.expander("📋 Preview & Extractie"):
                mijn_text = extract_and_analyze_content(mijn_argumentatie_file, "mijn")
                st.text_area("Geëxtraheerde tekst:", mijn_text[:500] + "...", height=150)
    
    # Extra afbeeldingen upload sectie
    st.divider()
    st.subheader("🖼️ Extra Afbeeldingen (Optioneel)")
    extra_images = st.file_uploader(
        "Upload extra afbeeldingen voor analyse",
        type=['jpg', 'png', 'gif'],
        accept_multiple_files=True,
        help="Upload losse afbeeldingen van bewijsstukken, handtekeningen, schema's, etc."
    )
    
    if extra_images:
        st.write(f"📷 {len(extra_images)} extra afbeelding(en) geüpload")
        with st.expander("🔍 Bekijk afbeeldinganalyses"):
            for img in extra_images:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(img, width=200)
                with col2:
                    if st.button(f"Analyseer {img.name}", key=f"analyze_{img.name}"):
                        with st.spinner("Analyseren..."):
                            analysis = process_image_with_gemini(img.read(), f"Extra bewijs: {img.name}")
                            st.text_area("Analyse:", analysis, height=200)
    
    # Check of alle files zijn geüpload
    all_files_uploaded = all([historiek_file, laatste_conclusie_file, mijn_argumentatie_file])
    
    if all_files_uploaded:
        st.success("✅ Alle documenten geüpload en verwerkt")
        
        # Analyse knop
        if st.button("🚀 Start Juridische Analyse", type="primary", use_container_width=True):
            # Progress container
            progress_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Log container
                with st.expander("📋 Analyse Log", expanded=True):
                    log_container = st.empty()
                    logs = []
                    
                    def update_progress(message):
                        logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {message}")
                        log_container.text("\n".join(logs))
                
                try:
                    # Gebruik geëxtraheerde teksten
                    update_progress("📖 Gebruik geëxtraheerde en geanalyseerde content...")
                    historiek = st.session_state.extracted_texts.get("historiek", "")
                    laatste_conclusie = st.session_state.extracted_texts.get("laatste", "")
                    mijn_argumentatie = st.session_state.extracted_texts.get("mijn", "")
                    
                    # Voeg extra afbeeldinganalyses toe indien aanwezig
                    if extra_images:
                        update_progress(f"🖼️ Verwerk {len(extra_images)} extra afbeeldingen...")
                        extra_content = "\n\n[EXTRA VISUELE BEWIJZEN]\n"
                        for img in extra_images:
                            img_bytes = img.read()
                            analysis = process_image_with_gemini(img_bytes, f"Extra bewijs: {img.name}")
                            extra_content += f"\n[AFBEELDING: {img.name}]\n{analysis}\n"
                        laatste_conclusie += extra_content
                    
                    progress_bar.progress(10)
                    
                    # Agent analyses
                    all_outputs = {}
                    
                    # Agent 0
                    status_text.text("🔍 Agent 0 - Nieuwheidsdetectie...")
                    agent0 = Agent0_NieuwheidsDetector()
                    agent0_output = agent0.analyze(historiek, laatste_conclusie, update_progress)
                    all_outputs['agent0'] = agent0_output
                    progress_bar.progress(25)
                    
                    if not agent0_output.get('nieuwe_elementen'):
                        st.warning("Geen nieuwe elementen gedetecteerd. Analyse gestopt.")
                        return
                    
                    # Agent 1
                    status_text.text("⚖️ Agent 1 - Procedurele analyse...")
                    agent1 = Agent1_ProcedureleStrateeg()
                    agent1_output = agent1.analyze(agent0_output, update_progress)
                    all_outputs['agent1'] = agent1_output
                    progress_bar.progress(40)
                    
                    # Agent 2
                    status_text.text("🔬 Agent 2 - Feitelijke analyse...")
                    agent2 = Agent2_FeitelijkeOnderzoeker()
                    agent2_output = agent2.analyze(agent0_output, mijn_argumentatie, update_progress)
                    all_outputs['agent2'] = agent2_output
                    progress_bar.progress(55)
                    
                    # Agent 3
                    status_text.text("📚 Agent 3 - Juridische analyse...")
                    agent3 = Agent3_JuridischeTacticus()
                    agent3_output = agent3.analyze(agent0_output, agent2_output, mijn_argumentatie, update_progress)
                    all_outputs['agent3'] = agent3_output
                    progress_bar.progress(70)
                    
                    # Agent 5
                    status_text.text("🎯 Agent 5 - Impact analyse...")
                    agent5 = Agent5_ImpactStrateeg()
                    agent5_output = agent5.analyze(
                        agent1_output, agent2_output, agent3_output, 
                        agent0_output, mijn_argumentatie, update_progress
                    )
                    all_outputs['agent5'] = agent5_output
                    progress_bar.progress(85)
                    
                    # Agent 4
                    status_text.text("🏆 Agent 4 - Finale synthese...")
                    agent4 = Agent4_Synthesizer()
                    agent4_output = agent4.synthesize(all_outputs, update_progress)
                    all_outputs['agent4'] = agent4_output
                    progress_bar.progress(100)
                    
                    # Sla resultaten op
                    st.session_state.results = all_outputs
                    st.session_state.analysis_complete = True
                    
                    status_text.text("✅ Analyse compleet!")
                    update_progress("🎉 Alle analyses succesvol afgerond!")
                    
                except Exception as e:
                    st.error(f"❌ Fout tijdens analyse: {str(e)}")
                    update_progress(f"ERROR: {str(e)}")
    
    # Toon resultaten
    if st.session_state.analysis_complete and st.session_state.results:
        st.divider()
        st.header("📊 Analyse Resultaten")
        
        results = st.session_state.results
        
        # Executive Summary
        st.subheader("📋 Executive Summary")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            nieuwe_elementen = len(results.get('agent0', {}).get('nieuwe_elementen', []))
            st.metric("Nieuwe Elementen", nieuwe_elementen)
        
        with col2:
            proc_risicos = len(results.get('agent1', {}).get('procedurele_analyse', []))
            st.metric("Procedurele Risico's", proc_risicos)
        
        with col3:
            feit_issues = len(results.get('agent2', {}).get('feitelijke_analyse', []))
            st.metric("Feitelijke Issues", feit_issues)
        
        with col4:
            jur_aanvallen = len(results.get('agent3', {}).get('juridische_analyse', []))
            st.metric("Juridische Aanvallen", jur_aanvallen)
        
        with col5:
            visuele_elementen = results.get('agent0', {}).get('samenvatting', {}).get('visuele_elementen_gevonden', 0)
            st.metric("Visuele Bewijzen", visuele_elementen)
        
        # Structurele integriteit
        integriteit = results.get('agent5', {}).get('structurele_integriteit', {})
        if integriteit.get('hoofdargumentatie_intact', False):
            st.success("✅ Hoofdargumentatie blijft intact")
        else:
            st.error("❌ Hoofdargumentatie aangetast - herstructurering nodig")
        
        # Visueel bewijs strategie
        visueel_strategie = results.get('agent5', {}).get('visueel_bewijs_strategie', '')
        if visueel_strategie:
            st.info(f"🖼️ **Strategie voor visuele bewijzen:** {visueel_strategie}")
        
        # Tabs voor gedetailleerde resultaten
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🔍 Nieuwe Elementen",
            "⚖️ Procedureel",
            "🔬 Feitelijk",
            "📚 Juridisch",
            "🎯 Impact",
            "🏆 Synthese"
        ])
        
        with tab1:
            st.subheader("Agent 0 - Nieuwheidsdetector")
            nieuwe = results.get('agent0', {}).get('nieuwe_elementen', [])
            for elem in nieuwe:
                with st.expander(f"{elem['id']} - {elem['categorie']} {'🖼️' if elem.get('bevat_visueel_element', False) else ''}"):
                    st.write(f"**Citaat:** {elem['citaat']}")
                    st.write(f"**Waarom nieuw:** {elem['waarom_nieuw']}")
                    st.write(f"**Confidence:** {elem['confidence']}")
                    if elem.get('bevat_visueel_element', False):
                        st.info("🖼️ Dit element bevat visuele componenten")
        
        with tab2:
            st.subheader("Agent 1 - Procedurele Strateeg")
            proc = results.get('agent1', {}).get('procedurele_analyse', [])
            for analyse in proc:
                with st.expander(f"{analyse['type_exceptie']} - Sterkte: {analyse['sterkte_beoordeling']['score']}"):
                    st.write(f"**Motivering:** {analyse['sterkte_beoordeling']['motivering']}")
                    st.write(f"**Artikelen:** {', '.join(analyse['juridische_basis']['artikelen'])}")
        
        with tab3:
            st.subheader("Agent 2 - Feitelijke Onderzoeker")
            feiten = results.get('agent2', {}).get('feitelijke_analyse', [])
            for feit in feiten:
                with st.expander(f"{feit['element_id']} - Bewijskracht: {feit['bewijs_analyse']['bewijskracht']}"):
                    st.write(f"**Type bewijs:** {feit.get('type_bewijs', 'Onbekend')}")
                    st.write(f"**Samenvatting:** {feit['feit_samenvatting']}")
                    st.write(f"**Bewijsgaten:** {', '.join(feit['bewijs_analyse']['bewijs_gaten'])}")
                    if feit['bewijs_analyse'].get('visuele_component'):
                        st.info(f"🖼️ Visuele component: {feit['bewijs_analyse']['visuele_component']}")
        
        with tab4:
            st.subheader("Agent 3 - Juridische Tacticus")
            jur = results.get('agent3', {}).get('juridische_analyse', [])
            for arg in jur:
                with st.expander(f"{arg['aangevallen_element']} - Impact: {arg['impact_op_mijn_vordering']['ernst']}"):
                    st.write(f"**Samenvatting:** {arg['argument_samenvatting']}")
                    st.write(f"**Verweer opties:** {', '.join(arg['verweer_opties'])}")
        
        with tab5:
            st.subheader("Agent 5 - Impact Strateeg")
            impact = results.get('agent5', {})
            if 'nieuwe_opportuniteiten' in impact:
                st.write("**Nieuwe Opportuniteiten:**")
                for opp in impact['nieuwe_opportuniteiten']:
                    st.write(f"- {opp}")
            if 'visueel_bewijs_strategie' in impact:
                st.write(f"\n**Visueel Bewijs Strategie:** {impact['visueel_bewijs_strategie']}")
        
        with tab6:
            st.subheader("Agent 4 - Eindverantwoordelijke Verdediger")
            synthese = results.get('agent4', {}).get('strategisch_memorandum', {})
            if 'executive_summary' in synthese:
                st.write("**Executive Summary:**")
                st.info(synthese['executive_summary'])
            if 'prioritaire_acties' in synthese:
                st.write("**Prioritaire Acties:**")
                for actie in synthese['prioritaire_acties']:
                    st.write(f"- {actie}")
        
        # Download opties
        st.divider()
        st.subheader("💾 Download Resultaten")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # JSON export
            json_str = json.dumps(results, indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 Download Volledige Analyse (JSON)",
                data=json_str,
                file_name=f"juridische_analyse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            # Markdown rapport
            visuele_info = f"\n- Visuele elementen gedetecteerd: {results.get('agent0', {}).get('samenvatting', {}).get('visuele_elementen_gevonden', 0)}" if results.get('agent0', {}).get('samenvatting', {}).get('visuele_elementen_gevonden', 0) > 0 else ""
            
            rapport = f"""# Juridische Multi-Agent Analyse Rapport
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
- Nieuwe elementen gedetecteerd: {len(results.get('agent0', {}).get('nieuwe_elementen', []))}
- Procedurele risico's: {len(results.get('agent1', {}).get('procedurele_analyse', []))}
- Feitelijke issues: {len(results.get('agent2', {}).get('feitelijke_analyse', []))}
- Juridische aanvallen: {len(results.get('agent3', {}).get('juridische_analyse', []))}{visuele_info}

## Structurele Integriteit
Hoofdargumentatie intact: {'Ja' if results.get('agent5', {}).get('structurele_integriteit', {}).get('hoofdargumentatie_intact', False) else 'Nee'}

## Prioritaire Acties
{chr(10).join('- ' + actie for actie in results.get('agent4', {}).get('strategisch_memorandum', {}).get('prioritaire_acties', []))}

## Visueel Bewijs Strategie
{results.get('agent5', {}).get('visueel_bewijs_strategie', 'Geen visuele bewijzen geanalyseerd')}
"""
            
            st.download_button(
                label="📥 Download Rapport (Markdown)",
                data=rapport,
                file_name=f"juridisch_rapport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
        
        with col3:
            # Geëxtraheerde content export
            extracted_content = f"""# Geëxtraheerde Document Content
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Historiek Tegenpartij
{st.session_state.extracted_texts.get('historiek', 'Geen content')}

## Laatste Conclusie Tegenpartij
{st.session_state.extracted_texts.get('laatste', 'Geen content')}

## Mijn Argumentatie
{st.session_state.extracted_texts.get('mijn', 'Geen content')}
"""
            
            st.download_button(
                label="📥 Download Geëxtraheerde Tekst",
                data=extracted_content,
                file_name=f"geextraheerde_content_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()