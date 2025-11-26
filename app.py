import streamlit as st
import cv2
import numpy as np
import pytesseract
import json
from fuzzywuzzy import process
from PIL import Image

# Konfiguracja strony
st.set_page_config(page_title="NMS Inventory Analyzer", page_icon="🚀")

# --- FUNKCJE ---

@st.cache_data
def load_db():
    try:
        with open('nms_items.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Błąd: Nie znaleziono pliku nms_items.json!")
        return {}

def process_image(image_file):
    # Konwersja wgranego pliku na format zrozumiały dla OpenCV
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Przetwarzanie obrazu (szarość + kontrast)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # OCR
    text = pytesseract.image_to_string(thresh, config='--psm 11')
    return text

def analyze_text(raw_text, db):
    results = []
    lines = [line.strip() for line in raw_text.split('\n') if len(line) > 3]
    db_keys = list(db.keys())
    
    for line in lines:
        # Fuzzy matching - szukamy podobieństwa
        match, score = process.extractOne(line.upper(), db_keys)
        if score >= 80: # Próg pewności 80%
            item_data = db[match]
            results.append({
                "Przedmiot": match,
                "Akcja": item_data['action'],
                "Typ": item_data['type'],
                "Rada": item_data['tip']
            })
    return results

# --- INTERFEJS (FRONTEND) ---

st.title("🚀 NMS Inventory Analyzer")
st.write("Wrzuć screen z PS App, a powiem Ci co sprzedać.")

# Wgrywanie pliku
uploaded_file = st.file_uploader("Wybierz zdjęcie...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Wyświetl obrazek (dla pewności)
    image = Image.open(uploaded_file)
    st.image(image, caption='Twój ekwipunek', use_column_width=True)
    
    st.write("🔍 Analizuję obraz...")
    
    # Logika
    database = load_db()
    raw_text = process_image(uploaded_file)
    found_items = analyze_text(raw_text, database)
    
    # Wyniki
    if found_items:
        st.success(f"Znaleziono {len(found_items)} pasujących przedmiotów!")
        
        for item in found_items:
            # Kolorowanie ramek w zależności od akcji
            color = "green" if item['Akcja'] == "TRZYMAJ" else "red"
            if "SPRZEDAJ" in item['Akcja']: color = "orange"
            
            with st.container():
                st.markdown(f"### :{color}[{item['Akcja']}] {item['Przedmiot']}")
                st.caption(f"Typ: {item['Typ']}")
                st.info(item['Rada'])
                st.divider()
    else:
        st.warning("Nie udało się rozpoznać znanych przedmiotów. Spróbuj wyraźniejsze zdjęcie lub zaktualizuj bazę danych.")

    # Debug (opcjonalnie - żeby widzieć co OCR przeczytał surowo)
    with st.expander("Pokaż surowy tekst z OCR (dla debugowania)"):
        st.text(raw_text)