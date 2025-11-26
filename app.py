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

def process_image(pil_image):
    # POPRAWKA: Konwersja bezpośrednio z obrazu PIL na format OpenCV (NumPy array)
    # Dzięki temu nie musimy czytać pliku drugi raz
    img_array = np.array(pil_image)
    
    # PIL używa RGB, OpenCV domyślnie BGR, ale my i tak robimy szarość
    # więc używamy COLOR_RGB2GRAY
    if len(img_array.shape) == 3: # Jeśli obraz jest kolorowy
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else: # Jeśli obraz już jest czarno-biały
        gray = img_array

    # Zwiększenie kontrastu (Binaryzacja)
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
        # Obniżyłem lekko próg do 75%, bo zdjęcia z TV mogą być mniej wyraźne
        if score >= 75: 
            item_data = db[match]
            # Sprawdzamy czy nie dodajemy tego samego przedmiotu kilka razy
            if not any(d['Przedmiot'] == match for d in results):
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
    # 1. Otwieramy obraz raz za pomocą PIL
    image = Image.open(uploaded_file)
    
    # Wyświetlamy obrazek
    st.image(image, caption='Twój ekwipunek', use_column_width=True)
    
    st.write("🔍 Analizuję obraz...")
    
    # Logika
    database = load_db()
    
    # POPRAWKA: Przekazujemy otwarty obiekt 'image', a nie plik 'uploaded_file'
    raw_text = process_image(image)
    
    found_items = analyze_text(raw_text, database)
    
    # Wyniki
    if found_items:
        st.success(f"Znaleziono {len(found_items)} pasujących przedmiotów!")
        
        for item in found_items:
            # Kolorowanie ramek w zależności od akcji
            color = "green" if item['Akcja'] == "TRZYMAJ" else "red"
            if "SPRZEDAJ" in item['Akcja'] or "HANDEL" in item['Akcja']: color = "orange"
            
            with st.container():
                # Używamy markdown do ładnego formatowania
                st.markdown(f"### :{color}[{item['Akcja']}] {item['Przedmiot']}")
                st.caption(f"Typ: {item['Typ']}")
                st.info(item['Rada'])
                st.divider()
    else:
        st.warning("Nie udało się rozpoznać znanych przedmiotów.")
        st.info("Wskazówka: Upewnij się, że zdjęcie jest wyraźne, a nazwy przedmiotów są w naszej bazie JSON.")

    # Debug (opcjonalnie)
    with st.expander("Pokaż surowy tekst z OCR (dla debugowania)"):
        st.text(raw_text)
