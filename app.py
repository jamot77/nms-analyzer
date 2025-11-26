import streamlit as st
import cv2
import numpy as np
import pytesseract
import json
from fuzzywuzzy import process
from PIL import Image

st.set_page_config(page_title="NMS Inventory Analyzer", page_icon="🚀")

@st.cache_data
def load_db():
    try:
        with open('nms_items.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def process_image(pil_image):
    # 1. Konwersja PIL -> OpenCV
    img_array = np.array(pil_image)
    if len(img_array.shape) == 3:
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img = img_array

    # 2. POWIĘKSZENIE (Upscaling) - Kluczowe dla małych napisów
    # Powiększamy obraz 2-krotnie, używając interpolacji sześciennej
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # 3. Konwersja na szarość
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 4. Odszumianie (Lekki Blur)
    # Usuwa "ziarno" ze zdjęcia zrobionego telefonem/screena
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 5. ADAPTIVE THRESHOLDING (To jest game changer)
    # Zamiast sztywnego progu, algorytm bada sąsiedztwo pikseli.
    # Sprawia, że biały tekst na ciemnym tle staje się czarnym tekstem na białym tle.
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 31, 2)

    # 6. OCR
    # psm 11 (sparse text) lub psm 3 (auto segmentation)
    # Dodajemy whitelist (opcjonalnie), żeby szukał tylko liter A-Z
    custom_config = r'--psm 11'
    text = pytesseract.image_to_string(thresh, config=custom_config)
    
    return text, thresh # Zwracamy też obrazek 'thresh' do podglądu

def analyze_text(raw_text, db):
    results = []
    # Filtrujemy bardzo krótkie śmieci (mniej niż 4 znaki)
    lines = [line.strip() for line in raw_text.split('\n') if len(line) > 3]
    db_keys = list(db.keys())
    
    for line in lines:
        # Usuwamy znaki specjalne, które OCR często dodaje (np. | [ ] { })
        clean_line = ''.join(e for e in line if e.isalnum() or e.isspace())
        
        match, score = process.extractOne(clean_line.upper(), db_keys)
        
        # Jeśli wynik jest wysoki, dodajemy
        if score >= 70: # Lekko obniżony próg dla trudnych screenów
            item_data = db[match]
            if not any(d['Przedmiot'] == match for d in results):
                results.append({
                    "Przedmiot": match,
                    "Akcja": item_data['action'],
                    "Typ": item_data['type'],
                    "Rada": item_data['tip'],
                    "Oryginał": line # Debug: co zobaczył OCR
                })
    return results

# --- FRONTEND ---

st.title("🚀 NMS Inventory Analyzer v2")
st.write("Wgraj screen z PS App.")

uploaded_file = st.file_uploader("Wybierz zdjęcie...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Oryginał', use_column_width=True)
    
    st.write("⚙️ Przetwarzam obraz...")
    
    database = load_db()
    
    # Pobieramy tekst ORAZ przetworzony obraz
    raw_text, processed_img = process_image(image)
    
    # --- DEBUG VIEW ---
    with st.expander("👁️ Zobacz jak komputer widzi Twój screen (Debug)"):
        st.write("Jeśli tutaj nie widzisz wyraźnych czarnych liter, OCR też ich nie zobaczy.")
        st.image(processed_img, caption='Obraz po filtrach', use_column_width=True)
        st.text("Surowy tekst:")
        st.text(raw_text)
    # ------------------

    found_items = analyze_text(raw_text, database)
    
    if found_items:
        st.success(f"Znaleziono {len(found_items)} przedmiotów!")
        for item in found_items:
            color = "green" if item['Akcja'] == "TRZYMAJ" else "red"
            if "SPRZEDAJ" in item['Akcja']: color = "orange"
            
            with st.container():
                st.markdown(f"### :{color}[{item['Akcja']}] {item['Przedmiot']}")
                st.caption(f"Typ: {item['Typ']} (Dopasowano z: '{item['Oryginał']}')")
                st.info(item['Rada'])
                st.divider()
    else:
        st.error("Nie znaleziono przedmiotów.")
        st.info("Spójrz w sekcję 'Debug' powyżej. Jeśli tekst na czarno-białym zdjęciu jest zamazany, spróbuj zrobić screena bliżej lub z innej zakładki ekwipunku.")
