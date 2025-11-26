import streamlit as st
import cv2
import numpy as np
import pytesseract
import json
from fuzzywuzzy import process
from PIL import Image

# Konfiguracja strony Streamlit
st.set_page_config(page_title="NMS Inventory Analyzer v3", page_icon="🚀")

# --- FUNKCJE DANYCH I PRZETWARZANIA ---

@st.cache_data
def load_db():
    try:
        # Ładowanie bazy danych przedmiotów
        with open('nms_items.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Błąd: Nie znaleziono pliku nms_items.json!")
        return {}

def process_image(pil_image):
    # 1. Konwersja PIL -> OpenCV
    img_array = np.array(pil_image)
    if len(img_array.shape) == 3:
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img = img_array
    
    # 2. Powiększenie (Upscaling) 2x
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # 3. Konwersja na szarość
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 4. Odszumianie (Blur)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 5. Adaptive Thresholding (Kluczowe dla UI gier)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 31, 2)

    # 6. OCR
    custom_config = r'--psm 11' # Tryb dla rzadkiego tekstu
    text = pytesseract.image_to_string(thresh, config=custom_config)
    
    return text, thresh # Zwracamy też obrazek 'thresh' do podglądu debug

def analyze_text(raw_text, db):
    results = []
    debug_matches = []
    
    lines = [line.strip() for line in raw_text.split('\n') if len(line) > 3]
    
    # KRYTYCZNA POPRAWKA: Filtrowanie kluczy
    # Wybieramy tylko klucze (nazwy przedmiotów), których wartość jest słownikiem (dict), 
    # co pozwala wykluczyć komentarze (stringi).
    db_keys = [key for key, value in db.items() if isinstance(value, dict)]
    
    for line in lines:
        # A. Agresywne czyszczenie: usuwamy znaki specjalne
        clean_line = ''.join(c for c in line if c.isalnum() or c.isspace()).strip()
        
        if not clean_line:
            continue
            
        # B. Szukamy najlepszego dopasowania w bazie (nawet jeśli ma niski wynik)
        best_match, score = process.extractOne(clean_line.upper(), db_keys)
        
        # Dodajemy informację do listy debugowania, ZAWSZE
        debug_matches.append({
            "OCR Saw": line,
            "Cleaned": clean_line.upper(),
            "Best Match": best_match,
            "Score": score
        })
        
        # C. Sprawdzamy, czy dopasowanie przekroczyło próg
        if score >= 70: 
            item_data = db[best_match]
            if not any(d['Przedmiot'] == best_match for d in results):
                results.append({
                    "Przedmiot": best_match,
                    "Akcja": item_data['action'], 
                    "Typ": item_data['type'],
                    "Rada": item_data['tip'],
                    "Oryginał": line
                })
    return results, debug_matches

# --- INTERFEJS UŻYTKOWNIKA (FRONTEND) ---

st.title("🚀 NMS Inventory Analyzer v3 (Final Fix)")
st.write("Wgraj screen z PS App.")

uploaded_file = st.file_uploader("Wybierz zdjęcie...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. Otwieramy i wyświetlamy oryginalny obraz
    image = Image.open(uploaded_file)
    st.image(image, caption='Oryginał', use_column_width=True)
    
    st.write("⚙️ Przetwarzam obraz i dopasowuję do bazy...")
    
    database = load_db()
    raw_text, processed_img = process_image(image)
    
    # 2. Analiza tekstu
    found_items, debug_matches = analyze_text(raw_text, database) 
    
    # --- DEBUG VIEW ---
    with st.expander("👁️ Zobacz diagnostykę OCR i dopasowania", expanded=False):
        st.write("Wiersze ze 'Score' poniżej 70 są odrzucane przez program.")
        st.dataframe(debug_matches)
        st.text("Surowy tekst (do weryfikacji błędów):")
        st.text(raw_text)
        st.image(processed_img, caption='Obraz po filtrach', use_column_width=True)

    # --- WYNIKI ---
    if found_items:
        st.success(f"Znaleziono {len(found_items)} unikalnych przedmiotów! Czas na porządki!")
        for item in found_items:
            # Ustawianie koloru na podstawie akcji
            color = "green" if item['Akcja'] == "TRZYMAJ" else "red"
            if "SPRZEDAJ" in item['Akcja'] or "HANDEL" in item['Akcja']: color = "orange"
            
            with st.container():
                st.markdown(f"### :{color}[{item['Akcja']}] {item['Przedmiot']}")
                st.caption(f"Typ: {item['Typ']}")
                st.info(item['Rada'])
                st.divider()
    else:
        st.error("Nie znaleziono przedmiotów w bazie (Score < 70).")
        st.info("Sprawdź tabelę w sekcji 'Diagnostyka dopasowania'. Jeśli widzisz przedmiot, którego nie ma w bazie lub jego Score jest zbyt niski, dodaj go do pliku nms_items.json.")
