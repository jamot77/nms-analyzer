import streamlit as st
import cv2
import numpy as np
import pytesseract
import json
from PIL import Image

# --- STAŁE KONFIGURACYJNE (USTALONE Z TWOJEGO SCREENA 1080p) ---

# --- WYJAŚNIENIE STAŁYCH (NOWY BLOK) ---
# SLOT_WIDTH/HEIGHT: Rozmiar jednego slotu ekwipunku w pikselach (np. 75x75).
# SPACING: Odległość między slotami w pikselach (np. 13px).
# GRID_COLS/ROWS: Wymiary siatki głównej (np. 8x6 dla Cargo).
# START_X/Y: Współrzędne (piksel) górnego lewego rogu PIERWSZEGO slotu siatki Cargo.
# SYMBOL_ROI_...: Współrzędne i rozmiar małego obszaru, z którego wycinamy symbol pierwiastka (np. 'Fe').
# --- KONIEC WYJAŚNIEŃ ---

# --- STAŁE KONFIGURACYJNE (FINALNA KALIBRACJA 4K / 3840x2160) ---

# Wymiary slotów i siatki (dane z Twojego 4K)
SLOT_WIDTH = 165
SLOT_HEIGHT = 165
SPACING = 20 # Odstęp między slotami
GRID_COLS = 10 # PRAWIDŁOWA LICZBA KOLUMN DLA TWOJEJ KONFIGURACJI
GRID_ROWS = 10  # PRAWIDŁOWA LICZBA RZĘDÓW DLA TWOJEJ KONFIGURACJI

# Współrzędne startowe siatki (dostosowane do 4K i celowania w symbol)
START_X = 350 # Na podstawie Twojego udanego testu z tą wartością
START_Y = 950 # Na podstawie Twojego udanego testu z tą wartością

# ROI (Region of Interest) dla symbolu pierwiastka (proporcjonalnie większe)
SYMBOL_ROI_OFFSET_X = 5 # Lekko zmniejszone, by uniknąć zaszumionych krawędzi
SYMBOL_ROI_OFFSET_Y = 5 # Lekko zmniejszone, by uniknąć zaszumionych krawędzi
SYMBOL_ROI_SIZE = 70 

# Baza symboli do konwersji (Musi pasować do kluczy z nms_items.json)
SYMBOL_TO_ITEM = {
    "C": "CARBON", "NA": "SODIUM", "FE": "FERRITE DUST",
    "O": "OXYGEN", "ZN": "ZINC", "CU": "COPPER",
    "H": "HYDROGEN", "CL": "CHLORINE", "CO": "COBALT",
    "FE+": "PURE FERRITE",      
    "O+": "CONDENSED OXYGEN",    
    "NA+": "DI-SODIUM",          
    "+": "PURE FERRITE" # Domyślne mapowanie dla symbolu plus, gdy litera jest ignorowana
}
# --- KONIEC STAŁYCH ---

st.set_page_config(page_title="🧪 NMS Symbol Analyzer", page_icon="🧪")

# --- FUNKCJE DANYCH I PRZETWARZANIA ---

@st.cache_data
def load_db():
    try:
        with open('nms_items.json', 'r', encoding='utf-8') as f:
            data = {k: v for k, v in json.load(f).items() if isinstance(v, dict)}
            return data
    except FileNotFoundError:
        st.error("Błąd: Nie znaleziono pliku nms_items.json!")
        return {}

def preprocess_image(img_cv):
    """
    Krok 0: Wstępne przetwarzanie obrazu (Adaptive Thresholding).
    Zwraca przetworzony obraz, który może być użyty do pełnego OCR.
    """
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    
    # Adaptive Thresholding lepiej radzi sobie ze zmiennym oświetleniem/kontrastem
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    return thresh

def find_symbol_slots(img_cv):
    """
    Krok 1: Wycina i wstępnie przetwarza maleńkie obszary symboli.
    """
    symbol_images = []
    
    # Przetwarzanie całego obrazu (Adaptive Thresholding)
    thresh = preprocess_image(img_cv)
    
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            # Obliczanie współrzędnych ROI symbolu
            x_start = START_X + col * (SLOT_WIDTH + SPACING) + SYMBOL_ROI_OFFSET_X
            y_start = START_Y + row * (SLOT_HEIGHT + SPACING) + SYMBOL_ROI_OFFSET_Y
            x_end = x_start + SYMBOL_ROI_SIZE
            y_end = y_start + SYMBOL_ROI_SIZE
            
            # Wycina ROI symbolu
            symbol_img = thresh[y_start:y_end, x_start:x_end]
            
            # Weryfikacja: upewniamy się, że slot został poprawnie wycięty
            if symbol_img.shape[0] == SYMBOL_ROI_SIZE and symbol_img.shape[1] == SYMBOL_ROI_SIZE:
                
                # --- WIZUALNY ZNACZNIK DEBUGOWANIA (BIAŁY KRZYŻYK) ---
                # Rysujemy biały krzyżyk na wycinanym obszarze.
                center = SYMBOL_ROI_SIZE // 2
                cv2.line(symbol_img, (center-5, center), (center+5, center), 255, 1)
                cv2.line(symbol_img, (center, center-5), (center, center+5), 255, 1)
                # ----------------------------------------------------

                symbol_images.append(symbol_img)
            else:
                symbol_images.append(None) 

    return symbol_images, thresh # Zwracamy również przetworzony obraz

def analyze_symbols(symbol_images, db):
    """
    Krok 2: Przeprowadza Localized OCR i mapuje symbole na pełne nazwy.
    """
    results = []
    
    # Konfiguracja OCR: brak PSM, lista dozwolonych znaków to litery, cyfry i znak plus (+)
    custom_config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+' 

    for i, symbol_img in enumerate(symbol_images):
        if symbol_img is None: continue
        
        # 1. LOCALIZED OCR
        raw_symbol = pytesseract.image_to_string(symbol_img, config=custom_config).replace(' ', '').replace('\n', '').upper()
        
        # 2. Lookup & Cleaning
        if raw_symbol in SYMBOL_TO_ITEM:
            item_key = SYMBOL_TO_ITEM[raw_symbol]
            
            # 3. Dopasowanie do bazy JSON
            if item_key in db:
                item_data = db[item_key]
                
                if not any(d['Przedmiot'] == item_key for d in results):
                    results.append({
                        "Przedmiot": item_key,
                        "Akcja": item_data['action'], 
                        "Typ": item_data['type'],
                        "Rada": item_data['tip'],
                        "Slot": i
                    })
        
    return results

# --- INTERFEJS UŻYTKOWNIKA (FRONTEND) ---

st.title("🧪 NMS Resource Analyzer (Symbol OCR)")
st.write("Wykrywanie zasobów na podstawie symboli z Tablicy Mendelejewa.")

uploaded_file = st.file_uploader("Wybierz zdjęcie...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. Konwersja
    image_pil = Image.open(uploaded_file)
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    st.write("⚙️ Wykrywam symbole z 48 slotów...")
    
    database = load_db()
    
    # 2. Cięcie i przetwarzanie
    symbol_slots, full_thresholded_image = find_symbol_slots(image_cv)
    
    # 3. Analiza
    found_resources = analyze_symbols(symbol_slots, database)
    
    # --- WYNIKI ---
    if found_resources:
        st.success(f"Znaleziono {len(found_resources)} unikalnych zasobów na podstawie symboli!")
        
        for item in found_resources:
            color = "green" if item['Akcja'] == "TRZYMAJ" else "orange"
            
            with st.container():
                st.markdown(f"### :{color}[{item['Akcja']}] {item['Przedmiot']}")
                st.caption(f"Typ: {item['Typ']} (Slot: {item['Slot']})")
                st.info(item['Rada'])
                st.divider()
    else:
        st.error("Nie znaleziono znanych zasobów. Jeśli widzisz symbole w diagnostyce, zaktualizuj SYMBOL_TO_ITEM.")

    # --- DEBUG VIEW ---
    with st.expander("👁️ DIAGNOSTYKA I WERYFIKACJA (Symbol OCR)", expanded=True):
        
        # 1. PEŁNY PRZETWORZONY OBRAZ (NOWY WYMAGANY BLOK)
        st.subheader("1. Pełny Przetworzony Obraz (Adaptive Threshold)")
        st.image(full_thresholded_image, caption="Cały obraz po filtrowaniu (tu symbole są bardzo wyraźne)", clamp=True)
        
        # 2. WYCINANE SLOTY (WIĘCEJ SLOTÓW)
        st.subheader("2. Wycinki Symboli (2 rzędy - 16 slotów)")
        # Wyświetlamy 16 slotów (2 pełne rzędy)
        if symbol_slots and all(s is not None for s in symbol_slots[:16]):
            row1 = np.hstack(symbol_slots[:8])
            row2 = np.hstack(symbol_slots[8:16])
            combined_symbols = np.vstack([row1, row2])
            st.image(combined_symbols, caption="Wycinki symboli z białymi krzyżykami (Sprawdź, czy celują w symbol)", clamp=True)
        
        # 3. ZAREJESTROWANE SYMBOLE
        st.subheader("3. Konfiguracja")
        st.write(f"Zarejestrowane symbole (w bazie): {list(SYMBOL_TO_ITEM.keys())}")
        st.caption("Jeśli OCR odczytuje '+' zamiast 'FE+', musimy dodać do bazy więcej symboli 'FE', 'NA' itp.")
