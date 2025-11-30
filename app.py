import streamlit as st
import cv2
import numpy as np
import pytesseract
import json
from PIL import Image

# --- STAŁE KONFIGURACYJNE ---
# Stałe związane z WYMIARAMI slotów (muszą być poprawne dla Twojej rozdzielczości 4K)
SLOT_WIDTH = 165
SLOT_HEIGHT = 165
SPACING = 20 

# PRAWIDŁOWY ROZMIAR SIATKI CARGO
GRID_COLS = 8 
GRID_ROWS = 6 

# ROI (Region of Interest) dla symbolu pierwiastka (proporcjonalnie większe)
SYMBOL_ROI_OFFSET_X = 30 
SYMBOL_ROI_OFFSET_Y = 30 
SYMBOL_ROI_SIZE = 70 

# Baza symboli (pozostała bez zmian)
SYMBOL_TO_ITEM = {
    "C": "CARBON", "NA": "SODIUM", "FE": "FERRITE DUST",
    "O": "OXYGEN", "ZN": "ZINC", "CU": "COPPER",
    "H": "HYDROGEN", "CL": "CHLORINE", "CO": "COBALT",
    "FE+": "PURE FERRITE",      
    "O+": "CONDENSED OXYGEN",    
    "NA+": "DI-SODIUM",          
    "+": "PURE FERRITE" 
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
    """Wstępne przetwarzanie obrazu (Adaptive Thresholding)."""
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    return thresh

def find_cargo_anchor(img_cv):
    """
    DYNAMICZNIE WYSZUKUJE NAPIS 'CARGO' NA CAŁYM EKRANIE i określa punkt startowy.
    """
    # Używamy konwersji do PIL dla pełnego OCR
    image_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    
    # Konfiguracja OCR do szukania tekstu (tryb PSM 3 jest dobry dla całej strony)
    full_config = r'--psm 3' 
    
    # Wykonujemy OCR i parsujemy dane
    data = pytesseract.image_to_data(image_pil, config=full_config, output_type=pytesseract.Output.DICT)
    
    # Szukamy słowa 'CARGO'
    for i, text in enumerate(data['text']):
        if text.upper().strip() == 'CARGO':
            # Znaleziono! Używamy współrzędnych tekstu
            x = data['left'][i]
            y = data['top'][i]
            w = data['width'][i]
            h = data['height'][i]
            
            # Punkt startowy dla cięcia siatki CARGO:
            # START_X: X napisu CARGO (lub jego lewa krawędź)
            # START_Y: Dolna krawędź napisu + stały margines (sloty zaczynają się tuż poniżej)
            
            # Zakładamy, że siatka Cargo zaczyna się na tej samej wysokości X
            start_x = x 
            
            # START_Y to dolna krawędź tekstu (y + h) + mały margines (np. 15 pikseli w 4K)
            start_y = y + h + 15
            
            # Zwracamy lewą krawędź i dół napisu jako początek siatki
            return start_x, start_y
            
    # Jeśli nie znaleziono, zwracamy stałe, które ustabilizowaliśmy
    return 350, 1050 

def find_symbol_slots(img_cv, START_X, START_Y):
    """
    Krok 1: Wycina i wstępnie przetwarza maleńkie obszary symboli, 
    używając dynamicznych współrzędnych.
    """
    symbol_images = []
    
    # Przetwarzanie całego obrazu
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
                
                # --- WIZUALNY ZNACZNIK DEBUGOWANIA ---
                center = SYMBOL_ROI_SIZE // 2
                cv2.line(symbol_img, (center-5, center), (center+5, center), 255, 1)
                cv2.line(symbol_img, (center, center-5), (center, center+5), 255, 1)
                # ------------------------------------

                symbol_images.append(symbol_img)
            else:
                symbol_images.append(None) 

    return symbol_images, thresh 

def analyze_symbols(symbol_images, db):
    """
    Krok 2: Przeprowadza Localized OCR i mapuje symbole na pełne nazwy.
    """
    results = []
    
    # Konfiguracja OCR
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

st.title("🧪 NMS Resource Analyzer (Dynamic OCR)")
st.write("Dynamiczne wykrywanie zasobów, start cięcia kotwiczony na nagłówku 'CARGO'.")

uploaded_file = st.file_uploader("Wybierz zdjęcie...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. Konwersja
    image_pil = Image.open(uploaded_file)
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    st.write("⚙️ Wyszukuję punkt kotwiczenia 'CARGO'...")
    
    # ** DYNAMICZNE WYSZUKIWANIE WSPÓŁRZĘDNYCH **
    dynamic_start_x, dynamic_start_y = find_cargo_anchor(image_cv)
    
    st.write(f"✅ Znaleziono punkt startowy (Anchor): X={dynamic_start_x}, Y={dynamic_start_y}")
    
    database = load_db()
    
    st.write("⚙️ Wycinam sloty na podstawie kotwicy...")

    # 2. Cięcie i przetwarzanie (używamy dynamicznych współrzędnych)
    symbol_slots, full_thresholded_image = find_symbol_slots(image_cv, dynamic_start_x, dynamic_start_y)
    
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
        st.error("Nie znaleziono znanych zasobów. Sprawdź diagnostykę poniżej.")

    # --- DEBUG VIEW ---
    with st.expander("👁️ DIAGNOSTYKA I WERYFIKACJA (Symbol OCR)", expanded=True):
        
        # 1. PEŁNY PRZETWORZONY OBRAZ
        st.subheader("1. Pełny Przetworzony Obraz (Adaptive Threshold)")
        st.image(full_thresholded_image, caption="Cały obraz po filtrowaniu", clamp=True)
        
        # 2. WYCINANE SLOTY
        st.subheader("2. Wycinki Symboli (2 rzędy - 16 slotów)")
        if symbol_slots and all(s is not None for s in symbol_slots[:16]):
            row1 = np.hstack(symbol_slots[:8])
            row2 = np.hstack(symbol_slots[8:16])
            combined_symbols = np.vstack([row1, row2])
            st.image(combined_symbols, caption="Wycinki symboli z białymi krzyżykami (muszą celować w symbol!)", clamp=True)
        
        # 3. KONFIGURACJA
        st.subheader("3. Konfiguracja")
        st.write(f"Zarejestrowane symbole (w bazie): {list(SYMBOL_TO_ITEM.keys())}")
        st.write(f"Wymiary Siatki (oczekiwane): {GRID_COLS}x{GRID_ROWS}")
