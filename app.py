import streamlit as st
import cv2
import numpy as np
import pytesseract
import json
from PIL import Image

# --- STAŁE KONFIGURACYJNE (USTALONE Z TWOJEGO SCREENA 1080p) ---

# Wymiary slotów i siatki
SLOT_WIDTH = 75
SLOT_HEIGHT = 75
SPACING = 13
GRID_COLS = 8
GRID_ROWS = 6

# Współrzędne startowe siatki Cargo (ustabilizowane po wielu testach)
START_X = 50 
START_Y = 265 

# ROI (Region of Interest) dla symbolu pierwiastka (wewnątrz slotu 75x75)
# Zwiększono rozmiar na 35x35 dla lepszego działania OCR
SYMBOL_ROI_OFFSET_X = 15 
SYMBOL_ROI_OFFSET_Y = 15 
SYMBOL_ROI_SIZE = 35 # Zwiększone z 25 na 35

# Baza symboli do konwersji (Musi pasować do kluczy z nms_items.json)
SYMBOL_TO_ITEM = {
    "C": "CARBON", "NA": "SODIUM", "FE": "FERRITE DUST",
    "O": "OXYGEN", "ZN": "ZINC", "CU": "COPPER",
    "H": "HYDROGEN", "CL": "CHLORINE", "CO": "COBALT",
    "FE+": "PURE FERRITE",      # Wersja z plusem
    "O+": "CONDENSED OXYGEN",    # Wersja z plusem
    "NA+": "DI-SODIUM",          # Wersja z plusem
    # Dodaj tutaj więcej symboli i ich ulepszonych wersji!
}
# --- KONIEC STAŁYCH ---

st.set_page_config(page_title="🧪 NMS Symbol Analyzer", page_icon="🧪")

# --- FUNKCJE DANYCH I PRZETWARZANIA ---

@st.cache_data
def load_db():
    try:
        with open('nms_items.json', 'r', encoding='utf-8') as f:
            # Wczytujemy tylko słowniki, aby uniknąć błędów
            data = {k: v for k, v in json.load(f).items() if isinstance(v, dict)}
            return data
    except FileNotFoundError:
        st.error("Błąd: Nie znaleziono pliku nms_items.json!")
        return {}

def find_symbol_slots(img_cv):
    """
    Krok 1: Wycina i wstępnie przetwarza maleńkie obszary symboli.
    Dodano znaczniki wizualne w debugu (krzyżyki).
    """
    symbol_images = []
    
    # Przetwarzanie całego obrazu: szarość i Binary Thresholding
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY) 
    
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
                # Rysujemy biały krzyżyk, aby potwierdzić, że obszar jest wycinany.
                center = SYMBOL_ROI_SIZE // 2
                cv2.line(symbol_img, (center-5, center), (center+5, center), 255, 1) # Linia pozioma
                cv2.line(symbol_img, (center, center-5), (center, center+5), 255, 1) # Linia pionowa
                # ----------------------------------------------------

                symbol_images.append(symbol_img)
            else:
                symbol_images.append(None) 

    return symbol_images

def analyze_symbols(symbol_images, db):
    """
    Krok 2: Przeprowadza Localized OCR i mapuje symbole na pełne nazwy.
    """
    results = []
    
    # Konfiguracja OCR: usunięto PSM 10, aby poprawić rozpoznawanie liter.
    # Lista dozwolonych znaków to litery, cyfry i znak plus (+)
    custom_config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+' 

    for i, symbol_img in enumerate(symbol_images):
        if symbol_img is None: continue
        
        # 1. LOCALIZED OCR
        # Usuwamy spacje i nowe linie, zostawiając znaki alfanumeryczne i plus
        raw_symbol = pytesseract.image_to_string(symbol_img, config=custom_config).replace(' ', '').replace('\n', '').upper()
        
        # 2. Lookup & Cleaning
        if raw_symbol in SYMBOL_TO_ITEM:
            item_key = SYMBOL_TO_ITEM[raw_symbol]
            
            # 3. Dopasowanie do bazy JSON
            if item_key in db:
                item_data = db[item_key]
                
                # Dodajemy tylko raz (pomijamy duplikaty)
                if not any(d['Przedmiot'] == item_key for d in results):
                    results.append({
                        "Przedmiot": item_key,
                        "Akcja": item_data['action'], 
                        "Typ": item_data['type'],
                        "Rada": item_data['tip'],
                        "Slot": i # Numer slotu dla debugowania
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
    symbol_slots = find_symbol_slots(image_cv)
    
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
    with st.expander("👁️ Zobacz diagnostykę cięcia symboli", expanded=True):
        st.write("Wycinek symboli z pierwszych 8 slotów:")
        if symbol_slots and all(s is not None for s in symbol_slots[:8]):
            # Łączymy pierwsze 8 symboli
            combined_symbols = np.hstack(symbol_slots[:8])
            # Zmieniamy kolor z powrotem na RGB, by streamlit mógł to wyświetlić (mimo że jest czarno-białe)
            st.image(combined_symbols, caption="Wycinek symboli (powinny być widoczne symbole i białe krzyżyki)", clamp=True)
        
        st.write(f"Zarejestrowane symbole (w bazie): {list(SYMBOL_TO_ITEM.keys())}")
