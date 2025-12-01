import streamlit as st
import cv2
import numpy as np
import pytesseract
import json
import os
import glob
from PIL import Image

# --- STAŁE KONFIGURACYJNE (Dostosowane do 4K) ---
# Te stałe odnoszą się do WYMIARÓW slotów, a nie ich pozycji.
SLOT_WIDTH = 165
SLOT_HEIGHT = 165
SPACING = 20
CONFIDENCE_THRESHOLD = 0.85 # Próg dopasowania szablonu (85%)

# Konfiguracja siatek (na podstawie Twojego screena 4K)
GRID_CONFIGS = {
    # Założenia: Siatka zaczyna się 10px pod napisem nagłówka.
    "TECHNOLOGY": {"COLS": 8, "ROWS": 2, "X_OFFSET": 0, "Y_OFFSET": 10}, 
    "CARGO": {"COLS": 8, "ROWS": 6, "X_OFFSET": 0, "Y_OFFSET": 10} 
}

# --- FOLDERY BAZY SZABLONÓW ---
TEMPLATE_DIR = "templates"
UNKNOWN_DIR = "unknown_templates"

if not os.path.exists(TEMPLATE_DIR):
    os.makedirs(TEMPLATE_DIR)
if not os.path.exists(UNKNOWN_DIR):
    os.makedirs(UNKNOWN_DIR)
# --- KONIEC STAŁYCH ---

st.set_page_config(page_title="🚀 NMS Scanner", page_icon="🧪")

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

@st.cache_data
def load_templates():
    """Ładuje wszystkie szablony ikon z folderu templates/."""
    templates = {}
    for filepath in glob.glob(os.path.join(TEMPLATE_DIR, "*.png")):
        filename = os.path.basename(filepath)
        item_name = os.path.splitext(filename)[0].upper()
        # Ważne: Wczytujemy szablony w skali szarości!
        templates[item_name] = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    return templates

def preprocess_image(img_cv):
    """
    Wstępne przetwarzanie obrazu (tylko skala szarości).
    UWAGA: W Template Matching często lepiej jest używać samego obrazu w skali szarości, 
    bez agresywnego Thresholdingu, aby zachować niuanse ikony.
    """
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Zwracamy szary obraz, ale konwertujemy go z powrotem na 3 kanały BGR (cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR))
    # jest to wymagane, aby Streamlit i dalsza logika cięcia (img_cv) działały bez problemów z wymiarami
    return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR) 


def find_anchors(img_cv):
    """DYNAMICZNIE WYSZUKUJE NAPISY 'TECHNOLOGY' i 'CARGO' i określa punkty startowe."""
    image_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    # Używamy PSM 3 (domyślny dla pełnej strony)
    data = pytesseract.image_to_data(image_pil, config=r'--psm 3', output_type=pytesseract.Output.DICT)
    anchors = {}
    
    for i, text in enumerate(data['text']):
        word = text.upper().strip()
        if word in GRID_CONFIGS:
            x = data['left'][i]
            y = data['top'][i]
            h = data['height'][i]
            
            anchor_x = x + GRID_CONFIGS[word]['X_OFFSET']
            anchor_y = y + h + GRID_CONFIGS[word]['Y_OFFSET']
            
            if word not in anchors:
                anchors[word] = {"x": anchor_x, "y": anchor_y}
                
    # Domyślne wartości dla 4K, jeśli OCR zawiedzie (na podstawie Twoich ostatnich testów)
    if 'CARGO' not in anchors:
        # Używamy ustalonych koordynatów, gdy dynamiczne wykrywanie zawiedzie
        anchors['CARGO'] = {"x": 350, "y": 1050} 
    
    return anchors

def process_grid(img_cv, anchor_name, anchor_coords):
    """Tnie całą siatkę slotów na podstawie kotwicy."""
    
    # 1. PRZETWARZANIE WSTĘPNE JEDYNIE DO SZAROŚCI
    # Używamy obrazu przetwarzanego wstępnie (szarość), aby z niego wycinać sloty
    img_processed = preprocess_image(img_cv)
    
    START_X = anchor_coords["x"]
    START_Y = anchor_coords["y"]
    config = GRID_CONFIGS[anchor_name]
    
    slots = []
    
    for row in range(config["ROWS"]):
        for col in range(config["COLS"]):
            # Obliczanie współrzędnych pełnego slotu
            x_start = START_X + col * (SLOT_WIDTH + SPACING)
            y_start = START_Y + row * (SLOT_HEIGHT + SPACING)
            x_end = x_start + SLOT_WIDTH
            y_end = y_start + SLOT_HEIGHT
            
            # Wycina pełny slot z obrazu w skali szarości (BGR z trzema kanałami)
            slot_img = img_processed[y_start:y_end, x_start:x_end]
            
            if slot_img.shape[0] == SLOT_HEIGHT and slot_img.shape[1] == SLOT_WIDTH:
                # Konwersja na czystą szarość (1 kanał) do Template Matching
                slot_gray = cv2.cvtColor(slot_img, cv2.COLOR_BGR2GRAY) 
                slots.append({"grid": anchor_name, "img": slot_gray, "index": row * config["COLS"] + col})
            
    return slots

def match_template(slot_img, templates):
    """Wykonuje Template Matching dla pojedynczego slotu."""
    best_match_name = None
    max_corr = -1
    
    for item_name, template in templates.items():
        # Upewnienie się, że szablon ma taki sam rozmiar jak slot (Template Matching wymaga tego)
        if template.shape != slot_img.shape:
            # Wymuszamy reskalowanie szablonu, chociaż idealnie powinny mieć ten sam rozmiar
            template_resized = cv2.resize(template, (slot_img.shape[1], slot_img.shape[0]))
        else:
            template_resized = template

        # Używamy CCorrNormed, który jest odporny na jasność i skalę
        result = cv2.matchTemplate(slot_img, template_resized, cv2.TM_CCOEFF_NORMED)
        
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        if max_val > max_corr:
            max_corr = max_val
            best_match_name = item_name

    if max_corr >= CONFIDENCE_THRESHOLD:
        return best_match_name, max_corr
    
    return None, max_corr

# --- INTERFEJS UŻYTKOWNIKA (FRONTEND) ---

st.title("🚀 NMS Inventory Scanner (Template Matching)")
st.write("Wykrywanie przedmiotów na podstawie ikon, zakotwiczenie na 'CARGO' / 'TECHNOLOGY'.")

uploaded_file = st.file_uploader("Wybierz zdjęcie...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. Konwersja
    image_pil = Image.open(uploaded_file)
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    database = load_db()
    templates = load_templates()

    st.write("⚙️ Wyszukuję punkty kotwiczenia i wycinam sloty...")
    
    # DYNAMICZNE WYSZUKIWANIE WSPÓŁRZĘDNYCH
    anchors = find_anchors(image_cv)
    
    all_slots = []
    
    # 2. Cięcie siatek
    for name, coords in anchors.items():
        st.write(f"✅ Znaleziono **{name}** (Kotwica: X={coords['x']}, Y={coords['y']})")
        all_slots.extend(process_grid(image_cv, name, coords))
        
    if not all_slots:
        st.error("Nie znaleziono siatek. Sprawdź, czy napisy są widoczne.")
        st.stop()
        
    st.write(f"Wycięto łącznie **{len(all_slots)}** slotów do analizy.")
    
    # 3. Analiza (Template Matching)
    found_resources = []
    unknown_slots_to_save = []
    
    for slot in all_slots:
        item_name, confidence = match_template(slot["img"], templates)
        
        if item_name:
            # Znany przedmiot - dodajemy do wyników
            if item_name in database:
                 # Zapisujemy tylko unikalne przedmioty do tabeli wyników
                 if not any(d['Przedmiot'] == item_name for d in found_resources):
                    found_resources.append({
                        "Przedmiot": item_name,
                        "Akcja": database[item_name]['action'], 
                        "Typ": database[item_name]['type'],
                        "Rada": database[item_name]['tip'],
                        "Slot": f"{slot['grid']}: {slot['index']}",
                        "Confidence": confidence
                    })
        else:
            # Nieznany przedmiot - dodajemy do listy do zapisania
            unknown_slots_to_save.append(slot)
            
    # Zapisywanie nieznanych slotów (aby uniknąć zapisywania pustych)
    unknown_count = 0
    if unknown_slots_to_save:
         # Używamy zbioru do przechowywania unikalnych histogramów, aby nie zapisywać duplikatów
        saved_histograms = set()
        
        for slot in unknown_slots_to_save:
            # Tworzymy histogram (unikalny odcisk palca) dla obrazu
            hist = cv2.calcHist([slot["img"]], [0], None, [256], [0, 256])
            hist_tuple = tuple(hist.flatten()) # Konwersja na hashable tuple
            
            if hist_tuple not in saved_histograms:
                # Jeśli histogram jest nowy, zapisujemy plik i dodajemy odcisk do zbioru
                filename = os.path.join(UNKNOWN_DIR, f"UNKNOWN_{slot['grid']}_{slot['index']}_{os.urandom(4).hex()}.png")
                cv2.imwrite(filename, slot["img"])
                saved_histograms.add(hist_tuple)
                unknown_count += 1
                
    # --- WYNIKI ---
    st.header("Wyniki Skanowania")
    if found_resources:
        st.success(f"Znaleziono {len(found_resources)} unikalnych, znanych zasobów!")
        
        for item in found_resources:
            color = "green" if item['Akcja'] == "TRZYMAJ" else "orange"
            
            with st.container():
                st.markdown(f"### :{color}[{item['Akcja']}] {item['Przedmiot']}")
                st.caption(f"Typ: {item['Typ']} (Siatka: {item['Slot']} | Zgodność: {item['Confidence']:.2f})")
                st.info(item['Rada'])
                st.divider()
    else:
        st.warning("Nie znaleziono znanych zasobów w ekwipunku.")
    
    if unknown_count > 0:
         st.error(f"Zapisano {unknown_count} unikalnych, nieznanych ikon do folderu `{UNKNOWN_DIR}/`! Opisz je i przenieś do `{TEMPLATE_DIR}/`.")


    # --- DEBUG VIEW ---
    with st.expander("👁️ DIAGNOSTYKA I GENEROWANIE SZABLONÓW", expanded=True):
        st.header("Instrukcja Generowania Bazy Ikon")
        st.markdown(f"""
        1.  **Zobacz folder `{UNKNOWN_DIR}/`:** Znajdziesz tam nowe pliki PNG.
        2.  **Opisz ikonę:** Jeśli ikona to np. **Chromatic Metal**, zmień nazwę pliku na **`CHROMATIC_METAL.png`**.
        3.  **Przenieś plik:** Przenieś plik do folderu **`{TEMPLATE_DIR}/`**.
        4.  **Odśwież aplikację:** Aplikacja załaduje nowy szablon i zacznie rozpoznawać ten przedmiot.
        """)
        
        if all_slots:
            st.subheader("Wycinki pierwszych 16 slotów (pełne ikony w skali szarości)")
            
            slots_to_display = [cv2.cvtColor(slot['img'], cv2.COLOR_GRAY2BGR) for slot in all_slots[:16]]
            
            if len(slots_to_display) >= 8:
                row1 = np.hstack(slots_to_display[:8])
                row2 = np.hstack(slots_to_display[8:16])
                combined_slots = np.vstack([row1, row2])
                st.image(combined_slots, caption="Wycinki pierwszych 16 slotów", clamp=True)
            elif slots_to_display:
                 st.image(np.hstack(slots_to_display), caption="Wycinki slotów", clamp=True)
