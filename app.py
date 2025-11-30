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
    "TECHNOLOGY": {"COLS": 8, "ROWS": 2, "X_OFFSET": 0, "Y_OFFSET": 10}, # Lekki margines 10px pod napisem
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
        # Kluczem jest nazwa pliku bez rozszerzenia (np. "CARBON")
        item_name = os.path.splitext(filename)[0].upper()
        templates[item_name] = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    return templates

def preprocess_image(img_cv):
    """Wstępne przetwarzanie obrazu (Adaptive Thresholding)."""
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    # W Template Matching często lepiej jest użyć czystej szarości, ale spróbujmy Adaptive Thresh dla lepszego kontrastu
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR) # Konwersja z powrotem na BGR, by pasowało do logiki dalszej

def find_anchors(img_cv):
    """DYNAMICZNIE WYSZUKUJE NAPISY 'TECHNOLOGY' i 'CARGO' i określa punkty startowe."""
    image_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    data = pytesseract.image_to_data(image_pil, config=r'--psm 3', output_type=pytesseract.Output.DICT)
    anchors = {}
    
    for i, text in enumerate(data['text']):
        word = text.upper().strip()
        if word in GRID_CONFIGS:
            x = data['left'][i]
            y = data['top'][i]
            h = data['height'][i]
            
            # Punkt kotwiczenia: X lewej krawędzi tekstu, Y tuż pod tekstem
            anchor_x = x + GRID_CONFIGS[word]['X_OFFSET']
            anchor_y = y + h + GRID_CONFIGS[word]['Y_OFFSET']
            
            # Zapisujemy tylko pierwsze wystąpienie (najbardziej wiarygodne)
            if word not in anchors:
                anchors[word] = {"x": anchor_x, "y": anchor_y}
                
    return anchors

def process_grid(img_cv, anchor_name, anchor_coords):
    """Tnie całą siatkę slotów na podstawie kotwicy."""
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
            
            # Wycina pełny slot
            slot_img = img_cv[y_start:y_end, x_start:x_end]
            
            if slot_img.shape[0] == SLOT_HEIGHT and slot_img.shape[1] == SLOT_WIDTH:
                 # Konwersja na szarość i zapisanie jako obiekt
                slot_gray = cv2.cvtColor(slot_img, cv2.COLOR_BGR2GRAY)
                slots.append({"grid": anchor_name, "img": slot_gray, "index": row * config["COLS"] + col})
            
    return slots

def match_template(slot_img, templates):
    """Wykonuje Template Matching dla pojedynczego slotu."""
    
    best_match_name = None
    max_corr = -1
    
    for item_name, template in templates.items():
        # Używamy CCorrNormed, który jest odporny na jasność
        result = cv2.matchTemplate(slot_img, template, cv2.TM_CCOEFF_NORMED)
        
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        if max_val > max_corr:
            max_corr = max_val
            best_match_name = item_name

    if max_corr >= CONFIDENCE_THRESHOLD:
        return best_match_name, max_corr
    
    return None, max_corr

# --- INTERFEJS UŻYTKOWNIKA (FRONTEND) ---

st.title("🚀 NMS Inventory Scanner (Template Matching)")
st.write("Wykrywanie przedmiotów na podstawie ikon w Twojej rozdzielczości 4K.")

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
        st.error("Nie znaleziono siatek CARGO ani TECHNOLOGY. Sprawdź, czy napisy są widoczne.")
        st.stop()
        
    st.write(f"Wycięto łącznie **{len(all_slots)}** slotów do analizy.")
    
    # 3. Analiza (Template Matching)
    found_resources = []
    unknown_count = 0
    
    for slot in all_slots:
        item_name, confidence = match_template(slot["img"], templates)
        
        if item_name:
            # Znany przedmiot - dodajemy do wyników
            if item_name in database:
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
            # Nieznany przedmiot - zapisujemy jako szablon do opisania
            # Zapobiegamy wielokrotnemu zapisywaniu tego samego (jeśli slot jest pusty, zapiszemy go raz jako PUSTY)
            if unknown_count < 10: # Ograniczamy zapis do 10 pierwszych nieznanych
                filename = os.path.join(UNKNOWN_DIR, f"UNKNOWN_{slot['grid']}_{slot['index']}_{int(confidence*100)}.png")
                cv2.imwrite(filename, slot["img"])
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
         st.error(f"Zapisano {unknown_count} nieznanych ikon do folderu `{UNKNOWN_DIR}/`! Opisz je i przenieś do `{TEMPLATE_DIR}/`.")


    # --- DEBUG VIEW ---
    with st.expander("👁️ DIAGNOSTYKA I GENEROWANIE SZABLONÓW", expanded=True):
        st.header("Instrukcja Generowania Bazy Ikon")
        st.markdown(f"""
        1.  **Zobacz folder `{UNKNOWN_DIR}/`:** Znajdziesz tam pliki typu `UNKNOWN_CARGO_5_80.png`.
        2.  **Opisz ikonę:** Jeśli ikona to **Chromatic Metal**, zmień nazwę pliku na **`CHROMATIC_METAL.png`**.
        3.  **Przenieś plik:** Przenieś plik do folderu **`{TEMPLATE_DIR}/`**.
        4.  **Odśwież aplikację:** Aplikacja automatycznie załaduje nowy szablon i zacznie rozpoznawać ten przedmiot!
        """)
        
        if all_slots:
            st.subheader("Wycinki pierwszych 16 slotów (pełne ikony)")
            
            # Wyświetlamy pierwsze 16 slotów
            slots_to_display = [cv2.cvtColor(slot['img'], cv2.COLOR_GRAY2BGR) for slot in all_slots[:16]]
            
            if len(slots_to_display) >= 8:
                row1 = np.hstack(slots_to_display[:8])
                row2 = np.hstack(slots_to_display[8:16])
                combined_slots = np.vstack([row1, row2])
                st.image(combined_slots, caption="Wycinki pierwszych 16 slotów", clamp=True)
            elif slots_to_display:
                 st.image(np.hstack(slots_to_display), caption="Wycinki slotów", clamp=True)
