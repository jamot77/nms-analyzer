import streamlit as st
import cv2
import numpy as np
import json
import base64
from PIL import Image
import io
import pytesseract

# --- STAŁE KONFIGURACYJNE (Dostosowane do 4K) ---
SLOT_WIDTH = 165
SLOT_HEIGHT = 165
SPACING = 20
CONFIDENCE_THRESHOLD = 0.85 

# NOWA TRWAŁA BAZA DANYCH SZABLONÓW
TEMPLATES_FILE = "templates.json"

# Konfiguracja siatek (X_OFFSET = -90 na Twoje życzenie)
GRID_CONFIGS = {
    "TECHNOLOGY": {"COLS": 8, "ROWS": 2, "X_OFFSET": -90, "Y_OFFSET": 10}, 
    "CARGO": {"COLS": 8, "ROWS": 6, "X_OFFSET": -90, "Y_OFFSET": 10} 
}
# --- KONIEC STAŁYCH ---

st.set_page_config(page_title="🚀 NMS Scanner", page_icon="🧪")

# --- FUNKCJE KODOWANIA/DEKODOWANIA (Base64) ---

def encode_template(image_array):
    """Koduje numpy array do Base64 String."""
    # Tworzenie tymczasowego pliku PNG w pamięci
    is_success, buffer = cv2.imencode(".png", image_array)
    if is_success:
        return base64.b64encode(buffer).decode('utf-8')
    return None

def decode_template(base64_string):
    """Dekoduje Base64 String do numpy array (cv2 image)."""
    img_bytes = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_bytes, np.uint8)
    # Wczytujemy jako szary obraz (1 kanał)
    return cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

# --- FUNKCJE DANYCH I PRZETWARZANIA ---

@st.cache_data(show_spinner="Ładowanie bazy danych przedmiotów...")
def load_db():
    try:
        with open('nms_items.json', 'r', encoding='utf-8') as f:
            data = {k: v for k, v in json.load(f).items() if isinstance(v, dict)}
            return data
    except FileNotFoundError:
        st.error("Błąd: Nie znaleziono pliku nms_items.json!")
        return {}

@st.cache_data(show_spinner="Ładowanie bazy szablonów ikon...")
def load_templates():
    """Ładuje szablony z pliku templates.json (Base64)."""
    templates = {}
    try:
        with open(TEMPLATES_FILE, 'r', encoding='utf-8') as f:
            template_data = json.load(f)
            for item_name, b64_string in template_data.items():
                templates[item_name] = decode_template(b64_string)
    except FileNotFoundError:
        st.warning(f"Brak pliku {TEMPLATES_FILE}. Rozpocznij tworzenie bazy.")
    except json.JSONDecodeError:
        st.error(f"Błąd odczytu pliku {TEMPLATES_FILE}. Upewnij się, że jest to poprawny JSON.")
    return templates

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
            
            anchor_x = x + GRID_CONFIGS[word]['X_OFFSET']
            anchor_y = y + h + GRID_CONFIGS[word]['Y_OFFSET']
            
            if word not in anchors:
                anchors[word] = {"x": anchor_x, "y": anchor_y}
                
    # Domyślne wartości dla 4K, jeśli OCR zawiedzie
    if 'CARGO' not in anchors:
        anchors['CARGO'] = {"x": 350, "y": 1050} 
    
    return anchors

def process_grid(img_cv, anchor_name, anchor_coords):
    """
    Tnie całą siatkę slotów na podstawie kotwicy, konwertuje na szarość i wyrównuje kontrast.
    """
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
            
            # Wycina pełny slot z ORYGINALNEGO obrazu BGR
            slot_img = img_cv[y_start:y_end, x_start:x_end] 
            
            if slot_img.shape[0] == SLOT_HEIGHT and slot_img.shape[1] == SLOT_WIDTH:
                # 1. Konwersja wyciętego slotu na czystą szarość (1 kanał)
                slot_gray = cv2.cvtColor(slot_img, cv2.COLOR_BGR2GRAY) 
                
                # 2. KLUCZOWY KROK: Wyrównanie histogramu dla stabilnego Template Matching
                slot_gray = cv2.equalizeHist(slot_gray) 
                
                slots.append({"grid": anchor_name, "img": slot_gray, "index": row * config["COLS"] + col})
            
    return slots

def match_template(slot_img, templates):
    """Wykonuje Template Matching dla pojedynczego slotu."""
    best_match_name = None
    max_corr = -1
    
    for item_name, template in templates.items():
        if template.shape != slot_img.shape:
            # Reskalowanie, jeśli rozmiary się nie zgadzają (awaryjnie)
            template_resized = cv2.resize(template, (slot_img.shape[1], slot_img.shape[0]))
        else:
            template_resized = template

        result = cv2.matchTemplate(slot_img, template_resized, cv2.TM_CCOEFF_NORMED)
        
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        if max_val > max_corr:
            max_corr = max_val
            best_match_name = item_name

    if max_corr >= CONFIDENCE_THRESHOLD:
        return best_match_name, max_corr
    
    return None, max_corr

# --- INTERFEJS UŻYTKOWNIKA (FRONTEND) ---

st.title("🚀 NMS Inventory Scanner (Chmurowe zarządzanie bazą)")
st.write("Wykrywanie przedmiotów na podstawie ikon. Zarządzaj szablonami bezpośrednio w tej aplikacji.")

uploaded_file = st.file_uploader("Wybierz zdjęcie...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. Konwersja
    image_pil = Image.open(uploaded_file)
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    database = load_db()
    templates = load_templates()
    
    # 2. Cięcie slotów
    anchors = find_anchors(image_cv)
    all_slots = []
    
    for name, coords in anchors.items():
        st.caption(f"Znaleziono **{name}** (Kotwica: X={coords['x']}, Y={coords['y']})")
        all_slots.extend(process_grid(image_cv, name, coords))
        
    if not all_slots:
        st.error("Nie znaleziono siatek. Sprawdź, czy napisy są widoczne.")
        st.stop()
        
    st.info(f"Wycięto łącznie **{len(all_slots)}** slotów do analizy.")
    
    # 3. Analiza (Template Matching)
    found_resources = []
    unknown_slots_data = {}
    
    for slot in all_slots:
        slot_img = slot["img"]
        item_name, confidence = match_template(slot_img, templates)
        
        if item_name:
            # Znany przedmiot
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
            # Nieznany przedmiot - kodujemy i przechowujemy w st.session_state
            b64_img = encode_template(slot_img)
            
            # Używamy unikalnego hasha obrazu do sprawdzenia duplikatów
            img_hash = hash(slot_img.tobytes())
            
            if img_hash not in st.session_state.get('known_hashes', set()):
                # Unikalna nieznana ikona
                unknown_slots_data[img_hash] = {"b64": b64_img, "grid": slot["grid"], "index": slot["index"]}
                if 'known_hashes' not in st.session_state:
                    st.session_state['known_hashes'] = set()
                st.session_state['known_hashes'].add(img_hash)

    # --- WYNIKI ---
    st.header("Wyniki Skanowania")
    if found_resources:
        st.success(f"Znaleziono {len(found_resources)} unikalnych, znanych zasobów!")
        for item in found_resources:
            color = "green" if item['Akcja'] == "TRZYMAJ" else "orange"
            with st.container(border=True):
                st.markdown(f"### :{color}[{item['Akcja']}] {item['Przedmiot']}")
                st.caption(f"Typ: {item['Typ']} (Siatka: {item['Slot']} | Zgodność: {item['Confidence']:.2f})")
                st.info(item['Rada'])
    else:
        st.warning("Nie znaleziono znanych zasobów w ekwipunku.")

    # --- PANEL ZARZĄDZANIA SZABLONAMI ---
    
    if unknown_slots_data:
        st.error(f"Znaleziono {len(unknown_slots_data)} unikalnych nieznanych ikon! Wymagają opisu.")
        
        with st.expander("📝 Zarządzanie Nowymi Ikonami i Aktualizacja Bazy", expanded=True):
            st.markdown("### Krok 1: Wprowadź Nazwy dla Nowych Ikon (WIELKIE LITERY, bez spacji)")
            
            # Przechowujemy dane ikon w st.session_state, aby przetrwały interakcje
            if 'new_templates_input' not in st.session_state:
                st.session_state['new_templates_input'] = {}

            cols = st.columns(4)
            
            for i, (img_hash, data) in enumerate(unknown_slots_data.items()):
                b64_img = data["b64"]
                
                # Wyświetlanie miniatury
                image_bytes = base64.b64decode(b64_img)
                image_pil = Image.open(io.BytesIO(image_bytes))
                
                with cols[i % 4]:
                    st.image(image_pil, use_column_width=True, caption=f"Siatka: {data['grid']} | Index: {data['index']}")
                    
                    # Pole tekstowe do wprowadzania nazwy
                    key = f"input_{img_hash}"
                    st.session_state['new_templates_input'][key] = st.text_input(
                        "Nazwa Ikonu (np. SODIUM)", 
                        key=key, 
                        value=st.session_state['new_templates_input'].get(key, "")
                    ).strip().upper().replace(" ", "_")

            st.markdown("---")
            st.markdown("### Krok 2: Generowanie Nowej Bazy")
            
            if st.button("💾 Generuj Zaktualizowany templates.json"):
                # Pobieramy obecną bazę (na wszelki wypadek, żeby nie nadpisać)
                new_templates = load_templates()
                
                updates = 0
                for key, item_name in st.session_state['new_templates_input'].items():
                    if item_name:
                        img_hash = key.replace("input_", "")
                        b64_img = unknown_slots_data[img_hash]["b64"]
                        
                        # Sprawdzamy, czy nazwa już istnieje w bazie
                        if item_name not in new_templates:
                            new_templates[item_name] = b64_img
                            updates += 1
                
                if updates > 0:
                    st.success(f"Pomyślnie dodano {updates} nowych ikon do bazy! Pobierz nowy plik poniżej.")
                    
                    # Generowanie pliku JSON do pobrania
                    json_data = json.dumps(new_templates, indent=4, ensure_ascii=False)
                    st.download_button(
                        label="Pobierz NOWY templates.json",
                        data=json_data.encode('utf-8'),
                        file_name=TEMPLATES_FILE,
                        mime="application/json"
                    )
                    st.markdown("""
                        **WAŻNE:** Po pobraniu pliku, **musisz go wgrać na GitHub** do głównego katalogu swojego repozytorium. Dopiero po tym aplikacja zacznie rozpoznawać nowe ikony!
                    """)
                else:
                    st.warning("Nie wprowadzono nowych nazw. Wprowadź nazwy, aby wygenerować bazę.")
    
    # --- DEBUG VIEW ---
    with st.expander("👁️ DIAGNOSTYKA (Wycinki do Template Matching)", expanded=True):
        if all_slots:
            st.subheader("Wycinki pierwszych 16 slotów (gotowe do TM)")
            
            slots_to_display = [cv2.cvtColor(slot['img'], cv2.COLOR_GRAY2BGR) for slot in all_slots[:16]]
            
            if len(slots_to_display) >= 8:
                row1 = np.hstack(slots_to_display[:8])
                row2 = np.hstack(slots_to_display[8:16])
                combined_slots = np.vstack([row1, row2])
                st.image(combined_slots, caption="Wycinki slotów (skala szarości, wyrównany kontrast)", clamp=True)
            elif slots_to_display:
                 st.image(np.hstack(slots_to_display), caption="Wycinki slotów", clamp=True)
