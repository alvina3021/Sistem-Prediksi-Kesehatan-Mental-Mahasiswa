import pandas as pd
import itertools
import streamlit as st
import altair as alt

# 1. KONFIGURASI HALAMAN & CSS
st.set_page_config(
    page_title="Sistem Prediksi Mental Health",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling CSS untuk tampilan form
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; color: #333; }
    .prediction-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 6px solid #008080;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .high-risk { border-left-color: #e74c3c; }
    .medium-risk { border-left-color: #f1c40f; }
    .safe { border-left-color: #2ecc71; }
    h1, h2, h3 { color: #008080; }
    /* Nonaktifkan input manual pada selectbox */
    [data-baseweb="select"] input {
        pointer-events: none !important;
        caret-color: transparent !important;
        user-select: none !important;
    }
</style>
""", unsafe_allow_html=True)

# 2. FUNGSI LOGIKA (BACKEND)
@st.cache_data
#data preprocessing
def load_and_preprocess_data(file_path):
    try:
        df = pd.read_csv(file_path) # Membaca data dari CSV
    except FileNotFoundError:
        return None, None
    
    # 1. Rename Kolom (Standardisasi)
    df.columns = ['Timestamp', 'Gender', 'Age', 'Course', 'Year', 'CGPA', 'Marital', 
                  'Depression', 'Anxiety', 'Panic', 'Treatment']
    df = df.drop(columns=['Timestamp']) 

    # 2. Pembersihan Data String
    obj_cols = df.select_dtypes(include=['object']).columns
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip().str.title()

    # 3. Binning Age (pengelompokan umur)
    def bin_age(x):
        try:
            x = float(x)
            if x <= 19: return 'Age_Teens (<=19)'
            elif x <= 22: return 'Age_YoungAdult (20-22)'
            else: return 'Age_Adult (>22)'
        except:
            return 'Age_Unknown'

    df['Age_Group'] = df['Age'].apply(bin_age) #kolom baru untuk Age_Group
    
    # Simpan data mentah untuk dropdown list di form input
    raw_data_for_ui = df.copy()

    # 4. Hapus kolom Age asli untuk proses Apriori (kita pakai Age_Group)
    df_apriori = df.drop(columns=['Age'])

    # 5. Buat List Transaksi
    transactions = []
    for index, row in df_apriori.iterrows():
        transaction = set()
        for col in df_apriori.columns:
            val = row[col]
            transaction.add(f"{col}={val}")
        transactions.append(transaction)
    
    return transactions, raw_data_for_ui

def calculate_support(itemset, all_transactions):
    count = sum(1 for t in all_transactions if itemset.issubset(t))
    return count / len(all_transactions)

@st.cache_data
def generate_rules(transactions, min_support, min_confidence):
    """Membuat Kamus Aturan (Rule Base)"""
    unique_items = set().union(*transactions)
    item_supports = {}
    l1_frequent = []

    # Level 1 (hitung support tiap item individu)
    for item in unique_items:
        itemset = {item}
        sup = calculate_support(itemset, transactions)
        if sup >= min_support: #ketika nilai support terpenuhi maka item dianggap sering muncul
            l1_frequent.append(item)
            item_supports[frozenset(itemset)] = sup

    # Level 2 & Rule Generation
    rules = []
    combinations = itertools.combinations(l1_frequent, 2)

    #hitung support gabungan
    for item1, item2 in combinations:
        itemset = {item1, item2}
        sup_both = calculate_support(itemset, transactions)
        
        if sup_both >= min_support:
            item_supports[frozenset(itemset)] = sup_both
            items = list(itemset)
            
            for i in range(2):
                ant = items[i]
                cons = items[1-i]
                
                sup_ant = item_supports.get(frozenset({ant}), 0)
                if sup_ant > 0:
                    conf = sup_both / sup_ant
                    sup_cons = item_supports.get(frozenset({cons}), 1) 
                    lift = conf / sup_cons
                    
                    if conf >= min_confidence:
                        rules.append({
                            'Antecedent': ant,
                            'Consequent': cons,
                            'Support': sup_both,
                            'Confidence': conf,
                            'Lift': lift
                        })
    
    return pd.DataFrame(rules)

# 3. ANTARMUKA PENGGUNA
# LOAD DATA 
transactions, df_ui = load_and_preprocess_data('data_bersih.csv')

if transactions is None:
    st.error("❌ File `data_bersih.csv` tidak ditemukan!")
    st.stop()

# SIDEBAR: FORMULIR INPUT PENGGUNA 
st.sidebar.title("👤 Profil Pengguna")
st.sidebar.markdown("Masukkan data diri untuk melihat prediksi risiko.")

# Input Dinamis Berdasarkan Data yang Ada
input_gender = st.sidebar.selectbox("Gender", df_ui['Gender'].unique())
input_age = st.sidebar.number_input("Umur (Angka)", min_value=17, max_value=30, value=20)
input_course = st.sidebar.selectbox("Jurusan (Course)", sorted(df_ui['Course'].unique()))
input_year = st.sidebar.selectbox("Tahun Studi", sorted(df_ui['Year'].unique()))
input_cgpa = st.sidebar.selectbox("CGPA (IPK)", sorted(df_ui['CGPA'].unique()))
input_marital = st.sidebar.selectbox("Status Pernikahan", sorted(df_ui['Marital'].unique()))

# Konversi Input Umur ke Kategori (Sesuai Logika Backend)
if input_age <= 19:
    age_group_val = 'Age_Teens (<=19)'
elif input_age <= 22:
    age_group_val = 'Age_YoungAdult (20-22)'
else:
    age_group_val = 'Age_Adult (>22)'

# Mengemas Input User menjadi "Itemset"
user_profile_items = [
    f"Gender={input_gender}",
    f"Age_Group={age_group_val}",
    f"Course={input_course}",
    f"Year={input_year}",
    f"CGPA={input_cgpa}",
    f"Marital={input_marital}"
]

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Parameter Analisis")
min_sup = st.sidebar.slider("Min Support", 0.05, 0.5, 0.10, 0.01) # Diturunkan agar sensitif
min_conf = st.sidebar.slider("Min Confidence", 0.1, 1.0, 0.40, 0.05)


# 4. HALAMAN UTAMA: HASIL PREDIKSI
st.title("Sistem Prediksi Kesehatan Mental Mahasiswa")
st.markdown("Menggunakan **Association Rule Mining (Apriori)** untuk mencocokkan profil dengan pola data historis.")

# Layout tab 1 dan 2
tab1, tab2 = st.tabs(["🔍 Prediksi", "📊 Analisis Data Keseluruhan"])

with tab1:
    st.subheader(f"Hasil Analisis untuk Profil:")
    
    # Profil user
    st.write("Profil Input:", user_profile_items)
    
    if st.button("Analisis Risiko", type="primary"):
        with st.spinner("Mencocokkan profil dengan ribuan pola..."):
            # 1. Generate Rules dari seluruh data
            all_rules = generate_rules(transactions, min_sup, min_conf)
            
            if all_rules.empty:
                st.warning("Data historis tidak memiliki pola yang cukup kuat dengan parameter Support/Confidence saat ini. Coba turunkan nilainya di Sidebar.")
            else:
                # 2. FILTER RULES: Cari aturan yang Antecedent-nya COCOK dengan Profil User
                # Contoh: Jika User 'Year 1', cari rule "Year=Year 1 -> Depression=Yes"
                
                matched_rules = []
                targets = ['Depression', 'Anxiety', 'Panic', 'Treatment']
                
                for index, row in all_rules.iterrows():
                    ant = row['Antecedent']
                    cons = row['Consequent']
                    
                    # Cek apakah Antecedent ada di dalam Input User
                    if ant in user_profile_items:
                        # Cek apakah Consequent-nya adalah tentang Mental Health (Target)
                        # Kita ingin tahu: Profil User -> MENYEBABKAN -> Mental Health?
                        cons_key = cons.split('=')[0] # Ambil nama kolomnya saja (misal 'Depression')
                        if cons_key in targets:
                            matched_rules.append(row)

                match_df = pd.DataFrame(matched_rules)

                # 3. Hasil kecocokan
                if not match_df.empty:
                    st.success(f"Ditemukan {len(match_df)} pola yang cocok dengan profil Anda!")
                    
                    # Urutkan berdasarkan Confidence tertinggi
                    match_df = match_df.sort_values(by='Confidence', ascending=False)
                    
                    for i, rule in match_df.iterrows():
                        target_val = rule['Consequent']
                        confidence_pct = rule['Confidence'] * 100
                        lift_val = rule['Lift']
                        
                        #warna border berdasarkan risiko
                        risk_class = "metric-card"
                        if "Yes" in target_val:
                            risk_class = "prediction-card high-risk"
                            icon = "⚠️"
                            msg = "BERISIKO TINGGI"
                        else:
                            risk_class = "prediction-card safe"
                            icon = "✅"
                            msg = "KEMUNGKINAN RENDAH"

                        st.markdown(f"""
                        <div class="{risk_class}">
                            <h4 style="margin:0;">{icon} Prediksi: {target_val}</h4>
                            <p style="color:#666; margin-bottom:5px;">Karena Anda memiliki faktor: <b>{rule['Antecedent']}</b></p>
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <span><b>{confidence_pct:.1f}%</b> Data historis mengalami hal ini.</span>
                                <span style="background:#eee; padding:2px 8px; border-radius:4px; font-size:0.8em;">Lift: {lift_val:.2f}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
 
                else:
                    st.info("Profil Anda unik atau aman! Tidak ditemukan aturan asosiasi spesifik yang menghubungkan input Anda dengan risiko kesehatan mental pada dataset ini.")
#tab 2
with tab2:
    st.subheader("Pola Data Global (Knowledge Base)")
    st.markdown("Ini adalah seluruh aturan yang ditemukan dari dataset tanpa memandang input Anda.")
    
    if st.button("Tampilkan Semua Aturan"):
        all_rules = generate_rules(transactions, min_sup, min_conf)
        if not all_rules.empty:
            # Filter hanya target mental health
            targets = ['Depression', 'Anxiety', 'Panic', 'Treatment']
            def is_relevant(row):
                return any(t in str(row['Antecedent']) for t in targets) or \
                       any(t in str(row['Consequent']) for t in targets)
            
            global_rules = all_rules[all_rules.apply(is_relevant, axis=1)]
            
            st.dataframe(global_rules.sort_values(by='Lift', ascending=False), use_container_width=True)
            
            # Scatter Plot
            c = alt.Chart(global_rules).mark_circle(size=60).encode(
                x='Support',
                y='Confidence',
                color='Lift',
                tooltip=['Antecedent', 'Consequent', 'Confidence']
            ).interactive()
            st.altair_chart(c, use_container_width=True)
        else:
            st.write("Tidak ada aturan.")