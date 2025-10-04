# Import library yang diperlukan
import streamlit as st
import yfinance as yf
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from exa_py import Exa

# --- 1. Konfigurasi Halaman dan Judul ---

st.set_page_config(page_title="🤖 Asisten Investasi Pribadi", layout="wide")

st.title("🤖 Asisten Investasi Pribadi")
st.caption("Chatbot untuk membantu Anda merencanakan investasi berdasarkan profil risiko dan tujuan keuangan.")

# --- 2. Sidebar untuk Pengaturan ---

# with st.sidebar:
#     st.subheader("Pengaturan API")
    
#     # Input untuk Google AI API Key
#     google_api_key = st.text_input("Google AI API Key", type="password", help="Dapatkan kunci API Anda dari Google AI Studio.")
    
#     # Input untuk Exa API Key
#     exa_api_key = st.text_input("Exa API Key", type="password", help="Dapatkan kunci API Anda dari dashboard Exa.")
    
#     st.markdown("---")
    
#     # Tombol untuk mereset percakapan
#     if st.button("Reset Percakapan", help="Mulai percakapan baru dari awal"):
#         st.session_state.pop("agent", None)
#         st.session_state.pop("messages", None)
#         st.rerun()

with st.sidebar:
    st.subheader("Menu")
    st.markdown("""
    Aplikasi ini ditenagai oleh **Google Gemini** dan **Exa Search**. 
    
    Cukup mulai percakapan untuk mendapatkan analisis investasi pribadi Anda.
    """)
    
    st.markdown("---")
    
    # Tombol untuk mereset percakapan tetap di sini
    if st.button("Reset Percakapan", help="Mulai percakapan baru dari awal"):
        # Hapus state yang relevan untuk reset
        for key in list(st.session_state.keys()):
            if key in ['agent', 'messages']:
                del st.session_state[key]
        st.rerun()

# Kode untuk mengambil API keys sekarang ada di sini.
try:
    google_api_key = st.secrets["GEMINI"]
    exa_api_key = st.secrets["EXA"]
except KeyError:
    st.error("🛑 API Key tidak ditemukan. Mohon pastikan file `.streamlit/secrets.toml` Anda sudah benar.")
    st.info("File `secrets.toml` harus berisi:\n\n`GOOGLE_API_KEY = '...'`\n`EXA_API_KEY = '...'`")
    st.stop()

# --- 3. Definisi Tools untuk AI ---
exa = Exa(api_key = exa_api_key)
# Tool 1: Pencarian Web menggunakan Exa
@tool
def search_the_web(query: str):
    """
    Gunakan tool ini untuk mencari informasi terkini di internet.
    Sangat berguna untuk mencari berita ekonomi, tren pasar, informasi detail tentang reksadana,
    atau prospek perusahaan tertentu.
    Contoh query: "prospek ekonomi Indonesia 2025", "kinerja reksadana saham terbaik 1 tahun terakhir".
    """
    try:
        search_results = exa.search_and_contents(
            query=query,
            type="auto",
            num_results=3,  # Get the top 3 results for a good summary
            text={"max_characters": 2000} # Limit content length per result
        )
        return search_results.results

        # retriever = ExaSearchRetriever(k=3, api_key=exa_api_key)
        # results = retriever.invoke(query)
        # return "\n\n".join([doc.page_content for doc in results])
    except Exception as e:
        return f"Gagal melakukan pencarian: {e}"

# Tool 2: Mendapatkan Harga Saham
@tool
def get_stock_price(ticker_symbol: str):
    """
    Mendapatkan harga terkini dan beberapa data historis untuk saham tertentu.
    Gunakan ticker symbol yang valid dari Yahoo Finance.
    - Untuk saham Indonesia (IDX), tambahkan akhiran '.JK'. Contoh: 'BBCA.JK', 'TLKM.JK'.
    - Untuk saham AS, gunakan ticker symbol biasa. Contoh: 'AAPL', 'GOOGL'.
    """
    try:
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period="1y")
        if hist.empty:
            return f"Data tidak ditemukan untuk ticker {ticker_symbol}. Pastikan ticker symbol valid."
        
        current_price = hist['Close'].iloc[-1]
        high_52wk = hist['High'].max()
        low_52wk = hist['Low'].min()
        
        return (
            f"Data untuk {ticker_symbol}:\n"
            f"- Harga Terkini: ${current_price:.2f}\n"
            f"- Tertinggi 52 Minggu: ${high_52wk:.2f}\n"
            f"- Terendah 52 Minggu: ${low_52wk:.2f}"
        )
    except Exception as e:
        return f"Error saat mengambil data saham: {e}"

# Tool 3: Mendapatkan Harga Kripto
@tool
def get_crypto_price(crypto_symbol: str):
    """
    Mendapatkan harga terkini untuk mata uang kripto.
    Gunakan format 'SIMBOL-USD'. Contoh: 'BTC-USD', 'ETH-USD'.
    """
    try:
        crypto = yf.Ticker(crypto_symbol)
        hist = crypto.history(period="7d") # Cukup 7 hari untuk kripto
        if hist.empty:
            return f"Data tidak ditemukan untuk simbol kripto {crypto_symbol}."
            
        current_price = hist['Close'].iloc[-1]
        return f"Harga terkini untuk {crypto_symbol} adalah ${current_price:,.2f}"
    except Exception as e:
        return f"Error saat mengambil data kripto: {e}"

# Tool 4: Mendapatkan Harga Emas
@tool
def get_gold_price():
    """
    Mendapatkan harga emas terkini di pasar global (berdasarkan ETF GLD).
    """
    try:
        gold = yf.Ticker("GLD")
        hist = gold.history(period="7d")
        current_price = hist['Close'].iloc[-1]
        return f"Harga emas (berdasarkan GLD) saat ini adalah ${current_price:.2f} per unit."
    except Exception as e:
        return f"Error saat mengambil data harga emas: {e}"

# --- 4. Inisialisasi API Key dan Agent ---

if not google_api_key or not exa_api_key:
    st.info("🔑 Harap masukkan Google AI dan Exa API key Anda di sidebar untuk memulai.")
    st.stop()

# Gabungkan semua tools ke dalam satu list
tools = [search_the_web, get_stock_price, get_crypto_price, get_gold_price]

# Inisialisasi agent hanya jika belum ada atau API key berubah
if ("agent" not in st.session_state) or (getattr(st.session_state, "_last_google_key", None) != google_api_key):
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",  # Gunakan model yang lebih kuat untuk analisis
            google_api_key=google_api_key,
            temperature=0.4
        )
        
        # System Prompt yang baru untuk menjadi asisten investasi
        system_prompt = """
        Anda adalah seorang Asisten Perencana Investasi AI yang ahli, ramah, dan sangat membantu.
        Tujuan utama Anda adalah untuk memahami situasi keuangan, tujuan, dan profil risiko pengguna untuk memberikan rekomendasi alokasi investasi yang dipersonalisasi.
        
        IKUTI PROSES INI DENGAN SEKSAMA:
        
        LANGKAH 1: Pengumpulan Informasi.
        - Mulailah dengan menyapa pengguna dengan ramah.
        - JANGAN memberikan saran apapun sebelum Anda memiliki semua informasi yang dibutuhkan.
        - Ajukan pertanyaan-pertanyaan berikut SATU PER SATU untuk membangun profil pengguna. Tunggu jawaban pengguna sebelum bertanya selanjutnya:
          1.  "Berapa usia Anda saat ini?"
          2.  "Berapa perkiraan pendapatan bulanan Anda (bisa dalam rentang, misal: 5-10 juta, 10-20 juta, dst)?"
          3.  "Apa tujuan utama investasi Anda dan dalam berapa lama Anda ingin mencapainya? (Contoh: Dana Pensiun dalam 20 tahun, DP Rumah dalam 5 tahun, atau lainnya)"
          4.  "Bagaimana toleransi Anda terhadap risiko? Pilih salah satu:
              a. Konservatif (Saya tidak ingin nilai investasi saya turun sama sekali, keuntungan stabil lebih utama)
              b. Moderat (Saya siap menerima sedikit fluktuasi untuk potensi keuntungan yang lebih tinggi)
              c. Agresif (Saya siap mengambil risiko tinggi untuk potensi keuntungan maksimal)"
        
        LANGKAH 2: Analisis dan Penggunaan Tools.
        - Setelah SEMUA pertanyaan di atas terjawab, rangkum profil pengguna yang telah Anda pahami.
        - SEKARANG, Anda boleh menggunakan tools yang tersedia.
        - Gunakan `search_the_web` untuk mencari kondisi ekonomi makro saat ini, tren sektor yang relevan, atau mencari informasi tentang produk investasi seperti reksadana.
        - Gunakan `get_stock_price`, `get_crypto_price`, atau `get_gold_price` untuk memeriksa harga terkini dari instrumen yang Anda pertimbangkan.
        
        LANGKAH 3: Memberikan Rekomendasi.
        - Berdasarkan profil pengguna dan informasi yang Anda kumpulkan dari tools, berikan rekomendasi yang terstruktur.
        - Format rekomendasi Anda sebagai berikut:
          - **Gunakan mata uang Rupiah (Indonesian Rupiah atau IDR atau Rp)
          - **Ringkasan Profil Anda:** (Sebutkan kembali usia, pendapatan, tujuan, dan profil risiko pengguna).
          - **Rekomendasi Alokasi Aset:** (Berikan persentase, misal: 60% Saham, 30% Reksadana Pendapatan Tetap, 10% Emas).
          - **Contoh Instrumen Spesifik:** (Berikan beberapa contoh konkret, misal: Saham: BBCA.JK, Reksadana: Sucorinvest Sharia Money Market Fund, Emas: Emas fisik atau Antam).
          - **Alasan Rekomendasi:** (Jelaskan secara singkat mengapa alokasi dan instrumen tersebut cocok untuk profil pengguna).
          - **Rekomendasi crypto juga
        
        LANGKAH 4: Disclaimer.
        - SELALU akhiri respons rekomendasi Anda dengan disclaimer penting berikut:
          "PENTING: Rekomendasi ini dibuat oleh AI berdasarkan informasi yang Anda berikan dan data pasar saat ini. Ini bukanlah nasihat keuangan profesional. Selalu lakukan riset Anda sendiri (DYOR - Do Your Own Research) dan/atau konsultasikan dengan perencana keuangan berlisensi sebelum membuat keputusan investasi."
        """
        
        st.session_state.agent = create_react_agent(model=llm, tools=tools, prompt=system_prompt)
        st.session_state._last_google_key = google_api_key
        st.session_state.pop("messages", None) # Hapus history jika agent dibuat ulang
        
    except Exception as e:
        st.error(f"Terjadi kesalahan saat inisialisasi model: {e}")
        st.stop()


# --- 5. Manajemen dan Tampilan Riwayat Chat ---

if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan pesan-pesan yang sudah ada
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 6. Input Pengguna dan Interaksi dengan Agent ---

prompt = st.chat_input("Coba disapa dong AI Agent-nya...")

if prompt:
    # Tambahkan dan tampilkan pesan pengguna
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Dapatkan respons dari asisten
    try:
        messages = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        # with st.chat_message("assistant"):
        #     with st.spinner("Berpikir..."):
        #         # Kirim riwayat chat ke agent
        #         response = st.session_state.agent.invoke({"messages": messages})
        #         answer = response["messages"][-1].content
        #         st.markdown(answer)
        with st.spinner("Berpikir..."):
            # Kirim riwayat chat ke agent
            response = st.session_state.agent.invoke({"messages": messages})
            answer = response["messages"][-1].content
            st.markdown(answer)
        
        # Tambahkan respons asisten ke riwayat
        st.session_state.messages.append({"role": "assistant", "content": answer})

    except Exception as e:
        st.error(f"Maaf, terjadi kesalahan: {e}")