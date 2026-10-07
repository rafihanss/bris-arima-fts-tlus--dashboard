# Dashboard ARIMA vs FTS-TLUS — Saham BRIS.JK

Dashboard interaktif bergaya Stockbit untuk membandingkan akurasi model ARIMA dan
FTS-TLUS pada data testing 2026, dibangun dari hasil notebook penelitian.

## Cara menjalankan

1. Pastikan Python 3.9+ sudah terpasang.
2. Buka terminal di folder ini, lalu jalankan:

   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

3. Browser akan otomatis terbuka ke `http://localhost:8501`.

## Struktur folder

```
dashboard/
├── app.py                  <- kode dashboard
├── requirements.txt
├── README.md
└── data/
    ├── forecast_comparison.csv
    ├── model_comparison.csv
    ├── BRIS_clean_data.csv
    ├── BRIS_train.csv
    ├── BRIS_test.csv
    ├── fts_fuzzy_sets.csv
    ├── fts_tlus_table.csv
    ├── FTS_TLUS_forecast.csv
    ├── ARIMA_forecast.csv
    ├── validation_checklist.csv
    └── arima_model_info.json
```

Seluruh file di folder `data/` diambil langsung dari output notebook
(`Laporan.zip`). Jangan mengubah nama kolom di dalamnya, karena `app.py`
membaca nama kolom tersebut secara eksplisit.

## Fitur dashboard

- **Header ala Stockbit**: nama ticker, harga penutupan terakhir pada data
  testing, dan indikator naik/turun berwarna hijau/merah.
- **Kartu metrik**: prediksi terakhir tiap model, model dengan MAE terendah,
  jumlah observasi testing.
- **Grafik utama interaktif** (Plotly): aktual vs ARIMA vs FTS-TLUS, dengan
  range slider, toggle tampilkan/sembunyikan tiap model di sidebar, dan opsi
  menambahkan riwayat training di latar belakang.
- **Tab "Akurasi Model"**: grafik batang MAE/RMSE/MAPE, dan interpretasi
  deskriptif otomatis.
- **Tab "Error Harian"**: grafik error absolut harian, serta 5 error terbesar
  masing-masing model.
- **Tab "Tabel Forecast"**: tabel lengkap dengan pewarnaan error (merah jika
  di atas median, hijau jika di bawah), plus tombol unduh CSV hasil filter.
- **Tab "Detail FTS-TLUS"**: tabel fuzzy set, tabel Table Look-Up Scheme, dan
  distribusi fuzzy state pada data training.
- **Sidebar validasi**: ringkasan checklist anti-data-leakage dari notebook
  (berapa item PASS dari total).

## Catatan

Dashboard ini murni menampilkan ulang hasil yang sudah dihitung di notebook —
tidak ada model yang di-fit ulang di sini, sehingga aman digunakan untuk
presentasi tanpa risiko angka berubah-ubah.
