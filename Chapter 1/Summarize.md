# Bagian 1: Dasar-Dasar TensorFlow 2 dan Deep Learning 
---
**Bab 1: Dunia TensorFlow yang Menakjubkan**
Bab pengantar ini memberikan gambaran umum tingkat tinggi tentang apa itu TensorFlow dan posisinya dalam lanskap machine learning.

# Teori
- **Definisi TensorFlow**: TensorFlow adalah sebuah kerangka kerja machine learning end-to-end yang dirancang untuk berjalan cepat pada perangkat keras yang dioptimalkan seperti GPU dan TPU. TensorFlow juga mendukung seluruh siklus hidup model, mulai dari pengembangan, deployment, hingga monitoring.
- **Perbandingan CPU vs GPU vs TPU**: 
    - CPU digambarkan sebagai komponen yang mampu menyelesaikan instruksi kompleks dengna waktu singkat.
    - GPU digambarkan sebagai sebuah komponen yang mampu untuk menjalankan sebuah instruksi sederhana dengan skala yang lebih besar.
    - TPU digambarkan sebagai komponen yang mampu untuk menyelesaikan sebuah instruksi yang spesifik, namun tidak fleksibel atau mudah untuk di aplikasikan jika dibandingkan dengan CPU dan GPU
- **Kapan Menggunakan TensorFlow?** Penggunaan TensorFlow optimal untuk,
    - prototyping model deep learning,
        Karena TensorFlow mendukung untuk Layer khusus untuk koneksi ke internet atau jaringan, konvolusi layer untuk neural network konvolusional, dan layer RNN (recurrent neural network)/LSTM (long short-term memory)/GRU (gated recurrent unit) untuk model sekuensial.
    - implementasi model yang butuh akselerasi perangkat keras
    - productionizing model di cloud
    - monitoring training
    - membuat data pipeline yang tangguh.
- **Kapan Tidak Menggunakan TensorFlow?** Sebaliknya hindari TensorFlow untuk,
    - model machine learning tradisional (seperti Decision Trees)
    - manipulasi data terstruktur skala kecil
    - pipeline NLP yang kompleks
**lalu apa solusinya?** Disarankan menggunakan pustaka lain seperti Scikit-learn, Pandas/NumPy, atau spaCy.
