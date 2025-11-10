# Bab 5 : State of the art in deep learning : Transformers

Bab ini akan membahas,
- Representasi teks dalam format numerik untuk model machine learning
- Membuat model transformasi menggunakan sub-classing API dari Keras

Transformasi adalah generasi terakhir dari deep networks. Jurnal Vaswani er al., yang berjudul "Attention is All You Need" (dapat di akses di : `https://arxiv.org/pdf/1706.03762.pdf`), mempopulerkan ide tersebut. Transformer ini di implementasikan ke company global seperi Google, OpenAI, dan Facebook, dimana model mereka terbilang besar dan melebihi kemampuan model lain di domain NLP (Natural Language Processing). Fokus dari bab ini akan mengacu pada bagaimana penggunaan Transformer di domain NLP, secara khusus di model yang bertugas utama mentranslasikan bahasa. 

---

## Representasi Teks sebagai nomor

Untuk merepresentasikan teks menjadi penomoran maka perlu untuk memahami struktur penomorannya, dan salah satu caranya adalah dengan NLP (Natural Language Processing).

Sebagai contoh, kita diberikan beberapa set kalimat, 
- I went to the beach.
- It was cold.
- I came back to the house.

Langkah pertama adalah me-nomori setiap kata yang similar dengan ID, dimulai dari 1 dengan 0 digunakan untuk karakter khusus, contohnya kita me-nomori dengan ID seperti berikut,
- I → 1
- went → 2
- to → 3
- the → 4
- beach → 5
- It → 6
- was → 7
- cold → 8
- came → 9
- back → 10
- house → 11

Sehingga diperoleh ID dan kalimat yang berkoresponden seperti berikut,
- [1, 2, 3, 4, 5]
- [6, 7, 8]
- [1, 9, 10, 3, 4, 11]

---

## Penjelasan Mendalam

### 1. Arsitektur Encoder-Decoder

Transformer dijelaskan sebagai arsitektur encoder-decoder. Encoder memproses seluruh sekuens input untuk membuat representasi kontekstual. Decoder menggunakan representasi dari encoder dan output yang telah dihasilkannya sejauh ini untuk menghasilkan token berikutnya secara autoregresif.

### 2. Self-Attention

Ini adalah mekanisme inti dari Transformer. Berbeda dengan RNN yang memproses sekuens kata per kata, self-attention memungkinkan model untuk melihat semua kata lain dalam sekuens saat memproses satu kata tertentu. Ini dilakukan dengan menghitung skor relevansi antara setiap pasangan kata menggunakan tiga vektor: Query (Q), Key (K), dan Value (V). 

### 3. Masked Self-Attention

Variasi dari self-attention yang digunakan di decoder. Masking ini mencegah decoder untuk "melihat" token-token di masa depan (posisi selanjutnya dalam sekuens) saat memprediksi token saat ini, yang sangat penting untuk tugas-tugas generatif.

### 4. Multi-Head Attention

Ide untuk menjalankan beberapa mekanisme self-attention secara paralel (disebut "heads") dan kemudian menggabungkan hasilnya. Ini memungkinkan model untuk secara bersamaan memperhatikan informasi dari subspace representasi yang berbeda.

---

## Gambar atau Grafik Representasi  

**1. Gambar 5.3**

Diagram tingkat tinggi yang sangat baik dari arsitektur encoder-decoder untuk tugas terjemahan mesin, menunjukkan alur dari teks input ke representasi numerik, output encoder, decoder, dan akhirnya output teks.

**2. Gambar 5.6**

Ilustrasi paling detail dan penting di bab ini. Gambar ini memecah komputasi self-attention menjadi langkah-langkah yang dapat dipahami, mulai dari input, pembuatan vektor Q, K, V, perhitungan matriks skor (Q.KT), normalisasi softmax, dan perkalian akhir dengan V untuk mendapatkan output53.

**3. Gambar 5.9**
Perbandingan visual antara self-attention standar dan masked self-attention, dengan jelas menunjukkan bagaimana masking mencegah koneksi ke token masa depan.

---

## Kode

- **Listing 5.1 & 5.2:** Implementasi `SelfAttentionLayer` sebagai lapisan Keras kustom menggunakan Sub-classing API. Kode ini secara langsung menerjemahkan teori self-attention menjadi kode TensorFlow yang fungsional, termasuk implementasi untuk masking opsional55555555.

- **Listing 5.5 & 5.6:** Kode untuk membangun `EncoderLayer` dan DecoderLayer` yang lebih lengkap, yang menggabungkan multi-head attention dan lapisan fully connected.

- **Listing 5.7:** Kode yang menyatukan semua komponen (`EncoderLayer`, `DecoderLayer`, `Embedding`) untuk membangun model Transformer mini menggunakan Keras Functional API56.

---

# KESIMPULAN AKHIR

- Jaringan transformator telah mengungguli model lain di hampir semua tugas NLP.
- Transformator adalah jaringan saraf tipe encoder-decoder yang terutama digunakan untuk mempelajari tugas-tugas NLP.
- Dengan Transformer, encoder dan dekoder terdiri dari dua sublapisan komputasional: lapisan self-attention dan lapisan terhubung penuh.
- Lapisan self-attention menghasilkan jumlah masukan terbobot untuk langkah waktu tertentu, berdasarkan seberapa penting memperhatikan posisi lain dalam urutan tersebut saat memproses posisi saat ini.
- Lapisan terhubung penuh menciptakan representasi nonlinier dari keluaran yang diperhatikan yang dihasilkan oleh lapisan self-attention.
- Dekoder menggunakan masking pada lapisan self-attention-nya untuk memastikan bahwa dekoder tidak melihat prediksi di masa mendatang saat menghasilkan prediksi saat ini.
