# Bab 9: Natural Language Processing dengan TensorFlow: Analisis Sentimen

Bab ini menandai transisi kita dari computer vision ke NLP, dimulai dengan tugas fundamental analisis sentimen.

---

## **Teori**
1. **Preprocessing Teks**: Menjelaskan langkah-langkah penting untuk membersihkan data teks sebelum pemodelan:
    - **Lowercasing**: Mengubah semua teks menjadi huruf kecil.
    - **Tokenization**: Memecah teks menjadi unit-unit individual (kata).
    - **Stop Word Removal**: Menghapus kata-kata umum yang tidak informatif (misalnya, 'the', 'a', 'is').  Penulis dengan cerdas menunjukkan untuk **tidak** menghapus kata-kata seperti 'not' atau 'no', karena mereka sangat penting untuk sentimen.
    - **Lemmatization**: Mengubah kata ke bentuk dasarnya (misalnya, 'walking' menjadi 'walk').  Ini memerlukan **Part-of-Speech (PoS) tagging** untuk bekerja dengan benar.
2. **Long Short-Term Memory (LSTM) Networks**: Diperkenalkan sebagai jenis RNN yang sangat efektif untuk tugas sekuensial. LSTMs mengatasi masalah vanishing gradient dari RNN sederhana dengan menggunakan mekanisme **gates** (input, forget, output) dan sebuah **cell state**.  Cell state bertindak sebagai memori jangka panjang, sementara gates secara selektif mengontrol informasi apa yang disimpan, dibuang, dan dikeluarkan.
3. **Word Embeddings**: Dijelaskan sebagai representasi vektor yang padat (dense) dan berdimensi rendah untuk kata-kata. Tidak seperti one-hot encoding, embeddings dapat **menangkap hubungan semantik** antara kata-kata (misalnya, vektor untuk 'king' dan 'queen' akan berdekatan di ruang vektor).  Embeddings ini dapat dipelajari dari awal bersama dengan model atau menggunakan versi pretrained seperti GloVe atau Word2Vec.

---

## **Gambar**

- **Gambar 9.5**: Diagram arsitektur model analisis sentimen, menunjukkan alur dari input ID token, melalui lapisan one-hot/embedding, lapisan LSTM, dan akhirnya melalui lapisan Dense untuk menghasilkan prediksi sentimen.
- **Gambar 9.7**: Ilustrasi yang sangat baik tentang cara kerja lapisan Embedding.  Ini menunjukkan bagaimana ID kata digunakan sebagai indeks untuk "mencari" vektor yang sesuai dari matriks embedding yang dapat dilatih.

---

## **Kode**

- **Listing 9.2**: Fungsi `clean_text` yang menggabungkan semua langkah preprocessing teks (lowercasing, tokenization, stop word removal, lemmatization) menggunakan pustaka **NLTK**.  Ini adalah resep praktis yang dapat digunakan kembali.
-  **Keras `Tokenizer`**: Menunjukkan cara menggunakan `tf.keras.preprocessing.text.Tokenizer` untuk membangun kamus kata dan mengubah sekuens teks menjadi sekuens integer (ID token).
- **Listing 9.4 & Pipeline Data dengan Bucketing**: Kode untuk membangun pipeline `tf.data` untuk data teks dengan panjang variabel. Ini memperkenalkan `tf.RaggedTensor` dan teknik **bucketing** menggunakan `tf.data.experimental.bucket_by_sequence_length`.  Bucketing mengelompokkan sekuens dengan panjang yang sama ke dalam batch yang sama, yang jauh lebih efisien daripada mem-padding semua sekuens ke panjang maksimum.
- **Listing 9.5**: Implementasi model LSTM untuk klasifikasi biner menggunakan Keras Sequential API.  Ini menunjukkan penggunaan `Masking` layer untuk mengabaikan token padding, diikuti oleh lapisan `LSTM` dan `Dense`.
-  **`class_weight`**: Menunjukkan cara menangani dataset yang tidak seimbang (lebih banyak ulasan positif daripada negatif) dengan memberikan bobot yang lebih tinggi pada kelas minoritas selama perhitungan loss, menggunakan argumen `class_weight` di `model.fit()`.
-  **Listing 9.7**: Model yang ditingkatkan yang menggantikan one-hot encoding dengan `Embedding` layer, yang biasanya memberikan performa yang lebih baik dan lebih efisien.