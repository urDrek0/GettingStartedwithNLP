# Bab 10: Natural Language Processing dengan TensorFlow: Pemodelan Bahasa

Bab ini membahas tugas pre-training fundamental dalam NLP, yaitu language modeling, dan memperkenalkan arsitektur RNN lain serta teknik decoding yang lebih canggih.

---

## **Teori**

1. **Language Modeling**: Didefinisikan sebagai tugas memprediksi kata berikutnya dalam sebuah sekuens, P(wn | w1, ..., wn-1).  Ini adalah tugas self-supervised (label dapat dibuat secara otomatis dari teks mentah) yang memaksa model untuk mempelajari tata bahasa, semantik, dan pengetahuan dunia.
2. **N-grams**: Untuk mengurangi ukuran vocabulary yang sangat besar, teks dapat dipecah menjadi n-gram karakter (misalnya, bi-gram atau tri-gram) alih-alih kata-kata utuh.  Ini membantu model menangani kata-kata yang jarang muncul atau salah ketik.
3. **Gated Recurrent Units (GRU)**: Diperkenalkan sebagai alternatif yang lebih sederhana dan lebih efisien secara komputasi dari LSTM. GRU menggabungkan forget dan input gates menjadi satu **update gate** dan juga menggabungkan cell state dan hidden state.  Meskipun lebih sederhana, performanya seringkali sebanding dengan LSTM .
4. **Perplexity**: Dijelaskan sebagai metrik evaluasi standar untuk language models, lebih baik daripada akurasi. Secara intuitif, perplexity mengukur seberapa "bingung" atau "terkejut" model saat melihat data uji.  Nilai perplexity yang lebih rendah menunjukkan model yang lebih baik.
5. **Decoding Strategies**:
    -  **Greedy Decoding**: Strategi decoding paling sederhana, di mana pada setiap langkah waktu, kita hanya memilih kata dengan probabilitas tertinggi sebagai kata berikutnya.
    - **Beam Search**: Teknik decoding yang lebih canggih. Alih-alih hanya memilih satu kata terbaik di setiap langkah, beam search mempertahankan sejumlah `k` (disebut *beam width*) hipotesis atau sekuens kandidat terbaik.  Ini secara signifikan meningkatkan kualitas teks yang dihasilkan dibandingkan dengan greedy decoding.

---

## **Gambar**

-  **Gambar 10.1**: Diagram alir yang menunjukkan bagaimana data mentah (cerita) diubah menjadi data training untuk language modeling: **windowing** (membuat sekuens dengan panjang tetap) dan kemudian **splitting** menjadi pasangan input dan target.
- **Gambar 10.3 & 10.4**: Visualisasi arsitektur sel GRU, menyoroti perbedaannya dengan LSTM (hanya dua gerbang dan satu state).
- **Gambar 10.6**: Ilustrasi yang sangat baik tentang cara kerja beam search, menunjukkan bagaimana beberapa jalur kandidat dieksplorasi secara paralel untuk menemukan sekuens dengan probabilitas gabungan tertinggi.

---

## **Kode**

- **Listing 10.3**: Implementasi pipeline `tf.data` untuk language modeling.  Ini menunjukkan teknik lanjutan seperti penggunaan `flat_map` dengan `window` untuk secara efisien membuat sekuens input-target yang tumpang tindih dari teks yang panjang.
- **Listing 10.4**: Implementasi model GRU untuk language modeling.  Penting untuk dicatat bahwa `return_sequences=True` digunakan pada lapisan GRU karena kita membutuhkan prediksi untuk setiap token dalam sekuens input (bukan hanya yang terakhir).
-  **Listing 10.5**: Kode untuk mengimplementasikan metrik **Perplexity** kustom dengan men-subclass `tf.keras.metrics.Mean` dan menggunakan `tf.keras.losses.SparseCategoricalCrossentropy` di dalamnya.
- **Listing 10.6 & 10.7**: Menunjukkan cara membangun **model inferensi** yang terpisah dari model training.  Model inferensi perlu menangani state RNN secara eksplisit untuk dapat menghasilkan teks satu token pada satu waktu secara rekursif.
- **Listing 10.8**: Implementasi **beam search** sebagai fungsi rekursif di Python.  Meskipun implementasi ini lebih konseptual, ini memberikan pemahaman yang kuat tentang logika di balik algoritma tersebut.
