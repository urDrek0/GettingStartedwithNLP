
# Bab 6: Mengajari Mesin untuk Melihat: Klasifikasi Gambar dengan CNN

Bab ini adalah penyelaman mendalam pertama kita ke dalam proyek klasifikasi gambar yang realistis, memperkenalkan analisis data eksplorasi (EDA) dan arsitektur CNN yang canggih. Bagian ini berfokus pada penerapan praktis model deep learning, terutama dalam domain computer vision dan natural language processing (NLP). Kita akan membangun model yang lebih kompleks untuk tugas-tugas seperti klasifikasi gambar, segmentasi, analisis sentimen, dan pemodelan bahasa.

---

## **Teori**

- **Analisis Data Eksplorasi (EDA)**: Bab ini menekankan pentingnya EDA sebagai langkah awal krusial dalam setiap proyek data science.  Tujuannya adalah untuk memahami data secara mendalam, mengidentifikasi potensi masalah seperti kelas yang tidak seimbang, data rusak, atau outlier, sebelum memulai pemodelan=.
- **Inception Net v1 (GoogLeNet)**: Arsitektur CNN canggih ini diperkenalkan sebagai solusi untuk membuat jaringan yang lebih "dalam" (deep) tanpa menyebabkan ledakan jumlah parameter. Ide utamanya adalah **Inception block**, yang melakukan beberapa operasi konvolusi dengan ukuran kernel yang berbeda (1x1, 3x3, 5x5) dan pooling secara paralel, lalu menggabungkan hasilnya.  Hal ini memungkinkan model untuk menangkap fitur pada berbagai skala secara efisien =.
- **Konvolusi 1x1**: Dijelaskan sebagai teknik cerdas untuk **pengurangan dimensi** pada channel depth.  Sebelum melakukan konvolusi 3x3 atau 5x5 yang mahal secara komputasi, input dilewatkan melalui konvolusi 1x1 untuk mengurangi jumlah feature map, sehingga secara drastis mengurangi jumlah parameter dan komputasi.
- **Auxiliary Outputs**: Inception Net v1 juga menggunakan "kepala" klasifikasi tambahan di lapisan-lapisan tengahnya.  Tujuannya adalah untuk memerangi masalah **vanishing gradient** dengan menyediakan jalur gradien yang lebih pendek ke lapisan-lapisan awal, sehingga membantu menstabilkan proses training jaringan yang sangat dalam.

---

## **Gambar**

-  **Gambar 6.4**: Visualisasi struktur folder dataset `tiny-imagenet-200`, yang sangat membantu untuk memahami bagaimana data training dan validasi diorganisir, suatu langkah penting sebelum menulis kode untuk memuat data.
-  **Gambar 6.10**: Diagram arsitektur tingkat tinggi dari Inception Net v1, yang dengan jelas memisahkan komponen-komponen utama: **Stem** (lapisan konvolusi awal), tumpukan **Inception blocks**, dan **Auxiliary outputs**.
- **Gambar 6.14**: Ilustrasi detail dari Inception block dengan konvolusi 1x1 untuk reduksi dimensi.  Ini adalah visualisasi kunci untuk memahami inovasi utama dari arsitektur Inception.

---

## **Kode**
-  **Listing 6.1**: Fungsi Python menggunakan Pandas untuk membaca file `wnids.txt` dan `words.txt` guna membuat pemetaan dari ID kelas ke deskripsi kelas yang dapat dibaca manusia, sebuah langkah EDA yang umum.
- **Pipeline Data dengan `ImageDataGenerator`**: Bab ini menunjukkan cara membuat pipeline data untuk training, validasi, dan testing menggunakan `ImageDataGenerator`.  Kode ini memanfaatkan `flow_from_directory` untuk data training dan `flow_from_dataframe` untuk data validasi/testing yang memiliki struktur file berbeda .
- **Listing 6.3, 6.4, 6.5, 6.6**: Implementasi modular dari Inception Net v1 menggunakan Keras Functional API. Penulis memecah arsitektur menjadi fungsi-fungsi terpisah untuk `stem`, `inception` block, dan `aux_out`, yang kemudian dirakit menjadi model akhir.  Ini adalah contoh yang sangat baik dari praktik rekayasa perangkat lunak yang baik dalam membangun model yang kompleks.