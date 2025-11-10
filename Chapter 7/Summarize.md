# Bab 7: Mengajari Mesin untuk Melihat Lebih Baik: Meningkatkan CNN dan Membuatnya Mengaku

Bab ini berfokus pada teknik-teknik untuk meningkatkan performa model klasifikasi gambar yang kita bangun di Bab 6, terutama dalam mengurangi overfitting dan menginterpretasikan keputusan model.

---

## **Teori**

### 1. **Teknik Mengurangi Overfitting**

- **Data Augmentation**: Menjelaskan cara menciptakan data training baru dari data yang ada dengan menerapkan transformasi acak seperti rotasi, pergeseran, zoom, dan penyesuaian kecerahan.  Ini membantu model untuk menggeneralisasi lebih baik.
- **Dropout**: Teknik regularisasi di mana neuron-neuron dinonaktifkan secara acak selama training.  Hal ini memaksa jaringan untuk mempelajari fitur-fitur yang redundan dan lebih kuat, sehingga tidak terlalu bergantung pada neuron tertentu.
-  **Early Stopping**: Sebuah strategi untuk menghentikan training ketika performa pada set data validasi berhenti meningkat, untuk mencegah model mulai menghafal data training.

### 2. **Arsitektur yang Ditingkatkan (Minception)**: Buku ini memperkenalkan "Minception", sebuah arsitektur yang terinspirasi dari **Inception-ResNet v2**.  Inovasi utamanya adalah pengenalan **Batch Normalization** dan **Residual Connections**.

- **Batch Normalization**: Menormalkan output dari lapisan sebelumnya. Ini menstabilkan dan mempercepat proses training.
- **Residual Connections (Skip Connections)**: Menambahkan input dari lapisan yang lebih awal ke output dari lapisan yang lebih dalam. Ini menciptakan "jalan pintas" untuk aliran gradien, memungkinkan training jaringan yang jauh lebih dalam tanpa masalah vanishing gradient.
- **Transfer Learning**: Konsep menggunakan model yang telah dilatih sebelumnya (pretrained) pada dataset besar (misalnya, ImageNet) sebagai titik awal.  Alih-alih melatih dari nol, kita hanya perlu "menyempurnakan" (fine-tune) bobot model pada dataset kita yang lebih kecil, yang secara dramatis dapat meningkatkan akurasi dan mengurangi waktu training.
- **Grad-CAM (Gradient-weighted Class Activation Mapping)**: Sebuah teknik visualisasi untuk "membuat model mengaku". Grad-CAM menyoroti area pada gambar input yang paling berpengaruh terhadap prediksi kelas tertentu dengan menggunakan gradien yang mengalir ke lapisan konvolusi terakhir.  Ini membantu kita memahami *mengapa* model membuat keputusan tertentu.

---

## **Gambar**

- **Gambar 7.2**: Galeri visual yang sangat komprehensif yang menunjukkan efek dari berbagai parameter augmentasi pada `ImageDataGenerator` (rotasi, pergeseran, shear, zoom, dll.).  Sangat berguna untuk referensi praktis.
- **Gambar 7.4**: Ilustrasi tentang bagaimana dropout bekerja.  Selama training, neuron-neuron (lingkaran) dijatuhkan secara acak, memaksa model untuk belajar fitur redundan (misalnya, kumis dan telinga kucing).
- **Gambar 7.10 & 7.12**: Perbandingan arsitektur blok `Inception-ResNet` A dan B dari Minception, dengan jelas menunjukkan bagaimana residual connection (garis melengkung) ditambahkan untuk menggabungkan input dengan output.
- **Gambar 7.15**: Contoh output dari Grad-CAM.  Peta panas (heatmap) yang ditumpangkan pada gambar asli menunjukkan area mana yang "dilihat" oleh model untuk membuat prediksi (misalnya, model fokus pada kepala anjing untuk mengidentifikasi "Labrador retriever").

---

## **Kode**

-  **Listing 7.1**: Menunjukkan cara mengkonfigurasi `ImageDataGenerator` dengan berbagai parameter augmentasi untuk data training .
-  **Listing 7.4**: Implementasi `aux_out` yang dimodifikasi dengan menambahkan lapisan `Dropout`.
-  **Callback `EarlyStopping` & `ReduceLROnPlateau`**: Kode menunjukkan cara menggunakan callback Keras ini untuk menghentikan training secara otomatis dan menyesuaikan learning rate saat performa stagnan.
-  **Listing 7.7, 7.8, & 7.9**: Implementasi modular dari Minception, memecah arsitektur menjadi blok `inception_resnet_a`, `inception_resnet_b`, dan `reduction` .
- **Transfer Learning dengan `InceptionResNetV2`**: Kode menunjukkan betapa mudahnya menggunakan model pretrained dari `tf.keras.applications`.  Dengan `include_top=False`, kita dapat membuang kepala klasifikasi asli dan menambahkan kepala kustom kita sendiri.
-  **Implementasi Grad-CAM**: Meskipun kode lengkapnya ada di apendiks, bab ini menjelaskan logika di baliknya: menggunakan `tf.GradientTape` untuk menghitung gradien dari output kelas yang diprediksi terhadap feature map dari lapisan konvolusi terakhir.
