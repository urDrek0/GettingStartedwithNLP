# Bab 8: Membedakan Benda: Segmentasi Gambar

Bab ini beralih dari klasifikasi gambar ke tugas yang lebih kompleks, yaitu segmentasi gambar, di mana tujuannya adalah untuk mengklasifikasikan setiap piksel dalam gambar.

---

## **Teori**

1. **Segmentasi Semantik vs. Instans**: Bab ini membedakan dua jenis segmentasi. **Segmentasi semantik** hanya peduli tentang kategori objek (misalnya, semua orang diberi label "orang").  **Segmentasi instans** membedakan setiap objek individual (misalnya, orang 1, orang 2, dst.). Bab ini berfokus pada segmentasi semantik.
2. **DeepLab v3**: Arsitektur canggih untuk segmentasi semantik diperkenalkan. Alih-alih arsitektur encoder-decoder simetris seperti U-Net, DeepLab v3 menggunakan **backbone CNN pretrained** (seperti ResNet-50) yang dimodifikasi.
3. **Atrous (Dilated) Convolution**: Inovasi kunci dalam DeepLab. Ini adalah konvolusi dengan "lubang" di antara bobot kernelnya.  Hal ini memungkinkan model untuk memiliki **bidang reseptif (receptive field) yang lebih besar** tanpa menambah jumlah parameter, sehingga dapat menangkap informasi kontekstual yang lebih luas tanpa kehilangan resolusi spasial.
4. **Atrous Spatial Pyramid Pooling (ASPP)**: Modul ini adalah inti dari DeepLab. ASPP menerapkan beberapa konvolusi atrous paralel dengan *dilation rate* yang berbeda pada feature map. Ini memungkinkan model untuk **menangkap konteks pada beberapa skala** secara bersamaan.  Hasil dari semua cabang ini kemudian digabungkan untuk membuat prediksi akhir yang kaya akan detail.
5. **Fungsi Loss dan Metrik Segmentasi**:
    - **Loss**: Karena ketidakseimbangan kelas yang parah (piksel latar belakang sering mendominasi), **weighted cross-entropy** dan **Dice loss** diperkenalkan.  Dice loss secara langsung mengoptimalkan tumpang tindih (overlap) antara prediksi dan ground truth.
    -  **Metrik**: **Mean Intersection over Union (mIoU)** dijelaskan sebagai metrik standar emas untuk segmentasi, yang mengukur rasio tumpang tindih antara area yang diprediksi dan area sebenarnya untuk setiap kelas, lalu dirata-ratakan.

---

## **Gambar**

- **Gambar 8.1**: Perbandingan visual yang jelas antara segmentasi semantik dan instans, menunjukkan bagaimana instans memberikan label unik untuk setiap objek.
- **Gambar 8.9**: Ilustrasi yang sangat baik tentang cara kerja atrous convolution.  Ini menunjukkan bagaimana kernel 3x3 dengan dilation rate yang berbeda dapat mencakup area input yang lebih besar sambil tetap menggunakan hanya 9 parameter.
- **Gambar 8.11**: Diagram arsitektur modul ASPP, menunjukkan beberapa cabang paralel dari konvolusi atrous dan global average pooling yang digabungkan.

---

## **Kode**

- **Listing 8.2**: Fungsi Python menggunakan pustaka PIL dan NumPy untuk memuat gambar target yang **terpaletisasi**.  Ini adalah langkah praktis yang penting karena banyak dataset segmentasi menyimpan mask dengan cara ini untuk menghemat ruang.
- **Listing 8.3 & 8.6**: Kode untuk membangun **pipeline `tf.data` yang lengkap** untuk tugas segmentasi.  Ini menunjukkan cara memuat pasangan gambar (input dan mask), menerapkan augmentasi (seperti flip horizontal), dan memastikan augmentasi yang sama diterapkan pada input dan mask.
- **`tf.numpy_function`**: Menunjukkan cara membungkus fungsi Python/NumPy kustom (seperti pemuat gambar palet) agar dapat digunakan di dalam graf `tf.data`, sebuah teknik yang sangat berguna.
- **Listing 8.8, 8.9, 8.10**: Implementasi modular dari **ResNet block yang dimodifikasi** dengan atrous convolution.
- **Listing 8.11**: Implementasi modul **ASPP** menggunakan Keras Functional API.
- **Listing 8.12**: Perakitan model **DeepLab v3** akhir, menggabungkan backbone ResNet-50 yang dimodifikasi dengan modul ASPP.
- **Listing 8.14, 8.15, & 8.17-8.19**: Implementasi **loss dan metrik kustom** untuk segmentasi (weighted cross-entropy, dice loss, pixel accuracy, mean accuracy, mIoU) dengan men-subclass kelas metrik Keras. Ini adalah contoh lanjutan yang sangat baik tentang cara memperluas fungsionalitas Keras.