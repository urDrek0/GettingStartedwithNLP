# Bab 3: Keras dan Pengambilan Data di TensorFlow 2
Bab ini berfokus pada API tingkat tinggi dari Keras untuk membangun model dengan mudah, serta berbagai cara untuk memuat dan memproses data untuk model TensorFlow.
---
# Teori
- **(API Pembangun Model Keras**: Penulis menjelaskan tiga API utama untuk membangun model di Keras:
  
  - **Sequential API**: Paling sederhana, digunakan untuk model dengan tumpukan lapisan linear (satu input, satu output).
  
  - **Functional API**: Lebih fleksibel, memungkinkan arsitektur kompleks seperti model dengan multi-input, multi-output, dan cabang paralel.
  
  - **Sub-classing API**: Memberikan kontrol penuh, di mana kita mendefinisikan model atau lapisan kustom sebagai sebuah kelas Python yang mewarisi dari tf.keras.Model atau tf.keras.layers.Layer.

- **Metode Pengambilan Data**:

  - `tf.data API`: Cara paling kuat dan efisien untuk membangun pipeline data yang kompleks dan dapat diskalakan. Sangat direkomendasikan untuk dataset besar.
  
  - **Keras DataGenerators**: Kelas utilitas seperti ImageDataGenerator yang memudahkan pemuatan jenis data tertentu (misalnya, gambar) langsung dari direktori.
  
  - **tensorflow-datasets**: Sebuah pustaka terpisah yang menyediakan akses mudah ke ratusan dataset populer dengan satu baris kode.
