# Bab 4 : Dipping toes in deep learning

Bab ini akan membahas,
- Implementasi dan keseluruhan pelatihan FCN (__Fully Connected Networks__) menggunakan Keras
- Implementasi dan pelatihan CNN (__Convolutional Nerual Networks__) untuk klasifikasi gambar
- Implementasi dan pelatihan RNN (__Recurrent Neural Networks__) untuk menyelesaikan permasalahn deret waktu (__time-series__)

---

## FCN (Fully Connected Networks)

Dalam buku ini diberikan narasi yang akan menjelaskan mengenai **Hyperparameter optimization**, yaitu sebuah proses yang cukup memakan sumber daya dan juga untuk evaluasinya perlu banyak model untuk dapat dipilih set parameter yang terbaik. hal ini membuatnya sulit diaplikasikan untuk metode deep learning, karena metode ini biasanya memiliki banyak data dan model dengan tingkat kompleksitas yang tinggi.

### 1. Memahami Data
Pada contoh proses dalam buku, digunakan MNIST digit data set, yaitu sebuah hand-drawn numbers dari 0 - 9. Proses Load data MNIST di TensorFlow dapat menggunakan

`from tensorflow.keras.datasets.minst import load_data`

`(x_train, y_train),  (x_text,y_test) = load_data()`

Code `load.data()` memungkinkan return dari data latihan dan data yang diujikan. Dalam kasus ini dilakukan __unsupervised task__, sehingga tidak ada label untuk data latihan, sehingga tidak perlu label untuk menyelesaikan tugas ini. MNIST kini dianggap terlalu mudah untuk di identifikasi oleh __computer vision__, sehingga untuk implementasi data set lain sangat dianjurkan. Selanjutnya, untuk memahami proses pelatihan dapat digunakan `print` untuk melihat prosesnya secara langsung, seperti contoh.

`print(x_train)`
`print('x_train has shape {}'.format(x_train.shape))`

begitu juga berlaku dengan `y_train`,

`print(y_train)`
`print('y_train has shape {}'.format(y_train.shape))`

selanjutnya bisa dilakukan normalisasi sepert contoh

`norm_x_train = ((x_train - 128.0)/128.0).reshape([-1,784])`

pada contoh kasus shape dari x dan y train adalah sebanyak 60000,28,28 sehingga dengan normalisasi mengaturnya ke satu dimensional vektor sepanjang 784 satuan. Apabila kita menampilkan data maka gambar akan terlihat bisa dikenali secara sekilas, lalu bagaimana cara kita untuk dapat membuat model yang dapat mengenali bahkan ketika gambar terlihat terdistorsi? dengan code berikut akan membuat gambar terdistorsi.

`import numpy as np`
`def generate_masked_inputs(x, p, seed=None):`
    `if seed:`
        `np.random.seed(seed)`
    `mask = np.random.binomial(n=1, p=p, size=x.shape).astype('float32')`
    `return x * mask`
`masked_x_train = generate_masked_inputs(norm_x_train, 0.5)`

pada `masked_x_train = generate_masked_inputs(norm_x_train, 0.5)` memberikan kita data yang hanya dapat dikenali 50% dari data aslinya.

### 2. Model Auto Encoder

FCN dan MLP(__Multilayer Perceptron__) beroprasi secara similar, namun MLP di desain untuk menyelesaikan supervised task, dimana autoencoder di desain untuk menyelesaikan unsupervised task, maka dari itu sebenarnya MLP dan FCN adalah FCN juga.

Perbedaan antara unsupervised dan supervised utamanya terletak di karakteristik dataset, dimana unsupervised tidak berlabel dalam datanya, sedangkan supervised berlabel. Beberapa sueprvised task adalah klasifikasi gambar, deteksi objek, pengenalan suara, dan analisis sentimen. Sedangkan contoh unsupervised adalah rekonstruksi gambar, image generation dengan generative adversarial networks, klustering text, dan language modeling.

Selain itu perlu juga untuk di lakukan "denosising autoencoders", jika kita kaitkan pada contoh maka tujuannya adalah,

- Produce gambar dengan kualitas yang lebih baik
- Menghilangkan randomness variations seperti pencahayaan atau warna
- Artifacts yang dikarenakan kompresi JPEG

## CNN (Convolutional Neural Networks)

CNN yang digunakan untuk mengklasifikasikan gambar diimplementasikan ke sebuah model dilengkapi dengan dataset yang mencukupi serta variansi data set yang tinggi. CNN berfokus pada akurasi pengklasifikasian gambar.

### 1. Memahami Data
### 2. Model Auto Encoder

## RNN (Recurrent Neural Networks)

RNN memiliki kemampuan khusus yang CNN dan FCN tidak miliki, yaitu dapat membaca dan mempelajari data dengan deret waktu (__time-series__). Sebenarnya FCN dan CNN dapat membaca data deret waktu, namun perlu perlakuan dan adaptasi yang khusus. RNN tidak hanya digunakan untuk membuat prediksi, tapi juga menggunakan ingatan dari jaringan lama untuk menentukan step pada waktu tertentu.

### 1. Memahami Data
### 2. Model Auto Encoder
