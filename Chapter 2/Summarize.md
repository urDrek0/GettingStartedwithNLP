# Bab 2: TensorFlow 2

Bab ini masuk lebih dalam ke aspek teknis TensorFlow 2, memperkenalkan blok bangunan fundamental dan operasi-operasi umum.
---
# Teori

- **Eager Execution**: Perbedaan fundamental antara TF1 dan TF2 adalah eager execution yang diaktifkan secara default di TF2. Ini berarti operasi dieksekusi secara imperatif (langsung), membuatnya lebih intuitif dan mudah untuk di-debug, mirip seperti kode Python pada umumnya.

- **AutoGraph (`@tf.function`)**: Meskipun eager, TensorFlow tetap dapat memperoleh keuntungan performa dari komputasi berbasis graf. Dekorator 

- `@tf.function` secara otomatis men-trace kode Python yang berisi operasi TensorFlow dan mengubahnya menjadi data-flow graph yang sangat teroptimalkan.

- Blok Bangunan Fundamental:

    - `tf.Variable`: Struktur data yang mutable (nilainya bisa berubah), digunakan untuk menyimpan parameter model (weights dan bias) yang akan dioptimalkan selama training.
    - `tf.Tensor`: Struktur data yang immutable (nilainya tidak bisa diubah setelah dibuat), digunakan untuk menyimpan data input, output intermediet, dan output final model.
    - `tf.Operation`: Representasi dari sebuah komputasi, seperti tf.matmul (perkalian matriks) atau tf.add.

- Operasi Jaringan Saraf: Bab ini memperkenalkan operasi-operasi kunci dalam deep learning seperti perkalian matriks, konvolusi, dan pooling.
