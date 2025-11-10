# Bab 11: Pembelajaran Sequence-to-Sequence: Bagian 1

Bab ini memperkenalkan arsitektur model yang kuat yang dirancang untuk memetakan sekuens input dengan panjang arbitrer ke sekuens output dengan panjang arbitrer. Fokus utamanya adalah membangun sebuah **penerjemah mesin** sederhana dari bahasa Inggris ke bahasa Jerman.

---

## **Teori**

1. **Model Sequence-to-Sequence (Seq2Seq)**: Model ini dijelaskan sebagai arsitektur **encoder-decoder**.
    - **Encoder**: Bagian ini (dalam bab ini, menggunakan **GRU**) memproses seluruh sekuens input (misalnya, kalimat bahasa Inggris) dan mengompresnya menjadi satu representasi vektor yang disebut **context vector** (atau "thought vector"). Vektor ini bertujuan untuk merangkum makna dari seluruh sekuens input.
    - **Decoder**: Bagian ini (juga menggunakan **GRU**) mengambil context vector dari encoder sebagai *initial state*-nya. Tujuannya adalah untuk menghasilkan sekuens output (misalnya, kalimat bahasa Jerman) satu token pada satu waktu, berdasarkan context vector tersebut.
2. **Token Khusus ('sos' & 'eos')**: Pentingnya menambahkan token khusus ke sekuens target (Jerman) ditekankan. Token **'sos' (start of sentence)** memberi sinyal kepada decoder untuk mulai menghasilkan kata, sementara token **'eos' (end of sentence)** memberi sinyal kapan harus berhent.
3. **Teacher Forcing**: Ini adalah strategi training yang penting untuk model seq2seq. Alih-alih membiarkan decoder menggunakan output prediksinya sendiri sebagai input untuk langkah waktu berikutnya (yang bisa jadi salah dan menyebabkan kesalahan yang menumpuk), kita "memaksa" decoder dengan memberinya **kata yang benar** dari sekuens target sebagai input pada setiap langkah waktu. Ini menstabilkan dan mempercepat training.
4. **Perbedaan Model Training vs. Inferensi**: Akibat dari teacher forcing, model yang digunakan untuk training **berbeda** dengan model yang digunakan untuk inferensi (prediksi).
    - Saat **training**, decoder memproses seluruh sekuens target secara paralel.
    - Saat **inferensi**, kita tidak memiliki sekuens target. Oleh karena itu, kita harus membangun decoder **rekursif** yang berbeda. Decoder ini mengambil 'sos' sebagai input pertama, menghasilkan prediksi kata pertama, lalu menggunakan prediksi tersebut sebagai input untuk langkah waktu kedua, dan seterusnya, hingga memprediksi 'eos'.

---

## **Gambar**

- **Gambar 11.1**: Diagram konseptual tingkat tinggi dari arsitektur encoder-decoder, yang secara jelas menunjukkan *context vector* sebagai jembatan informasi antara encoder dan decoder.
- **Gambar 11.3**: Ilustrasi terperinci dari model seq2seq selama **fase training** (menggunakan teacher forcing). Ini menunjukkan bagaimana sekuens input dan sekuens target (yang di-offset) dimasukkan ke dalam model.
- **Gambar 11.4**: Diagram yang sangat penting yang mengkontraskan model training dengan **model inferensi**. Ini secara visual menunjukkan loop rekursif di mana output yang diprediksi oleh decoder pada satu langkah waktu menjadi input untuk langkah waktu berikutnya.

---

## **Kode**

- **`TextVectorization` Layer**: Bab ini memperkenalkan lapisan `tf.keras.layers.experimental.preprocessing.TextVectorization`. Ini adalah komponen yang sangat kuat yang mengintegrasikan preprocessing teks (tokenisasi, pembuatan kamus, dan konversi ke ID integer) langsung ke dalam model Keras, sehingga model dapat menerima input string mentah.
- **Listing 11.3**: Fungsi `get_vectorizer` menunjukkan cara menginisialisasi `TextVectorization`, mengadaptasinya (melatihnya) pada korpus teks, dan membungkusnya dalam `tf.keras.Model` yang dapat digunakan kembali.
- **Listing 11.4**: Fungsi `get_encoder`, yang mendefinisikan encoder menggunakan Keras Functional API. Ini menumpuk lapisan `Input` (untuk string mentah), `TextVectorization`, `Embedding`, dan `Bidirectional` `GRU`.
- **Listing 11.5**: Fungsi `get_final_seq2seq_model` yang merakit model training lengkap. Poin kuncinya adalah bagaimana output encoder (`d_init_state`) secara eksplisit diumpankan sebagai `initial_state` ke lapisan GRU decoder.
- **Listing 11.6**: Fungsi `prepare_data` yang sangat penting, yang menunjukkan bagaimana mempersiapkan data untuk teacher forcing. Ini membagi data target Jerman menjadi `decoder_inputs` (misalnya, "sos Ich möchte") dan `decoder_labels` (misalnya, "Ich möchte ein").
- **Listing 11.11**: Fungsi `get_inference_model`. Kode ini mendemonstrasikan cara **membongkar model training yang sudah disimpan** (`model.get_layer(...)`) dan **merakit kembali lapisannya** menjadi arsitektur inferensi rekursif yang baru, sambil tetap menggunakan bobot (weights) yang sudah terlatih. Ini adalah teknik praktis yang sangat penting.