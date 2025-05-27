# Laporan Proyek Machine Learning - Zuhair Nashif Abdurrohim

## Project Overview

Rekomendasi buku adalah salah satu solusi yang banyak dicari oleh pengguna platform buku digital maupun fisik untuk menemukan buku yang relevan dengan preferensi mereka. Seiring berkembangnya teknologi, sistem rekomendasi berbasis machine learning menjadi alat yang efektif untuk menyarankan buku yang sesuai dengan selera pengguna. Proyek ini bertujuan untuk membuat sistem rekomendasi buku menggunakan dataset yang berisi informasi tentang berbagai buku beserta rating dari pengguna.

Pentingnya proyek ini terletak pada kemampuannya untuk meningkatkan pengalaman pengguna dalam menemukan buku yang sesuai dengan preferensi mereka, sekaligus meningkatkan penjualan dan kepuasan pelanggan di platform buku. Sistem rekomendasi yang akurat dapat membantu pengguna menghemat waktu dan usaha dalam memilih buku yang tepat.

## Business Understanding

### Problem Statements

1. Pengguna sering kesulitan menemukan buku yang sesuai dengan minat mereka dari koleksi yang luas.
2. Sistem rekomendasi yang ada saat ini kurang personal dan tidak dapat menyesuaikan preferensi pengguna secara tepat.

### Goals

1. Mengembangkan sistem rekomendasi buku yang dapat menyarankan buku berdasarkan preferensi pengguna.
2. Meningkatkan kepuasan pengguna dengan menyediakan rekomendasi yang relevan dan personal.

### Solution Approach

Untuk mencapai tujuan tersebut, dua pendekatan akan diterapkan dalam proyek ini:

1. **Content-Based Filtering**: Menggunakan fitur buku (seperti genre, penulis, atau deskripsi) untuk memberi rekomendasi buku yang serupa dengan buku yang telah dinilai tinggi oleh pengguna.
2. **Collaborative Filtering**: Menganalisis perilaku pengguna yang serupa dan memberikan rekomendasi berdasarkan buku yang disukai oleh pengguna lain dengan preferensi serupa.

## Data Understanding

Dataset yang digunakan adalah **Book Recommendation Dataset by arashanic** yang tersedia di [Link Sumber dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset). Dataset ini terdiri dari informasi tentang buku dan rating yang diberikan oleh pengguna. Dataset ini memiliki sekitar 10.000 buku dan lebih dari 50.000 rating dari pengguna.

### Visualisasi dan Exploratory Data Analysis (EDA)

- Mengetahui jumlah data <br> ![image](https://github.com/user-attachments/assets/fe6280ef-1cc8-45b9-a315-42f7402ab2a3)
- Mengetahui informasi data users <br> ![image](https://github.com/user-attachments/assets/1236c45d-c391-49a8-aac9-b761e381af78)
- Mengetahui informasi data books <br> ![image](https://github.com/user-attachments/assets/8fb87427-852a-4056-a00b-b261981060c8)
- Mengetahui informasi data rating <br> ![image](https://github.com/user-attachments/assets/35dda53f-c4e1-45b2-986a-d67e421fdc37)

Jadi dataset ini berisi
- Users.csv : 278858 baris dan 3 kolom
- Books.csv : 271360 baris dan 8 kolom
- Ratings.csv : 1149780 baris dan 3 kolom

#### Uraian Fitur
Users.csv
- User-ID : ID unik yang dianonimkan untuk setiap pengguna.
- Location : Lokasi pengguna dalam format "Kota, Provinsi/Negara Bagian, Negara".
- Age : Usia pengguna dalam tahun.

Books.csv
- ISBN : Nomor ISBN unik yang mengidentifikasi buku.
- Book-Title : Judul buku.
- Book-Author : Nama penulis buku.
- Year-Of-Publication : Tahun buku tersebut diterbitkan.
- Publisher : Nama penerbit buku.
- Image-URL-S : URL gambar sampul buku berukuran kecil.
- Image-URL-M : URL gambar sampul buku berukuran sedang.
- Image-URL-L : URL gambar sampul buku berukuran besar.

Ratings.csv
- User-ID : ID pengguna yang memberikan rating.
- ISBN : ISBN buku yang diberi rating.
- Book-Rating : Nilai rating yang diberikan pengguna terhadap buku, biasanya dalam skala 0–10.

#### Kondisi data. (missing value, duplikat, dan sebagainya)
- Missing Value
  - Terdapat 2 data Book-Author, 2 data Publisher hilang di data Books
  - Terdapat 110762 data Age hilng di data Users
  - Terdapat 0 data hilang di data ratings
- Duplicate Datae
  - Terdapat 0 data duplikat pada data books, users dan ratings


## Data Preparation dan Preprocessing
Tahapan data preparation dan preprocessing meliputi beberapa langkah berikut:

1. Menghapus Kolom Gambar dari Data Buku:

- Menghapus kolom Image-URL-S, Image-URL-M, dan Image-URL-L dari DataFrame books karena URL gambar tidak diperlukan untuk model rekomendasi yang akan dibangun.
2. Menggabungkan Data:
- Menggabungkan DataFrame ratings dengan DataFrame users berdasarkan User-ID untuk mendapatkan informasi pengguna terkait dengan setiap rating.
- Menggabungkan hasil penggabungan sebelumnya (merge_df) dengan DataFrame books berdasarkan ISBN untuk menambahkan detail buku (judul, penulis, penerbit, tahun terbit) ke setiap rating.
3. Menangani Missing Values Metadata Buku:
- Mengidentifikasi dan menghapus baris dalam DataFrame gabungan (merge_df) yang memiliki missing values pada kolom metadata buku (Book-Title, Book-Author, Year-Of-Publication, Publisher). Ini dilakukan karena rating tanpa informasi buku yang lengkap tidak dapat digunakan untuk rekomendasi berbasis konten.
4. Menghapus Kolom yang Kurang Relevan:
- Menghapus kolom 'Age', 'Location', 'Year-Of-Publication', dan 'Publisher' dari DataFrame merge_df. Keputusan ini dibuat berdasarkan asumsi bahwa informasi ini kurang penting untuk model rekomendasi yang dikembangkan dibandingkan dengan rating, user ID, dan detail buku utama.
5. Membuat Subset Data untuk Content-Based Filtering:
- Mengambil 10.000 baris pertama dari DataFrame merge_df yang sudah dibersihkan dan menyimpannya dalam variabel data. Subset ini digunakan secara spesifik untuk pengembangan model Content-Based Filtering guna mengurangi waktu komputasi.
6. Persiapan Data untuk Content-Based Filtering (TF-IDF):
- Menginisialisasi objek TfidfVectorizer.
- Melakukan proses fitting dan transforming pada kolom Book-Title dari subset data (data) untuk membuat matriks TF-IDF (tfidf_matrix). Matriks ini merepresentasikan setiap judul buku sebagai vektor numerik berdasarkan bobot Term Frequency-Inverse Document Frequency.
- Membuat DataFrame dari matriks TF-IDF (cosine_sim_df) dengan indeks dan kolom berdasarkan judul buku, yang akan digunakan untuk menghitung kesamaan antar buku.
7. Persiapan Data untuk Collaborative Filtering:
- Membuat salinan DataFrame ratings yang sudah dibersihkan dan menamakannya dc.
- Melakukan encoding pada User-ID dan ISBN yang unik menjadi indeks numerik untuk digunakan dalam model. Ini disimpan dalam dictionary user_encoded dan book_encoded, beserta dictionary decoding user_decode dan book_decode.
- Menambahkan kolom 'user' dan 'book' ke DataFrame dc yang berisi hasil encoding numerik.
- Mengonversi kolom Book-Rating menjadi tipe data float dan menormalisasikannya ke rentang 0-1.
- Mengacak urutan baris dalam DataFrame dc untuk memastikan distribusi data yang baik saat pembagian.
- Membagi data menjadi set training (x_train, y_train) dan validation set (x_val, y_val) untuk melatih dan mengevaluasi model Collaborative Filtering.

## Model Development

Dalam proyek ini, dua skema sistem rekomendasi dikembangkan: Content-Based Filtering dan Collaborative Filtering.

### 1. Content-Based Filtering

Model Content-Based Filtering merekomendasikan buku kepada pengguna berdasarkan kesamaan konten atau fitur dari buku-buku yang disukai pengguna di masa lalu. Dalam implementasi ini, kesamaan diukur berdasarkan judul buku.

*   **Konsep Dasar**: Jika pengguna menyukai suatu buku, model akan mencari buku lain yang memiliki fitur konten serupa (dalam hal ini, kata-kata dalam judul buku).
*   **Algoritma yang Digunakan**:
        *   **Implementasi dalam Kode**: Objek `TfidfVectorizer` diinisialisasi (`tf = TfidfVectorizer()`) dan kemudian diterapkan pada kolom `Book-Title` dari subset data (`data`) menggunakan `tf.fit_transform(data['Book-Title'])`. Hasilnya adalah matriks TF-IDF (`tfidf_matrix`).
    *   **Cosine Similarity**: Digunakan untuk mengukur tingkat kesamaan antara dua vektor TF-IDF (dalam hal ini, vektor judul buku). Nilai Cosine Similarity berkisar antara 0 (tidak ada kesamaan) hingga 1 (sangat mirip). Semakin tinggi nilai Cosine Similarity antara dua judul buku, semakin mirip kedua buku tersebut dianggap.
        *   **Implementasi dalam Kode**: Fungsi `cosine_similarity()` dari `sklearn.metrics.pairwise` digunakan untuk menghitung matriks kesamaan antara semua pasangan vektor dalam `tfidf_matrix` (`cosine_sim = cosine_similarity(tfidf_matrix)`). Matriks ini kemudian diubah menjadi DataFrame (`cosine_sim_df`) dengan indeks dan kolom berupa judul buku.
*   **Cara Kerja Rekomendasi**:
    *   Fungsi `book_recommendations` mengambil judul buku input.
    *   Mencari skor kesamaan buku input dengan semua buku lain menggunakan `cosine_sim_df`.
    *   Mengurutkan buku berdasarkan skor kesamaan secara menurun.
    *   Mengambil `k` buku teratas yang paling mirip (mengecualikan buku input itu sendiri).
    *   Mengembalikan informasi (ISBN, Judul, Penulis) dari buku-buku yang direkomendasikan.
*   **Kelebihan Content-Based Filtering**:
    *   Tidak memerlukan data tentang pengguna lain; rekomendasi hanya didasarkan pada preferensi pengguna target dan fitur item.
    *   Dapat merekomendasikan item baru yang belum pernah dinilai oleh siapa pun (masalah *cold-start* pada item).
    *   Mampu merekomendasikan item kepada pengguna baru asalkan ada data preferensi minimal (masalah *cold-start* pada pengguna dapat dikurangi jika ada interaksi awal).
    *   Rekomendasi mudah dijelaskan berdasarkan fitur item.
*   **Kekurangan Content-Based Filtering**:
    *   Kualitas rekomendasi sangat bergantung pada kekayaan dan struktur fitur item.
    *   Cenderung merekomendasikan item yang sangat mirip dengan yang sudah disukai pengguna, membatasi penemuan item baru yang berbeda (*serendipity* rendah).
    *   Membutuhkan analisis konten yang mendalam untuk setiap item.

### 2. Collaborative Filtering

Model Collaborative Filtering merekomendasikan buku kepada pengguna berdasarkan perilaku (rating) dari pengguna lain yang memiliki selera serupa.

*   **Konsep Dasar**: Pengguna yang memiliki pola rating yang mirip cenderung memiliki selera yang sama. Model ini mencari pengguna yang mirip dengan pengguna target dan merekomendasikan buku yang disukai oleh pengguna serupa tersebut, tetapi belum pernah dibaca atau dinilai oleh pengguna target.
*   **Algoritma yang Digunakan**:
    *   **Neural Network dengan Embedding Layers**: Model ini menggunakan arsitektur Neural Network sederhana dengan embedding layer untuk merepresentasikan pengguna dan buku dalam ruang vektor dimensi rendah.
        *   **Implementasi dalam Kode**:
            *   **Embedding Layers**: Dua `Embedding` layer dibuat, satu untuk pengguna (`user_embedding`) dan satu untuk buku (`book_embedding`), dengan dimensi embedding sebesar 32 (`embedding_dim = 32`). Embedding layer mengubah input user ID dan book ID yang sudah di-encode menjadi vektor padat (dense vector) yang menangkap karakteristik pengguna dan buku.
            *   **Flatten Layers**: Output dari embedding layer di-flatten menjadi vektor satu dimensi (`user_vec`, `book_vec`).
            *   **Dot Product Layer**: Sebuah layer `dot` digunakan untuk menghitung produk titik (dot product) antara vektor pengguna dan vektor buku. Produk titik ini merepresentasikan skor prediksi rating atau tingkat kecocokan antara pengguna dan buku.
            *   **Model Keras**: Model didefinisikan menggunakan Keras Functional API, mengambil input user dan book, dan menghasilkan skor produk titik. Model dikompilasi dengan optimizer 'adam' dan loss function 'mean_squared_error' (MSE), yang umum digunakan untuk tugas prediksi rating.
*   **Cara Kerja Rekomendasi**:
    *   Fungsi `recommend_books` mengambil `user_id` target.
    *   Mengidentifikasi buku-buku yang belum dinilai oleh pengguna target.
    *   Menggunakan model Neural Network yang sudah dilatih untuk memprediksi rating yang mungkin diberikan pengguna target untuk buku-buku yang belum dinilai tersebut.
    *   Mengurutkan buku berdasarkan prediksi rating tertinggi.
    *   Mengambil `k` buku dengan prediksi rating tertinggi sebagai rekomendasi.
    *   Mengembalikan informasi (ISBN, Judul, Penulis) dari buku-buku yang direkomendasikan.
*   **Kelebihan Collaborative Filtering**:
    *   Mampu merekomendasikan item yang benar-benar baru atau berbeda dari yang sebelumnya disukai pengguna (*serendipity* tinggi).
    *   Tidak memerlukan analisis konten item secara mendalam.
    *   Mampu menangani item dengan struktur data yang minimal, asalkan ada data interaksi pengguna.
*   **Kekurangan Collaborative Filtering**:
    *   Mengalami masalah *cold-start* baik untuk pengguna baru (tidak ada data interaksi untuk dicocokkan) maupun item baru (belum ada interaksi rating).
    *   Masalah *sparsity* data: Sulit menemukan pengguna yang mirip atau membuat prediksi akurat jika data interaksi pengguna-item sangat jarang.
    *   Rekomendasi mungkin bias terhadap item populer.
    *   Sulit memberikan penjelasan yang intuitif mengapa suatu item direkomendasikan.

Kedua model ini menawarkan pendekatan yang berbeda untuk rekomendasi buku, masing-masing dengan kekuatan dan kelemahannya sendiri, dan dapat digunakan secara independen atau digabungkan (Hybrid Recommendation System) untuk meningkatkan kualitas rekomendasi dan mengatasi keterbatasan masing-masing.
## Uji
- Content Based <br> ![image](https://github.com/user-attachments/assets/f1ed207a-bc8f-44ed-aad1-d9f5987951a9)

- Collaborative Filtering <br> ![image](https://github.com/user-attachments/assets/71383918-9f7b-4040-b98c-7b87078b797e)



## Evaluation

Berikut adalah rangkuman evaluasi dari dua pendekatan sistem rekomendasi yang dikembangkan, yaitu Content-Based Filtering dan Collaborative Filtering, serta keterkaitannya dengan pemahaman bisnis (Business Understanding) yang telah ditetapkan sebelumnya.

### Content-Based Filtering

* **Matriks Evaluasi yang Digunakan:**

  * **Cosine Similarity:** Digunakan untuk mengukur kesamaan konten antara buku input ('Flesh Tones: A Novel') dengan buku-buku lainnya.
  * **Precision\@10:** 0.9000
  * **Recall\@10:** 1.0000

* **Hasil Evaluasi:**

  * Sistem berhasil merekomendasikan 9 dari 10 buku yang relevan berdasarkan konten.
  * Precision yang tinggi menunjukkan bahwa sebagian besar buku yang direkomendasikan memang relevan.
  * Recall sempurna menunjukkan bahwa semua buku relevan berhasil ditemukan dalam 10 rekomendasi teratas.
  * Rekomendasi ditampilkan berdasarkan kemiripan konten dengan buku input, seperti:

    * Jonathan Kellerman : *Flesh and Blood*
    * Michael Cunningham : *The Hours: A Novel*
    * Clive Barker : *IN THE FLESH*

* **Keterkaitan dengan Business Understanding:**

  * **Menjawab Problem Statement 1:** Membantu pengguna menemukan buku yang mirip dengan buku yang mereka sukai.
  * **Mencapai Goal 1 & 2:** Menyediakan rekomendasi yang relevan dan meningkatkan pengalaman pengguna.
  * **Solusi yang Diberikan Berdampak:** Precision dan recall yang tinggi menunjukkan bahwa solusi ini berhasil dalam memenuhi ekspektasi dan kebutuhan pengguna terkait relevansi konten.

### Collaborative Filtering

* **Matriks Evaluasi yang Digunakan:**

  * **Training Loss (MSE):** 0.0349
  * **Validation Loss (MSE):** 0.2210
  * **Training RMSE:** 0.1867
  * **Validation RMSE:** 0.4701

* **Hasil Evaluasi:**

  * Nilai error yang rendah pada data pelatihan menunjukkan model belajar dengan baik dari data tersebut.
  * Perbedaan signifikan antara error training dan validasi mengindikasikan adanya overfitting.
  * Model tetap mampu memberikan rekomendasi yang sesuai untuk pengguna baru, contohnya:

    * Sam Siciliano : *Darkness*
    * Emma McLaughlin : *The Nanny Diaries: A Novel*
    * William Gibson : *Mona Lisa Overdrive*

* **Keterkaitan dengan Business Understanding:**

  * **Menjawab Problem Statement 2:** Mengatasi keterbatasan sistem rekomendasi yang tidak personal dengan memanfaatkan data perilaku pengguna.
  * **Mencapai Goal 1 & 2:** Memberikan rekomendasi berdasarkan preferensi kolektif pengguna serupa, menjadikan pengalaman lebih personal.
  * **Solusi yang Diberikan Berdampak:** Walaupun ada indikasi overfitting, model tetap dapat memberikan rekomendasi yang relevan, mendekati kebutuhan pengguna.

## Kesimpulan

Kedua pendekatan memberikan kontribusi terhadap pemecahan masalah bisnis:

* **Content-Based Filtering** unggul dalam menemukan buku-buku serupa dengan minat pengguna berdasarkan konten buku.
* **Collaborative Filtering** menawarkan pendekatan yang lebih personal berdasarkan perilaku pengguna lain.

Dengan hasil evaluasi ini, dapat disimpulkan bahwa:

* Sistem rekomendasi telah berhasil menjawab semua problem statement yang diajukan.
* Goals bisnis tercapai melalui akurasi rekomendasi yang baik dan pengalaman pengguna yang ditingkatkan.
* Kedua solusi terbukti berdampak positif dan dapat dikembangkan lebih lanjut, terutama untuk meningkatkan generalisasi model collaborative filtering agar tidak overfitting.

---

![image](https://github.com/user-attachments/assets/a0893a07-5f80-4733-8c8e-b3902397826c)



