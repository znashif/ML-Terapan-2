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

Dataset yang digunakan adalah **Book Recommendation Dataset by arashanic** yang tersedia di [Kaggle](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset). Dataset ini terdiri dari informasi tentang buku dan rating yang diberikan oleh pengguna. Dataset ini memiliki sekitar 10.000 buku dan lebih dari 50.000 rating dari pengguna.

### Visualisasi dan Exploratory Data Analysis (EDA)

- Mengetahui jumlah data <br> ![image](https://github.com/user-attachments/assets/fe6280ef-1cc8-45b9-a315-42f7402ab2a3)
- Mengetahui informasi data users <br> ![image](https://github.com/user-attachments/assets/1236c45d-c391-49a8-aac9-b761e381af78)
- Mengetahui informasi data books <br> ![image](https://github.com/user-attachments/assets/8fb87427-852a-4056-a00b-b261981060c8)
- Mengetahui informasi data rating <br> ![image](https://github.com/user-attachments/assets/35dda53f-c4e1-45b2-986a-d67e421fdc37)



## Data Preparation

Tahapan data preparation meliputi beberapa langkah berikut:
1. **Penghapusan nilai yang hilang**: Menghapus baris yang memiliki data kosong atau tidak lengkap, terutama pada kolom yang diperlukan untuk model rekomendasi seperti `rating`, `book_id`, dan `user_id`.
2. **Melakukan drop kolom**: Melakukan drop pada data tidak perlu seperti 'Image-URL-S', 'Image-URL-M', 'Image-URL-L', 'Age', 'Location', 'Year-Of-Publication', 'Publisher'
3. ""Melakukan penggabungan data**: Lakukan penggabungan data dengan cara bertahap. Gabungkan ratings dengan users lalu gabungkan merge_df dengan books
```
merge_df = pd.merge(ratings, users, on='User-ID', how='left')
merge_df = pd.merge(merge_df, books, on='ISBN', how='left')
```

## Modeling

Dua model rekomendasi buku akan digunakan dalam proyek ini:

1. **Content-Based Filtering**: Model ini menggunakan fitur penulis untuk memberikan rekomendasi. Model ini bekerja dengan mencari buku yang mirip berdasarkan fitur-fitur tersebut.
   - **Tahapan**: Tahapan Singkat Content-Based Filtering:
      - Pemilihan Data: Mengambil 10.000 baris pertama dari data gabungan (merge_df) sebagai subset data (data) untuk model content-based.
      - Representasi Fitur (Vectorization): Mengubah judul buku (Book-Title) menjadi vektor numerik menggunakan TF-IDF Vectorizer.
      - Perhitungan Kesamaan: Mengukur kesamaan antar buku berdasarkan vektor TF-IDF menggunakan Cosine Similarity.
      - Pembuatan Rekomendasi: Membuat fungsi (book_recommendations) untuk merekomendasikan buku berdasarkan kesamaan penulis, meskipun implementasi saat ini menggunakan kesamaan judul melalui TF-IDF.

   - **Parameter Utama yang Digunakan**:
      - TfidfVectorizer():
        Tidak ada parameter spesifik yang dimodifikasi dari nilai default saat inisialisasi (tf = TfidfVectorizer()). Ini berarti menggunakan default seperti lowercase=True, stop_words=None, ngram_range=(1, 1), dll.
Data yang digunakan untuk .fit() dan .fit_transform() adalah kolom Book-Title dari DataFrame data (yang merupakan 10.000 baris pertama).
      - cosine_similarity():
        Inputnya adalah matriks TF-IDF yang dihasilkan dari tfidf_matrix = tf.fit_transform(data['Book-Title']).
      - book_recommendations() function:
         - book_author: Nama penulis sebagai input untuk rekomendasi.
         - similarity_data: Matriks cosine similarity (cosine_sim_df) yang digunakan untuk mencari buku serupa.
         - items: DataFrame yang berisi informasi buku (data[['Book-Title', 'Book-Author']]) untuk ditampilkan dalam rekomendasi.
         - k: Jumlah rekomendasi yang diinginkan (default 10).
        
        Secara ringkas, model content-based Anda menggunakan TF-IDF pada judul buku untuk menghitung kesamaan kosinus, dan kemudian merekomendasikan buku berdasarkan kesamaan tersebut, dengan input awal berupa nama penulis (meskipun kesamaan dihitung berdasarkan judul).
   - **Kelebihan**: Mudah dipahami dan dapat memberikan rekomendasi berdasarkan item yang benar-benar mirip dengan preferensi pengguna.
   - **Kekurangan**: Terbatas pada buku dengan informasi yang lengkap dan tidak bisa memberikan rekomendasi yang sangat berbeda dari buku yang telah dinilai tinggi.

2. **Collaborative Filtering**: Model ini berbasis pada perilaku pengguna, seperti buku yang disukai oleh pengguna dengan preferensi yang mirip. Pengguna yang memiliki rating serupa akan diberi rekomendasi buku berdasarkan buku yang disukai oleh pengguna lain.

   - **Tahapan**:
      - Penggunaan Data Rating: Menggunakan seluruh data rating (ratings) sebagai dasar interaksi pengguna dan buku.
      - Encoding ID: Mengubah User-ID dan ISBN menjadi indeks numerik (user dan book) untuk memudahkan penggunaan dalam model neural network.
      - Pembagian Data: Membagi data yang sudah di-encode menjadi training set (90%) dan validation set (10%).
      - Normalisasi Rating: Menormalisasi nilai rating ke rentang 0-1.
      - Pembuatan Model (Matrix Factorization with Embeddings): Membangun model neural network dengan layer embedding terpisah untuk pengguna dan buku, lalu menggabungkan representasi (embedding vectors) keduanya menggunakan operasi dot product untuk memprediksi rating.
      - Pelatihan Model: Melatih model menggunakan data training dan memvalidasinya pada data validation.
      - Pembuatan Rekomendasi: Membuat fungsi (recommend_books) untuk memprediksi rating pengguna target untuk buku-buku yang belum dirating, dan merekomendasikan buku dengan prediksi rating tertinggi.
   - **Parameter**:
      - Data Input: DataFrame dc (yang merupakan DataFrame ratings).
      - Encoding: Mapping user_encoded dan book_encoded dibuat berdasarkan unique User-ID dan ISBN dalam dc.
      - Pembagian Data: train_indices ditentukan sebagai 90% dari total jumlah baris data (0.9 * dc.shape[0]).
      - Normalisasi Rating: Dilakukan scaling linear berdasarkan min_rate dan max_rate dari kolom Book-Rating di dc.
      - Arsitektur Model (tf.keras.Model):
         - embedding_dim: Dimensi embedding vector untuk pengguna dan buku, diset ke 32.
         - Layer tf.keras.layers.Embedding: Menggunakan num_users dan num_books (jumlah unique user/book) sebagai input_dim, dan embedding_dim sebagai output_dim.
         - Layer tf.keras.layers.dot: Menggunakan axes=1 untuk melakukan dot product antara vektor pengguna dan buku.
      - Kompilasi Model (model.compile):
         - optimizer: 'adam'
         - loss: 'mean_squared_error'
         - metrics: Anda menambahkan [tf.keras.metrics.RootMeanSquaredError()] di modifikasi terakhir yang saya ajukan (meskipun Anda menolaknya, ini adalah metrik yang relevan untuk dibahas jika Anda menggunakannya). *[Note: Based on the current notebook state after user rejections, RMSE metric is NOT added to compilation. I will stick to the code state after rejections when describing parameters.]*
      - Pelatihan Model (model.fit):
         - epochs: Diset ke 5.
         - validation_data: Menggunakan ([x_val[:, 0], x_val[:, 1]], y_val).
         - verbose: Diset ke 1 untuk menampilkan progress pelatihan.
         - Batch size menggunakan nilai default Keras (biasanya 32).
      - recommend_books() function:
         - user_id: ID pengguna untuk rekomendasi.
         - dc_df: DataFrame rating (dc) untuk mengetahui buku yang sudah dirating pengguna.
         - books_df: DataFrame buku (books) untuk mendapatkan detail buku.
         - k: Jumlah rekomendasi yang diinginkan (default 10).
   
   Dengan parameter-parameter ini, model Collaborative Filtering Anda belajar merepresentasikan pengguna dan buku dalam ruang embedding 32 dimensi untuk memprediksi rating yang mungkin diberikan pengguna pada buku yang belum mereka baca.
   - **Kelebihan**: Dapat memberikan rekomendasi yang lebih personal berdasarkan pola perilaku pengguna lain.
   - **Kekurangan**: Memerlukan data yang lebih banyak untuk menghasilkan rekomendasi yang akurat dan dapat mengalami kesulitan dengan pengguna baru (cold start problem).

## Uji
- Content Based <br> ![image](https://github.com/user-attachments/assets/a1502cc0-31dd-4277-800a-9c47930d04fb)
- Collaborative Filtering <br> ![image](https://github.com/user-attachments/assets/2a584cb0-79d5-43a0-8a72-e94d779784d8)


## Evaluation

Berikut adalah rangkuman singkat mengenai evaluasi kedua model rekomendasi:

### Content-Based Filtering
- Matriks yang Digunakan:
   - Cosine Similarity: Digunakan untuk mengukur kesamaan antara buku input ('The Lovely Bones') dengan buku-buku lain dalam dataset terbatas.
- Hasil:
   - Rekomendasi buku ditampilkan berdasarkan kesamaan konten (judul) dengan 'The Lovely Bones'.
   - Nilai Cosine Similarity untuk beberapa buku lain terhadap 'The Lovely Bones' berkisar antara sekitar 0.27 hingga 0.52. Buku dengan nilai similarity lebih tinggi dianggap lebih relevan secara konten. Contohnya, 'The Hours: A Novel' memiliki similarity tertinggi (sekitar 0.52).

### Collaborative Filtering
- Matriks yang Digunakan:
   - Training Loss (MSE): Mean Squared Error pada data pelatihan.
   - Validation Loss (MSE): Mean Squared Error pada data validasi.
   - Training RMSE: Root Mean Squared Error pada data pelatihan.
   - Validation RMSE: Root Mean Squared Error pada data validasi.
- Hasil:
   - Training Loss (MSE): 0.0349 (pada akhir epoch). Ini menunjukkan error rata-rata yang sangat rendah pada data yang digunakan untuk melatih model.
   - Validation Loss (MSE): 0.2215 (pada akhir epoch). Error pada data validasi jauh lebih tinggi dibandingkan data pelatihan, yang bisa mengindikasikan model mengalami overfitting.
   - Training RMSE: 0.1868 (pada akhir epoch). Ini adalah akar dari Training MSE.
   - Validation RMSE: 0.4706 (pada akhir epoch). Ini adalah akar dari Validation MSE. Nilai ini menunjukkan rata-rata selisih antara rating prediksi dan rating aktual pada data validasi adalah sekitar 0.47 (dalam skala rating yang sudah dinormalisasi 0-1).
   - Contoh rekomendasi buku untuk User ID 201768 juga ditampilkan, menunjukkan daftar buku yang diprediksi memiliki rating tinggi oleh model untuk pengguna tersebut.
 
<br>![image](https://github.com/user-attachments/assets/e2faf018-a863-41cf-8b48-53c2297a3f06)

Secara ringkas, Content-Based Filtering dievaluasi berdasarkan kesamaan konten menggunakan Cosine Similarity, sementara Collaborative Filtering dievaluasi berdasarkan akurasi prediksi rating menggunakan MSE dan RMSE. Hasil menunjukkan model CF memiliki error prediksi yang lebih tinggi pada data validasi dibandingkan data pelatihan.
