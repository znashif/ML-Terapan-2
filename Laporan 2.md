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

Variabel-variabel pada dataset ini adalah sebagai berikut:
- **book_id**: ID unik untuk setiap buku.
- **title**: Judul buku.
- **author**: Penulis buku.
- **genre**: Genre buku.
- **user_id**: ID unik untuk setiap pengguna.
- **rating**: Rating yang diberikan oleh pengguna pada buku.

### Visualisasi dan Exploratory Data Analysis (EDA)

Beberapa tahapan visualisasi dan analisis dilakukan untuk memahami distribusi rating, genre buku yang paling sering dinilai, dan hubungan antara pengguna dengan buku yang telah mereka beri rating.

## Data Preparation

Tahapan data preparation meliputi beberapa langkah berikut:
1. **Penghapusan nilai yang hilang**: Menghapus baris yang memiliki data kosong atau tidak lengkap, terutama pada kolom yang diperlukan untuk model rekomendasi seperti `rating`, `book_id`, dan `user_id`.
2. **Pengkodean kategori**: Mengubah genre buku yang berbentuk teks menjadi format numerik agar bisa diproses oleh model machine learning.
3. **Normalisasi rating**: Mengubah rating yang diberikan pengguna menjadi rentang yang lebih seragam agar lebih mudah dianalisis oleh model.

Alasan untuk tahapan ini adalah untuk memastikan kualitas data yang baik sebelum diterapkan pada model, serta agar model dapat berfungsi dengan optimal.

## Modeling

Dua model rekomendasi buku akan digunakan dalam proyek ini:

1. **Content-Based Filtering**: Model ini menggunakan fitur buku seperti genre, penulis, dan deskripsi untuk memberikan rekomendasi. Model ini bekerja dengan mencari buku yang mirip berdasarkan fitur-fitur tersebut.
   
   - **Kelebihan**: Mudah dipahami dan dapat memberikan rekomendasi berdasarkan item yang benar-benar mirip dengan preferensi pengguna.
   - **Kekurangan**: Terbatas pada buku dengan informasi yang lengkap dan tidak bisa memberikan rekomendasi yang sangat berbeda dari buku yang telah dinilai tinggi.

2. **Collaborative Filtering**: Model ini berbasis pada perilaku pengguna, seperti buku yang disukai oleh pengguna dengan preferensi yang mirip. Pengguna yang memiliki rating serupa akan diberi rekomendasi buku berdasarkan buku yang disukai oleh pengguna lain.
   
   - **Kelebihan**: Dapat memberikan rekomendasi yang lebih personal berdasarkan pola perilaku pengguna lain.
   - **Kekurangan**: Memerlukan data yang lebih banyak untuk menghasilkan rekomendasi yang akurat dan dapat mengalami kesulitan dengan pengguna baru (cold start problem).

## Evaluation

Metrik evaluasi yang digunakan untuk menilai performa sistem rekomendasi adalah **Root Mean Square Error (RMSE)** dan **Precision at k (P@k)**.

1. **RMSE**: Mengukur seberapa besar perbedaan antara rating yang diprediksi dan rating aktual. Formula RMSE adalah:
   \[
   RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
   \]
   dimana \( y_i \) adalah rating aktual dan \( \hat{y}_i \) adalah rating yang diprediksi.

2. **Precision at k (P@k)**: Mengukur seberapa banyak rekomendasi yang relevan berada di antara top-k rekomendasi yang diberikan oleh model.

Hasil evaluasi akan menunjukkan sejauh mana model dapat memberikan rekomendasi yang relevan dan tepat waktu kepada pengguna.
