import pandas as pd
import argparse
import os
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    """Fungsi untuk memuat dataset raw."""
    print(f"Memuat data dari: {file_path}")
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Fungsi untuk membersihkan dan memproses data."""
    print("Memulai preprocessing data...")

    df = df.drop_duplicates()

    df = df[(df['person_age'] <= 100) & (df['person_emp_length'] <= 50)]

    df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
    df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())

    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        
    print("Preprocessing selesai!")
    return df

def save_data(df, output_path):
    """Fungsi untuk menyimpan data yang sudah bersih."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data bersih berhasil disimpan ke: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script untuk otomatisasi data preprocessing")
    parser.add_argument("--input", type=str, required=True, help="Path ke file raw data (CSV)")
    parser.add_argument("--output", type=str, required=True, help="Path untuk menyimpan clean data (CSV)")
    args = parser.parse_args()

    raw_df = load_data(args.input)
    clean_df = preprocess_data(raw_df)
    save_data(clean_df, args.output)