print("🔥 NEW CHECK_MODEL FILE RUNNING ✅")

import os
import joblib
import pandas as pd
import numpy as np


def main():
    print("\n🔍 Model Health Check Started...\n")

   
    # Paths
   
    data_path = os.path.join("data", "eda_clean.csv")
    model_path = os.path.join("outputs", "best_model.pkl")

    print(f"📌 Dataset path: {os.path.abspath(data_path)}")
    print(f"📌 Model path:   {os.path.abspath(model_path)}\n")

    if not os.path.exists(data_path):
        print(f"❌ Dataset not found: {data_path}")
        print("👉 Fix: Make sure 'eda_clean.csv' is inside data/ folder.\n")
        return

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("👉 Fix: Make sure 'best_model.pkl' is inside outputs/ folder.\n")
        return

    print(f"✅ Dataset found: {data_path}")
    print(f"✅ Model found:   {model_path}\n")

    
    # Load dataset
   
    print("📦 Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"✅ Dataset loaded | Shape: {df.shape}")

    
    # Normalize column names
   
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    print("✅ Column names normalized (lowercase + underscores)")

    
    # Load model
   
    print("\n🧠 Loading model...")
    model = joblib.load(model_path)
    print("✅ Model loaded successfully!")
    print("Model type:", type(model))

   
    # Create text_combined if missing
   
    if "text_combined" not in df.columns:
        print("\n⚠️ text_combined missing — creating it automatically...")

        subject_col = None
        desc_col = None

        for c in df.columns:
            if "subject" in c:
                subject_col = c
            if "description" in c:
                desc_col = c

        if subject_col and desc_col:
            df["text_combined"] = (
                df[subject_col].astype(str).fillna("") + " " +
                df[desc_col].astype(str).fillna("")
            ).str.lower()

            print(f"✅ text_combined created using: {subject_col} + {desc_col}")
        else:
            df["text_combined"] = ""
            print("⚠️ subject/description columns not found → empty text_combined created.")

   
    # Required columns used in model training
   
    required_cols = [
        "ticket_type",
        "priority",
        "channel",
        "ticket_status",
        "customer_gender",
        "product_purchased",
        "customer_age",
        "first_response_time",
        "time_to_resolution",
        "text_combined"
    ]

    # Add missing required columns
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan
            print(f"⚠️ Missing column added: {col}")

   
    # Ensure numeric columns are numeric
    
    numeric_cols = ["customer_age", "first_response_time", "time_to_resolution"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print("✅ Numeric columns converted:", numeric_cols)


    # Sample prediction test
   
    print("\n⚡ Running sample prediction test...")
    sample = df[required_cols].sample(5, random_state=42)

    preds = model.predict(sample)
    print("✅ Prediction successful!")
    print("🔮 Sample predictions:", preds)

    # Pipeline steps
    if hasattr(model, "named_steps"):
        print("\n🧩 Pipeline steps:")
        for step in model.named_steps:
            print(" -", step)

    print("\n🎉 Model Health Check Completed Successfully!\n")


if __name__ == "__main__":
    main()
