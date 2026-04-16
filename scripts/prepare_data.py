import pandas as pd
import os

def main():
    fake_path = "data/News _dataset/Fake.csv"
    true_path = "data/News _dataset/True.csv"
    output_path = "data/raw/news.csv"
    
    print("Loading datasets...")
    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        print(f"Error: Could not find datasets in expected paths '{fake_path}' or '{true_path}'")
        return

    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)
    
    print("Assigning labels (1=Fake, 0=Real)...")
    df_fake['label'] = 1  
    df_true['label'] = 0  
    
    print("Concatenating datasets...")
    df = pd.concat([df_fake, df_true], ignore_index=True)
    
    print("Shuffling combined dataset...")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Saving to {output_path}...")
    df.to_csv(output_path, index=False)
    
    print("Data preparation complete! Dataset shape:", df.shape)

if __name__ == "__main__":
    main()
