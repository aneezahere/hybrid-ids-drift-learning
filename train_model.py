"""
Training script for Hybrid IDS Model
"""

import pandas as pd
import numpy as np
from hybrid_ids import HybridIDSModel
from sklearn.model_selection import train_test_split
import os


def extract_label_from_filename(filename):
    """Extract attack type from filename"""
    if 'Benign' in filename:
        return 'Benign'
    elif 'ARP_Spoofing' in filename:
        return 'ARP_Spoofing'
    elif 'MQTT-DDoS' in filename:
        return 'MQTT-DDoS'
    elif 'MQTT-DoS' in filename:
        return 'MQTT-DoS'
    elif 'MQTT-Malformed' in filename:
        return 'MQTT-Malformed'
    elif 'TCP_IP-DDoS' in filename:
        return 'TCP_IP-DDoS'
    elif 'TCP_IP-DoS' in filename:
        return 'TCP_IP-DoS'
    elif 'Recon' in filename:
        return 'Reconnaissance'
    else:
        return 'Unknown'


def load_dataset(data_path):
    """Load and combine all attack datasets"""
    all_files = [
        'Benign_train.pcap.csv',
        'ARP_Spoofing_train.pcap.csv', 
        'MQTT-DDoS-Connect_Flood_train.pcap.csv',
        'MQTT-DoS-Connect_Flood_train.pcap.csv',
        'TCP_IP-DoS-ICMP1_train.pcap.csv',
        'TCP_IP-DDoS-SYN1_train.pcap.csv',
        'Recon-Port_Scan_train.pcap.csv',
        'MQTT-Malformed_Data_train.pcap.csv'
    ]
    
    combined_data = []
    combined_labels = []
    
    for file in all_files:
        file_path = os.path.join(data_path, file)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            
            # Sample for training efficiency
            sample_size = min(3000, len(df))
            df_sample = df.sample(n=sample_size, random_state=42)
            
            label = extract_label_from_filename(file)
            
            combined_data.append(df_sample)
            combined_labels.extend([label] * len(df_sample))
            
            print(f"Loaded {len(df_sample)} samples of {label}")
    
    X = pd.concat(combined_data, ignore_index=True)
    y = np.array(combined_labels)
    
    return X, y


def main():
    """Main training function"""
    print("Training Hybrid IDS Model")
    print("=" * 50)
    
    # Load dataset
    data_path = "data/"  # Adjust path as needed
    X, y = load_dataset(data_path)
    
    print(f"Dataset loaded: {X.shape}")
    print(f"Attack types: {np.unique(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = HybridIDSModel()
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Save model
    model.save_model("models/hybrid_ids_model.pkl")
    print("Model saved successfully")


if __name__ == "__main__":
    main()
