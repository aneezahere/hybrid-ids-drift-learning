"""
Hybrid IDS with Concept Drift Learning
Main implementation of the adaptive intrusion detection system
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb
import joblib


class SensitiveDriftDetector:
    """Sensitivity-based drift detector using accuracy monitoring"""
    
    def __init__(self, threshold=0.05, window_size=10):
        self.threshold = threshold
        self.window_size = window_size
        self.accuracy_window = []
        self.baseline_accuracy = None
        
    def add_element(self, accuracy):
        """Add new accuracy and check for drift"""
        self.accuracy_window.append(accuracy)
        
        if len(self.accuracy_window) > self.window_size:
            self.accuracy_window.pop(0)
            
        if self.baseline_accuracy is None and len(self.accuracy_window) >= 5:
            self.baseline_accuracy = np.mean(self.accuracy_window[:5])
            
        if len(self.accuracy_window) >= self.window_size and self.baseline_accuracy is not None:
            current_avg = np.mean(self.accuracy_window)
            accuracy_drop = self.baseline_accuracy - current_avg
            
            if accuracy_drop > self.threshold:
                print(f"Drift detected: accuracy dropped from {self.baseline_accuracy:.3f} to {current_avg:.3f}")
                self.baseline_accuracy = current_avg
                return True
                
        return False


class HybridIDSModel:
    """Hybrid Intrusion Detection System with adaptive drift learning"""
    
    def __init__(self, drift_threshold=0.05, buffer_size=1500):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        self.drift_detector = SensitiveDriftDetector(threshold=drift_threshold)
        self.retraining_buffer = {'X': [], 'y': []}
        self.buffer_size = buffer_size
        self.is_trained = False
        
    def fit(self, X, y):
        """Train the initial model"""
        print("Training Hybrid IDS Model...")
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        print("Training completed")
        
    def predict(self, X):
        """Make predictions without adaptation"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
        
    def predict_and_adapt(self, X_batch, y_batch_true):
        """Make predictions and adapt to concept drift"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        predictions = self.predict(X_batch)
        accuracy = accuracy_score(y_batch_true, predictions)
        
        self._update_buffer(X_batch, y_batch_true)
        
        drift_detected = self.drift_detector.add_element(accuracy)
        
        if drift_detected and len(self.retraining_buffer['X']) >= 500:
            self._retrain_model()
            
        return predictions, accuracy, drift_detected
        
    def _update_buffer(self, X_batch, y_batch):
        """Update the retraining buffer with new samples"""
        X_array = X_batch.values if hasattr(X_batch, 'values') else np.array(X_batch)
        
        for i in range(len(X_array)):
            self.retraining_buffer['X'].append(X_array[i])
            self.retraining_buffer['y'].append(y_batch[i])
        
        if len(self.retraining_buffer['X']) > self.buffer_size:
            excess = len(self.retraining_buffer['X']) - self.buffer_size
            self.retraining_buffer['X'] = self.retraining_buffer['X'][excess:]
            self.retraining_buffer['y'] = self.retraining_buffer['y'][excess:]
            
    def _retrain_model(self):
        """Retrain the model with recent samples"""
        print(f"Retraining with {len(self.retraining_buffer['X'])} samples...")
        
        X_retrain = np.array(self.retraining_buffer['X'])
        y_retrain = np.array(self.retraining_buffer['y'])
        
        X_retrain_scaled = self.scaler.fit_transform(X_retrain)
        self.model.fit(X_retrain_scaled, y_retrain)
        
        print("Retraining completed")
        
    def save_model(self, filepath):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
            
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'drift_detector': self.drift_detector,
            'buffer_size': self.buffer_size
        }
        joblib.dump(model_data, filepath)
        
    @classmethod
    def load_model(cls, filepath):
        """Load a saved model"""
        model_data = joblib.load(filepath)
        
        instance = cls(buffer_size=model_data['buffer_size'])
        instance.model = model_data['model']
        instance.scaler = model_data['scaler'] 
        instance.drift_detector = model_data['drift_detector']
        instance.is_trained = True
        
        return instance
