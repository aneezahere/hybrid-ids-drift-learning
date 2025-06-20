# Hybrid IDS with Concept Drift Learning

## Abstract

This repository implements a novel hybrid intrusion detection system that addresses the critical challenge of concept drift in cybersecurity environments. By combining ensemble learning with adaptive drift detection, our approach achieves 91.8% accuracy while maintaining real-time adaptation capabilities for evolving attack patterns.

## Problem Statement

Traditional intrusion detection systems suffer from:
- **Concept drift** - attack patterns evolve over time
- **Static models** - unable to adapt to new threats
- **Poor performance** on IoT specific attacks
- **Limited real-time adaptation** capabilities

## Our Solution

### Hybrid Architecture
We propose a multi-component system that combines:

1. **Ensemble Learning**: XGBoost + LightGBM + Random Forest for robust classification
2. **Drift Detection**: Sensitivity based accuracy monitoring to detect performance degradation
3. **Adaptive Retraining**: Dynamic model updates using recent samples when drift is detected
4. **Real time Processing**: Streaming data handling with batch-wise adaptation

### Key Innovations

- **Sensitivity-based drift detector** that monitors accuracy drops rather than statistical changes
- **Weighted ensemble approach** that adapts model contributions based on recent performance
- **Incremental learning strategy** that maintains knowledge while adapting to new patterns
- **IoT-aware feature engineering** for modern network security threats

## Methodology

### Dataset
- **Source**: New Brunswick University IoT Security Dataset
- **Size**: 787K+ network traffic samples
- **Classes**: 8 attack types (Benign, ARP Spoofing, MQTT DDoS/DoS, TCP/IP DDoS/DoS, Reconnaissance, MQTT Malformed)
- **Features**: 45 network flow characteristics

### Experimental Setup
1. **Baseline Comparison**: Random Forest + Isolation Forest (62% accuracy)
2. **Static Models**: Individual and ensemble approaches without adaptation
3. **Drift Simulation**: Introduced artificial concept drift at regular intervals
4. **Performance Metrics**: Accuracy, F1 score, drift detection rate, adaptation time

### Algorithm Workflow
Initial Training: Train ensemble on balanced multi-class dataset
Streaming Prediction: Process incoming batches
Performance Monitoring: Track accuracy using sliding window
Drift Detection: Compare current vs baseline performance
Adaptive Retraining: Update models when drift threshold exceeded
Model Update: Retrain with recent samples while preserving core knowledge

## Results

### Performance Improvements
- **91.8% accuracy** vs 62% baseline (+48% improvement)
- **Real-time drift detection** with average detection delay of 2 3 batches
- **Successful adaptation** across all 8 attack categories
- **Maintained performance** during concept drift scenarios

### Comparative Analysis
| Approach | Accuracy | Drift Handling | Real-time |
|----------|----------|----------------|-----------|
| Baseline (RF+IF) | 62.0% | None | No |
| Static Ensemble | 99.4% | None | No |
| **Our Hybrid** | **91.8%** | **Yes** | **Yes** |

## Technical Contributions

1. **Novel drift detection method** using accuracy-based sensitivity monitoring
2. **Hybrid ensemble architecture** optimized for cybersecurity applications
3. **Adaptive retraining strategy** that balances stability and plasticity
4. **Comprehensive evaluation** on IoT and traditional network attacks

## Implementation Details

### Core Components
- `HybridIDSModel`: Main ensemble classifier with drift adaptation
- `SensitiveDriftDetector`: Accuracy based drift detection algorithm
- `AdaptiveRetraining`: Incremental learning with sample buffer management

### Key Features
- Configurable drift sensitivity thresholds
- Dynamic model weight adjustment
- Memory-efficient sample buffering
- Real-time prediction with adaptation

Future Work
Short term Enhancements

Multi-objective optimization for accuracy-speed trade-offs
Advanced drift detectors (ADWIN, Page Hinkley integration)
Feature importance analysis during drift periods
Automated hyperparameter tuning for adaptation

Long term Research Directions

Federated learning for distributed IDS deployment
Adversarial robustness against concept drift attacks
Explainable AI for drift reasoning and decision transparency
Zero-day attack detection using unsupervised drift methods
Edge computing optimization for IoT deployment scenarios

Research Impact
This work contributes to:

Adaptive cybersecurity research community
Concept drift handling in security applications
IoT security with real time threat detection
Ensemble learning optimization for dynamic environments
