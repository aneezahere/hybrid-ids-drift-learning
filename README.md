# Hybrid IDS with Concept Drift Learning

A novel intrusion detection system combining ensemble learning with adaptive drift detection for IoT and network security.

## Key Results
- 91.8% accuracy (vs 62% baseline)
- 48% improvement over existing approaches
- Real-time concept drift detection and adaptation
- Support for 8 different attack types

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Training
```bash
python train_model.py
```

### Web Demo
```bash
python app.py
```

## Performance Comparison
| Method | Accuracy |
|--------|-----------|
| Original Baseline | 62.0% |
| Static Random Forest | 99.5% |
| **Adaptive Hybrid** | **91.8%** |

## Research Project
This supports the senior research project on Hybrid Intrusion Detection with Concept Drift Learning for IoT Security

## Files
- hybrid_ids.py - Main model implementation
- train_model.py - Training script
- app.py - Web application
- templates/ - HTML templates
- requirements.txt - Dependencies

## Dataset
New Brunswick University IoT Security Dataset
- 8 attack categories
- 787K+ samples
- IoT and traditional network attacks
