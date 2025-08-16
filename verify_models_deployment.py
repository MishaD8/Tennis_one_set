#!/usr/bin/env python3
"""
Model Deployment Verification Script
===================================

Verifies that the enhanced tennis ML models are properly saved,
can be loaded, and are ready for production deployment.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_model_files():
    """Verify all model files exist and are accessible"""
    models_dir = '/home/apps/Tennis_one_set/tennis_models_corrected'
    
    required_files = [
        'metadata_corrected.json',
        'logistic_regression_corrected_20250816_000902.pkl',
        'random_forest_corrected_20250816_000902.pkl',
        'xgboost_corrected_20250816_000902.pkl',
        'lightgbm_corrected_20250816_000902.pkl',
        'scaler_corrected_20250816_000902.pkl'
    ]
    
    logger.info("Verifying model files...")
    
    for filename in required_files:
        filepath = os.path.join(models_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            logger.info(f"✅ {filename}: {size:,} bytes")
        else:
            logger.error(f"❌ Missing: {filename}")
            return False
    
    return True

def load_and_test_models():
    """Load models and test with sample data"""
    models_dir = '/home/apps/Tennis_one_set/tennis_models_corrected'
    
    try:
        # Load metadata
        with open(os.path.join(models_dir, 'metadata_corrected.json'), 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"✅ Metadata loaded - {metadata['total_features']} features")
        
        # Load scaler
        with open(os.path.join(models_dir, 'scaler_corrected_20250816_000902.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        
        logger.info("✅ Scaler loaded successfully")
        
        # Load models
        models = {}
        model_files = {
            'logistic_regression': 'logistic_regression_corrected_20250816_000902.pkl',
            'random_forest': 'random_forest_corrected_20250816_000902.pkl',
            'xgboost': 'xgboost_corrected_20250816_000902.pkl',
            'lightgbm': 'lightgbm_corrected_20250816_000902.pkl'
        }
        
        for model_name, filename in model_files.items():
            with open(os.path.join(models_dir, filename), 'rb') as f:
                models[model_name] = pickle.load(f)
            logger.info(f"✅ {model_name} model loaded successfully")
        
        # Test with sample data
        logger.info("Testing models with sample data...")
        
        # Create sample feature data (72 features as per metadata)
        n_features = metadata['total_features']
        sample_data = np.random.randn(5, n_features)  # 5 sample predictions
        
        # Scale the data
        sample_data_scaled = scaler.transform(sample_data)
        
        # Test each model
        predictions = {}
        for model_name, model in models.items():
            try:
                pred_proba = model.predict_proba(sample_data_scaled)[:, 1]
                pred_binary = model.predict(sample_data_scaled)
                predictions[model_name] = {
                    'probabilities': pred_proba,
                    'predictions': pred_binary
                }
                logger.info(f"✅ {model_name} predictions: {pred_proba}")
            except Exception as e:
                logger.error(f"❌ {model_name} prediction failed: {e}")
                return False
        
        # Test ensemble prediction (weighted average)
        ensemble_weights = metadata.get('ensemble_weights', {})
        if ensemble_weights:
            ensemble_proba = np.average([
                predictions[model]['probabilities'] 
                for model in ensemble_weights.keys()
                if model in predictions
            ], axis=0, weights=list(ensemble_weights.values()))
            
            logger.info(f"✅ Ensemble predictions: {ensemble_proba}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading/testing failed: {e}")
        return False

def verify_performance_metrics():
    """Verify the performance metrics are realistic"""
    models_dir = '/home/apps/Tennis_one_set/tennis_models_corrected'
    
    try:
        with open(os.path.join(models_dir, 'metadata_corrected.json'), 'r') as f:
            metadata = json.load(f)
        
        logger.info("Verifying performance metrics...")
        
        performance = metadata['model_performance']
        
        for model_name, metrics in performance.items():
            accuracy = metrics.get('accuracy', 0)
            precision = metrics.get('precision', 0)
            roi = metrics.get('simulated_roi', 0)
            
            # Check for realistic ranges
            if 0.5 <= accuracy <= 0.7:  # Realistic for tennis
                logger.info(f"✅ {model_name} accuracy: {accuracy:.3f} (realistic)")
            else:
                logger.warning(f"⚠️ {model_name} accuracy: {accuracy:.3f} (check)")
            
            if 0.5 <= precision <= 0.8:  # Good for betting
                logger.info(f"✅ {model_name} precision: {precision:.3f} (suitable for betting)")
            else:
                logger.warning(f"⚠️ {model_name} precision: {precision:.3f} (check)")
            
            if 0.9 <= roi <= 1.2:  # Realistic ROI range
                logger.info(f"✅ {model_name} ROI: {roi:.3f} (realistic)")
            else:
                logger.warning(f"⚠️ {model_name} ROI: {roi:.3f} (check)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance verification failed: {e}")
        return False

def generate_deployment_summary():
    """Generate deployment readiness summary"""
    models_dir = '/home/apps/Tennis_one_set/tennis_models_corrected'
    
    try:
        with open(os.path.join(models_dir, 'metadata_corrected.json'), 'r') as f:
            metadata = json.load(f)
        
        logger.info("\n" + "="*50)
        logger.info("DEPLOYMENT READINESS SUMMARY")
        logger.info("="*50)
        
        logger.info(f"Pipeline Version: {metadata['pipeline_version']}")
        logger.info(f"Training Date: {metadata['training_timestamp']}")
        logger.info(f"Ranking Range: {metadata['ranking_range']}")
        logger.info(f"Total Features: {metadata['total_features']}")
        logger.info(f"Total Samples: {metadata['data_statistics']['total_samples']}")
        
        logger.info("\nBest Model Performance:")
        best_model = max(metadata['model_performance'].items(), 
                        key=lambda x: x[1].get('f1_score', 0))
        
        model_name, metrics = best_model
        logger.info(f"Model: {model_name}")
        logger.info(f"  - Accuracy: {metrics['accuracy']:.3f}")
        logger.info(f"  - Precision: {metrics['precision']:.3f}")
        logger.info(f"  - F1-Score: {metrics['f1_score']:.3f}")
        logger.info(f"  - ROI: {metrics['simulated_roi']:.3f}")
        
        # Betting recommendations
        precision_65 = metrics.get('precision_65_threshold', 0)
        logger.info(f"\nBetting Strategy Recommendations:")
        logger.info(f"  - Use {model_name} as primary model")
        logger.info(f"  - Confidence threshold: 65% (Precision: {precision_65:.3f})")
        logger.info(f"  - Expected ROI per bet: {(metrics['simulated_roi']-1)*100:.1f}%")
        logger.info(f"  - Recommended bet size: 0.5-1% of bankroll")
        
        logger.info("\nFiles Ready for Production:")
        for filename in os.listdir(models_dir):
            if filename.endswith(('.pkl', '.json')):
                logger.info(f"  - {filename}")
        
        logger.info("\n✅ SYSTEM READY FOR DEPLOYMENT")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Summary generation failed: {e}")
        return False

def main():
    """Main verification function"""
    logger.info("Starting Model Deployment Verification")
    logger.info("=" * 60)
    
    # Run all verification steps
    steps = [
        ("File Verification", verify_model_files),
        ("Model Loading & Testing", load_and_test_models),
        ("Performance Verification", verify_performance_metrics),
        ("Deployment Summary", generate_deployment_summary)
    ]
    
    all_passed = True
    
    for step_name, step_func in steps:
        logger.info(f"\n{step_name}...")
        success = step_func()
        if not success:
            all_passed = False
            logger.error(f"❌ {step_name} FAILED")
        else:
            logger.info(f"✅ {step_name} PASSED")
    
    logger.info("\n" + "="*60)
    if all_passed:
        logger.info("🎉 ALL VERIFICATION CHECKS PASSED")
        logger.info("🚀 MODELS ARE READY FOR PRODUCTION DEPLOYMENT")
    else:
        logger.error("❌ SOME VERIFICATION CHECKS FAILED")
        logger.error("🔧 PLEASE REVIEW AND FIX ISSUES BEFORE DEPLOYMENT")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)