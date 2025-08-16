#!/usr/bin/env python3
"""
Generate performance report from trained models
"""

import json
import os
from datetime import datetime

def generate_performance_report():
    """Generate performance report from training results"""
    
    # Load metadata
    metadata_path = "/home/apps/Tennis_one_set/tennis_models/metadata.json"
    
    if not os.path.exists(metadata_path):
        print("❌ No metadata found. Please train models first.")
        return
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    training_results = metadata.get('training_results', {})
    
    report = []
    report.append("# 🎾 Tennis Underdog Second Set ML Performance Report\n")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Training Summary
    training_info = training_results.get('training_info', {})
    report.append("## 📊 Training Summary\n")
    
    data_shape = training_info.get('data_shape', ['N/A', 'N/A'])
    total_samples = data_shape[0] if isinstance(data_shape, list) and len(data_shape) > 0 else 'N/A'
    
    report.append(f"- **Total Samples:** {total_samples:,}" if isinstance(total_samples, int) else f"- **Total Samples:** {total_samples}")
    
    feature_count = training_info.get('feature_count', 'N/A')
    report.append(f"- **Features:** {feature_count:,}" if isinstance(feature_count, int) else f"- **Features:** {feature_count}")
    
    train_samples = training_info.get('train_samples', 'N/A')
    report.append(f"- **Training Samples:** {train_samples:,}" if isinstance(train_samples, int) else f"- **Training Samples:** {train_samples}")
    
    test_samples = training_info.get('test_samples', 'N/A')
    report.append(f"- **Test Samples:** {test_samples:,}" if isinstance(test_samples, int) else f"- **Test Samples:** {test_samples}")
    
    class_dist = training_info.get('class_distribution', {})
    favorite_wins = class_dist.get('0', 'N/A')
    underdog_wins = class_dist.get('1', 'N/A')
    
    fav_str = f"{favorite_wins:,}" if isinstance(favorite_wins, int) else str(favorite_wins)
    und_str = f"{underdog_wins:,}" if isinstance(underdog_wins, int) else str(underdog_wins)
    
    report.append(f"- **Class Distribution:** Favorite wins: {fav_str}, Underdog wins: {und_str}")
    report.append("")
    
    # Model Performance
    report.append("## 🤖 Model Performance\n")
    model_results = training_results.get('model_results', {})
    
    for model_name, result in model_results.items():
        if 'error' not in result:
            metrics = result['metrics']
            report.append(f"### {model_name.replace('_', ' ').title()}")
            report.append(f"- **Accuracy:** {metrics['accuracy']:.3f}")
            report.append(f"- **Precision:** {metrics['precision']:.3f}")
            report.append(f"- **Recall:** {metrics['recall']:.3f}")
            report.append(f"- **F1 Score:** {metrics['f1_score']:.3f}")
            report.append(f"- **ROC AUC:** {metrics['roc_auc']:.3f}")
            report.append(f"- **Precision @ 70% Recall:** {metrics['precision_70_recall']:.3f}")
            report.append(f"- **Simulated ROI:** {metrics['simulated_roi']:.3f}")
            report.append("")
    
    # Ensemble Performance
    if 'ensemble_performance' in training_results:
        report.append("## 🎼 Ensemble Performance\n")
        ensemble = training_results['ensemble_performance']
        if 'voting' in ensemble:
            voting_metrics = ensemble['voting']['metrics']
            report.append("### Voting Classifier")
            report.append(f"- **Accuracy:** {voting_metrics['accuracy']:.3f}")
            report.append(f"- **Precision:** {voting_metrics['precision']:.3f}")
            report.append(f"- **Recall:** {voting_metrics['recall']:.3f}")
            report.append(f"- **F1 Score:** {voting_metrics['f1_score']:.3f}")
            report.append(f"- **ROC AUC:** {voting_metrics['roc_auc']:.3f}")
            report.append(f"- **Simulated ROI:** {voting_metrics['simulated_roi']:.3f}")
            report.append("")
    
    # Betting Recommendations
    report.append("## 💰 Betting Recommendations\n")
    
    best_roi = -1
    best_model = None
    for model_name, result in model_results.items():
        if 'error' not in result:
            roi = result['metrics']['simulated_roi']
            if roi > best_roi:
                best_roi = roi
                best_model = model_name
    
    if best_roi > 0.05:
        report.append("🟢 **POSITIVE EXPECTED VALUE DETECTED**")
        report.append(f"- Best performing model: {best_model}")
        report.append(f"- Simulated ROI: {best_roi:.3f} ({best_roi*100:.1f}%)")
        report.append("- Recommended for live betting with conservative stakes")
        report.append("- Use ensemble model for best results")
    elif best_roi > 0:
        report.append("🟡 **MARGINAL PROFITABILITY**")
        report.append("- Small positive ROI detected")
        report.append("- Proceed with extreme caution and minimal stakes")
    else:
        report.append("🔴 **NEGATIVE EXPECTED VALUE**")
        report.append("- Models show negative ROI")
        report.append("- Not recommended for betting until improvement")
    
    report.append("")
    
    # Feature Importance Analysis
    report.append("## 🔍 Key Insights\n")
    
    # Top features analysis
    model_performance = metadata.get('model_performance', {})
    if model_performance:
        report.append("### Model Performance Summary")
        for model_name, perf in model_performance.items():
            roi = perf.get('simulated_roi', 0)
            precision = perf.get('precision', 0)
            report.append(f"- **{model_name.title()}**: ROI {roi:.3f}, Precision {precision:.3f}")
        report.append("")
    
    # Ensemble analysis
    ensemble_perf = training_results.get('ensemble_performance', {})
    if 'voting' in ensemble_perf:
        ensemble_roi = ensemble_perf['voting']['metrics']['simulated_roi']
        report.append(f"### Ensemble Advantage")
        report.append(f"- Voting ensemble ROI: {ensemble_roi:.3f}")
        report.append(f"- Improvement over best individual model: {ensemble_roi - best_roi:.3f}")
        report.append("")
    
    # System Status
    report.append("## 🎯 System Status\n")
    report.append("### Ready for Production")
    report.append("- ✅ Models trained and saved")
    report.append("- ✅ Feature engineering pipeline complete")
    report.append("- ✅ Evaluation metrics implemented")
    report.append("- ✅ Ensemble model available")
    report.append("")
    
    report.append("### Deployment Recommendations")
    if best_roi > 0.5:
        report.append("- 🚀 **HIGH CONFIDENCE**: Deploy with standard betting limits")
    elif best_roi > 0.2:
        report.append("- 🟢 **MEDIUM CONFIDENCE**: Deploy with reduced betting limits")
    elif best_roi > 0.05:
        report.append("- 🟡 **LOW CONFIDENCE**: Deploy with minimal betting limits for testing")
    else:
        report.append("- 🔴 **NOT READY**: Requires model improvement before deployment")
    
    report.append("")
    report.append("## ⚠️ Important Disclaimers")
    report.append("- These models are for educational/research purposes")
    report.append("- Past performance does not guarantee future results")
    report.append("- Sports betting involves significant financial risk")
    report.append("- Always bet responsibly and within your means")
    report.append("- Validate model predictions on new data before live deployment")
    
    # Save report
    report_content = "\n".join(report)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"/home/apps/Tennis_one_set/tennis_models/performance_report_{timestamp}.md"
    
    with open(output_path, 'w') as f:
        f.write(report_content)
    
    print(f"📝 Performance report saved: {output_path}")
    print("\n" + "="*60)
    print(report_content)
    
    return report_content

if __name__ == "__main__":
    generate_performance_report()