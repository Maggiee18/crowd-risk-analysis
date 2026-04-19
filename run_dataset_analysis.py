"""
Dataset Risk Analysis Script
Runs complete crowd risk detection on the mall dataset
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from predictor import RiskPredictor
from anomaly_detector import AnomalyDetector
from analyzer import CrowdAnalyzer

def load_dataset():
    """Load the mall dataset"""
    print("Loading mall dataset...")
    df = pd.read_csv('output/mall_detection_counts.csv')
    print(f"Loaded {len(df)} frames with people count data")
    
    # Show basic statistics
    print(f"People count statistics:")
    print(f"  Min: {df['people_count'].min()}")
    print(f"  Max: {df['people_count'].max()}")
    print(f"  Mean: {df['people_count'].mean():.1f}")
    print(f"  Std: {df['people_count'].std():.1f}")
    
    return df

def create_features_from_counts(df):
    """Create comprehensive features from people counts"""
    print("Creating features from people counts...")
    
    features = []
    
    for i in range(len(df)):
        count = df.iloc[i]['people_count']
        
        # Calculate density (assuming frame area)
        frame_area = 480 * 640  # pixels
        density = count / (frame_area / 10000)  # people per 10000 pixels
        
        # Calculate growth rate (change from previous frames)
        if i >= 5:
            recent_counts = df.iloc[i-5:i]['people_count'].values
            growth_rate = np.polyfit(range(5), recent_counts, 1)[0]
        else:
            growth_rate = 0
        
        # Calculate volatility (standard deviation of recent counts)
        if i >= 10:
            recent_counts = df.iloc[i-10:i]['people_count'].values
            volatility = np.std(recent_counts)
        else:
            volatility = 0.5
        
        # Simulate movement features based on count changes
        if i > 0:
            count_change = abs(count - df.iloc[i-1]['people_count'])
            avg_magnitude = count_change * 2.0
            movement_entropy = min(count_change * 0.3, 3.0)
        else:
            avg_magnitude = 1.0
            movement_entropy = 0.5
        
        # Simulate other features
        angle_consistency = 0.7 + np.random.normal(0, 0.1)
        angle_consistency = np.clip(angle_consistency, 0, 1)
        
        movement_concentration = 1.0 + np.random.exponential(0.5)
        
        feature_dict = {
            'frame_number': i + 1,
            'people_count': count,
            'density_per_frame': density,
            'growth_rate': growth_rate,
            'volatility': volatility,
            'avg_magnitude': avg_magnitude,
            'movement_entropy': movement_entropy,
            'angle_consistency': angle_consistency,
            'movement_concentration': movement_concentration
        }
        
        features.append(feature_dict)
    
    features_df = pd.DataFrame(features)
    print(f"Created {len(features_df)} feature vectors")
    
    return features_df

def train_risk_models(features_df):
    """Train risk prediction and anomaly detection models"""
    print("\nTraining risk prediction models...")
    
    # Train risk predictor
    predictor = RiskPredictor(model_type='random_forest')
    X_train, X_test, y_train, y_test = predictor.prepare_data(features_df)
    training_results = predictor.train(X_train, y_train)
    evaluation_results = predictor.evaluate(X_test, y_test)
    
    print(f"Risk Predictor Results:")
    print(f"  Accuracy: {evaluation_results['accuracy']:.3f}")
    print(f"  F1-Score: {evaluation_results['f1_score']:.3f}")
    print(f"  Precision: {evaluation_results['precision']:.3f}")
    print(f"  Recall: {evaluation_results['recall']:.3f}")
    
    # Train anomaly detector
    print("\nTraining anomaly detection model...")
    anomaly_detector = AnomalyDetector(method='isolation_forest', contamination=0.1)
    anomaly_results = anomaly_detector.train(features_df)
    
    print(f"Anomaly Detector Results:")
    print(f"  Training samples: {anomaly_results['n_samples']}")
    print(f"  Anomaly rate: {anomaly_results['anomaly_rate']:.3f}")
    
    return predictor, anomaly_detector

def analyze_dataset_risk(df, features_df, predictor, anomaly_detector):
    """Analyze risk across the entire dataset"""
    print("\nAnalyzing crowd risk across dataset...")
    
    results = []
    alerts = []
    
    for i in range(len(features_df)):
        features = features_df.iloc[i].values
        frame_num = i + 1
        
        # Risk prediction
        risk_level, confidence = predictor.predict(features)
        
        # Anomaly detection
        is_anomaly, anomaly_score, anomaly_details = anomaly_detector.detect_anomaly(features)
        
        # Determine risk category
        if risk_level == 'high' and confidence > 0.7:
            risk_category = 'CRITICAL'
        elif risk_level == 'medium' or (risk_level == 'high' and confidence > 0.5):
            risk_category = 'HIGH'
        elif risk_level == 'low' and confidence > 0.6:
            risk_category = 'LOW'
        else:
            risk_category = 'NORMAL'
        
        # Generate alerts
        frame_alerts = []
        
        if risk_category == 'CRITICAL':
            frame_alerts.append({
                'type': 'critical_risk',
                'message': f'Critical crowd risk detected! People: {int(features[1])}',
                'severity': 'critical'
            })
        elif risk_category == 'HIGH':
            frame_alerts.append({
                'type': 'high_risk',
                'message': f'High crowd risk detected! People: {int(features[1])}',
                'severity': 'warning'
            })
        
        if is_anomaly and anomaly_score > 0.5:
            frame_alerts.append({
                'type': 'anomaly',
                'message': f'Anomalous crowd behavior detected! Score: {anomaly_score:.2f}',
                'severity': 'warning'
            })
        
        # Check for sudden crowd surges
        if i >= 5:
            recent_counts = df.iloc[max(0,i-5):i]['people_count'].values
            if len(recent_counts) >= 3:
                recent_trend = np.polyfit(range(len(recent_counts)), recent_counts, 1)[0]
                if recent_trend > 3:  # Sudden increase
                    frame_alerts.append({
                        'type': 'crowd_surge',
                        'message': f'Sudden crowd surge detected! Growth rate: {recent_trend:.1f}',
                        'severity': 'warning'
                    })
        
        result = {
            'frame': frame_num,
            'people_count': int(features[1]),
            'risk_level': risk_level,
            'confidence': confidence,
            'risk_category': risk_category,
            'is_anomaly': is_anomaly,
            'anomaly_score': anomaly_score,
            'alerts': frame_alerts,
            'density': features[2],
            'growth_rate': features[3],
            'volatility': features[4]
        }
        
        results.append(result)
        alerts.extend(frame_alerts)
    
    print(f"Analyzed {len(results)} frames")
    print(f"Generated {len(alerts)} alerts")
    
    return results, alerts

def generate_summary_report(results, alerts):
    """Generate comprehensive summary report"""
    print("\n" + "="*60)
    print("CROWD RISK DETECTION SUMMARY REPORT")
    print("="*60)
    
    # Risk distribution
    risk_counts = {}
    for result in results:
        category = result['risk_category']
        risk_counts[category] = risk_counts.get(category, 0) + 1
    
    print(f"\nRisk Distribution:")
    for category, count in sorted(risk_counts.items()):
        percentage = (count / len(results)) * 100
        print(f"  {category}: {count} frames ({percentage:.1f}%)")
    
    # Alert summary
    alert_types = {}
    for alert in alerts:
        alert_type = alert['type']
        alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
    
    print(f"\nAlert Summary:")
    for alert_type, count in sorted(alert_types.items()):
        print(f"  {alert_type}: {count} occurrences")
    
    # High-risk frames
    high_risk_frames = [r for r in results if r['risk_category'] in ['HIGH', 'CRITICAL']]
    print(f"\nHigh-Risk Frames: {len(high_risk_frames)}")
    
    if high_risk_frames:
        print(f"Top 10 High-Risk Frames:")
        sorted_high_risk = sorted(high_risk_frames, 
                               key=lambda x: (x['people_count'], x['confidence']), 
                               reverse=True)[:10]
        
        for i, frame in enumerate(sorted_high_risk, 1):
            print(f"  {i}. Frame {frame['frame']}: {frame['people_count']} people, "
                  f"{frame['risk_level']} risk (conf: {frame['confidence']:.2f})")
    
    # Anomaly summary
    anomaly_frames = [r for r in results if r['is_anomaly']]
    print(f"\nAnomalous Frames: {len(anomaly_frames)}")
    
    if anomaly_frames:
        avg_anomaly_score = np.mean([r['anomaly_score'] for r in anomaly_frames])
        print(f"Average anomaly score: {avg_anomaly_score:.3f}")
    
    # Statistics
    people_counts = [r['people_count'] for r in results]
    print(f"\nCrowd Statistics:")
    print(f"  Average people count: {np.mean(people_counts):.1f}")
    print(f"  Peak crowd: {np.max(people_counts)} people")
    print(f"  Minimum crowd: {np.min(people_counts)} people")
    
    return risk_counts, alert_types

def create_visualizations(results, alerts):
    """Create visualizations of the analysis"""
    print("\nCreating visualizations...")
    
    # Create output directory
    os.makedirs('output/visualizations', exist_ok=True)
    
    # 1. People count over time with risk levels
    plt.figure(figsize=(15, 8))
    frames = [r['frame'] for r in results]
    counts = [r['people_count'] for r in results]
    
    # Color by risk level
    colors = []
    for r in results:
        if r['risk_category'] == 'CRITICAL':
            colors.append('red')
        elif r['risk_category'] == 'HIGH':
            colors.append('orange')
        elif r['risk_category'] == 'LOW':
            colors.append('yellow')
        else:
            colors.append('green')
    
    plt.scatter(frames, counts, c=colors, alpha=0.6, s=20)
    plt.plot(frames, counts, 'b-', alpha=0.3, linewidth=1)
    
    # Mark alerts
    alert_frames = [r['frame'] for r in results if r['alerts']]
    alert_counts = [r['people_count'] for r in results if r['alerts']]
    plt.scatter(alert_frames, alert_counts, c='red', s=100, marker='x', linewidth=2)
    
    plt.xlabel('Frame Number')
    plt.ylabel('People Count')
    plt.title('Crowd Count Over Time with Risk Levels')
    plt.grid(True, alpha=0.3)
    
    # Add legend
    legend_elements = [
        plt.scatter([], [], c='green', label='Normal'),
        plt.scatter([], [], c='yellow', label='Low Risk'),
        plt.scatter([], [], c='orange', label='High Risk'),
        plt.scatter([], [], c='red', label='Critical Risk'),
        plt.scatter([], [], c='red', marker='x', s=100, label='Alerts')
    ]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig('output/visualizations/crowd_risk_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Risk distribution pie chart
    risk_counts = {}
    for r in results:
        category = r['risk_category']
        risk_counts[category] = risk_counts.get(category, 0) + 1
    
    plt.figure(figsize=(10, 6))
    colors_pie = ['green', 'yellow', 'orange', 'red']
    plt.pie(risk_counts.values(), labels=risk_counts.keys(), colors=colors_pie[:len(risk_counts)], 
            autopct='%1.1f%%', startangle=90)
    plt.title('Risk Distribution Across All Frames')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('output/visualizations/risk_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Alert timeline
    if alerts:
        plt.figure(figsize=(15, 6))
        alert_frames = [int(r['frame']) for r in results for _ in r['alerts']]
        alert_types = [alert['type'] for r in results for alert in r['alerts']]
        
        alert_colors = {'critical_risk': 'red', 'high_risk': 'orange', 
                       'anomaly': 'purple', 'crowd_surge': 'brown'}
        
        for i, (frame, alert_type) in enumerate(zip(alert_frames, alert_types)):
            color = alert_colors.get(alert_type, 'gray')
            plt.scatter(frame, 1, c=color, s=50, alpha=0.7)
        
        plt.xlabel('Frame Number')
        plt.ylabel('Alerts')
        plt.title('Alert Timeline Throughout Dataset')
        plt.yticks([])
        plt.grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = [plt.scatter([], [], c=color, label=alert_type.replace('_', ' ').title()) 
                          for alert_type, color in alert_colors.items() if alert_type in set(alert_types)]
        plt.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.savefig('output/visualizations/alert_timeline.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Visualizations saved to 'output/visualizations/' directory")

def save_results(results, alerts, risk_counts, alert_types):
    """Save analysis results to files"""
    print("\nSaving results...")
    
    # Create output directory
    os.makedirs('output/analysis_results', exist_ok=True)
    
    # Save detailed results
    results_df = pd.DataFrame(results)
    results_df.to_csv('output/analysis_results/detailed_analysis.csv', index=False)
    
    # Save alerts
    alerts_df = pd.DataFrame(alerts)
    alerts_df.to_csv('output/analysis_results/alerts.csv', index=False)
    
    # Save summary
    summary = {
        'analysis_date': datetime.now().isoformat(),
        'total_frames': len(results),
        'risk_distribution': risk_counts,
        'alert_summary': alert_types,
        'high_risk_frames': len([r for r in results if r['risk_category'] in ['HIGH', 'CRITICAL']]),
        'anomalous_frames': len([r for r in results if r['is_anomaly']]),
        'total_alerts': len(alerts)
    }
    
    with open('output/analysis_results/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("Results saved to 'output/analysis_results/' directory")

def main():
    """Main analysis function"""
    print("Crowd Risk Detection System - Dataset Analysis")
    print("="*60)
    print("Analyzing mall dataset for crowd risks and generating warnings...")
    print("="*60)
    
    try:
        # Load dataset
        df = load_dataset()
        
        # Create features
        features_df = create_features_from_counts(df)
        
        # Train models
        predictor, anomaly_detector = train_risk_models(features_df)
        
        # Analyze risk
        results, alerts = analyze_dataset_risk(df, features_df, predictor, anomaly_detector)
        
        # Generate report
        risk_counts, alert_types = generate_summary_report(results, alerts)
        
        # Create visualizations
        create_visualizations(results, alerts)
        
        # Save results
        save_results(results, alerts, risk_counts, alert_types)
        
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"Key Findings:")
        print(f"  - Total frames analyzed: {len(results)}")
        print(f"  - High-risk periods: {len([r for r in results if r['risk_category'] in ['HIGH', 'CRITICAL']])}")
        print(f"  - Alerts generated: {len(alerts)}")
        print(f"  - Anomalous events: {len([r for r in results if r['is_anomaly']])}")
        print(f"\nCheck 'output/' directory for detailed results and visualizations!")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
