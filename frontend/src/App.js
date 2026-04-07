import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { AlertOutlined, UserOutlined, EyeOutlined, WarningOutlined, CheckCircleOutlined, ExclamationCircleOutlined } from '@ant-design/icons';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

function App() {
  const [systemStatus, setSystemStatus] = useState(null);
  const [currentData, setCurrentData] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [historicalData, setHistoricalData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);

  // Fetch system status on component mount
  useEffect(() => {
    fetchSystemStatus();
    fetchCurrentState();
    fetchAlerts();
    
    // Set up polling for real-time updates
    const interval = setInterval(() => {
      fetchCurrentState();
      fetchAlerts();
    }, 2000); // Update every 2 seconds

    return () => clearInterval(interval);
  }, []);

  const fetchSystemStatus = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/status`);
      setSystemStatus(response.data);
      setLoading(false);
    } catch (err) {
      setError('Failed to connect to the backend system');
      setLoading(false);
    }
  };

  const fetchCurrentState = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/current_state`);
      setCurrentData(response.data);
      
      // Update historical data for charts
      if (response.data.features) {
        setHistoricalData(prev => {
          const newData = {
            timestamp: new Date().toLocaleTimeString(),
            peopleCount: response.data.features.people_count || 0,
            density: (response.data.features.density_per_frame || 0) * 10, // Scale for visibility
            riskScore: getRiskScore(response.data.risk_prediction?.risk_level)
          };
          
          const updated = [...prev, newData];
          // Keep only last 20 data points
          return updated.slice(-20);
        });
      }
    } catch (err) {
      console.error('Error fetching current state:', err);
    }
  };

  const fetchAlerts = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/alerts`);
      setAlerts(response.data.alerts || []);
    } catch (err) {
      console.error('Error fetching alerts:', err);
    }
  };

  const getRiskScore = (riskLevel) => {
    switch (riskLevel) {
      case 'low': return 1;
      case 'medium': return 2;
      case 'high': return 3;
      default: return 0;
    }
  };

  const getRiskColor = (riskLevel) => {
    switch (riskLevel) {
      case 'low': return '#28a745';
      case 'medium': return '#ffc107';
      case 'high': return '#dc3545';
      default: return '#6c757d';
    }
  };

  const getRiskClass = (riskLevel) => {
    switch (riskLevel) {
      case 'low': return 'risk-low';
      case 'medium': return 'risk-medium';
      case 'high': return 'risk-high';
      default: return '';
    }
  };

  const handleVideoUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setIsProcessing(true);
    const formData = new FormData();
    formData.append('video', file);

    try {
      // Convert video to base64 for API
      const reader = new FileReader();
      reader.onload = async (e) => {
        const base64Data = e.target.result.split(',')[1];
        
        await axios.post(`${API_BASE_URL}/process_frame`, {
          image: base64Data
        });
        
        setIsProcessing(false);
      };
      reader.readAsDataURL(file);
    } catch (err) {
      setError('Failed to process video');
      setIsProcessing(false);
    }
  };

  const renderSystemStatus = () => {
    if (!systemStatus) return null;

    const isOnline = systemStatus.status === 'ok';
    const isInitialized = systemStatus.system?.initialized;

    return (
      <div className="status-bar">
        <span className={`status-indicator ${isOnline ? 'status-online' : 'status-offline'}`}></span>
        System Status: {isOnline ? 'Online' : 'Offline'}
        {isInitialized && <span className="status-indicator status-online"></span>}
        Initialized: {isInitialized ? 'Yes' : 'No'}
      </div>
    );
  };

  const renderVideoFeed = () => {
    if (!currentData?.current_frame) {
      return (
        <div className="video-container">
          <div className="loading">
            Waiting for video feed...
          </div>
        </div>
      );
    }

    return (
      <div className="video-container">
        <img 
          src={`data:image/jpeg;base64,${currentData.current_frame}`} 
          alt="Live feed" 
          className="video-element"
        />
      </div>
    );
  };

  const renderStats = () => {
    if (!currentData?.features) return null;

    const { features, risk_prediction, anomaly_detection } = currentData;

    return (
      <div className="stats-grid">
        <div className="stat-item">
          <div className="stat-value">{features.people_count || 0}</div>
          <div className="stat-label">People Count</div>
        </div>
        <div className="stat-item">
          <div className="stat-value">{(features.density_per_frame || 0).toFixed(2)}</div>
          <div className="stat-label">Density</div>
        </div>
        <div className="stat-item">
          <div className="stat-value">{(features.growth_rate || 0).toFixed(2)}</div>
          <div className="stat-label">Growth Rate</div>
        </div>
        <div className="stat-item">
          <div className="stat-value">{(features.volatility || 0).toFixed(2)}</div>
          <div className="stat-label">Volatility</div>
        </div>
      </div>
    );
  };

  const renderRiskIndicator = () => {
    if (!currentData?.risk_prediction) return null;

    const { risk_level, confidence } = currentData.risk_prediction;
    const riskClass = getRiskClass(risk_level);

    return (
      <div className={`risk-indicator ${riskClass}`}>
        <div className="risk-level">
          Risk Level: {risk_level?.toUpperCase() || 'UNKNOWN'}
        </div>
        <div className="confidence">
          Confidence: {((confidence || 0) * 100).toFixed(1)}%
        </div>
      </div>
    );
  };

  const renderAlerts = () => {
    if (alerts.length === 0) {
      return (
        <div className="card">
          <h3 className="card-title">
            <CheckCircleOutlined style={{ color: '#28a745', marginRight: '8px' }} />
            System Status
          </h3>
          <div style={{ textAlign: 'center', padding: '20px', color: '#28a745' }}>
            <CheckCircleOutlined style={{ fontSize: '48px' }} />
            <p style={{ marginTop: '10px' }}>No active alerts</p>
          </div>
        </div>
      );
    }

    return (
      <div className="card">
        <h3 className="card-title">
          <WarningOutlined style={{ color: '#ffc107', marginRight: '8px' }} />
          Active Alerts ({alerts.length})
        </h3>
        {alerts.map((alert, index) => (
          <div key={index} className={`alert alert-${alert.severity}`}>
            {alert.severity === 'critical' ? (
              <ExclamationCircleOutlined style={{ fontSize: '20px' }} />
            ) : (
              <AlertOutlined style={{ fontSize: '20px' }} />
            )}
            <div>
              <strong>{alert.type.replace('_', ' ').toUpperCase()}</strong>
              <p>{alert.message}</p>
              <small>{new Date(alert.timestamp).toLocaleString()}</small>
            </div>
          </div>
        ))}
      </div>
    );
  };

  const renderCharts = () => {
    if (historicalData.length === 0) return null;

    return (
      <div className="chart-container">
        <h3 className="card-title">Real-time Analytics</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={historicalData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="peopleCount" stroke="#007bff" name="People Count" />
            <Line type="monotone" dataKey="density" stroke="#28a745" name="Density (x10)" />
            <Line type="monotone" dataKey="riskScore" stroke="#dc3545" name="Risk Score" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="container">
        <div className="loading">
          Initializing Crowd Risk Detection System...
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="container">
        <div className="card" style={{ textAlign: 'center', padding: '40px' }}>
          <ExclamationCircleOutlined style={{ fontSize: '48px', color: '#dc3545' }} />
          <h2 style={{ color: '#dc3545', marginTop: '20px' }}>System Error</h2>
          <p>{error}</p>
          <button className="button" onClick={() => window.location.reload()}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="App">
      <header className="header">
        <h1>Crowd Risk Detection System</h1>
        <p>Real-time crowd analysis and risk prediction powered by AI</p>
        {renderSystemStatus()}
      </header>

      <div className="container">
        <div className="dashboard">
          {/* Video Feed */}
          <div className="card">
            <h3 className="card-title">
              <EyeOutlined style={{ marginRight: '8px' }} />
              Live Video Feed
            </h3>
            {renderVideoFeed()}
            <div style={{ marginTop: '15px', textAlign: 'center' }}>
              <input
                type="file"
                accept="video/*,image/*"
                onChange={handleVideoUpload}
                style={{ display: 'none' }}
                id="video-upload"
              />
              <label htmlFor="video-upload" className="button" style={{ display: 'inline-block' }}>
                {isProcessing ? 'Processing...' : 'Upload Video/Image'}
              </label>
            </div>
          </div>

          {/* Current Stats */}
          <div className="card">
            <h3 className="card-title">
              <UserOutlined style={{ marginRight: '8px' }} />
              Current Statistics
            </h3>
            {renderStats()}
            {renderRiskIndicator()}
          </div>

          {/* Alerts */}
          {renderAlerts()}

          {/* Charts */}
          {renderCharts()}
        </div>
      </div>
    </div>
  );
}

export default App;
