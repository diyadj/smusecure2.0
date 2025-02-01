import React, { useState, useEffect } from 'react';
import { BarChart, LineChart, Line, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import Papa from 'papaparse';
import { BarChart2, PieChart, FileText, Clock } from 'lucide-react';

const Card = ({ title, children }) => (
  <div className="bg-white rounded-lg shadow p-4">
    <h3 className="text-lg font-semibold mb-4">{title}</h3>
    {children}
  </div>
);

const Tabs = ({ tabs, activeTab, onTabChange }) => (
  <div className="flex space-x-2 bg-gray-100 p-2 rounded-lg mb-4">
    {tabs.map(tab => (
      <button
        key={tab.value}
        onClick={() => onTabChange(tab.value)}
        className={`flex items-center gap-2 px-4 py-2 rounded-md ${
          activeTab === tab.value 
            ? 'bg-white text-blue-600 shadow' 
            : 'text-gray-600 hover:bg-gray-200'
        }`}
      >
        {tab.icon}
        {tab.label}
      </button>
    ))}
  </div>
);

const Dashboard = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [data, setData] = useState([]);

  const tabs = [
    { value: 'overview', label: 'Overview', icon: <BarChart2 className="w-4 h-4" /> },
    { value: 'topics', label: 'Topics', icon: <PieChart className="w-4 h-4" /> },
    { value: 'entities', label: 'Entities', icon: <FileText className="w-4 h-4" /> },
    { value: 'timeline', label: 'Timeline', icon: <Clock className="w-4 h-4" /> }
  ];

  useEffect(() => {
    const loadData = async () => {
      try {
        const response = await fetch('processed_data.csv');
        const csvText = await response.text();
        Papa.parse(csvText, {
          header: true,
          complete: (results) => setData(results.data)
        });
      } catch (error) {
        console.error('Error loading data:', error);
      }
    };
    loadData();
  }, []);

  const renderContent = () => {
    switch(activeTab) {
      case 'overview':
        return (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card title="Text Statistics">
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span>Total Documents</span>
                  <span className="font-semibold">{data?.length || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span>Average Word Count</span>
                  <span className="font-semibold">248</span>
                </div>
              </div>
            </Card>
            <Card title="Sentiment Distribution">
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={[
                    { sentiment: 'Positive', count: 450 },
                    { sentiment: 'Neutral', count: 800 },
                    { sentiment: 'Negative', count: 259 }
                  ]}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="sentiment" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="count" fill="#4f46e5" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Card>
          </div>
        );

      case 'topics':
        return (
          <Card title="Topic Distribution">
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={[
                  { topic: 'CRIMINAL JUSTICE SYSTEM', documents: 320 },
                  { topic: 'HEALTHCARE', documents: 280 },
                  { topic: 'CORPORATE', documents: 250 },
                  { topic: 'ECONOMIC RS (CHINA)', documents: 200 },
                  { topic: 'SANCTIONS & WAR', documents: 150 }
                ]}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="topic" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="documents" fill="#6366f1" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>
        );

      case 'entities':
        return (
          <Card title="Named Entity Recognition">
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart layout="vertical" data={[
                  { entity: 'PERSON', count: 450 },
                  { entity: 'ORG', count: 380 },
                  { entity: 'GPE', count: 320 },
                  { entity: 'DATE', count: 280 },
                  { entity: 'NORP', count: 220 }
                ]}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis dataKey="entity" type="category" />
                  <Tooltip />
                  <Bar dataKey="count" fill="#8b5cf6" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>
        );

      case 'timeline':
        return (
          <Card title="ISD Timeline">
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={[
                  { date: 'Jan', count: 120 },
                  { date: 'Feb', count: 180 },
                  { date: 'Mar', count: 150 },
                  { date: 'Apr', count: 200 },
                  { date: 'May', count: 250 }
                ]}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="count" stroke="#3b82f6" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Card>
        );
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-4">
      <div className="max-w-7xl mx-auto">
        <header className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">ISD Analytics Dashboard</h1>
          <p className="text-gray-600">Intelligent Insights</p>
        </header>

        <Tabs 
          tabs={tabs}
          activeTab={activeTab}
          onTabChange={setActiveTab}
        />

        {renderContent()}
      </div>
    </div>
  );
};

export default Dashboard;