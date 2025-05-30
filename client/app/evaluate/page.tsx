import React, { useState, ChangeEvent } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from "recharts";
import Papa from "papaparse";

type MetricData = {
  question: string;
  bleu?: number;
  rouge1?: number;
  rougeL?: number;
  [key: string]: string | number | undefined;
};

export default function ResultsDashboard() {
  const [data, setData] = useState<MetricData[]>([]);

  const handleUpload = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    Papa.parse<MetricData>(file, {
      header: true,
      dynamicTyping: true,
      complete: (result) => {
        setData(result.data.filter(row => row.bleu !== undefined));
      },
    });
  };

  return (
    <div className="p-4 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Evaluation Metrics Dashboard</h1>
      <input type="file" accept=".csv" onChange={handleUpload} className="mb-6" />

      {data.length > 0 && (
        <div className="grid gap-6 md:grid-cols-2">
          <MetricChart data={data} metric="bleu" color="#8884d8" />
          <MetricChart data={data} metric="rouge1" color="#82ca9d" />
          <MetricChart data={data} metric="rougeL" color="#ffa500" />
        </div>
      )}
    </div>
  );
}

type MetricChartProps = {
  data: MetricData[];
  metric: "bleu" | "rouge1" | "rougeL";
  color: string;
};

function MetricChart({ data, metric, color }: MetricChartProps) {
  return (
    <div className="bg-white rounded-2xl shadow p-4">
      <h2 className="text-xl font-semibold mb-2">{metric.toUpperCase()} Scores</h2>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="question" tick={false} label={{ value: "Samples", position: "insideBottomRight", offset: -5 }} />
          <YAxis domain={[0, 1]} />
          <Tooltip />
          <Line type="monotone" dataKey={metric} stroke={color} strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}