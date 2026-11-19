import { useState } from "react";
import PatientsPage from "./pages/PatientsPage";
import DiagnosisPage from "./pages/DiagnosisPage";

export default function App() {
  const [selectedPatient, setSelectedPatient] = useState(null);

  return (
    <div className="min-h-screen bg-slate-100">
      <header className="bg-blue-900 text-white px-6 py-3.5 shadow">
        <div className="max-w-6xl mx-auto flex justify-between items-center">
          <span className="font-bold tracking-wide">
            MED-SEG AI • Medical Lesion Segmentation & Quantification System
          </span>
          <span className="text-xs bg-blue-800 px-2.5 py-1 rounded">
            Phase 1 • UI Prototype
          </span>
        </div>
      </header>

      {selectedPatient ? (
        <DiagnosisPage
          patient={selectedPatient}
          onBack={() => setSelectedPatient(null)}
        />
      ) : (
        <PatientsPage onSelectPatient={(p) => setSelectedPatient(p)} />
      )}
    </div>
  );
}