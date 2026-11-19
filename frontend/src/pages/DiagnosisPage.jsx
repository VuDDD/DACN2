import { useState } from "react";
import { ArrowLeft, Upload, Activity, FileDown } from "lucide-react";

export default function DiagnosisPage({ patient, onBack }) {
  const [imagePreview, setImagePreview] = useState(null);
  const [doctorNotes, setDoctorNotes] = useState("");
  const [metrics, setMetrics] = useState(null);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setImagePreview(URL.createObjectURL(file));
    setMetrics({
      model_version: "UNet-Baseline-v1",
      lesion_count: 14,
      lesion_pixels: 3420,
      area_mm2: 1.85,
      lesion_ratio_percent: 2.41,
      severity_level: "Moderate",
      inference_time_ms: 142,
    });
  };

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <button
        onClick={onBack}
        className="flex items-center gap-1.5 text-sm text-slate-600 hover:text-blue-700 mb-4"
      >
        <ArrowLeft size={16} /> Back to Patient Registry
      </button>

      <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 mb-6 flex justify-between items-center">
        <div>
          <span className="text-xs font-bold uppercase text-blue-700">
            {patient.patient_code}
          </span>
          <h2 className="text-lg font-bold text-slate-800">
            {patient.full_name} ({patient.gender} • {patient.dob})
          </h2>
          <p className="text-xs text-slate-600">
            Medical History: {patient.medical_history || "None recorded"}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
          <h3 className="font-semibold text-slate-800 mb-3">
            1. Fundus Image & Lesion Segmentation
          </h3>
          <label className="flex flex-col items-center justify-center border-2 border-dashed border-slate-300 rounded-xl p-6 cursor-pointer hover:border-blue-600 bg-slate-50">
            <Upload className="text-blue-600 mb-2" size={28} />
            <span className="text-sm font-medium text-slate-700">
              Upload Retinal Fundus Image (JPG / PNG)
            </span>
            <input
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              className="hidden"
            />
          </label>

          {imagePreview && (
            <div className="mt-4">
              <p className="text-xs text-slate-500 mb-2">Uploaded Retinal Scan:</p>
              <img
                src={imagePreview}
                alt="Retinal Scan"
                className="w-full h-72 object-contain bg-black rounded-lg"
              />
            </div>
          )}
        </div>

        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm flex flex-col justify-between">
          <div>
            <h3 className="font-semibold text-slate-800 mb-3 flex items-center gap-2">
              <Activity size={18} className="text-emerald-600" /> 2. AI Quantitative Lesion Metrics
            </h3>

            {!metrics ? (
              <div className="text-center py-12 text-sm text-slate-400 border rounded-lg bg-slate-50">
                Upload a fundus image to run AI lesion segmentation and quantification
              </div>
            ) : (
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-3">
                  <div className="p-3 bg-slate-50 rounded-lg border">
                    <p className="text-xs text-slate-500">Detected Lesions</p>
                    <p className="text-xl font-bold text-slate-800">
                      {metrics.lesion_count} lesions
                    </p>
                  </div>
                  <div className="p-3 bg-slate-50 rounded-lg border">
                    <p className="text-xs text-slate-500">Total Lesion Area</p>
                    <p className="text-xl font-bold text-blue-700">
                      {metrics.area_mm2} mm²
                    </p>
                  </div>
                  <div className="p-3 bg-slate-50 rounded-lg border">
                    <p className="text-xs text-slate-500">Affected Retina Ratio</p>
                    <p className="text-xl font-bold text-slate-800">
                      {metrics.lesion_ratio_percent}%
                    </p>
                  </div>
                  <div className="p-3 bg-amber-50 rounded-lg border border-amber-200">
                    <p className="text-xs text-amber-700">Severity Grade</p>
                    <p className="text-xl font-bold text-amber-800">
                      {metrics.severity_level}
                    </p>
                  </div>
                </div>

                <div>
                  <label className="block text-xs font-medium text-slate-700 mb-1">
                    Physician Clinical Notes & Assessment
                  </label>
                  <textarea
                    rows={3}
                    value={doctorNotes}
                    onChange={(e) => setDoctorNotes(e.target.value)}
                    placeholder="Enter clinical findings and treatment recommendations..."
                    className="w-full border border-slate-300 rounded-lg p-2.5 text-sm"
                  />
                </div>
              </div>
            )}
          </div>

          <button
            disabled={!metrics}
            onClick={() =>
              alert("Diagnostic PDF Report will be generated once connected to the Backend API!")
            }
            className="mt-4 w-full flex items-center justify-center gap-2 bg-slate-800 disabled:bg-slate-300 hover:bg-slate-900 text-white py-2.5 rounded-lg text-sm font-medium"
          >
            <FileDown size={18} /> Export Diagnostic Report (PDF)
          </button>
        </div>
      </div>
    </div>
  );
}