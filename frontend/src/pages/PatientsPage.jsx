import { useState } from "react";
import { UserPlus, Search, Stethoscope } from "lucide-react";

const INITIAL_PATIENTS = [
  {
    id: 1,
    patient_code: "PT-2026-001",
    full_name: "Nguyen Van An",
    gender: "Male",
    dob: "1968-05-12",
    phone: "0905123456",
    medical_history: "Type 2 Diabetes (8 years), Hypertension",
  },
  {
    id: 2,
    patient_code: "PT-2026-002",
    full_name: "Tran Thi Binh",
    gender: "Female",
    dob: "1975-11-20",
    phone: "0914987654",
    medical_history: "History of Gestational Diabetes",
  },
];

export default function PatientsPage({ onSelectPatient }) {
  const [patients, setPatients] = useState(INITIAL_PATIENTS);
  const [search, setSearch] = useState("");
  const [showModal, setShowModal] = useState(false);
  const [form, setForm] = useState({
    patient_code: "",
    full_name: "",
    gender: "Male",
    dob: "",
    phone: "",
    medical_history: "",
  });

  const handleAdd = (e) => {
    e.preventDefault();
    if (!form.patient_code || !form.full_name) return;
    setPatients([{ id: Date.now(), ...form }, ...patients]);
    setForm({
      patient_code: "",
      full_name: "",
      gender: "Male",
      dob: "",
      phone: "",
      medical_history: "",
    });
    setShowModal(false);
  };

  const filtered = patients.filter(
    (p) =>
      p.full_name.toLowerCase().includes(search.toLowerCase()) ||
      p.patient_code.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-2xl font-bold text-slate-800">Patient Records Management</h1>
          <p className="text-sm text-slate-500">
            Diabetic Retinopathy Screening & Clinical Registry
          </p>
        </div>
        <button
          onClick={() => setShowModal(true)}
          className="flex items-center gap-2 bg-blue-700 hover:bg-blue-800 text-white px-4 py-2.5 rounded-lg font-medium shadow-sm"
        >
          <UserPlus size={18} /> Add New Patient
        </button>
      </div>

      <div className="mb-4 relative max-w-md">
        <Search className="absolute left-3 top-3 text-slate-400" size={18} />
        <input
          type="text"
          placeholder="Search by Patient ID or Full Name..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full pl-10 pr-4 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600"
        />
      </div>

      <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
        <table className="w-full text-left border-collapse">
          <thead className="bg-slate-50 border-b border-slate-200 text-slate-600 text-sm">
            <tr>
              <th className="py-3 px-4">Patient ID</th>
              <th className="py-3 px-4">Full Name</th>
              <th className="py-3 px-4">Gender / DOB</th>
              <th className="py-3 px-4">Phone Number</th>
              <th className="py-3 px-4">Medical History</th>
              <th className="py-3 px-4 text-right">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 text-sm">
            {filtered.map((p) => (
              <tr key={p.id} className="hover:bg-slate-50">
                <td className="py-3 px-4 font-semibold text-blue-700">{p.patient_code}</td>
                <td className="py-3 px-4 font-medium text-slate-800">{p.full_name}</td>
                <td className="py-3 px-4 text-slate-600">
                  {p.gender} • {p.dob}
                </td>
                <td className="py-3 px-4 text-slate-600">{p.phone}</td>
                <td className="py-3 px-4 text-slate-600 max-w-xs truncate">
                  {p.medical_history}
                </td>
                <td className="py-3 px-4 text-right">
                  <button
                    onClick={() => onSelectPatient(p)}
                    className="inline-flex items-center gap-1.5 bg-emerald-600 hover:bg-emerald-700 text-white px-3 py-1.5 rounded-lg text-xs font-medium"
                  >
                    <Stethoscope size={15} /> AI Diagnosis
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {showModal && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center p-4 z-50">
          <form
            onSubmit={handleAdd}
            className="bg-white rounded-xl p-6 w-full max-w-md shadow-xl space-y-4"
          >
            <h2 className="text-lg font-bold text-slate-800">Register New Patient</h2>
            <input
              required
              placeholder="Patient ID (e.g., PT-2026-003)"
              value={form.patient_code}
              onChange={(e) => setForm({ ...form, patient_code: e.target.value })}
              className="w-full border p-2.5 rounded-lg text-sm"
            />
            <input
              required
              placeholder="Patient Full Name"
              value={form.full_name}
              onChange={(e) => setForm({ ...form, full_name: e.target.value })}
              className="w-full border p-2.5 rounded-lg text-sm"
            />
            <div className="grid grid-cols-2 gap-3">
              <select
                value={form.gender}
                onChange={(e) => setForm({ ...form, gender: e.target.value })}
                className="border p-2.5 rounded-lg text-sm"
              >
                <option value="Male">Male</option>
                <option value="Female">Female</option>
              </select>
              <input
                type="date"
                value={form.dob}
                onChange={(e) => setForm({ ...form, dob: e.target.value })}
                className="border p-2.5 rounded-lg text-sm"
              />
            </div>
            <input
              placeholder="Phone Number"
              value={form.phone}
              onChange={(e) => setForm({ ...form, phone: e.target.value })}
              className="w-full border p-2.5 rounded-lg text-sm"
            />
            <textarea
              placeholder="Clinical & Medical History..."
              rows={3}
              value={form.medical_history}
              onChange={(e) => setForm({ ...form, medical_history: e.target.value })}
              className="w-full border p-2.5 rounded-lg text-sm"
            />
            <div className="flex justify-end gap-2 pt-2">
              <button
                type="button"
                onClick={() => setShowModal(false)}
                className="px-4 py-2 text-sm text-slate-600 hover:bg-slate-100 rounded-lg"
              >
                Cancel
              </button>
              <button
                type="submit"
                className="px-4 py-2 text-sm bg-blue-700 text-white rounded-lg font-medium"
              >
                Save Record
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
}