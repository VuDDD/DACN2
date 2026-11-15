import React, { useEffect, useState } from 'react';
import api from '../services/api';
import {
  Users,
  Search,
  Plus,
  Edit2,
  Trash2,
  Calendar,
  Phone,
  FileText,
  Activity,
  X
} from 'lucide-react';
import { Link } from 'react-router-dom';

export const Patients: React.FC = () => {
  const [patients, setPatients] = useState<any[]>([]);
  const [search, setSearch] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(true);
  const [showModal, setShowModal] = useState<boolean>(false);
  const [editingPatient, setEditingPatient] = useState<any | null>(null);

  // Form State
  const [fullName, setFullName] = useState<string>('');
  const [gender, setGender] = useState<string>('Male');
  const [dob, setDob] = useState<string>('1975-01-01');
  const [phone, setPhone] = useState<string>('');
  const [history, setHistory] = useState<string>('Tiểu đường type 2 (10 năm), tăng huyết áp nhẹ');
  const [notes, setNotes] = useState<string>('');

  useEffect(() => {
    fetchPatients();
  }, [search]);

  const fetchPatients = async () => {
    setLoading(true);
    try {
      const res = await api.get('/patients', {
        params: { search: search || undefined },
      });
      setPatients(res.data.items);
    } catch (err) {
      console.error('Error fetching patients:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleOpenCreate = () => {
    setEditingPatient(null);
    setFullName('');
    setGender('Male');
    setDob('1975-01-01');
    setPhone('');
    setHistory('Tiểu đường type 2 (10 năm), tăng huyết áp');
    setNotes('');
    setShowModal(true);
  };

  const handleOpenEdit = (p: any) => {
    setEditingPatient(p);
    setFullName(p.full_name);
    setGender(p.gender);
    setDob(p.date_of_birth || '1975-01-01');
    setPhone(p.phone_number || '');
    setHistory(p.medical_history || '');
    setNotes(p.notes || '');
    setShowModal(true);
  };

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const payload = {
        full_name: fullName,
        gender,
        date_of_birth: dob || null,
        phone_number: phone || null,
        medical_history: history || null,
        notes: notes || null,
      };

      if (editingPatient) {
        await api.put(`/patients/${editingPatient.id}`, payload);
      } else {
        await api.post('/patients', payload);
      }

      setShowModal(false);
      fetchPatients();
    } catch (err: any) {
      alert('Lỗi lưu thông tin bệnh nhân: ' + (err.response?.data?.detail || err.message));
    }
  };

  const handleDelete = async (id: number) => {
    if (window.confirm('Bạn có chắc chắn muốn xóa hồ sơ bệnh nhân này cùng các dữ liệu phân tích liên quan?')) {
      try {
        await api.delete(`/patients/${id}`);
        fetchPatients();
      } catch (err) {
        alert('Lỗi xóa bệnh nhân');
      }
    }
  };

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
            Quản lý Bệnh nhân (Patient Management)
          </h1>
          <p className="text-xs text-slate-500 mt-1">
            Hồ sơ y bạ điện tử lưu trữ lịch sử chụp đáy mắt và kết quả định lượng tổn thương.
          </p>
        </div>
        <button
          onClick={handleOpenCreate}
          className="inline-flex items-center space-x-2 px-4 py-2 bg-sky-600 hover:bg-sky-700 text-white text-xs font-semibold rounded-lg shadow-md shadow-sky-600/20 transition"
        >
          <Plus className="w-4 h-4" />
          <span>Thêm bệnh nhân mới</span>
        </button>
      </div>

      {/* Search Bar */}
      <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex items-center">
        <Search className="w-4 h-4 text-slate-400 mr-2" />
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Tìm kiếm bệnh nhân theo tên, mã bệnh nhân (PAT-...), hoặc số điện thoại..."
          className="w-full text-xs text-slate-800 placeholder-slate-400 focus:outline-none"
        />
      </div>

      {/* Patient Table */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 text-slate-500 uppercase border-b border-slate-200">
              <tr>
                <th className="py-3 px-4 font-semibold">Mã Bệnh nhân</th>
                <th className="py-3 px-4 font-semibold">Họ và tên</th>
                <th className="py-3 px-4 font-semibold">Giới tính</th>
                <th className="py-3 px-4 font-semibold">Ngày sinh</th>
                <th className="py-3 px-4 font-semibold">Tiền sử bệnh lý</th>
                <th className="py-3 px-4 font-semibold text-center">Số ca chụp</th>
                <th className="py-3 px-4 font-semibold text-center">Hành động</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {loading ? (
                <tr>
                  <td colSpan={7} className="py-8 text-center text-slate-400">
                    Đang tải danh sách bệnh nhân...
                  </td>
                </tr>
              ) : patients.length === 0 ? (
                <tr>
                  <td colSpan={7} className="py-8 text-center text-slate-400">
                    Không tìm thấy bệnh nhân nào. Bấm "Thêm bệnh nhân mới" để tạo hồ sơ.
                  </td>
                </tr>
              ) : (
                patients.map((p) => (
                  <tr key={p.id} className="hover:bg-slate-50/80 transition">
                    <td className="py-3 px-4 font-mono font-bold text-sky-700">
                      {p.patient_code}
                    </td>
                    <td className="py-3 px-4 font-semibold text-slate-800">
                      {p.full_name}
                    </td>
                    <td className="py-3 px-4 text-slate-600">
                      {p.gender === 'Male' ? 'Nam' : p.gender === 'Female' ? 'Nữ' : p.gender}
                    </td>
                    <td className="py-3 px-4 text-slate-600 font-mono">
                      {p.date_of_birth || 'N/A'}
                    </td>
                    <td className="py-3 px-4 text-slate-600 max-w-xs truncate" title={p.medical_history}>
                      {p.medical_history || 'Chưa ghi nhận'}
                    </td>
                    <td className="py-3 px-4 text-center font-mono font-semibold text-slate-800">
                      {p.total_analyses || 0}
                    </td>
                    <td className="py-3 px-4 text-center">
                      <div className="flex items-center justify-center space-x-2">
                        <Link
                          to={`/studio?patient_id=${p.id}`}
                          title="Phân tích ảnh cho bệnh nhân này"
                          className="p-1 rounded hover:bg-sky-50 text-sky-600 hover:text-sky-800 transition"
                        >
                          <Activity className="w-4 h-4" />
                        </Link>
                        <button
                          onClick={() => handleOpenEdit(p)}
                          title="Chỉnh sửa thông tin"
                          className="p-1 rounded hover:bg-slate-100 text-slate-500 hover:text-slate-700 transition"
                        >
                          <Edit2 className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => handleDelete(p.id)}
                          title="Xóa hồ sơ"
                          className="p-1 rounded hover:bg-red-50 text-red-500 hover:text-red-700 transition"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Modal Add/Edit Patient */}
      {showModal && (
        <div className="fixed inset-0 z-50 bg-slate-900/50 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl max-w-lg w-full shadow-2xl border border-slate-200 overflow-hidden">
            <div className="bg-gradient-to-r from-sky-600 to-teal-600 px-6 py-4 flex items-center justify-between text-white">
              <h3 className="font-bold text-sm">
                {editingPatient ? 'Chỉnh sửa Hồ sơ Bệnh nhân' : 'Tạo Hồ sơ Bệnh nhân Mới'}
              </h3>
              <button
                onClick={() => setShowModal(false)}
                className="text-white/80 hover:text-white transition"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <form onSubmit={handleSave} className="p-6 space-y-4">
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Họ và tên bệnh nhân <span className="text-red-500">*</span>
                </label>
                <input
                  type="text"
                  required
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  placeholder="Trần Thị B"
                  className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg focus:ring-2 focus:ring-sky-500 focus:outline-none"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">Giới tính</label>
                  <select
                    value={gender}
                    onChange={(e) => setGender(e.target.value)}
                    className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg focus:ring-2 focus:ring-sky-500 focus:outline-none"
                  >
                    <option value="Male">Nam</option>
                    <option value="Female">Nữ</option>
                    <option value="Other">Khác</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">Ngày sinh</label>
                  <input
                    type="date"
                    value={dob}
                    onChange={(e) => setDob(e.target.value)}
                    className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg focus:ring-2 focus:ring-sky-500 focus:outline-none"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Số điện thoại liên lạc</label>
                <input
                  type="text"
                  value={phone}
                  onChange={(e) => setPhone(e.target.value)}
                  placeholder="0905 123 456"
                  className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg focus:ring-2 focus:ring-sky-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Tiền sử bệnh lý (Tiểu đường, Tim mạch, Mắt)</label>
                <textarea
                  rows={2}
                  value={history}
                  onChange={(e) => setHistory(e.target.value)}
                  placeholder="Ghi nhận tiền sử bệnh tiểu đường, thời gian mắc bệnh, thị lực giảm sút..."
                  className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg focus:ring-2 focus:ring-sky-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Ghi chú lâm sàng</label>
                <textarea
                  rows={2}
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  placeholder="Các ghi chú bổ sung..."
                  className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg focus:ring-2 focus:ring-sky-500 focus:outline-none"
                />
              </div>

              <div className="flex justify-end space-x-3 pt-3 border-t border-slate-100">
                <button
                  type="button"
                  onClick={() => setShowModal(false)}
                  className="px-4 py-2 border border-slate-300 text-slate-600 rounded-lg text-xs font-semibold hover:bg-slate-50 transition"
                >
                  Hủy bỏ
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 bg-sky-600 hover:bg-sky-700 text-white rounded-lg text-xs font-semibold shadow-md shadow-sky-600/20 transition"
                >
                  {editingPatient ? 'Lưu thay đổi' : 'Tạo hồ sơ'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
