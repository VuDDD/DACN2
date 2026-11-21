import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import api from '../services/api';
import {
  FileText,
  Download,
  Eye,
  Calendar,
  User,
  Activity,
  CheckCircle2,
  Clock,
  AlertCircle
} from 'lucide-react';

export const Reports: React.FC = () => {
  const [analyses, setAnalyses] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    fetchAnalyses();
  }, []);

  const fetchAnalyses = async () => {
    try {
      const res = await api.get('/analysis?limit=50');
      setAnalyses(res.data.items);
    } catch (e) {
      console.error('Error fetching analyses:', e);
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadPDF = async (analysisId: number) => {
    try {
      const res = await api.post(`/reports/generate/${analysisId}`, {
        doctor_notes: 'Báo cáo định lượng phân vùng tổn thương võng mạc xuất bản tự động.',
      });
      window.open(`/api/reports/${res.data.id}/download`, '_blank');
    } catch (err: any) {
      alert('Lỗi tải báo cáo: ' + (err.response?.data?.detail || err.message));
    }
  };

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
          Hồ sơ Lâm sàng & Báo cáo Xét nghiệm (Medical Reports)
        </h1>
        <p className="text-xs text-slate-500 mt-1">
          Lịch sử kết quả định lượng tổn thương và các phiếu kết quả dạng PDF tiêu chuẩn y khoa.
        </p>
      </div>

      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 text-slate-500 uppercase border-b border-slate-200">
              <tr>
                <th className="py-3 px-4 font-semibold">Mã xét nghiệm</th>
                <th className="py-3 px-4 font-semibold">Nhiệm vụ</th>
                <th className="py-3 px-4 font-semibold">Mô hình AI</th>
                <th className="py-3 px-4 font-semibold text-center">Trạng thái</th>
                <th className="py-3 px-4 font-semibold text-right">Tỷ lệ tổn thương</th>
                <th className="py-3 px-4 font-semibold text-right">Thời gian tạo</th>
                <th className="py-3 px-4 font-semibold text-center">Tùy chọn</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {loading ? (
                <tr>
                  <td colSpan={7} className="py-8 text-center text-slate-400">
                    Đang tải danh sách hồ sơ phân tích...
                  </td>
                </tr>
              ) : analyses.length === 0 ? (
                <tr>
                  <td colSpan={7} className="py-8 text-center text-slate-400">
                    Chưa có hồ sơ xét nghiệm nào trong hệ thống.
                  </td>
                </tr>
              ) : (
                analyses.map((a) => (
                  <tr key={a.id} className="hover:bg-slate-50/80 transition">
                    <td className="py-3 px-4 font-mono font-bold text-sky-700">
                      #{a.id}
                    </td>
                    <td className="py-3 px-4 font-medium text-slate-800">
                      {a.task_type === 'IDRID_LESIONS' ? 'Tổn thương võng mạc (IDRiD)' : 'Cây mạch máu (DRIVE)'}
                    </td>
                    <td className="py-3 px-4 text-slate-600">
                      {a.model_name}
                    </td>
                    <td className="py-3 px-4 text-center">
                      <span
                        className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-semibold ${
                          a.analysis_status === 'COMPLETED'
                            ? 'bg-emerald-100 text-emerald-800'
                            : a.analysis_status === 'FAILED'
                            ? 'bg-red-100 text-red-800'
                            : 'bg-amber-100 text-amber-800'
                        }`}
                      >
                        {a.analysis_status}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-right font-mono font-bold text-slate-800">
                      {a.lesion_percentage ? `${a.lesion_percentage.toFixed(3)}%` : '0.000%'}
                    </td>
                    <td className="py-3 px-4 text-right text-slate-500 font-mono">
                      {new Date(a.created_at).toLocaleString('vi-VN')}
                    </td>
                    <td className="py-3 px-4 text-center">
                      <div className="flex items-center justify-center space-x-2">
                        <Link
                          to={`/studio?analysis_id=${a.id}`}
                          title="Xem trực quan trên Canvas"
                          className="p-1.5 rounded hover:bg-sky-50 text-sky-600 hover:text-sky-800 transition"
                        >
                          <Eye className="w-4 h-4" />
                        </Link>
                        {a.analysis_status === 'COMPLETED' && (
                          <button
                            onClick={() => handleDownloadPDF(a.id)}
                            title="Tải báo cáo y khoa PDF"
                            className="p-1.5 rounded hover:bg-emerald-50 text-emerald-600 hover:text-emerald-800 transition"
                          >
                            <Download className="w-4 h-4" />
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

