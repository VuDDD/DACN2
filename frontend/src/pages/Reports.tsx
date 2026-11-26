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
  AlertCircle,
  Search
} from 'lucide-react';

export const Reports: React.FC = () => {
  const [analyses, setAnalyses] = useState<any[]>([]);
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    fetchAnalyses();
  }, []);

  const fetchAnalyses = async () => {
    setErrorMsg(null);
    try {
      const res = await api.get('/analysis?limit=100');
      setAnalyses(res.data.items || []);
    } catch (e: any) {
      console.error('Error fetching analyses:', e);
      setErrorMsg(
        e.response?.data?.detail ||
          'Không tải được danh sách hồ sơ bệnh nhân từ máy chủ Backend.'
      );
    } finally {
      setLoading(false);
    }
  };

  const triggerBlobDownload = (blobData: Blob, filename: string) => {
    const url = window.URL.createObjectURL(blobData);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', filename);
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
  };

  const handleDownloadPDF = async (analysisId: number) => {
    try {
      const res = await api.post(`/reports/generate/${analysisId}`, {
        doctor_notes: 'Báo cáo định lượng phân vùng tổn thương võng mạc xuất bản tự động.',
      });
      const pdfRes = await api.get(`/reports/${res.data.id}/download`, {
        responseType: 'blob',
      });
      triggerBlobDownload(
        new Blob([pdfRes.data], { type: 'application/pdf' }),
        `${res.data.report_code}.pdf`
      );
    } catch (err: any) {
      alert('Lỗi tải báo cáo PDF: ' + (err.response?.data?.detail || err.message));
    }
  };

  const handleDownloadExcel = async (analysisId: number) => {
    try {
      const res = await api.post(
        `/reports/export-excel/${analysisId}`,
        { doctor_notes: 'Báo cáo định lượng phân vùng tổn thương võng mạc xuất bản tự động.' },
        { responseType: 'blob' }
      );
      const reportCode = res.headers['x-report-code'] || `REP-${analysisId}`;
      triggerBlobDownload(
        new Blob([res.data], {
          type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        }),
        `${reportCode}.xlsx`
      );
    } catch (err: any) {
      alert('Lỗi tải báo cáo Excel: ' + (err.response?.data?.detail || err.message));
    }
  };

  const handleExportAllExcel = async () => {
    try {
      const res = await api.get('/reports/export-all-excel', {
        responseType: 'blob',
      });
      triggerBlobDownload(
        new Blob([res.data], {
          type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        }),
        'RetinaAI_Patients_Diagnoses.xlsx'
      );
    } catch (err: any) {
      alert('Lỗi xuất tổng hợp Excel: ' + (err.response?.data?.detail || err.message));
    }
  };

  const filteredAnalyses = analyses.filter((a) => {
    if (!searchTerm.trim()) return true;
    const q = searchTerm.trim().toLowerCase();
    return (
      String(a.id).includes(q) ||
      (a.patient_code || '').toLowerCase().includes(q) ||
      (a.patient_name || '').toLowerCase().includes(q) ||
      (a.task_type || '').toLowerCase().includes(q) ||
      (a.model_name || '').toLowerCase().includes(q)
    );
  });

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
            Hồ sơ Lâm sàng & Báo cáo Xét nghiệm (Medical Reports)
          </h1>
          <p className="text-xs text-slate-500 mt-1">
            Danh sách bệnh nhân kèm lịch sử kết quả định lượng tổn thương và phiếu kết quả PDF / Excel.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <div className="relative">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Tìm tên, mã BN hoặc mã XN..."
              className="pl-9 pr-3 py-2 text-xs bg-white border border-slate-300 rounded-lg focus:ring-2 focus:ring-sky-500 focus:outline-none w-56"
            />
          </div>
          <button
            onClick={handleExportAllExcel}
            className="inline-flex items-center space-x-2 px-4 py-2 bg-sky-600 hover:bg-sky-700 text-white text-xs font-semibold rounded-lg shadow-md shadow-sky-600/20 transition"
          >
            <FileText className="w-4 h-4" />
            <span>Xuất toàn bộ ra Excel (.xlsx)</span>
          </button>
        </div>
      </div>

      {errorMsg && (
        <div className="p-3 rounded-xl bg-red-50 border border-red-200 text-red-700 text-xs flex items-center space-x-2">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 text-slate-500 uppercase border-b border-slate-200">
              <tr>
                <th className="py-3 px-4 font-semibold">Mã xét nghiệm</th>
                <th className="py-3 px-4 font-semibold">Bệnh nhân</th>
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
                  <td colSpan={8} className="py-8 text-center text-slate-400">
                    Đang tải danh sách hồ sơ phân tích...
                  </td>
                </tr>
              ) : filteredAnalyses.length === 0 ? (
                <tr>
                  <td colSpan={8} className="py-8 text-center text-slate-400">
                    Chưa có hồ sơ xét nghiệm nào phù hợp trong hệ thống.
                  </td>
                </tr>
              ) : (
                filteredAnalyses.map((a) => (
                  <tr key={a.id} className="hover:bg-slate-50/80 transition">
                    <td className="py-3 px-4 font-mono font-bold text-sky-700">
                      #{a.id}
                    </td>
                    <td className="py-3 px-4">
                      <div className="font-semibold text-slate-800">
                        {a.patient_name || `Bệnh nhân #${a.patient_id}`}
                      </div>
                      <div className="text-[10px] font-mono text-slate-500">
                        {a.patient_code || `ID-${a.patient_id}`}
                      </div>
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
                          <>
                            <button
                              onClick={() => handleDownloadPDF(a.id)}
                              title="Tải báo cáo y khoa PDF"
                              className="p-1.5 rounded hover:bg-emerald-50 text-emerald-600 hover:text-emerald-800 transition"
                            >
                              <Download className="w-4 h-4" />
                            </button>
                            <button
                              onClick={() => handleDownloadExcel(a.id)}
                              title="Tải báo cáo Excel (.xlsx)"
                              className="p-1.5 rounded hover:bg-sky-50 text-sky-600 hover:text-sky-800 transition"
                            >
                              <FileText className="w-4 h-4" />
                            </button>
                          </>
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

