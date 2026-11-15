import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import api from '../services/api';
import {
  Users,
  Activity,
  CheckCircle2,
  Clock,
  AlertTriangle,
  ArrowUpRight,
  TrendingUp,
  BarChart2,
  FileSpreadsheet
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  PieChart,
  Pie
} from 'recharts';

export const Dashboard: React.FC = () => {
  const [stats, setStats] = useState<any>(null);
  const [lesionDist, setLesionDist] = useState<any[]>([]);
  const [modelPerf, setModelPerf] = useState<any[]>([]);
  const [recentAnalyses, setRecentAnalyses] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      const [statsRes, distRes, perfRes, recentRes] = await Promise.all([
        api.get('/dashboard/statistics'),
        api.get('/dashboard/lesion-distribution'),
        api.get('/dashboard/model-performance'),
        api.get('/dashboard/recent-analyses'),
      ]);

      setStats(statsRes.data);
      setLesionDist(distRes.data);
      setModelPerf(perfRes.data);
      setRecentAnalyses(recentRes.data);
    } catch (err) {
      console.error('Error fetching dashboard data:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="p-8 flex items-center justify-center min-h-[400px]">
        <div className="flex items-center space-x-3 text-sky-600">
          <Activity className="w-6 h-6 animate-spin" />
          <span className="font-semibold text-slate-700">Đang tải dữ liệu lâm sàng...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      {/* Title & Welcome */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
            Bảng điều khiển Trung tâm (Dashboard)
          </h1>
          <p className="text-xs text-slate-500 mt-1">
            Theo dõi tổng quan hoạt động phân vùng và định lượng tổn thương võng mạc tiểu đường.
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <Link
            to="/studio"
            className="inline-flex items-center space-x-2 px-4 py-2 bg-sky-600 hover:bg-sky-700 text-white text-xs font-semibold rounded-lg shadow-md shadow-sky-600/20 transition"
          >
            <Activity className="w-4 h-4" />
            <span>Phân tích ảnh mới</span>
          </Link>
        </div>
      </div>

      {/* Metric Stat Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm flex items-center justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">Tổng Bệnh nhân</p>
            <p className="text-2xl font-bold text-slate-900 mt-1">{stats?.total_patients || 0}</p>
            <p className="text-[11px] text-emerald-600 font-medium mt-1 flex items-center">
              <TrendingUp className="w-3 h-3 mr-1" /> Quản lý hồ sơ bệnh án
            </p>
          </div>
          <div className="w-12 h-12 rounded-xl bg-sky-50 text-sky-600 flex items-center justify-center">
            <Users className="w-6 h-6" />
          </div>
        </div>

        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm flex items-center justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">Ảnh đã phân tích</p>
            <p className="text-2xl font-bold text-slate-900 mt-1">{stats?.total_analyses || 0}</p>
            <p className="text-[11px] text-sky-600 font-medium mt-1">
              {stats?.completed_analyses || 0} ca hoàn thành
            </p>
          </div>
          <div className="w-12 h-12 rounded-xl bg-teal-50 text-teal-600 flex items-center justify-center">
            <Activity className="w-6 h-6" />
          </div>
        </div>

        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm flex items-center justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">Ổ tổn thương phát hiện</p>
            <p className="text-2xl font-bold text-slate-900 mt-1">{stats?.total_lesions_detected || 0}</p>
            <p className="text-[11px] text-amber-600 font-medium mt-1">
              Connected Components
            </p>
          </div>
          <div className="w-12 h-12 rounded-xl bg-amber-50 text-amber-600 flex items-center justify-center">
            <AlertTriangle className="w-6 h-6" />
          </div>
        </div>

        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm flex items-center justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">Thời gian xử lý TB</p>
            <p className="text-2xl font-bold text-slate-900 mt-1">
              {stats?.average_processing_time_ms ? `${Math.round(stats.average_processing_time_ms)} ms` : 'N/A'}
            </p>
            <p className="text-[11px] text-slate-400 font-medium mt-1">
              GPU CUDA Acceleration
            </p>
          </div>
          <div className="w-12 h-12 rounded-xl bg-indigo-50 text-indigo-600 flex items-center justify-center">
            <Clock className="w-6 h-6" />
          </div>
        </div>
      </div>

      {/* Visual Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Lesion Distribution */}
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-sm font-bold text-slate-800">Phân bố số lượng ổ tổn thương (IDRiD & DRIVE)</h2>
              <p className="text-xs text-slate-500">Thống kê số lượng vùng liên thông theo từng loại tổn thương</p>
            </div>
            <BarChart2 className="w-5 h-5 text-slate-400" />
          </div>

          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={lesionDist} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="label" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', color: '#fff', fontSize: '11px' }}
                />
                <Bar dataKey="total_count" name="Số lượng ổ tổn thương" radius={[4, 4, 0, 0]}>
                  {lesionDist.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* AI Model Performance Comparison */}
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-sm font-bold text-slate-800">Hiệu năng Mô hình (U-Net vs Attention U-Net)</h2>
              <p className="text-xs text-slate-500">Chỉ số thực nghiệm Dice & IoU so với Ground Truth chuẩn</p>
            </div>
            <Activity className="w-5 h-5 text-sky-600" />
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-50 text-slate-500 uppercase border-b border-slate-200">
                <tr>
                  <th className="py-2.5 px-3 font-semibold">Mô hình AI</th>
                  <th className="py-2.5 px-3 font-semibold text-center">Mean Dice</th>
                  <th className="py-2.5 px-3 font-semibold text-center">Mean IoU</th>
                  <th className="py-2.5 px-3 font-semibold text-center">Precision</th>
                  <th className="py-2.5 px-3 font-semibold text-center">Recall</th>
                  <th className="py-2.5 px-3 font-semibold text-center">Mẫu test</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {modelPerf.map((m) => (
                  <tr key={m.model_name} className="hover:bg-slate-50/80 transition">
                    <td className="py-3 px-3 font-semibold text-slate-800">
                      {m.model_name === 'Attention_UNet' ? 'Attention U-Net (with AG)' : 'Baseline U-Net'}
                    </td>
                    <td className="py-3 px-3 text-center font-mono font-bold text-sky-600">
                      {m.mean_dice > 0 ? m.mean_dice.toFixed(4) : 'Đang huấn luyện'}
                    </td>
                    <td className="py-3 px-3 text-center font-mono text-slate-700">
                      {m.mean_iou > 0 ? m.mean_iou.toFixed(4) : '—'}
                    </td>
                    <td className="py-3 px-3 text-center font-mono text-slate-700">
                      {m.mean_precision > 0 ? m.mean_precision.toFixed(4) : '—'}
                    </td>
                    <td className="py-3 px-3 text-center font-mono text-slate-700">
                      {m.mean_recall > 0 ? m.mean_recall.toFixed(4) : '—'}
                    </td>
                    <td className="py-3 px-3 text-center font-mono text-slate-500">
                      {m.sample_count}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="p-3 bg-sky-50 border border-sky-100 rounded-lg text-[11px] text-sky-800 leading-relaxed">
            <b>Lưu ý học thuật:</b> Theo nguyên tắc nghiên cứu y khoa của đề tài, các chỉ số trên phản ánh kết quả thực nghiệm thực tế, hoàn toàn không sử dụng số liệu mô phỏng.
          </div>
        </div>
      </div>

      {/* Recent Activity Table */}
      <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-sm font-bold text-slate-800">Lịch sử phân tích gần đây</h2>
            <p className="text-xs text-slate-500">Danh sách các ca chụp đáy mắt đã được đưa vào hệ thống</p>
          </div>
          <Link to="/reports" className="text-xs font-semibold text-sky-600 hover:text-sky-700 flex items-center">
            <span>Xem toàn bộ hồ sơ</span>
            <ArrowUpRight className="w-4 h-4 ml-0.5" />
          </Link>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 text-slate-500 uppercase border-b border-slate-200">
              <tr>
                <th className="py-2.5 px-3 font-semibold">Mã ca (#)</th>
                <th className="py-2.5 px-3 font-semibold">Tác vụ</th>
                <th className="py-2.5 px-3 font-semibold">Mô hình AI</th>
                <th className="py-2.5 px-3 font-semibold text-center">Trạng thái</th>
                <th className="py-2.5 px-3 font-semibold text-right">Tỷ lệ tổn thương (FOV %)</th>
                <th className="py-2.5 px-3 font-semibold text-right">Thời gian</th>
                <th className="py-2.5 px-3 font-semibold text-center">Hành động</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {recentAnalyses.length === 0 ? (
                <tr>
                  <td colSpan={7} className="py-8 text-center text-slate-400">
                    Chưa có ca phân tích nào. Hãy tải ảnh võng mạc lên để bắt đầu!
                  </td>
                </tr>
              ) : (
                recentAnalyses.map((item) => (
                  <tr key={item.id} className="hover:bg-slate-50/80 transition">
                    <td className="py-3 px-3 font-mono font-medium text-slate-800">
                      #{item.id}
                    </td>
                    <td className="py-3 px-3 font-medium text-slate-700">
                      {item.task_type === 'IDRID_LESIONS' ? 'Tổn thương võng mạc (IDRiD)' : 'Cây mạch máu (DRIVE)'}
                    </td>
                    <td className="py-3 px-3 text-slate-600">
                      {item.model_name}
                    </td>
                    <td className="py-3 px-3 text-center">
                      <span
                        className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-semibold ${
                          item.analysis_status === 'COMPLETED'
                            ? 'bg-emerald-100 text-emerald-800'
                            : item.analysis_status === 'FAILED'
                            ? 'bg-red-100 text-red-800'
                            : 'bg-amber-100 text-amber-800'
                        }`}
                      >
                        {item.analysis_status}
                      </span>
                    </td>
                    <td className="py-3 px-3 text-right font-mono font-semibold text-slate-800">
                      {item.lesion_percentage ? `${item.lesion_percentage.toFixed(3)}%` : '0.000%'}
                    </td>
                    <td className="py-3 px-3 text-right text-slate-500 font-mono">
                      {new Date(item.created_at).toLocaleDateString('vi-VN')}
                    </td>
                    <td className="py-3 px-3 text-center">
                      <Link
                        to={`/studio?analysis_id=${item.id}`}
                        className="text-xs font-semibold text-sky-600 hover:text-sky-800 inline-flex items-center"
                      >
                        Xem chi tiết
                      </Link>
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

