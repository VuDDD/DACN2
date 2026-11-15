import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import api from '../services/api';
import { Activity, Lock, User, Mail, ShieldCheck, AlertCircle } from 'lucide-react';

export const Login: React.FC = () => {
  const navigate = useNavigate();
  const { login } = useAuth();

  const [isRegister, setIsRegister] = useState<boolean>(false);
  const [username, setUsername] = useState<string>('admin');
  const [password, setPassword] = useState<string>('admin123');
  const [email, setEmail] = useState<string>('');
  const [fullName, setFullName] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      if (isRegister) {
        // Register new account
        await api.post('/auth/register', {
          username,
          email,
          password,
          full_name: fullName,
          role: 'DOCTOR',
        });
        // Auto-login after registration
        const res = await api.post('/auth/login-json', { username, password });
        login(res.data.access_token, res.data.user);
        navigate('/');
      } else {
        // Login
        const res = await api.post('/auth/login-json', { username, password });
        login(res.data.access_token, res.data.user);
        navigate('/');
      }
    } catch (err: any) {
      setError(
        err.response?.data?.detail || 'Đăng nhập không thành công. Vui lòng kiểm tra lại thông tin.'
      );
    } finally {
      setLoading(false);
    }
  };

  const handleFillDemo = () => {
    setIsRegister(false);
    setUsername('admin');
    setPassword('admin123');
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-sky-950 flex items-center justify-center p-4">
      <div className="max-w-md w-full bg-white rounded-2xl shadow-2xl overflow-hidden border border-slate-100">
        {/* Banner Header */}
        <div className="bg-gradient-to-r from-sky-600 to-teal-600 p-6 text-white text-center relative">
          <div className="w-14 h-14 mx-auto rounded-2xl bg-white/20 backdrop-blur-md flex items-center justify-center mb-3 shadow-inner">
            <Activity className="w-8 h-8 text-white" />
          </div>
          <h2 className="text-xl font-bold tracking-tight">RETINA MEDICAL AI PLATFORM</h2>
          <p className="text-xs text-sky-100 mt-1">
            Trường Đại học Công nghệ Thông tin & Truyền thông Việt - Hàn (VKU)
          </p>
        </div>

        {/* Tab Selection */}
        <div className="flex border-b border-slate-200 bg-slate-50">
          <button
            type="button"
            onClick={() => { setIsRegister(false); setError(null); }}
            className={`flex-1 py-3 text-xs font-semibold tracking-wide transition ${
              !isRegister
                ? 'bg-white text-sky-600 border-b-2 border-sky-600'
                : 'text-slate-500 hover:text-slate-800'
            }`}
          >
            ĐĂNG NHẬP (DOCTOR / ADMIN)
          </button>
          <button
            type="button"
            onClick={() => { setIsRegister(true); setError(null); }}
            className={`flex-1 py-3 text-xs font-semibold tracking-wide transition ${
              isRegister
                ? 'bg-white text-sky-600 border-b-2 border-sky-600'
                : 'text-slate-500 hover:text-slate-800'
            }`}
          >
            ĐĂNG KÝ MỚI
          </button>
        </div>

        {/* Form Body */}
        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          {error && (
            <div className="p-3 rounded-lg bg-red-50 border border-red-200 text-red-700 text-xs flex items-center space-x-2">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {isRegister && (
            <>
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Họ và tên bác sĩ</label>
                <div className="relative">
                  <User className="w-4 h-4 text-slate-400 absolute left-3 top-3" />
                  <input
                    type="text"
                    required
                    value={fullName}
                    onChange={(e) => setFullName(e.target.value)}
                    placeholder="BS. Nguyễn Văn A"
                    className="w-full pl-9 pr-3 py-2 text-xs border border-slate-300 rounded-lg focus:ring-2 focus:ring-sky-500 focus:outline-none"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Địa chỉ Email</label>
                <div className="relative">
                  <Mail className="w-4 h-4 text-slate-400 absolute left-3 top-3" />
                  <input
                    type="email"
                    required
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="doctor@hospital.vku.vn"
                    className="w-full pl-9 pr-3 py-2 text-xs border border-slate-300 rounded-lg focus:ring-2 focus:ring-sky-500 focus:outline-none"
                  />
                </div>
              </div>
            </>
          )}

          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">Tên tài khoản</label>
            <div className="relative">
              <User className="w-4 h-4 text-slate-400 absolute left-3 top-3" />
              <input
                type="text"
                required
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="admin hoặc mã định danh"
                className="w-full pl-9 pr-3 py-2 text-xs border border-slate-300 rounded-lg focus:ring-2 focus:ring-sky-500 focus:outline-none"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">Mật khẩu</label>
            <div className="relative">
              <Lock className="w-4 h-4 text-slate-400 absolute left-3 top-3" />
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full pl-9 pr-3 py-2 text-xs border border-slate-300 rounded-lg focus:ring-2 focus:ring-sky-500 focus:outline-none"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full py-2.5 px-4 bg-sky-600 hover:bg-sky-700 text-white font-semibold text-xs rounded-lg shadow-md shadow-sky-600/30 transition duration-150 disabled:opacity-50"
          >
            {loading ? 'Đang xử lý...' : isRegister ? 'HOÀN TẤT ĐĂNG KÝ' : 'ĐĂNG NHẬP VÀO HỆ THỐNG'}
          </button>

          {!isRegister && (
            <div className="pt-3 border-t border-slate-100">
              <button
                type="button"
                onClick={handleFillDemo}
                className="w-full py-2 px-3 border border-dashed border-sky-300 bg-sky-50 text-sky-700 text-xs font-medium rounded-lg hover:bg-sky-100 transition flex items-center justify-center space-x-2"
              >
                <ShieldCheck className="w-4 h-4 text-sky-600" />
                <span>Nạp tài khoản kiểm thử: admin / admin123</span>
              </button>
            </div>
          )}
        </form>

        {/* Footer Disclaimer */}
        <div className="p-3 bg-slate-50 text-[10px] text-slate-400 text-center border-t border-slate-100">
          Đề tài Đồ án chuyên ngành 2 • Hệ thống hỗ trợ nghiên cứu & phân tích ảnh y sinh
        </div>
      </div>
    </div>
  );
};
