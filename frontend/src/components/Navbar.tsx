import React from 'react';
import { useAuth } from '../context/AuthContext';
import { Activity, User, LogOut, ShieldAlert } from 'lucide-react';

export const Navbar: React.FC = () => {
  const { user, logout } = useAuth();

  return (
    <header className="bg-white border-b border-slate-200 sticky top-0 z-30 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand & Institution */}
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-sky-600 to-teal-500 flex items-center justify-center text-white shadow-md shadow-sky-500/20">
            <Activity className="w-6 h-6" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-bold text-slate-900 text-lg tracking-tight">RETINA AI</span>
              <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-sky-100 text-sky-800 border border-sky-200">
                VKU Medical Lab
              </span>
            </div>
            <p className="text-xs text-slate-500 hidden sm:block">
              Nền tảng Phân vùng & Định lượng Tổn thương Võng mạc
            </p>
          </div>
        </div>

        {/* User Session & Logout */}
        <div className="flex items-center space-x-4">
          {user && (
            <div className="flex items-center space-x-3 bg-slate-50 border border-slate-200 py-1.5 px-3 rounded-lg">
              <div className="w-8 h-8 rounded-full bg-sky-600 text-white flex items-center justify-center font-bold text-xs">
                {user.full_name ? user.full_name.charAt(0).toUpperCase() : 'D'}
              </div>
              <div className="text-left hidden md:block">
                <p className="text-xs font-semibold text-slate-800">{user.full_name || user.username}</p>
                <p className="text-[10px] text-sky-600 font-medium capitalize">{user.role}</p>
              </div>
              <button
                onClick={logout}
                title="Đăng xuất"
                className="text-slate-400 hover:text-red-600 transition-colors ml-2 p-1 rounded-md hover:bg-slate-200"
              >
                <LogOut className="w-4 h-4" />
              </button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
};
