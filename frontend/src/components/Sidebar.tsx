import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Users,
  Eye,
  BarChart3,
  FileText,
  HelpCircle
} from 'lucide-react';

export const Sidebar: React.FC = () => {
  const links = [
    { to: '/', label: 'Tổng quan (Dashboard)', icon: LayoutDashboard },
    { to: '/patients', label: 'Bệnh nhân (Patients)', icon: Users },
    { to: '/studio', label: 'Phòng phân tích (Studio)', icon: Eye },
    { to: '/reports', label: 'Hồ sơ & Báo cáo (Reports)', icon: FileText },
  ];

  return (
    <aside className="w-64 bg-slate-900 text-slate-300 flex flex-col justify-between shrink-0 min-h-[calc(100vh-4rem)]">
      <div className="py-6 px-4 space-y-6">
        <div className="px-3">
          <p className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
            Hệ thống Chuyên ngành 2
          </p>
        </div>

        <nav className="space-y-1.5">
          {links.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === '/'}
                className={({ isActive }) =>
                  `flex items-center space-x-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all ${
                    isActive
                      ? 'bg-sky-600 text-white shadow-md shadow-sky-600/30'
                      : 'hover:bg-slate-800 hover:text-white'
                  }`
                }
              >
                <Icon className="w-5 h-5 shrink-0" />
                <span>{item.label}</span>
              </NavLink>
            );
          })}
        </nav>
      </div>

      {/* Dataset & VKU Info Widget */}
      <div className="p-4 m-4 rounded-xl bg-slate-800/80 border border-slate-700/60 text-xs space-y-2">
        <div className="flex items-center space-x-2 text-sky-400 font-semibold">
          <HelpCircle className="w-4 h-4" />
          <span>Datasets Tích hợp</span>
        </div>
        <p className="text-slate-400 text-[11px] leading-relaxed">
          • <b>IDRiD:</b> MA, HE, EX, SE, OD<br/>
          • <b>DRIVE:</b> Cây mạch máu trong FOV
        </p>
        <div className="pt-2 border-t border-slate-700 text-[10px] text-slate-400">
          VKU • Trường ĐH CNTT&TT Việt - Hàn
        </div>
      </div>
    </aside>
  );
};
