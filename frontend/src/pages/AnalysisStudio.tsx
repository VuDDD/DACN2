import React, { useState, useEffect, useRef } from 'react';
import { useSearchParams } from 'react-router-dom';
import api from '../services/api';
import { CanvasViewer } from '../components/CanvasViewer';
import {
  Upload,
  Activity,
  Layers,
  FileText,
  Download,
  CheckCircle2,
  AlertCircle,
  Clock,
  Sparkles,
  Info,
  ChevronRight,
  RefreshCw,
  FolderOpen
} from 'lucide-react';

export const AnalysisStudio: React.FC = () => {
  const [searchParams] = useSearchParams();
  const initialAnalysisId = searchParams.get('analysis_id');
  const initialPatientId = searchParams.get('patient_id');

  // Input States
  const [patients, setPatients] = useState<any[]>([]);
  const [selectedPatientId, setSelectedPatientId] = useState<number | string>(initialPatientId || '');
  const [taskType, setTaskType] = useState<string>('IDRID_LESIONS');
  const [modelName, setModelName] = useState<string>('Attention_UNet');
  const [sampleImages, setSampleImages] = useState<any[]>([]);

  // Selected / Uploaded Image
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageMeta, setImageMeta] = useState<any | null>(null);
  const [groundTruthUrl, setGroundTruthUrl] = useState<string | null>(null);

  // Analysis Lifecycle
  const [analysisId, setAnalysisId] = useState<number | null>(initialAnalysisId ? parseInt(initialAnalysisId) : null);
  const [analysisResult, setAnalysisResult] = useState<any | null>(null);
  const [status, setStatus] = useState<string>('IDLE'); // IDLE, PENDING, PROCESSING, COMPLETED, FAILED
  const [progress, setProgress] = useState<number>(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Report Generation
  const [generatingReport, setGeneratingReport] = useState<boolean>(false);
  const [reportUrl, setReportUrl] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Fetch initial data
  useEffect(() => {
    fetchPatients();
    fetchSampleImages();
    if (initialAnalysisId) {
      loadExistingAnalysis(parseInt(initialAnalysisId));
    }
  }, [initialAnalysisId]);

  const fetchPatients = async () => {
    try {
      const res = await api.get('/patients?limit=50');
      setPatients(res.data.items);
      if (!selectedPatientId && res.data.items.length > 0) {
        setSelectedPatientId(res.data.items[0].id);
      }
    } catch (e) {
      console.error('Error loading patients:', e);
    }
  };

  const fetchSampleImages = async () => {
    try {
      const res = await api.get('/images/samples');
      setSampleImages(res.data);
    } catch (e) {
      console.error('Error loading samples:', e);
    }
  };

  const loadExistingAnalysis = async (id: number) => {
    try {
      const res = await api.get(`/analysis/${id}`);
      setAnalysisResult(res.data);
      setAnalysisId(res.data.id);
      setImageUrl(res.data.original_image_url);
      setGroundTruthUrl(res.data.ground_truth_mask_url);
      setTaskType(res.data.task_type);
      setModelName(res.data.model_name);
      setStatus(res.data.analysis_status);
      setSelectedPatientId(res.data.patient_id);
    } catch (e) {
      console.error('Error loading analysis:', e);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await api.post('/images/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setImageUrl(res.data.url);
      setImageMeta(res.data);
      setAnalysisResult(null);
      setAnalysisId(null);
      setStatus('IDLE');
    } catch (err: any) {
      alert('Lỗi tải ảnh lên: ' + (err.response?.data?.detail || err.message));
    }
  };

  const handleSelectSample = async (sample: any) => {
    try {
      const res = await api.post('/images/load-sample', null, {
        params: {
          dataset: sample.dataset,
          filename: sample.filename,
        },
      });
      setImageUrl(res.data.url);
      setImageMeta(res.data);
      setTaskType(sample.task_type);
      setAnalysisResult(null);
      setAnalysisId(null);
      setStatus('IDLE');
    } catch (err: any) {
      alert('Lỗi tải ảnh mẫu benchmark: ' + (err.response?.data?.detail || err.message));
    }
  };

  // Trigger Prediction
  const handleStartAnalysis = async () => {
    if (!imageUrl || !selectedPatientId) {
      alert('Vui lòng chọn bệnh nhân và ảnh võng mạc trước khi phân tích.');
      return;
    }

    setErrorMessage(null);
    setStatus('PENDING');
    setProgress(15);

    try {
      const payload = {
        patient_id: parseInt(selectedPatientId.toString()),
        image_url: imageUrl,
        task_type: taskType,
        model_name: modelName,
        ground_truth_url: groundTruthUrl,
      };

      const res = await api.post('/analysis/predict', payload);
      const newId = res.data.id;
      setAnalysisId(newId);

      // Start polling status
      pollAnalysisStatus(newId);
    } catch (err: any) {
      setStatus('FAILED');
      setErrorMessage(err.response?.data?.detail || 'Lỗi điều phối tác vụ phân tích AI.');
    }
  };

  const pollAnalysisStatus = (id: number) => {
    const interval = setInterval(async () => {
      try {
        const res = await api.get(`/analysis/${id}/status`);
        setStatus(res.data.status);
        setProgress(res.data.progress || 50);

        if (res.data.completed) {
          clearInterval(interval);
          if (res.data.status === 'COMPLETED') {
            const resultRes = await api.get(`/analysis/${id}/result`);
            setAnalysisResult(resultRes.data);
            setProgress(100);
          } else {
            setErrorMessage(res.data.message || 'Phân tích thất bại.');
          }
        }
      } catch (err) {
        clearInterval(interval);
        setStatus('FAILED');
        setErrorMessage('Lỗi kết nối kiểm tra tiến trình phân tích.');
      }
    }, 1500);
  };

  // Export PDF Report
  const handleExportPDF = async () => {
    if (!analysisId) return;
    setGeneratingReport(true);
    try {
      const res = await api.post(`/reports/generate/${analysisId}`, {
        doctor_notes: 'Hình ảnh tổn thương võng mạc và đĩa thị được định lượng tự động bằng mô hình Deep Learning.',
      });
      const downloadUrl = `/api/reports/${res.data.id}/download`;
      window.open(downloadUrl, '_blank');
    } catch (err: any) {
      alert('Lỗi xuất báo cáo PDF: ' + (err.response?.data?.detail || err.message));
    } finally {
      setGeneratingReport(false);
    }
  };

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
            Phòng Phân tích Y sinh (Analysis Studio)
          </h1>
          <p className="text-xs text-slate-500 mt-1">
            Môi trường tương tác đa lớp phân vùng tổn thương vi mô (IDRiD) và hệ thống mạch máu (DRIVE).
          </p>
        </div>
        {analysisResult && (
          <button
            onClick={handleExportPDF}
            disabled={generatingReport}
            className="inline-flex items-center space-x-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold rounded-lg shadow-md shadow-emerald-600/20 transition disabled:opacity-50"
          >
            <Download className="w-4 h-4" />
            <span>{generatingReport ? 'Đang kết xuất PDF...' : 'Xuất Báo cáo Y khoa (PDF)'}</span>
          </button>
        )}
      </div>

      {/* Main Studio Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Configuration & Upload (4 cols) */}
        <div className="lg:col-span-4 space-y-5">
          {/* Patient Selector */}
          <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm space-y-3">
            <label className="block text-xs font-bold uppercase tracking-wider text-slate-700">
              1. Chọn Bệnh nhân
            </label>
            <select
              value={selectedPatientId}
              onChange={(e) => setSelectedPatientId(e.target.value)}
              className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg focus:ring-2 focus:ring-sky-500 focus:outline-none"
            >
              {patients.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.patient_code} — {p.full_name} ({p.gender === 'Male' ? 'Nam' : 'Nữ'})
                </option>
              ))}
            </select>
          </div>

          {/* Task & AI Model Selection */}
          <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm space-y-3">
            <label className="block text-xs font-bold uppercase tracking-wider text-slate-700">
              2. Cấu hình Nhiệm vụ & Mô hình AI
            </label>

            <div>
              <span className="text-[11px] font-semibold text-slate-500">Nhiệm vụ phân tích:</span>
              <div className="grid grid-cols-2 gap-2 mt-1">
                <button
                  type="button"
                  onClick={() => setTaskType('IDRID_LESIONS')}
                  className={`py-2 px-2.5 rounded-lg text-xs font-semibold border text-left transition ${
                    taskType === 'IDRID_LESIONS'
                      ? 'bg-sky-50 border-sky-500 text-sky-800'
                      : 'border-slate-200 text-slate-600 hover:bg-slate-50'
                  }`}
                >
                  <div className="font-bold">IDRiD Lesions</div>
                  <div className="text-[10px] text-slate-400">MA, HE, EX, SE, OD</div>
                </button>
                <button
                  type="button"
                  onClick={() => setTaskType('DRIVE_VESSELS')}
                  className={`py-2 px-2.5 rounded-lg text-xs font-semibold border text-left transition ${
                    taskType === 'DRIVE_VESSELS'
                      ? 'bg-sky-50 border-sky-500 text-sky-800'
                      : 'border-slate-200 text-slate-600 hover:bg-slate-50'
                  }`}
                >
                  <div className="font-bold">DRIVE Vessels</div>
                  <div className="text-[10px] text-slate-400">Cây mạch máu / FOV</div>
                </button>
              </div>
            </div>

            <div>
              <span className="text-[11px] font-semibold text-slate-500">Kiến trúc Deep Learning:</span>
              <div className="grid grid-cols-2 gap-2 mt-1">
                <button
                  type="button"
                  onClick={() => setModelName('Attention_UNet')}
                  className={`py-2 px-2.5 rounded-lg text-xs font-semibold border text-left transition ${
                    modelName === 'Attention_UNet'
                      ? 'bg-sky-50 border-sky-500 text-sky-800'
                      : 'border-slate-200 text-slate-600 hover:bg-slate-50'
                  }`}
                >
                  <div className="font-bold">Attention U-Net</div>
                  <div className="text-[10px] text-slate-400">Attention Gates (Khuyên dùng)</div>
                </button>
                <button
                  type="button"
                  onClick={() => setModelName('UNet_Baseline')}
                  className={`py-2 px-2.5 rounded-lg text-xs font-semibold border text-left transition ${
                    modelName === 'UNet_Baseline'
                      ? 'bg-sky-50 border-sky-500 text-sky-800'
                      : 'border-slate-200 text-slate-600 hover:bg-slate-50'
                  }`}
                >
                  <div className="font-bold">Baseline U-Net</div>
                  <div className="text-[10px] text-slate-400">Standard U-Net 2015</div>
                </button>
              </div>
            </div>
          </div>

          {/* Image Upload & Benchmark Samples */}
          <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm space-y-3">
            <label className="block text-xs font-bold uppercase tracking-wider text-slate-700">
              3. Chọn Ảnh Đáy Mắt
            </label>

            {/* Dropzone */}
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              accept=".jpg,.jpeg,.png,.tif,.tiff"
              className="hidden"
            />
            <div
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-slate-300 hover:border-sky-500 bg-slate-50 hover:bg-sky-50/50 p-5 rounded-xl text-center cursor-pointer transition"
            >
              <Upload className="w-6 h-6 text-slate-400 mx-auto mb-2" />
              <p className="text-xs font-semibold text-slate-700">Bấm để tải ảnh võng mạc từ máy tính</p>
              <p className="text-[11px] text-slate-400 mt-0.5">Hỗ trợ JPG, PNG, TIF y tế (tối đa 30MB)</p>
            </div>

            {/* Preloaded Benchmark Samples */}
            <div className="space-y-1.5 pt-2 border-t border-slate-100">
              <span className="text-[11px] font-semibold text-slate-600 flex items-center">
                <FolderOpen className="w-3.5 h-3.5 mr-1 text-sky-600" />
                Mẫu thử nghiệm Dataset thực tế (1-Click Test):
              </span>
              <div className="grid grid-cols-2 gap-2 max-h-40 overflow-y-auto pr-1">
                {sampleImages.map((s, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => handleSelectSample(s)}
                    className="p-2 rounded-lg border border-slate-200 hover:border-sky-500 bg-white hover:bg-sky-50 text-left text-xs transition"
                  >
                    <div className="font-semibold text-slate-800 truncate">{s.filename}</div>
                    <div className="text-[10px] text-sky-600">{s.dataset} Benchmark</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Start Button */}
            <button
              onClick={handleStartAnalysis}
              disabled={!imageUrl || status === 'PROCESSING' || status === 'PENDING'}
              className="w-full py-2.5 px-4 bg-gradient-to-r from-sky-600 to-teal-600 hover:from-sky-700 hover:to-teal-700 text-white font-bold text-xs rounded-xl shadow-lg shadow-sky-600/30 transition disabled:opacity-50 flex items-center justify-center space-x-2"
            >
              {status === 'PROCESSING' || status === 'PENDING' ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Đang suy luận Deep Learning ({progress}%)...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  <span>BẮT ĐẦU PHÂN TÍCH AI</span>
                </>
              )}
            </button>
          </div>

          {/* Progress / Error Box */}
          {errorMessage && (
            <div className="p-3 rounded-xl bg-red-50 border border-red-200 text-red-700 text-xs flex items-center space-x-2">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{errorMessage}</span>
            </div>
          )}
        </div>

        {/* Right Column: Canvas Medical Viewer & Quantitative Findings (8 cols) */}
        <div className="lg:col-span-8 space-y-5">
          {/* Canvas Viewer Container */}
          <div className="h-[520px]">
            {imageUrl ? (
              <CanvasViewer
                originalImageUrl={imageUrl}
                predictedMaskUrl={analysisResult?.predicted_mask_url || null}
                groundTruthUrl={groundTruthUrl}
                taskType={taskType}
              />
            ) : (
              <div className="h-full bg-slate-900 rounded-xl border border-slate-800 flex flex-col items-center justify-center text-slate-500 space-y-3 p-8 text-center">
                <div className="w-16 h-16 rounded-2xl bg-slate-800 flex items-center justify-center text-slate-400">
                  <Activity className="w-8 h-8" />
                </div>
                <div>
                  <h3 className="text-slate-300 font-bold text-sm">Chưa có ảnh võng mạc nào được tải</h3>
                  <p className="text-xs text-slate-500 max-w-sm mt-1">
                    Hãy chọn một mẫu benchmark IDRiD/DRIVE bên trái hoặc kéo thả file ảnh từ máy tính để bắt đầu quan sát.
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Quantitative Clinical Findings & Metrics */}
          {analysisResult && (
            <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between border-b border-slate-100 pb-3 gap-2">
                <div>
                  <h2 className="text-sm font-bold text-slate-900 flex items-center">
                    <CheckCircle2 className="w-4 h-4 text-emerald-600 mr-1.5" />
                    Kết quả Định lượng Lâm sàng (Quantitative Medical Findings)
                  </h2>
                  <p className="text-xs text-slate-500">
                    Thời gian thực thi: <span className="font-mono font-semibold text-slate-700">{analysisResult.processing_time_ms} ms</span> | 
                    Diện tích FOV hợp lệ: <span className="font-mono font-semibold text-slate-700">{analysisResult.retinal_fov_pixels?.toLocaleString()} px</span>
                  </p>
                </div>
                <div className="text-right">
                  <span className="text-[11px] font-semibold text-slate-500">Tổng diện tích bất thường:</span>
                  <div className="text-lg font-bold font-mono text-sky-700">
                    {analysisResult.lesion_percentage?.toFixed(3)}% FOV
                  </div>
                </div>
              </div>

              {/* Lesion breakdown table */}
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-50 text-slate-500 uppercase border-b border-slate-200">
                    <tr>
                      <th className="py-2.5 px-3 font-semibold">Tổn thương / Cấu trúc</th>
                      <th className="py-2.5 px-3 font-semibold text-center">Số ổ tổn thương (Foci)</th>
                      <th className="py-2.5 px-3 font-semibold text-right">Diện tích (Pixel)</th>
                      <th className="py-2.5 px-3 font-semibold text-right">Tỷ lệ võng mạc (FOV %)</th>
                      <th className="py-2.5 px-3 font-semibold">Ý nghĩa bệnh học</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 font-mono">
                    {analysisResult.lesion_details?.map((item: any) => {
                      const metaInfo: Record<string, [string, string, string]> = {
                        MA: ['Vi phình mạch (MA)', '#ef4444', 'Dấu hiệu sớm của thoái hóa mao mạch võng mạc tiểu đường'],
                        HE: ['Xuất huyết (HE)', '#f97316', 'Tổn thương vỡ mạch máu, nguy cơ xuất huyết dịch kính'],
                        EX: ['Xuất tiết cứng (EX)', '#eab308', 'Lắng đọng lipid rỉ ra từ các mao mạch rò rỉ'],
                        SE: ['Xuất tiết mềm (SE)', '#22c55e', 'Thiếu máu cục bộ tại lớp sợi thần kinh thị giác'],
                        OD: ['Đĩa thị giác (OD)', '#3b82f6', 'Cấu trúc giải phẫu chuẩn (Anatomical landmark)'],
                        VESSEL: ['Hệ thống mạch máu', '#a855f7', 'Cây mao mạch đáy mắt hoàn chỉnh trong FOV'],
                      };
                      const info = metaInfo[item.lesion_type] || [item.lesion_type, '#64748b', 'Bất thường'];

                      return (
                        <tr key={item.id} className="hover:bg-slate-50/80 transition">
                          <td className="py-2.5 px-3 font-sans font-semibold text-slate-800 flex items-center space-x-2">
                            <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: info[1] }} />
                            <span>{info[0]}</span>
                          </td>
                          <td className="py-2.5 px-3 text-center font-bold text-slate-800">
                            {item.count}
                          </td>
                          <td className="py-2.5 px-3 text-right text-slate-700">
                            {item.area_pixels?.toLocaleString()} px
                          </td>
                          <td className="py-2.5 px-3 text-right font-bold text-sky-700">
                            {item.area_percentage?.toFixed(4)} %
                          </td>
                          <td className="py-2.5 px-3 font-sans text-slate-500 text-[11px]">
                            {info[2]}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              {/* Validation Metrics (If Ground Truth provided) */}
              {analysisResult.dice_score !== null && (
                <div className="pt-3 border-t border-slate-100 space-y-2">
                  <span className="text-xs font-bold text-slate-700">Chỉ số Đánh giá so với Ground Truth chuẩn:</span>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                    <div className="p-2.5 rounded-lg bg-sky-50 border border-sky-100 text-center">
                      <div className="text-[10px] text-sky-600 font-semibold uppercase">Dice Score</div>
                      <div className="text-base font-bold font-mono text-sky-900 mt-0.5">
                        {analysisResult.dice_score?.toFixed(4)}
                      </div>
                    </div>
                    <div className="p-2.5 rounded-lg bg-teal-50 border border-teal-100 text-center">
                      <div className="text-[10px] text-teal-600 font-semibold uppercase">IoU (Jaccard)</div>
                      <div className="text-base font-bold font-mono text-teal-900 mt-0.5">
                        {analysisResult.iou_score?.toFixed(4)}
                      </div>
                    </div>
                    <div className="p-2.5 rounded-lg bg-indigo-50 border border-indigo-100 text-center">
                      <div className="text-[10px] text-indigo-600 font-semibold uppercase">Precision</div>
                      <div className="text-base font-bold font-mono text-indigo-900 mt-0.5">
                        {analysisResult.precision?.toFixed(4)}
                      </div>
                    </div>
                    <div className="p-2.5 rounded-lg bg-purple-50 border border-purple-100 text-center">
                      <div className="text-[10px] text-purple-600 font-semibold uppercase">Recall (Sensitivity)</div>
                      <div className="text-base font-bold font-mono text-purple-900 mt-0.5">
                        {analysisResult.recall?.toFixed(4)}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

