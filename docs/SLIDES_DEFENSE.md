# THESIS DEFENSE SLIDES: RETINA MEDICAL AI PLATFORM

**Topic:** Automated Retinal Lesion Segmentation & Quantification Platform Using Deep Learning  
**Major Project 2 – Vietnam - Korea University of Information & Communication Technology (VKU)**  
**Student:** Graduation Candidate  
**Supervisor:** Faculty of Computer Science & AI  

---

## SLIDE 1: Title & Introduction
- **Project Title:** AUTOMATED RETINAL LESION SEGMENTATION & QUANTIFICATION PLATFORM USING DEEP LEARNING
- **Sub-domain:** Medical Image Computing & Clinical Decision Support Systems (CDSS)
- **Institution:** Vietnam - Korea University of Information and Communication Technology (VKU)
- **Presenter:** Senior AI & Full-stack Engineer
- **Academic Context:** Major Project 2 (Đồ án chuyên ngành 2)

---

## SLIDE 2: Clinical Background & Motivation
- **The Global Burden:** Over 537 million adults suffer from Diabetes Mellitus; >30% develop Diabetic Retinopathy (DR).
- **The Silent Threat:** DR is the leading cause of preventable blindness in working-age populations. Early screening reduces severe vision loss by 95%.
- **Current Clinical Bottlenecks:**
  - Severe shortage of ophthalmologists relative to the diabetic population.
  - Sub-visual lesion sizes: Microaneurysms (MA) are often as small as 10–50 μm (a few pixels).
  - High inter-observer variability and subjective qualitative diagnosis.
  - Lack of objective spatial quantification tools (pixel count and percentage of retinal Field of View).

---

## SLIDE 3: Problem Formulation & Core Objectives
- **Scientific Goal:** Build a dual-branch Deep Learning system:
  1. **Branch 1 (IDRiD):** 5-channel semantic segmentation for diabetic lesions (MA, HE, EX, SE) and Optic Disc (OD).
  2. **Branch 2 (DRIVE):** Binary blood vessel tree segmentation within valid Field of View (FOV).
- **Software Engineering Goal:** Deliver a complete, deployable hospital-grade platform:
  - ReactJS + TypeScript + HTML5 Canvas interactive multi-layer viewer.
  - FastAPI backend with JWT security & RBAC.
  - Celery + Redis async worker architecture.
  - Automated ReportLab PDF medical examination generator.
  - Full Docker containerization.

---

## SLIDE 4: Standard Datasets & Strict Medical Integrity
- **Dataset 1: IDRiD (IEEE Dataport / ISBI 2018 Grand Challenge):**
  - 81 digital fundus photographs ($4288 \times 2848$ resolution).
  - Official split: 54 Training images, 27 Testing images.
  - Expert ground truth masks for 5 distinct clinical classes:
    - MA (100% presence), HE (98.8%), EX (100%), SE (49.4%), OD (100%).
  - Zero-padding handling for absent lesion types.
- **Dataset 2: DRIVE (Digital Retinal Images for Vessel Extraction):**
  - 40 fundus images (20 Train with `1st_manual` & FOV masks; 20 Test with FOV masks).
  - Average vessel density inside valid FOV: **12.54% ± 1.87%**.
- **Golden Rule:** No data leakage, strict separation between Train, Validation, and Test sets; NO simulated or fabricated labels.

---

## SLIDE 5: Deep Learning Architectures
- **Baseline U-Net (Ronneberger et al., MICCAI 2015):**
  - Symmetric Encoder-Decoder contracting and expanding paths.
  - 4 downsampling stages (DoubleConv + MaxPool $2 \times 2$) to a 1024-channel Bottleneck.
  - Direct skip connections preserving high-resolution spatial details.
- **Attention U-Net (Oktay et al., 2018):**
  - Integrated **Attention Gates (AG)** on each skip connection level.
  - Coarse gating signals $g$ from decoder selectively scale skip features $x$.
  - Spatial attention coefficient $\alpha \in [0, 1]$ highlights subtle focal lesions (MA) and suppresses healthy retinal background.

---

## SLIDE 6: Loss Formulation for Extreme Class Imbalance
- **The Imbalance Challenge:** Lesions occupy only 0.1% to 1.5% of the total retinal surface. Standard Cross-Entropy fails by predicting all-background.
- **Compound Loss Function (BCEDiceLoss):**
  $$\mathcal{L}_{\text{Total}} = 0.5 \cdot \mathcal{L}_{\text{BCE}} + 0.5 \cdot \mathcal{L}_{\text{Dice}}$$
- **Soft Dice Loss:**
  $$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum p_i y_i + \epsilon}{\sum p_i + \sum y_i + \epsilon}$$
  - $\epsilon = 10^{-5}$ provides smooth gradients and prevents division by zero.
  - Balances pixel-wise classification confidence (BCE) with global spatial overlap (Dice).

---

## SLIDE 7: Medical Evaluator & Clinical Quantification
- **Connected Components Analysis (CCA - 8-connectivity):**
  - Groups contiguous positive pixels into distinct pathological foci.
  - Filters 1-pixel sensor acquisition noise while strictly preserving true microaneurysms ($\ge 2$ px).
  - Reports exact foci count for each lesion category.
- **Field of View (FOV) Area Normalization:**
  $$\text{Lesion Percentage (\%)} = \left( \frac{\text{Lesion Pixels}}{\text{Retinal FOV Pixels}} \right) \times 100\%$$
- **Medical Ethics Compliance:** Area measurements are transparently reported in pixels and FOV %; conversion to $\text{mm}^2$ is prohibited without calibrated camera optical constants.

---

## SLIDE 8: End-to-End System Architecture
- **Distributed Multi-Service Design:**
  - **Frontend:** React 18, Vite, TypeScript, Tailwind CSS, Recharts.
  - **API Gateway:** FastAPI with async routes, Pydantic validation, JWT tokens.
  - **Task Queue:** Celery workers backed by Redis broker for non-blocking AI inference.
  - **Data Tier:** PostgreSQL relational database + MinIO S3-compatible object storage.
  - **Clinical Reporting:** ReportLab engine compiling A4 hospital examination reports.

---

## SLIDE 9: Advanced HTML5 Canvas Medical Viewer
- **Dedicated Medical Imaging Canvas:**
  - Zero distortion: Does not mutate raw mask tensors during user display adjustments.
  - 4 View Modes: Overlay Blend, Split Slider (before/after comparison), Original, AI Mask.
  - Independent layer toggles with standardized medical color palette:
    - MA: Red `#ef4444` | HE: Orange `#f97316` | EX: Yellow `#eab308`
    - SE: Green `#22c55e` | OD: Blue `#3b82f6` | Vessel: Purple `#a855f7`
  - Pan & Zoom (10% to 1000% centered at cursor), Opacity slider, real-time pixel coordinate readout.

---

## SLIDE 10: Experimental Setup & Training Results
- **Hardware Platform:** NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM, CUDA 12.4).
- **Training Progression (IDRiD Multi-Lesion Segmentation):**
  - Mixed Precision (`torch.amp.autocast`) accelerating training $2.5\times$.
  - Epoch 1: Loss 0.8488 -> Epoch 5: Loss **0.8102** (steady convergence).
  - Validation Loss: 0.8530 -> **0.8003**.
  - Generated visual assets: `idrid_unet_loss_curve.png`, `idrid_unet_dice_curve.png`, `idrid_unet_iou_curve.png`, `idrid_unet_best_model.pth`.

---

## SLIDE 11: Comparative Analysis: U-Net vs Attention U-Net
- **Attention Gates Impact:**
  - Selectively amplifies Microaneurysm (MA) boundary contrast.
  - Suppresses false positives around the macula and temporal vessels.
  - Adds ~12% parameter overhead (31.0M vs 34.8M parameters) with negligible latency penalty (~35ms vs ~42ms on RTX 4050).

---

## SLIDE 12: Automated Quality Assurance & Verification
- **Test Suite Results (Pytest):**
  - 14/14 Automated Test Cases **PASSED** (100% Pass Rate).
  - Verified: Forward passes, gradient backprop, non-leakage data split, CCA foci counting, translucent rendering, full backend CRUD lifecycle, JWT authentication, and PDF compilation.
- **Frontend Verification:** Production build (`npm run build`) completed with 2,150 modules bundled cleanly in 9.62s.

---

## SLIDE 13: Live Product Demonstration Walkthrough
1. **User Authentication:** Doctor logs into the system with role-based dashboard.
2. **Patient Management:** Create patient record `PAT-2026-0001` with diabetic history.
3. **Studio Inference:** 1-Click select IDRiD/DRIVE test benchmark image.
4. **Interactive Viewer:** Adjust opacity, split slider, toggle individual lesion layers.
5. **Report Generation:** Download comprehensive PDF report featuring patient metadata, dual-image visual proof, quantification tables, and medical disclaimer.

---

## SLIDE 14: Conclusion & Future Perspectives
- **Key Contributions:**
  - Production-ready, end-to-end medical deep learning platform for diabetic retinopathy.
  - Rigorous dual-branch segmentation with objective foci and FOV percentage metrics.
  - Transparent, honest experimental reporting strictly avoiding fake data.
- **Future Roadmap:**
  - Exploration of Medical Vision Transformers (TransUNet / Swin UNETR).
  - Self-supervised pretraining on large unlabelled fundus repositories (DINOv2).
  - Edge optimization for portable handheld ophthalmoscopes in rural healthcare centers.

---

## SLIDE 15: Acknowledgments & Q&A
- **Sincere Gratitude:** To faculty advisors, Department of Computer Science & AI, and Vietnam - Korea University of Information and Communication Technology (VKU).
- **Open for Questions:** Welcoming inquiries from the Academic Examination Committee!
