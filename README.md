# Neural Stem Cell Differentiation Classifier (AMRANET & Xception)

An interactive, high-performance web interface designed to showcase deep learning classification of Neural Stem Cell (NSC) differentiation lineages from brightfield microscopy images.

Powered by **AMRANET** and **Xception** backbones trained to distinguish 4 key differentiation phenotypes:
- **Astrocyte**
- **NT-3 Induced**
- **Oligodendrocyte**
- **Retinoic Acid Induced**

Model Weights & Architecture Repository: [Hugging Face: MuhammadZulfikar020606/Klasifikasi-Hasil-Diferensiasi-Neural-Stem-Cell](https://huggingface.co/MuhammadZulfikar020606/Klasifikasi-Hasil-Diferensiasi-Neural-Stem-Cell)

---

## Features

- **Microscopy Image Input**: Drag & drop or upload brightfield micrographs (`.jpg`, `.jpeg`, `.png`, `.webp`).
- **Interactive Benchmark Presets**: Instant synthetic benchmark generators demonstrating morphological signatures for each lineage (Astrocyte star processes, NT-3 elongated axons, Oligodendrocyte ramified myelin networks, Retinoic Acid bipolar spindles).
- **Exact Pipeline Inspection**: Real-time visualization of image transformation identical to the model training pipeline:
  1. Grayscale 1-channel conversion
  2. 224&times;224 Bilinear interpolation
  3. 3&times;3 Median blur denoising
  4. Z-Score normalization: $(I - \mu) / \sigma$
  5. Min-Max normalization to $[0, 255]$ uint8 & 3-channel broadcast
  6. Standardized tensor scaling to $[-1, 1]$
- **Multi-Class Output Analytics**:
  - Dominant lineage classification with confidence percentage
  - Softmax distribution progress metrics for all 4 classes
  - Interactive Polar Radar chart for multi-lineage affinity
  - Shannon Entropy uncertainty evaluation
  - Biological marker guides (GFAP, S100B, MAP2, Olig2, NeuN, etc.)
- **Modern Responsive Design**: Tailwind CSS, Lucide icons, Chart.js, dark/light mode toggle, and glassmorphic aesthetic.

---

## Deployment to Vercel

This application is ready to deploy directly to Vercel.

### Method 1: Deploy with Git (Recommended)
1. Commit and push your code to your GitHub repository:
   ```bash
   git add .
   git commit -m "Add stem cell classification interface and Vercel config"
   git push origin main
   ```
2. Go to [Vercel Dashboard](https://vercel.com/new).
3. Import your GitHub repository (`muhammadzulfikar020606/klasifikasi-stem-cell`).
4. Click **Deploy**. Vercel will automatically detect the static project and deploy it with global edge CDN.

### Method 2: Deploy with Vercel CLI
```bash
npx vercel
```
Follow the interactive prompts to log in and deploy.

---

## Local Development

You can run the web interface locally using any lightweight HTTP server:

```bash
# Using Python
python -m http.server 3000

# Using Node.js
npx serve .
```

Then open `http://localhost:3000` in your browser.

---

## Model Architecture Details

- **Input Dimension**: `(None, 224, 224, 3)`
- **Data Augmentations**: Horizontal/Vertical RandomFlip, RandomRotation ($\pm 0.08$), RandomZoom ($0.08$)
- **Feature Extractor**: Xception / AMRANET functional backbone
- **Classification Head**:
  - GlobalAveragePooling2D
  - BatchNormalization
  - Dropout
  - Dense (256 units, ReLU)
  - Dropout
  - Dense (4 units, Softmax)
- **Target Classes**:
  1. `Astrocyte`
  2. `NT-3 Induced`
  3. `Oligodendrocyte`
  4. `Retinoic Acid Induced`

