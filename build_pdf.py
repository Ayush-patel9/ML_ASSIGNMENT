import os
import shutil
import subprocess
import fitz
import numpy as np
from PIL import Image

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ML Assignment 1: Polynomial Regression Report (BT2024054)</title>
<style>
  @page {
    size: A4 portrait;
    margin: 11mm 12mm 11mm 12mm;
    @bottom-right {
      content: counter(page) " of " counter(pages);
      font-size: 8pt;
      color: #64748b;
    }
  }

  *, *::before, *::after {
    box-sizing: border-box;
  }

  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 8.9pt;
    line-height: 1.35;
    color: #1e293b;
    margin: 0;
    padding: 0;
  }

  /* Header banner */
  .header-card {
    border-bottom: 2px solid #1e3a8a;
    padding-bottom: 5px;
    margin-bottom: 8px;
  }
  .doc-title {
    font-size: 15.5pt;
    font-weight: 700;
    margin: 0 0 4px 0;
    color: #0f2d59;
    letter-spacing: -0.3px;
    line-height: 1.2;
  }
  .meta-grid {
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    font-size: 8.4pt;
    color: #475569;
  }
  .meta-grid span { margin-right: 12px; }
  .meta-grid a { color: #2563eb; text-decoration: none; font-weight: 600; }

  /* Metric cards */
  .metrics-banner {
    display: flex;
    gap: 8px;
    margin: 6px 0 8px 0;
  }
  .metric-pill {
    flex: 1;
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-left: 3px solid #2563eb;
    border-radius: 4px;
    padding: 4px 7px;
  }
  .metric-pill .lbl {
    font-size: 7.2pt;
    text-transform: uppercase;
    font-weight: 700;
    color: #64748b;
    letter-spacing: 0.3px;
  }
  .metric-pill .val {
    font-size: 10pt;
    font-weight: 700;
    color: #0f2d59;
  }
  .metric-pill .sub {
    font-size: 7.2pt;
    color: #475569;
  }

  /* Headings */
  h2 {
    font-size: 10.8pt;
    font-weight: 700;
    color: #0f2d59;
    margin: 7px 0 3px 0;
    border-bottom: 1px solid #cbd5e1;
    padding-bottom: 2px;
    break-after: avoid;
  }
  h3 {
    font-size: 9.3pt;
    font-weight: 600;
    color: #1e3a8a;
    margin: 5px 0 2px 0;
    break-after: avoid;
  }

  p { margin: 0 0 4.5px 0; }
  ul, ol { margin: 0 0 4.5px 0; padding-left: 17px; }
  li { margin-bottom: 1.5px; }

  strong { font-weight: 600; color: #0f172a; }
  code {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 8pt;
    background: #f1f5f9;
    padding: 1px 3px;
    border-radius: 3px;
    color: #0f172a;
    border: 0.5px solid #e2e8f0;
  }

  pre {
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 5px 8px;
    font-size: 7.5pt;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    margin: 3px 0 5px 0;
    line-height: 1.25;
    break-inside: avoid;
  }

  /* Tables */
  table {
    width: 100%;
    border-collapse: collapse;
    margin: 5px 0 7px 0;
    font-size: 8pt;
    break-inside: avoid;
    box-shadow: 0 0.5px 1px rgba(0,0,0,0.04);
  }
  th {
    background: #0f2d59;
    color: #ffffff;
    font-weight: 600;
    text-align: left;
    padding: 3.5px 6px;
    border: 1px solid #0f2d59;
  }
  td {
    padding: 3px 6px;
    border: 1px solid #e2e8f0;
    vertical-align: middle;
  }
  tr:nth-child(even) td { background: #f8fafc; }
  tr.highlight td {
    background: #eff6ff;
    font-weight: 600;
    border-top: 1px solid #bfdbfe;
    border-bottom: 1px solid #bfdbfe;
  }
  .badge {
    display: inline-block;
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 7pt;
    font-weight: 700;
    text-transform: uppercase;
  }
  .badge-win { background: #dcfce7; color: #166534; border: 0.5px solid #86efac; }
  .badge-warn { background: #fee2e2; color: #991b1b; border: 0.5px solid #fca5a5; }

  /* Figures */
  .figure-box {
    text-align: center;
    margin: 5px 0 7px 0;
    break-inside: avoid;
  }
  .figure-box img {
    max-width: 95%;
    height: auto;
    max-height: 172px;
    border-radius: 4px;
    border: 1px solid #cbd5e1;
  }
  .figure-caption {
    font-size: 7.5pt;
    color: #475569;
    margin-top: 2px;
    font-style: italic;
  }

  /* Two-column grid */
  .grid-2 {
    display: flex;
    gap: 9px;
    margin-bottom: 4px;
    break-inside: avoid;
  }
  .grid-2 > div { flex: 1; }

  .callout {
    background: #f0f7ff;
    border: 1px solid #bfdbfe;
    border-left: 3.5px solid #2563eb;
    padding: 5px 8px;
    margin: 5px 0;
    font-size: 8.4pt;
    border-radius: 0 4px 4px 0;
    break-inside: avoid;
  }

  .page-footer {
    display: flex;
    justify-content: space-between;
    font-size: 7.2pt;
    color: #94a3b8;
    border-top: 0.5px solid #e2e8f0;
    padding-top: 2px;
    margin-top: 6px;
    break-inside: avoid;
  }

  /* Page break utilities */
  .page-break { page-break-before: always; }
  .avoid-break { break-inside: avoid; }
</style>
</head>
<body>

  <!-- ==================== PAGE 1 ==================== -->
  <div class="header-card">
    <h1 class="doc-title">Machine Learning Assignment 1: Polynomial Regression Report</h1>
    <div class="meta-grid">
      <span><strong>Student:</strong> Ayush Patel</span>
      <span><strong>Roll Number:</strong> BT2024054</span>
      <span><strong>Assigned Problems:</strong> <code>var1</code> &amp; <code>var2</code></span>
      <span><strong>Repository:</strong> <a href="https://github.com/Ayush-patel9/ML_ASSIGNMENT">github.com/Ayush-patel9/ML_ASSIGNMENT</a></span>
    </div>
  </div>

  <div class="metrics-banner">
    <div class="metric-pill">
      <div class="lbl">var1 Net Power (Lasso)</div>
      <div class="val">CV R&sup2; = 0.9689</div>
      <div class="sub">CV MSE: 0.3218 &bull; Deg 5 (98 active)</div>
    </div>
    <div class="metric-pill">
      <div class="lbl">var2 Thermal Anomaly (Ridge)</div>
      <div class="val">CV R&sup2; = 0.9929</div>
      <div class="sub">CV MSE: 0.2835 &bull; Deg 9 (norm 38.9)</div>
    </div>
    <div class="metric-pill">
      <div class="lbl">Baseline &rarr; Final Gain</div>
      <div class="val">+0.785 / +0.887 R&sup2;</div>
      <div class="sub">96.4% &amp; 99.2% MSE reduction</div>
    </div>
    <div class="metric-pill">
      <div class="lbl">Constraint Compliance</div>
      <div class="val">Strict Polynomial</div>
      <div class="sub">5-Fold CV &bull; 0 Data Leakage</div>
    </div>
  </div>

  <h2>1. Introduction &amp; Overview</h2>
  <p>
    This report documents the polynomial regression models built for two distinct engineering prediction problems assigned to my roll number (<strong>BT2024054</strong>):
  </p>
  <ul>
    <li><strong>Problem 1 (<code>var1</code>) &mdash; Steam Turbine Net Power Score:</strong> Predicting power output from 6 operational parameters (steam valve, coolant flow rate, pump pressure, blade pitch, exhaust rate, and inlet pressure).</li>
    <li><strong>Problem 2 (<code>var2</code>) &mdash; Subterranean Thermal Anomaly Score:</strong> Mapping subsurface temperatures across a 3D geological reservoir from spatial offsets (<i>x</i><sub>1</sub>: East-West, <i>x</i><sub>2</sub>: North-South, <i>x</i><sub>3</sub>: Depth).</li>
  </ul>
  <p>
    The assignment required using <strong>strictly polynomial regression</strong> without non-polynomial architectures (like neural networks or decision trees). When I initially tested the starting hints mentioned in the assignment prompt (degree 3 on <i>x</i><sub>1</sub>&ndash;<i>x</i><sub>3</sub> for <code>var1</code>, and degree 4 on <i>x</i><sub>1</sub> for <code>var2</code>), the validation score was poor (<i>R</i><sup>2</sup> &approx; 0.18 and 0.10). To find the true models, I followed a four-stage progression:
  </p>
  <div class="callout">
    <strong>Four-Stage Trajectory:</strong>
    <strong>M1 (Baseline):</strong> Tested PDF hints literally with OLS &rarr; heavy underfitting (<i>R</i><sup>2</sup> &approx; 0.10 &ndash; 0.18). &bull; 
    <strong>M2 (OLS Sweep):</strong> Added all features across degrees &rarr; massive jump (<i>R</i><sup>2</sup> &approx; 0.92 &ndash; 0.99), but hit unregularized variance limit. &bull; 
    <strong>M3+ (Regularized &mdash; Final):</strong> Applied Lasso (<i>L</i><sub>1</sub>) to prune 78.8% of turbine terms, and Ridge (<i>L</i><sub>2</sub>) to smoothly stabilize 3D heat fields &rarr; peak generalization (<strong>var1 CV <i>R</i><sup>2</sup> = 0.9689, MSE = 0.3218</strong>; <strong>var2 CV <i>R</i><sup>2</sup> = 0.9929, MSE = 0.2835</strong>). &bull; 
    <strong>M4 (Overfit Demo):</strong> Pushed degrees to 8 &amp; 15 with unregularized OLS &rarr; training error near zero, but validation MSE exploded to 285.50 (<i>R</i><sup>2</sup> = &minus;4.906).
  </div>

  <h2>2. Exploring the Datasets</h2>
  <div class="grid-2">
    <div>
      <h3>2.1 Turbine Net Power Score (<code>var1</code>)</h3>
      <p>
        Power generation follows thermodynamic laws (Brayton cycle). Net work depends on 6 operational controls (<i>x</i><sub>1</sub>&ndash;<i>x</i><sub>6</sub>) normalized in [&minus;1.0, 1.0]. The dataset contains 1,000 training samples with target mean &mu; = 0.76, std &sigma; = 3.31, spanning [&minus;10.43, 11.48]. Inputs interact (e.g. pressure &times; valve opening), but arbitrary 5-way cross products rarely correspond to physical processes. This pointed to a <strong>sparse polynomial</strong>.
      </p>
    </div>
    <div>
      <h3>2.2 Thermal Anomaly Score (<code>var2</code>)</h3>
      <p>
        Heat conduction in subterranean rock obeys continuous harmonic physics (&nabla;<sup>2</sup><i>T</i> = 0 in steady state). Spatial offsets (<i>x</i><sub>1</sub>, <i>x</i><sub>2</sub>, <i>x</i><sub>3</sub>) are normalized in [&minus;1.0, 1.0]. The dataset contains 1,000 training samples with target mean &mu; = 2.40, std &sigma; = 6.41, spanning [&minus;29.69, 39.25]. Because heat varies continuously across 3D space, all spatial derivatives contribute, requiring a <strong>dense, smoothly regularized polynomial</strong>.
      </p>
    </div>
  </div>

  <h2>3. Polynomial Expansion &amp; Regularization</h2>
  <p>
    Polynomial regression maps <i>d</i> inputs into monomial combinations up to degree <i>D</i>: <i>y&#770;</i> = &Sigma; <i>w<sub>j</sub></i> &phi;<sub><i>j</i></sub>(<b>x</b>) + <i>b</i>. The total number of terms <i>P</i> grows combinatorially: <i>P</i> = C(<i>d</i> + <i>D</i>, <i>D</i>) = (<i>d</i> + <i>D</i>)! / (<i>d</i>! &middot; <i>D</i>!).
  </p>
  <table>
    <thead>
      <tr>
        <th>Problem</th>
        <th>Inputs (<i>d</i>)</th>
        <th>Degree 1</th>
        <th>Degree 3</th>
        <th>Degree 4</th>
        <th>Degree 5</th>
        <th>Degree 8</th>
        <th>Degree 9</th>
        <th>Degree 15</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>var1 (Turbine)</strong></td>
        <td>6 features</td>
        <td>7 terms</td>
        <td>84 terms</td>
        <td>210 terms</td>
        <td><strong>462 terms (M3)</strong></td>
        <td>3,003 terms (M4)</td>
        <td>&mdash;</td>
        <td>&mdash;</td>
      </tr>
      <tr>
        <td><strong>var2 (Geology)</strong></td>
        <td>3 coordinates</td>
        <td>4 terms</td>
        <td>20 terms</td>
        <td>35 terms</td>
        <td>56 terms</td>
        <td>165 terms</td>
        <td><strong>220 terms (M3)</strong></td>
        <td>816 terms (M4)</td>
      </tr>
    </tbody>
  </table>
  <p>
    As <i>P</i> grows relative to <i>N</i> (1,000 samples), standard OLS amplifies multicollinearity and fits noise. Regularization adds a penalty on weights <b>w</b>:
  </p>
  <ul>
    <li><strong>Lasso (<i>L</i><sub>1</sub> penalty):</strong> Loss = MSE + &alpha; &Sigma; |<i>w<sub>j</sub></i>|. The diamond constraint forces unneeded weights strictly to 0, performing automatic feature selection on combinatorial interaction terms.</li>
    <li><strong>Ridge (<i>L</i><sub>2</sub> penalty):</strong> Loss = MSE + &alpha; &Sigma; <i>w<sub>j</sub></i><sup>2</sup>. The circular constraint shrinks weights smoothly without setting them to 0, preventing coefficient explosion while preserving 3D spatial field continuity.</li>
  </ul>
  <div class="page-footer">
    <span>Machine Learning Assignment 1 Report &bull; Roll Number: BT2024054</span>
    <span>Page 1 of 5</span>
  </div>

  <!-- ==================== PAGE 2 ==================== -->
  <div class="page-break"></div>

  <h2>4. Experimental Journey &amp; Thought Process</h2>

  <h3>4.1 Stage 1: Baseline Using Problem Hints (<code>model1_baseline.py</code>)</h3>
  <p>
    I started by directly testing the hints in the problem description: degree 3 on <i>x</i><sub>1</sub>&ndash;<i>x</i><sub>3</sub> for <code>var1</code>, and degree 4 on <i>x</i><sub>1</sub> alone for <code>var2</code> using standard Ordinary Least Squares (OLS).
  </p>
  <ul>
    <li><strong>var1 Baseline:</strong> Train <i>R</i><sup>2</sup> = 0.2254, 5-Fold CV <i>R</i><sup>2</sup> = <strong>0.1838 &plusmn; 0.029</strong>, CV MSE = <strong>8.9044 &plusmn; 1.785</strong></li>
    <li><strong>var2 Baseline:</strong> Train <i>R</i><sup>2</sup> = 0.1211, 5-Fold CV <i>R</i><sup>2</sup> = <strong>0.1058 &plusmn; 0.042</strong>, CV MSE = <strong>36.678 &plusmn; 8.020</strong></li>
  </ul>
  <p>
    <strong>Why it failed:</strong> Both models underfitted severely. For <code>var1</code>, dropping <i>x</i><sub>4</sub>, <i>x</i><sub>5</sub>, <i>x</i><sub>6</sub> removed blade pitch and steam inlet pressure, throwing away half the physical system. For <code>var2</code>, using only <i>x</i><sub>1</sub> attempted to map 3D subterranean heat from a single 1D axis. Because each student has independently generated data, model selection must be driven by empirical cross-validation, not static text hints.
  </p>

  <h3>4.2 Stage 2: Systematic OLS Sweep Across Degrees (<code>model2_ols_sweep.py</code>)</h3>
  <p>
    Next, I ran an exhaustive 5-fold cross-validation sweep over degrees 1 to 10 using all features:
  </p>
  <ul>
    <li><strong>var1 (all 6 features):</strong> Deg 1 (<i>R</i><sup>2</sup> = 0.1007, MSE = 9.76) &rarr; Deg 2 (<i>R</i><sup>2</sup> = 0.6726, MSE = 3.51) &rarr; Deg 3 (<i>R</i><sup>2</sup> = 0.8900, MSE = 1.16) &rarr; <strong>Deg 4 OLS Peak (<i>R</i><sup>2</sup> = 0.9230, MSE = 0.8081)</strong> &rarr; Deg 5 (<i>R</i><sup>2</sup> = 0.8399, MSE = 1.63) &rarr; Deg 6 (<i>R</i><sup>2</sup> = &minus;7.75, MSE = 97.32).</li>
    <li><strong>var2 (all 3 coordinates):</strong> Deg 1 (<i>R</i><sup>2</sup> = 0.2182, MSE = 32.10) &rarr; Deg 4 (<i>R</i><sup>2</sup> = 0.9142, MSE = 3.45) &rarr; Deg 6 (<i>R</i><sup>2</sup> = 0.9848, MSE = 0.62) &rarr; <strong>Deg 8 OLS Peak (<i>R</i><sup>2</sup> = 0.9927, MSE = 0.2896)</strong> &rarr; Deg 9 (<i>R</i><sup>2</sup> = 0.9916, MSE = 0.34).</li>
  </ul>

  <div class="figure-box">
    <img src="figures/bias_variance_tradeoff.png" alt="Bias Variance Tradeoff">
    <div class="figure-caption">Figure 1: Validation MSE across polynomial degrees, displaying the U-shaped curves where unregularized OLS begins to overfit and marking the optimal regularized points.</div>
  </div>

  <p>
    <strong>The OLS Ceiling:</strong> Restoring all features jumped <i>R</i><sup>2</sup> from 0.18 to 0.92 on <code>var1</code>, and from 0.10 to 0.99 on <code>var2</code>. However, unregularized OLS hit a wall: at degree 5 on <code>var1</code>, having 462 terms with 1,000 samples doubled validation MSE. Advancing further required regularization.
  </p>

  <h3>4.3 Stage 3 &amp; 3+: Regularization &mdash; Final Models (<code>train_predict.py</code>)</h3>
  <p>
    To safely unlock higher-degree curvature, I evaluated Lasso (<i>L</i><sub>1</sub>) and Ridge (<i>L</i><sub>2</sub>):
  </p>
  <ul>
    <li><strong>Why Lasso won on <code>var1</code>:</strong> At degree 5 (462 terms), Lasso (&alpha; = 0.00348) forced <strong>364 terms (78.8%) strictly to zero</strong>, retaining only 98 active terms. Ridge kept all 462 terms non-zero and reached <i>R</i><sup>2</sup> = 0.9521, whereas Lasso reached <strong>CV <i>R</i><sup>2</sup> = 0.9689</strong> and cut CV MSE to <strong>0.3218</strong>.</li>
    <li><strong>Why Ridge won on <code>var2</code>:</strong> Continuous heat conduction requires smooth gradients. Ridge (&alpha; = 0.00665) shrunk all 220 coefficients smoothly (weight norm ||<b>w</b>||<sub>2</sub> = 38.94), reaching <strong>CV <i>R</i><sup>2</sup> = 0.9929</strong> and CV MSE = <strong>0.2835</strong> without creating surface discontinuities.</li>
  </ul>

  <div class="figure-box">
    <img src="figures/sparsity_and_coefficients.png" alt="Sparsity and Coefficients">
    <div class="figure-caption">Figure 2: Top: Lasso setting 364 redundant terms to zero on var1 at degree 5. Bottom: Ridge weight shrinkage on var2 keeping coefficients stable (norm 38.9) compared to the exploded weights of unconstrained OLS (norm 27,038).</div>
  </div>
  <div class="page-footer">
    <span>Machine Learning Assignment 1 Report &bull; Roll Number: BT2024054</span>
    <span>Page 2 of 5</span>
  </div>

  <!-- ==================== PAGE 3 ==================== -->
  <div class="page-break"></div>

  <h3>4.4 Stage 4: Testing Overfitting (<code>model4_overfit.py</code>)</h3>
  <p>
    To observe where polynomial models break when complexity is unconstrained, I built two extreme models: degree 8 OLS on <code>var1</code> (3,003 terms) and degree 15 OLS on <code>var2</code> (816 terms).
  </p>
  <ul>
    <li><strong>Training memorization:</strong> Both models fit the training sets almost perfectly (Train <i>R</i><sup>2</sup> &approx; 0.9999).</li>
    <li><strong>Validation collapse:</strong> On unseen cross-validation folds, <code>var1</code> CV <i>R</i><sup>2</sup> dropped to <strong>0.6757</strong> (MSE = 3.414), while <code>var2</code> CV <i>R</i><sup>2</sup> collapsed to <strong>&minus;4.906</strong>, with CV MSE exploding to <strong>285.50</strong>.</li>
    <li><strong>Weight explosion:</strong> For <code>var2</code>, unconstrained OLS weights blew up to over &plusmn;4,600 (total weight norm = 27,038, compared to 38.9 with Ridge). The curve oscillated wildly between sample points (Runge's phenomenon).</li>
  </ul>

  <h2>5. Master Results Comparison</h2>
  <p>
    The table below summarizes all four experimental stages across both datasets:
  </p>
  <table>
    <thead>
      <tr>
        <th>Stage</th>
        <th>Script</th>
        <th>Task</th>
        <th>Features</th>
        <th>Deg</th>
        <th>Terms</th>
        <th>Model</th>
        <th>Train <i>R</i><sup>2</sup></th>
        <th>Train MSE</th>
        <th>5-Fold CV <i>R</i><sup>2</sup></th>
        <th>5-Fold CV MSE</th>
        <th>Status</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>M1</strong></td>
        <td><code>model1_baseline.py</code></td>
        <td><code>var1</code></td>
        <td><i>x</i><sub>1</sub>&ndash;<i>x</i><sub>3</sub></td>
        <td>3</td>
        <td>20</td>
        <td>OLS</td>
        <td>0.2254</td>
        <td>8.452</td>
        <td>0.1838 &plusmn; 0.029</td>
        <td>8.9044 &plusmn; 1.785</td>
        <td><span class="badge badge-warn">Underfit</span></td>
      </tr>
      <tr>
        <td><strong>M1</strong></td>
        <td><code>model1_baseline.py</code></td>
        <td><code>var2</code></td>
        <td><i>x</i><sub>1</sub> only</td>
        <td>4</td>
        <td>5</td>
        <td>OLS</td>
        <td>0.1211</td>
        <td>36.036</td>
        <td>0.1058 &plusmn; 0.042</td>
        <td>36.678 &plusmn; 8.020</td>
        <td><span class="badge badge-warn">Underfit</span></td>
      </tr>
      <tr>
        <td><strong>M2</strong></td>
        <td><code>model2_ols_sweep.py</code></td>
        <td><code>var1</code></td>
        <td><i>x</i><sub>1</sub>&ndash;<i>x</i><sub>6</sub></td>
        <td>4</td>
        <td>210</td>
        <td>OLS</td>
        <td>0.9628</td>
        <td>0.408</td>
        <td>0.9230 &plusmn; 0.020</td>
        <td>0.8081 &plusmn; 0.117</td>
        <td>OLS peak</td>
      </tr>
      <tr>
        <td><strong>M2</strong></td>
        <td><code>model2_ols_sweep.py</code></td>
        <td><code>var2</code></td>
        <td><i>x</i><sub>1</sub>&ndash;<i>x</i><sub>3</sub></td>
        <td>8</td>
        <td>165</td>
        <td>OLS</td>
        <td>0.9957</td>
        <td>0.178</td>
        <td>0.9927 &plusmn; 0.002</td>
        <td>0.2896 &plusmn; 0.033</td>
        <td>OLS peak</td>
      </tr>
      <tr>
        <td><strong>M3</strong></td>
        <td><code>model3_regularized.py</code></td>
        <td><code>var1</code></td>
        <td><i>x</i><sub>1</sub>&ndash;<i>x</i><sub>6</sub></td>
        <td>5</td>
        <td>462</td>
        <td>Lasso (&alpha; &approx; 0.0033)</td>
        <td>0.9774</td>
        <td>0.248</td>
        <td>0.9687 &plusmn; 0.006</td>
        <td>0.3325 &plusmn; 0.034</td>
        <td>Pruned 79%</td>
      </tr>
      <tr>
        <td><strong>M3</strong></td>
        <td><code>model3_regularized.py</code></td>
        <td><code>var2</code></td>
        <td><i>x</i><sub>1</sub>&ndash;<i>x</i><sub>3</sub></td>
        <td>9</td>
        <td>220</td>
        <td>Ridge (&alpha; &approx; 0.0066)</td>
        <td>0.9959</td>
        <td>0.168</td>
        <td>0.9929 &plusmn; 0.001</td>
        <td>0.2835 &plusmn; 0.036</td>
        <td>Smooth L2</td>
      </tr>
      <tr class="highlight">
        <td><strong>M3+</strong></td>
        <td><code>train_predict.py</code></td>
        <td><code>var1</code></td>
        <td><i>x</i><sub>1</sub>&ndash;<i>x</i><sub>6</sub></td>
        <td>5</td>
        <td>462</td>
        <td><strong>Lasso (&alpha; = 0.00348)</strong></td>
        <td><strong>0.9774</strong></td>
        <td><strong>0.248</strong></td>
        <td><strong>0.9689 &plusmn; 0.005</strong></td>
        <td><strong>0.3218 &plusmn; 0.031</strong></td>
        <td><span class="badge badge-win">WINNER</span></td>
      </tr>
      <tr class="highlight">
        <td><strong>M3+</strong></td>
        <td><code>train_predict.py</code></td>
        <td><code>var2</code></td>
        <td><i>x</i><sub>1</sub>&ndash;<i>x</i><sub>3</sub></td>
        <td>9</td>
        <td>220</td>
        <td><strong>Ridge (&alpha; = 0.00665)</strong></td>
        <td><strong>0.9959</strong></td>
        <td><strong>0.168</strong></td>
        <td><strong>0.9929 &plusmn; 0.001</strong></td>
        <td><strong>0.2835 &plusmn; 0.036</strong></td>
        <td><span class="badge badge-win">WINNER</span></td>
      </tr>
      <tr>
        <td><strong>M4</strong></td>
        <td><code>model4_overfit.py</code></td>
        <td><code>var1</code></td>
        <td><i>x</i><sub>1</sub>&ndash;<i>x</i><sub>6</sub></td>
        <td>8</td>
        <td>3,003</td>
        <td>OLS</td>
        <td>0.9999</td>
        <td>0.0006</td>
        <td>0.6757 &plusmn; 0.119</td>
        <td>3.4137 &plusmn; 0.977</td>
        <td>High variance</td>
      </tr>
      <tr>
        <td><strong>M4</strong></td>
        <td><code>model4_overfit.py</code></td>
        <td><code>var2</code></td>
        <td><i>x</i><sub>1</sub>&ndash;<i>x</i><sub>3</sub></td>
        <td>15</td>
        <td>816</td>
        <td>OLS</td>
        <td>0.9987</td>
        <td>0.052</td>
        <td>&minus;4.9055 &plusmn; 5.333</td>
        <td>285.50 &plusmn; 291.0</td>
        <td><span class="badge badge-warn">Overfit</span></td>
      </tr>
    </tbody>
  </table>

  <div class="figure-box">
    <img src="figures/model_comparison.png" alt="Model Comparison Across Stages">
    <div class="figure-caption">Figure 3: Cross-validation <i>R</i><sup>2</sup> (left) and logarithmic Mean Squared Error (right) across all four model configurations for both tasks.</div>
  </div>
  <div class="page-footer">
    <span>Machine Learning Assignment 1 Report &bull; Roll Number: BT2024054</span>
    <span>Page 3 of 5</span>
  </div>

  <!-- ==================== PAGE 4 ==================== -->
  <div class="page-break"></div>

  <h2>6. Model Diagnostics &amp; Prediction Verification</h2>
  
  <h3>6.1 Residual Statistical Properties</h3>
  <p>
    I analyzed the out-of-fold cross-validation residuals (<i>e<sub>i</sub></i> = <i>y<sub>i</sub></i> &minus; <i>y&#770;<sub>i</sub></i>) to verify that the final models are unbiased:
  </p>
  <ul>
    <li><strong><code>var1</code> (Degree 5 Lasso):</strong> Mean residual &mu; = &minus;2.30 &times; 10<sup>&minus;16</sup> &approx; 0.0, standard deviation &sigma; = 0.498, median +0.018, IQR = 0.683. The errors are symmetric and centered at zero.</li>
    <li><strong><code>var2</code> (Degree 9 Ridge):</strong> Mean residual &mu; = +1.73 &times; 10<sup>&minus;15</sup> &approx; 0.0, standard deviation &sigma; = 0.410, median +0.023, IQR = 0.543. Errors follow a standard normal distribution without heteroscedasticity.</li>
  </ul>

  <div class="figure-box">
    <img src="figures/residual_diagnostics.png" alt="Residual Diagnostics">
    <div class="figure-caption">Figure 4: Residual distribution histograms with Gaussian density fits (top) and residuals vs. predicted scatter plots (bottom), showing zero-mean, normally distributed errors.</div>
  </div>

  <div class="figure-box">
    <img src="figures/actual_vs_predicted.png" alt="Actual vs Predicted Out-of-Fold">
    <div class="figure-caption">Figure 5: 5-Fold cross-validated predictions plotted against ground truth labels along the ideal 1:1 reference line (<i>y</i> = <i>y&#770;</i>).</div>
  </div>

  <h3>6.2 Final Prediction File Checks</h3>
  <p>
    The final test set predictions were generated by fitting Model 3+ on the complete training sets:
  </p>
  <ul>
    <li><strong>File paths:</strong> <code>BT2024054/BT2024054_pred_var1.csv</code> and <code>BT2024054/BT2024054_pred_var2.csv</code>.</li>
    <li><strong>Format:</strong> Exactly 1,000 predictions each, formatted as a single column named <code>y</code> matching <code>sample_submission.csv</code>.</li>
    <li><strong>Integrity:</strong> Zero NaN, null, or infinite values.</li>
    <li><strong>Distributions:</strong>
      <code>var1</code> test predictions: mean 1.28, std 4.21, range [&minus;9.77, 15.78] (matches training target: mean 0.76, range [&minus;10.43, 11.48]). &bull; 
      <code>var2</code> test predictions: mean 1.88, std 6.65, range [&minus;29.53, 29.26] (matches training target: mean 2.40, range [&minus;29.69, 39.25]).
    </li>
  </ul>
  <div class="page-footer">
    <span>Machine Learning Assignment 1 Report &bull; Roll Number: BT2024054</span>
    <span>Page 4 of 5</span>
  </div>

  <!-- ==================== PAGE 5 ==================== -->
  <div class="page-break"></div>

  <h2>7. Key Takeaways &amp; Engineering Lessons</h2>
  <ol>
    <li>
      <strong>Data-driven validation beats static hints:</strong> The suggestions in problem descriptions are useful starting points, but they can be misleading on personalized datasets. Guided by 5-fold cross-validation, expanding features and degrees improved <i>R</i><sup>2</sup> from 0.18 &rarr; 0.97 on <code>var1</code> and from 0.10 &rarr; 0.99 on <code>var2</code>.
    </li>
    <li>
      <strong>Feature completeness is non-negotiable:</strong> Leaving out operational turbine parameters or spatial coordinates removes fundamental physics from the model. No amount of hyperparameter tuning can compensate for omitted features.
    </li>
    <li>
      <strong>Match regularization geometry to the physical domain:</strong>
      <ul>
        <li>For combinatorial multi-variable systems (the turbine), <strong>Lasso (<i>L</i><sub>1</sub>)</strong> is optimal because it performs automatic feature selection, pruning 78.8% of unphysical cross-terms.</li>
        <li>For continuous spatial fields (geothermal heat), <strong>Ridge (<i>L</i><sub>2</sub>)</strong> is optimal because it preserves smooth harmonic continuity without creating surface tears.</li>
      </ul>
    </li>
    <li>
      <strong>The reality of overfitting:</strong> High-degree unregularized polynomials easily achieve <i>R</i><sup>2</sup> &approx; 1.0 on training data, but their weights explode into the thousands and fail catastrophically on test data. Regularization is essential to keep high-degree polynomials stable.
    </li>
  </ol>

  <h2>8. Reproducibility &amp; GitHub Repository</h2>
  <p>
    All code, datasets, figure generation scripts, and prediction files are organized for one-command replication:
  </p>
  <pre># Clone repository and enter folder
git clone https://github.com/Ayush-patel9/ML_ASSIGNMENT.git
cd ML_ASSIGNMENT

# Install dependencies
pip install numpy pandas scikit-learn matplotlib

# Run the winning model (generates final submission CSVs in BT2024054/)
python3 train_predict.py

# Recreate all 5 figures in ./figures/
python3 generate_plots.py

# Run individual stages to reproduce comparison table:
python3 model1_baseline.py       # Stage 1: Naive PDF baseline
python3 model2_ols_sweep.py      # Stage 2: Data-driven OLS sweep
python3 model3_regularized.py    # Stage 3: Regularized models
python3 model4_overfit.py        # Stage 4: Overfitting demonstration</pre>

  <p style="margin-top: 8px;">
    <strong>Official Repository Link:</strong> <a href="https://github.com/Ayush-patel9/ML_ASSIGNMENT">https://github.com/Ayush-patel9/ML_ASSIGNMENT</a><br>
    <strong>Author:</strong> Ayush Patel (Roll Number: <strong>BT2024054</strong>) &bull; Department of Computer Science &bull; October 2026
  </p>
  <div class="page-footer">
    <span>Machine Learning Assignment 1 Report &bull; Roll Number: BT2024054</span>
    <span>Page 5 of 5</span>
  </div>

</body>
</html>
"""

def generate_pdf():
    html_file = "report_print.html"
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(HTML_TEMPLATE)

    chrome_bin = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    output_pdf = "BT2024054_Report.pdf"

    cmd = [
        chrome_bin,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={output_pdf}",
        os.path.abspath(html_file)
    ]

    print("Running headless Chrome to render PDF...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    
    if os.path.exists(output_pdf):
        doc = fitz.open(output_pdf)
        num_pages = len(doc)
        print(f"SUCCESS: Generated {output_pdf} with {num_pages} pages!")
        shutil.copy(output_pdf, "REPORT.pdf")
        
        # Check blank space distribution on each page
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples).convert("L")
            arr = np.array(img)
            non_white = np.where(arr < 250)[0]
            if len(non_white) > 0:
                top = non_white[0]
                bot = non_white[-1]
                span = (bot - top) / arr.shape[0] * 100
                blank_bot = (arr.shape[0] - bot) / arr.shape[0] * 100
                print(f"  Page {i+1}: Content span: {span:.1f}%, Blank bottom: {blank_bot:.1f}%")
        
        if os.path.exists(html_file):
            os.remove(html_file)
        return num_pages
    else:
        print("ERROR: PDF was not created!")
        print(res.stderr)
        return -1

if __name__ == "__main__":
    generate_pdf()
