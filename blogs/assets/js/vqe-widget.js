(function (global) {
  function createVqeWidget(containerSelectorOrElement) {
    const container = typeof containerSelectorOrElement === "string"
      ? document.querySelector(containerSelectorOrElement)
      : containerSelectorOrElement;

    if (!container) {
      console.error("createVqeWidget: container not found:", containerSelectorOrElement);
      return;
    }

    // Inject styles once per page
    if (!document.getElementById("vqe-widget-styles")) {
      const style = document.createElement("style");
      style.id = "vqe-widget-styles";
      style.textContent = `
        .vqe-widget-root {
          font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          margin: 1.5rem 0;
          padding: 1rem 1.25rem;
          background: #ffffff;
          border-radius: 8px;
          box-shadow: 0 2px 6px rgba(15, 23, 42, 0.08);
          max-width: 900px;
        }

        .vqe-title {
          font-size: 1.1rem;
          margin: 0 0 .5rem 0;
        }

        .vqe-small {
          font-size: .85rem;
          color: #4b5563;
          margin: 0 0 1rem 0;
        }

        .vqe-row {
          display: flex;
          flex-wrap: wrap;
          gap: 1rem;
        }

        .vqe-column {
          flex: 1;
          min-width: 260px;
        }

        .vqe-label {
          font-weight: 600;
          display: block;
          margin-bottom: .25rem;
          font-size: .9rem;
        }

        .vqe-slider {
          width: 100%;
        }

        .vqe-button {
          padding: 6px 12px;
          border-radius: 4px;
          border: 1px solid #3b82f6;
          background: #3b82f6;
          color: white;
          cursor: pointer;
          font-size: .85rem;
          margin-right: .25rem;
        }

        .vqe-button:disabled {
          opacity: 0.5;
          cursor: default;
        }

        .vqe-value-box {
          background: #f1f5f9;
          border-radius: 4px;
          padding: 6px 8px;
          font-size: .8rem;
          line-height: 1.4;
        }

        .vqe-status-text {
          font-size: .8rem;
          color: #4b5563;
          margin-top: .35rem;
          min-height: 1.2em;
        }

        .vqe-circuit {
          font-family: "Courier New", monospace;
          white-space: pre;
          background: #0f172a;
          color: #e5e7eb;
          padding: 8px 10px;
          border-radius: 4px;
          font-size: .8rem;
          margin-top: .35rem;
        }

        .vqe-canvas {
          border: 1px solid #e5e7eb;
          border-radius: 4px;
          background: #ffffff;
          width: 100%;
          max-width: 380px;
          height: auto;
        }

        .vqe-note {
          font-size: .8rem;
          color: #4b5563;
          margin-top: .25rem;
        }
      `;
      document.head.appendChild(style);
    }

    // Widget markup
    container.innerHTML = `
      <div class="vqe-widget-root">
        <div class="vqe-row">
          <div class="vqe-column">
            <label class="vqe-label">Ansatz parameter θ (radians)</label>
            <input type="range"
                   class="vqe-slider vqe-theta-slider"
                   min="0" max="628" step="1" />

            <div class="vqe-value-box vqe-theta-info"></div>

            <div style="margin-top: .6rem;">
              <button class="vqe-button vqe-run-btn">Run VQE (gradient descent)</button>
              <button class="vqe-button vqe-reset-btn" style="background:#e5e7eb;color:#111827;border-color:#d1d5db;">
                Reset
              </button>
            </div>

            <div class="vqe-status-text vqe-status-text-el"></div>

            <div style="margin-top: .7rem;">
              <label class="vqe-label">Current circuit</label>
              <div class="vqe-circuit vqe-circuit-view"></div>
            </div>
          </div>

          <div class="vqe-column">
            <label class="vqe-label">Energy landscape E(θ) = cos(θ)</label>
            <canvas class="vqe-canvas vqe-energy-canvas" width="380" height="260"></canvas>
            <div class="vqe-note">
              The red dot shows the current parameter θ and corresponding energy.
              VQE moves the dot downhill to approach the minimum energy.
            </div>
          </div>
        </div>
      </div>
    `;

    // Grab elements (scoped to this container)
    const thetaSlider  = container.querySelector(".vqe-theta-slider");
    const thetaInfo    = container.querySelector(".vqe-theta-info");
    const energyCanvas = container.querySelector(".vqe-energy-canvas");
    const runVqeBtn    = container.querySelector(".vqe-run-btn");
    const resetBtn     = container.querySelector(".vqe-reset-btn");
    const statusText   = container.querySelector(".vqe-status-text-el");
    const circuitView  = container.querySelector(".vqe-circuit-view");

    if (!thetaSlider || !energyCanvas) {
      console.error("createVqeWidget: missing internal elements");
      return;
    }

    const ctx = energyCanvas.getContext("2d");
    const TWO_PI = Math.PI * 2;

    // Plot ranges
    const plot = {
      paddingLeft: 40,
      paddingRight: 16,
      paddingTop: 16,
      paddingBottom: 30,
      xMin: 0,
      xMax: TWO_PI,
      yMin: -1,
      yMax: 1
    };

    let optimizing = false;
    let optTimer = null;

    function valueToCanvas(x, y) {
      const w = energyCanvas.width;
      const h = energyCanvas.height;
      const px = plot.paddingLeft +
        (x - plot.xMin) / (plot.xMax - plot.xMin) * (w - plot.paddingLeft - plot.paddingRight);
      const py = h - plot.paddingBottom -
        (y - plot.yMin) / (plot.yMax - plot.yMin) * (h - plot.paddingTop - plot.paddingBottom);
      return { x: px, y: py };
    }

    function drawAxes() {
      ctx.clearRect(0, 0, energyCanvas.width, energyCanvas.height);

      const w = energyCanvas.width;
      const h = energyCanvas.height;

      ctx.strokeStyle = "#9ca3af";
      ctx.lineWidth = 1;

      // X-axis (theta)
      const origin = valueToCanvas(0, 0);
      const xEnd = valueToCanvas(TWO_PI, 0);
      ctx.beginPath();
      ctx.moveTo(origin.x, origin.y);
      ctx.lineTo(xEnd.x, origin.y);
      ctx.stroke();

      // Y-axis (energy)
      const yMinP = valueToCanvas(0, plot.yMin);
      const yMaxP = valueToCanvas(0, plot.yMax);
      ctx.beginPath();
      ctx.moveTo(yMinP.x, yMinP.y);
      ctx.lineTo(yMaxP.x, yMaxP.y);
      ctx.stroke();

      ctx.fillStyle = "#4b5563";
      ctx.font = "11px system-ui, sans-serif";
      ctx.textAlign = "center";

      // X labels: 0, π, 2π
      ctx.fillText("0", origin.x, origin.y + 14);
      const piPos = valueToCanvas(Math.PI, 0);
      ctx.fillText("π", piPos.x, origin.y + 14);
      ctx.fillText("2π", xEnd.x, origin.y + 14);

      // Y labels: 1, -1
      const y1 = valueToCanvas(0, 1);
      const y_1 = valueToCanvas(0, -1);
      ctx.textAlign = "right";
      ctx.fillText("1", y1.x - 6, y1.y + 3);
      ctx.fillText("-1", y_1.x - 6, y_1.y + 3);

      // Y axis label
      ctx.save();
      ctx.translate(10, (h / 2));
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.fillText("Energy E(θ)", 0, 0);
      ctx.restore();

      ctx.textAlign = "center";
      ctx.fillText("θ (radians)", (plot.paddingLeft + w - plot.paddingRight) / 2, h - 8);
    }

    function drawEnergyCurve() {
      ctx.strokeStyle = "#60a5fa";
      ctx.lineWidth = 2;
      ctx.beginPath();
      const steps = 200;
      for (let i = 0; i <= steps; i++) {
        const theta = plot.xMin + (i / steps) * (plot.xMax - plot.xMin);
        const E = Math.cos(theta);
        const p = valueToCanvas(theta, E);
        if (i === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      }
      ctx.stroke();
    }

    function drawCurrentTheta(theta) {
      const E = Math.cos(theta);
      const p = valueToCanvas(theta, E);

      // Vertical helper line
      ctx.strokeStyle = "#fecaca";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(p.x, valueToCanvas(theta, plot.yMin).y);
      ctx.lineTo(p.x, valueToCanvas(theta, plot.yMax).y);
      ctx.stroke();
      ctx.setLineDash([]);

      // Point
      ctx.fillStyle = "#ef4444";
      ctx.beginPath();
      ctx.arc(p.x, p.y, 4, 0, TWO_PI);
      ctx.fill();
    }

    function updateThetaInfo(theta) {
      const E = Math.cos(theta);
      const amp0 = Math.cos(theta / 2);
      const amp1 = Math.sin(theta / 2);

      thetaInfo.innerHTML =
        "θ ≈ " + theta.toFixed(3) + " rad" +
        " (" + (theta / Math.PI).toFixed(3) + " π)<br>" +
        "Energy E(θ) = cos(θ) ≈ " + E.toFixed(4) + "<br>" +
        "|ψ(θ)⟩ = " +
        amp0.toFixed(3) + " |0⟩ + " +
        amp1.toFixed(3) + " |1⟩";

      const thetaStr = "θ ≈ " + theta.toFixed(3);
      circuitView.textContent =
        "|0⟩ ──[ Ry(" + thetaStr + ") ]────●── measure Z\n" +
        "                                  |\n" +
        " Energy: ⟨ψ(θ)|Z|ψ(θ)⟩ = cos(θ) ≈ " + E.toFixed(4);
    }

    function renderAll(theta) {
      drawAxes();
      drawEnergyCurve();
      drawCurrentTheta(theta);
      updateThetaInfo(theta);
    }

    function sliderToTheta(sliderValue) {
      // slider 0..628 -> θ in [0, 2π]
      const t = parseFloat(sliderValue) / 628;
      return t * TWO_PI;
    }

    function thetaToSlider(theta) {
      // θ in [0, 2π] -> 0..628
      return Math.round((theta / TWO_PI) * 628);
    }

    thetaSlider.addEventListener("input", function () {
      if (optimizing) return;
      const theta = sliderToTheta(thetaSlider.value);
      renderAll(theta);
      if (statusText) statusText.textContent = "";
    });

    runVqeBtn.addEventListener("click", function () {
      if (optimizing) return;
      optimizing = true;
      runVqeBtn.disabled = true;
      thetaSlider.disabled = true;
      if (statusText) {
        statusText.textContent = "Running VQE (gradient descent on E(θ) = cos θ)...";
      }

      let theta = sliderToTheta(thetaSlider.value);

      // Very simple optimizer
      const learningRate = 0.15;
      const maxIters = 80;
      let iter = 0;

      // d/dθ cos θ = -sin θ; gradient descent step: θ <- θ - η * dE/dθ
      optTimer = setInterval(function () {
        const grad = -Math.sin(theta);
        theta = theta - learningRate * grad;

        // keep θ in [0, 2π]
        theta = ((theta % TWO_PI) + TWO_PI) % TWO_PI;

        thetaSlider.value = thetaToSlider(theta);
        renderAll(theta);

        const E = Math.cos(theta);
        iter++;

        if (Math.abs(Math.sin(theta)) < 1e-3 || iter >= maxIters) {
          clearInterval(optTimer);
          optimizing = false;
          runVqeBtn.disabled = false;
          thetaSlider.disabled = false;
          if (statusText) {
            statusText.textContent =
              "Optimization finished after " + iter +
              " iterations. Approx. minimum energy E(θ) ≈ " + E.toFixed(4) +
              " at θ ≈ " + theta.toFixed(3) + " rad (~" +
              (theta / Math.PI).toFixed(3) + " π).";
          }
        }
      }, 70);
    });

    resetBtn.addEventListener("click", function () {
      if (optTimer) clearInterval(optTimer);
      optimizing = false;
      runVqeBtn.disabled = false;
      thetaSlider.disabled = false;
      thetaSlider.value = thetaToSlider(Math.PI / 4);  // default start
      const theta = sliderToTheta(thetaSlider.value);
      renderAll(theta);
      if (statusText) statusText.textContent = "";
    });

    // Initial render
    thetaSlider.value = thetaToSlider(Math.PI / 4);
    renderAll(sliderToTheta(thetaSlider.value));
  }

  // Expose a global factory function
  global.createVqeWidget = createVqeWidget;
})(window);
