const form = document.getElementById("prediction-form");
const submitButton = document.getElementById("submit-button");
const resultDiv = document.getElementById("result");
const featureImportanceDiv = document.getElementById("feature-importance");
const modelMetadataPre = document.getElementById("model-metadata");
const formErrorDiv = document.getElementById("form-error");

function setFieldError(fieldName, message) {
  const input = document.getElementById(fieldName);
  const errorDiv = document.querySelector(`[data-error-for="${fieldName}"]`);

  if (!input || !errorDiv) return;

  if (message) {
    input.classList.add("invalid");
    errorDiv.textContent = message;
  } else {
    input.classList.remove("invalid");
    errorDiv.textContent = "";
  }
}

function validateForm() {
  let isValid = true;
  formErrorDiv.textContent = "";

  const fields = [
    { name: "mileage", min: 0, max: 300000 },
    { name: "age", min: 0, max: 30 },
    { name: "engine_size", min: 0.8, max: 6.0 },
    { name: "horsepower", min: 40, max: 600 },
    { name: "doors", min: 2, max: 5 },
  ];

  for (const { name, min, max } of fields) {
    const input = document.getElementById(name);
    const rawValue = input.value.trim();

    if (!rawValue) {
      setFieldError(name, "This field is required.");
      isValid = false;
      continue;
    }

    const value = Number(rawValue);
    if (Number.isNaN(value)) {
      setFieldError(name, "Please enter a valid number.");
      isValid = false;
      continue;
    }

    if (value < min || value > max) {
      setFieldError(name, `Value must be between ${min} and ${max}.`);
      isValid = false;
      continue;
    }

    setFieldError(name, "");
  }

  const brandSelect = document.getElementById("brand");
  if (!brandSelect.value) {
    setFieldError("brand", "Please select a brand.");
    isValid = false;
  } else {
    setFieldError("brand", "");
  }

  const fuelSelect = document.getElementById("fuel_type");
  if (!fuelSelect.value) {
    setFieldError("fuel_type", "Please select a fuel type.");
    isValid = false;
  } else {
    setFieldError("fuel_type", "");
  }

  if (!isValid) {
    formErrorDiv.textContent = "Please fix the highlighted fields.";
  }

  return isValid;
}

function renderResult(prediction) {
  const usdtPrice = prediction.predicted_price;
  const etbPriceRaw = prediction.etb_price;
  const rate = prediction.usdt_to_etb_rate || 155.95;
  const etbPrice =
    typeof etbPriceRaw === "number" && !Number.isNaN(etbPriceRaw)
      ? etbPriceRaw
      : usdtPrice * rate;

  resultDiv.innerHTML = `
    <div class="result-label">Estimated market price</div>
    <div class="result-row">
      <div class="result-col">
        <div class="result-currency-label">USDT</div>
        <div class="result-value">
          ${usdtPrice.toLocaleString(undefined, {
            maximumFractionDigits: 2,
          })}
        </div>
      </div>
      <div class="result-col">
        <div class="result-currency-label">ETB</div>
        <div class="result-value">
          ${etbPrice.toLocaleString(undefined, {
            maximumFractionDigits: 2,
          })}
        </div>
      </div>
    </div>
    <div class="result-rate">
      1 USDT = ${rate.toLocaleString(undefined, {
        maximumFractionDigits: 2,
      })} ETB
    </div>
  `;

  const importance = prediction.feature_importance || {};
  featureImportanceDiv.innerHTML = "";
  const entries = Object.entries(importance).sort((a, b) => b[1] - a[1]);

  if (entries.length === 0) {
    featureImportanceDiv.innerHTML =
      '<p class="placeholder">Feature importance is not available for this model.</p>';
  } else {
    for (const [name, score] of entries) {
      const percent = (score * 100).toFixed(1);
      const row = document.createElement("div");
      row.className = "feature-row";
      row.innerHTML = `
        <div class="feature-label">${name}</div>
        <div class="feature-bar-wrapper">
          <div class="feature-bar" style="width: ${percent}%;"></div>
        </div>
        <div class="feature-value">${percent}%</div>
      `;
      featureImportanceDiv.appendChild(row);
    }
  }

  modelMetadataPre.textContent = JSON.stringify(
    prediction.model_metadata || {},
    null,
    2
  );
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!validateForm()) {
    return;
  }

  submitButton.disabled = true;
  submitButton.textContent = "Predicting...";
  formErrorDiv.textContent = "";

  const payload = {
    mileage: Number(document.getElementById("mileage").value),
    age: Number(document.getElementById("age").value),
    engine_size: Number(document.getElementById("engine_size").value),
    horsepower: Number(document.getElementById("horsepower").value),
    doors: Number(document.getElementById("doors").value),
    brand: document.getElementById("brand").value,
    fuel_type: document.getElementById("fuel_type").value,
  };

  try {
    const response = await fetch("/predict/regression", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      const detail =
        (Array.isArray(errorData.detail)
          ? errorData.detail.map((d) => d.msg).join(", ")
          : errorData.detail) || "Prediction failed.";
      throw new Error(detail);
    }

    const data = await response.json();
    renderResult(data);
  } catch (err) {
    formErrorDiv.textContent = err.message || "Prediction failed.";
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Predict Price";
  }
});

