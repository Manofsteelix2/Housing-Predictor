"use strict";

// The API supplies the only predictor list and its display metadata.
const API_BASE = "http://127.0.0.1:5000";
const API_URL = `${API_BASE}/predict`;
let predictors = [];
const form = document.querySelector("#prediction-form");
const button = document.querySelector("#predict-button");
const errorMessage = document.querySelector("#error-message");
const result = document.querySelector("#prediction-result");
const statusMessage = document.querySelector("#result-status");
const price = document.querySelector("#predicted-value");
const predictionInterval = document.querySelector("#prediction-interval");
const predictionIntervalValue = document.querySelector("#prediction-interval-value");
const currency = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });

// Build accessible controls from backend configuration; use textContent so
// display labels/options are rendered as text, never interpreted as HTML.
async function loadFields() {
  const formStatus = document.querySelector("#form-status");
  const container = document.querySelector("#predictor-fields");
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);
  try {
    const response = await fetch(`${API_BASE}/model-info`, { signal: controller.signal, cache: "no-store" });
    const data = await response.json().catch(() => null);
    if (!response.ok) throw new Error(data?.error || `Unable to load form (HTTP ${response.status}).`);
    if (!Array.isArray(data?.features) || !data.features.length) throw new Error("The API returned no configured fields.");
    const targetRange = document.querySelector("#model-target-range");
    if (Number.isFinite(data.target_min) && Number.isFinite(data.target_max)) {
      targetRange.textContent = `Training target range: ${currency.format(data.target_min)} to ${currency.format(data.target_max)} (inclusive).`;
      targetRange.hidden = false;
    } else if (Number.isFinite(data.target_min) || Number.isFinite(data.target_max)) {
      targetRange.textContent = Number.isFinite(data.target_min)
        ? `Training target minimum: ${currency.format(data.target_min)} (inclusive).`
        : `Training target maximum: ${currency.format(data.target_max)} (inclusive).`;
      targetRange.hidden = false;
    }
    const fragment = document.createDocumentFragment();
    const names = new Set();
    data.features.forEach((field, index) => {
      if (typeof field.name !== "string" || !field.name || names.has(field.name) ||
          typeof field.label !== "string" || !["number", "select"].includes(field.input_type) ||
          !["number", "string"].includes(field.value_type)) {
        throw new Error("The API returned invalid field metadata.");
      }
      names.add(field.name);
      const wrapper = document.createElement("div");
      wrapper.className = "field";
      const label = document.createElement("label");
      const input = document.createElement(field.input_type === "select" ? "select" : "input");
      input.id = `feature-${index}`;
      input.name = field.name;
      input.required = field.required === true;
      label.htmlFor = input.id;
      label.textContent = field.label;
      if (field.input_type === "select") {
        if (!Array.isArray(field.options) || !field.options.length) throw new Error(`Missing options for ${field.label}.`);
        input.add(new Option("Not specified", ""));
        field.options.forEach((option) => {
          const value = typeof option === "object" && option !== null ? option.value : option;
          const label = typeof option === "object" && option !== null ? option.label : option;
          // Survey missing codes are never choices, including in generated metadata.
          const code = String(value).trim();
          if (!code || Number(code) < 0 || /^\.[a-z_]*$/i.test(code)) return;
          input.add(new Option(String(label), String(value)));
        });
      } else {
        input.type = "number";
        ["min", "max", "step"].forEach((key) => {
          if (field[key] !== undefined) input.setAttribute(key, String(field[key]));
        });
      }
      wrapper.append(label, input);
      if (field.help) {
        const help = document.createElement("small");
        help.id = `${input.id}-help`;
        help.textContent = field.help;
        input.setAttribute("aria-describedby", help.id);
        wrapper.append(help);
      }
      if (field.name === "DIVISION" && field.input_type === "select" &&
          typeof attachCensusDivisionMap === "function") {
        attachCensusDivisionMap(wrapper, input, field);
      }
      fragment.append(wrapper);
    });
    container.replaceChildren(fragment);
    predictors = data.features;
    console.debug("Generated predictor fields (required only when explicitly configured):");
    console.table(predictors.map((field) => ({
      name: field.name,
      required: form.elements.namedItem(field.name).required,
    })));
    formStatus.hidden = true;
    button.disabled = false;
  } catch (error) {
    container.replaceChildren();
    formStatus.textContent = "Property fields could not be loaded. Reload the page to retry.";
    errorMessage.textContent = error.name === "AbortError" ? "Loading form configuration timed out. Check the backend and reload."
      : error instanceof TypeError ? "Cannot load form configuration. Check that Flask is running and reload the page." : error.message;
    errorMessage.hidden = false;
    button.disabled = true;
  } finally {
    clearTimeout(timeout);
  }
}
loadFields();

// Clear outdated results when the user changes property details.
form.addEventListener("input", (event) => {
  event.target.removeAttribute("aria-invalid");
  errorMessage.hidden = true;
  price.hidden = true;
  predictionInterval.hidden = true;
  statusMessage.textContent = "Submit your property details to calculate an estimate.";
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (button.disabled) return;
  errorMessage.hidden = true;
  price.hidden = true;
  predictionInterval.hidden = true;

  // Validate supplied values; only explicit schema overrides require a value.
  const payload = {};
  for (const field of predictors) {
    const input = form.elements.namedItem(field.name);
    input.removeAttribute("aria-invalid");
    const value = input.value.trim();
    // Blank optional fields are missing, never numeric zero.
    if (value === "" && !input.validity.badInput) {
      if (input.required) {
        input.setAttribute("aria-invalid", "true");
        errorMessage.textContent = `${field.label} is required.`;
        errorMessage.hidden = false;
        statusMessage.textContent = "Check the property details and try again.";
        input.focus();
        return;
      }
      payload[field.name] = null;
      continue;
    }
    const parsed = field.value_type === "number" ? Number(value) : value;
    if (input.validity.badInput || !input.checkValidity() || (field.value_type === "number" && !Number.isFinite(parsed))) {
      input.setAttribute("aria-invalid", "true");
      errorMessage.textContent = `Enter a valid value for ${field.label.toLowerCase()}. ${input.validationMessage}`;
      errorMessage.hidden = false;
      statusMessage.textContent = "Check the property details and try again.";
      input.focus();
      return;
    }
    payload[field.name] = parsed;
  }

  // Lock inputs while sending this snapshot; enforce a timeout for stalled requests.
  const controls = Array.from(form.querySelectorAll("input, select, button"));
  controls.forEach((control) => { control.disabled = true; });
  button.textContent = "Calculating…";
  result.setAttribute("aria-busy", "true");
  statusMessage.textContent = "Calculating your estimated home value…";
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);
  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    const data = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error(typeof data?.error === "string" ? data.error : `Prediction request failed (HTTP ${response.status}). Please try again.`);
    }
    if (typeof data?.predicted_value !== "number" || !Number.isFinite(data.predicted_value)) {
      throw new Error("The API returned an invalid prediction. Please try again.");
    }
    const interval = data.prediction_interval_95;
    if (!interval || !Number.isFinite(interval.lower) || !Number.isFinite(interval.upper)
        || interval.lower < 0 || interval.upper < interval.lower) {
      throw new Error("The API returned no valid 95% prediction range. Retrain the model and restart Flask.");
    }
    // Render API content as text and format dollar predictions for display.
    price.textContent = currency.format(data.predicted_value);
    price.hidden = false;
    predictionIntervalValue.textContent = `${currency.format(interval.lower)} – ${currency.format(interval.upper)}`;
    predictionInterval.hidden = false;
    statusMessage.textContent = "Estimated value based on the submitted details.";
  } catch (error) {
    errorMessage.textContent = error.name === "AbortError"
      ? "The request timed out. Check that the backend is running and try again."
      : error instanceof TypeError
        ? "Could not connect to the API. Check that Flask is running at http://127.0.0.1:5000 and try again."
        : error.message;
    errorMessage.hidden = false;
    statusMessage.textContent = "No estimate available. Please try again.";
  } finally {
    clearTimeout(timeout);
    controls.forEach((control) => { control.disabled = false; });
    button.textContent = "Predict Home Value →";
    result.removeAttribute("aria-busy");
  }
});
