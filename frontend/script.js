"use strict";

// The API supplies the only predictor list and its display metadata.
const API_BASE = "http://127.0.0.1:5000";
const API_URL = `${API_BASE}/predict`;
// Temporary diagnostics while investigating frontend startup/request failures.
console.info("[Housing] Page initialization", { apiBase: API_BASE });
let predictors = [];
const form = document.querySelector("#prediction-form");
const button = document.querySelector("#predict-button");
const errorMessage = document.querySelector("#error-message");
const result = document.querySelector("#prediction-result");
const statusMessage = document.querySelector("#result-status");
const price = document.querySelector("#predicted-value");
const predictionInterval = document.querySelector("#prediction-interval");
const predictionIntervalValue = document.querySelector("#prediction-interval-value");
const resetButton = document.querySelector("#reset-button");
const fieldCounter = document.querySelector("#field-counter");
const resultDetails = document.querySelector("#result-details");
const resultCount = document.querySelector("#result-count");
const resultDivision = document.querySelector("#result-division");
const currency = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });

// Groups describe presentation only; schema metadata assigns active predictors.
const GROUPS = [
  { key: "location", title: "Where does the story begin?", note: "01 / Choose a region", help: "Select a state to choose its Census division, or use the dropdown." },
  { key: "layout", title: "Home size & layout", note: "02 / Describe the home", help: "Start with the space and how it is arranged." },
  { key: "amenities", title: "Everyday comforts", note: "Amenities", help: "The practical details that help describe this home." },
  { key: "quality", title: "How does home feel?", note: "Quality & opinion", help: "Overall ratings of the home and its neighborhood, from worst to best." },
  { key: "other", title: "More about the home", note: "Additional characteristics", help: "Any other details you know." },
];
function selectedDivision(value) {
  return predictors.find(field => field.name === "DIVISION")?.options
    ?.find(option => String(option.value ?? option) === value)?.label || "Not specified";
}
function updateFormSummary() {
  const count = predictors.filter(field => form.elements.namedItem(field.name).value.trim() !== "").length;
  fieldCounter.textContent = `${count} of ${predictors.length} characteristics provided`;
  const division = form.elements.namedItem("DIVISION");
  const label = selectedDivision(division?.value ?? "");
  document.body.className = label === "Not specified" ? "theme-neutral"
    : `theme-${label.toLowerCase().replaceAll(" ", "-")}`;
  document.querySelector("#region-caption").textContent = label === "Not specified"
    ? "Every home has a story. Start with its region." : `A home in the ${label} division.`;
}
function clearEstimate() {
  errorMessage.hidden = true;
  price.hidden = true;
  predictionInterval.hidden = true;
  resultDetails.hidden = true;
  statusMessage.textContent = "Share what you know, then estimate the home's value.";
}
function focusField(input) {
  (input.choiceButtons?.find(choice => choice.getAttribute("aria-checked") === "true") || input).focus();
}
function attachBinaryChoices(wrapper, input, field, label) {
  const options = Array.from(input.options);
  if (options.length !== 3 || !options.some(option => option.text === "Yes") ||
      !options.some(option => option.text === "No")) return;
  const group = document.createElement("div");
  group.className = "segmented";
  group.setAttribute("role", "radiogroup");
  label.id = `${input.id}-label`;
  group.setAttribute("aria-labelledby", label.id);
  group.setAttribute("aria-required", String(input.required));
  if (field.help) group.setAttribute("aria-describedby", `${input.id}-help`);
  input.className = "binary-value";
  input.tabIndex = -1;
  input.setAttribute("aria-hidden", "true");
  input.choiceButtons = [];
  const choices = [...options.slice(1), options[0]];
  const synchronize = () => input.choiceButtons.forEach((choice, index) => {
    const checked = input.value === choices[index].value;
    choice.setAttribute("aria-checked", String(checked));
    choice.tabIndex = checked ? 0 : -1;
  });
  choices.forEach((option, index) => {
    const choice = document.createElement("button");
    choice.type = "button";
    choice.textContent = option.text;
    choice.setAttribute("role", "radio");
    choice.addEventListener("click", () => {
      if (input.disabled) return;
      input.value = option.value;
      input.dispatchEvent(new Event("input", { bubbles: true }));
      input.dispatchEvent(new Event("change", { bubbles: true }));
    });
    choice.addEventListener("keydown", event => {
      const direction = ["ArrowRight", "ArrowDown"].includes(event.key) ? 1
        : ["ArrowLeft", "ArrowUp"].includes(event.key) ? -1 : 0;
      if (direction && !input.disabled) {
        event.preventDefault();
        const next = input.choiceButtons[(index + direction + choices.length) % choices.length];
        next.click();
        next.focus();
      }
    });
    input.choiceButtons.push(choice);
    group.append(choice);
  });
  input.addEventListener("input", synchronize);
  input.addEventListener("change", synchronize);
  synchronize();
  wrapper.append(group);
}

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
    console.info("[Housing] /model-info response", { status: response.status, data });
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
    const sections = new Map();
    GROUPS.forEach(group => {
      const section = document.createElement("fieldset");
      section.className = `form-section group-${group.key}`;
      const legend = document.createElement("legend");
      legend.textContent = group.title;
      const eyebrow = document.createElement("p");
      eyebrow.className = "eyebrow";
      eyebrow.textContent = group.note;
      const description = document.createElement("p");
      description.className = "section-note";
      description.textContent = group.help;
      const fields = document.createElement("div");
      fields.className = "fields";
      section.append(legend, eyebrow, description, fields);
      sections.set(group.key, { section, fields, count: 0 });
    });
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
      if (field.name === "DIVISION" && field.input_type === "select") {
        console.info("[Housing] Map initialization", { available: typeof attachCensusDivisionMap === "function" });
        if (typeof attachCensusDivisionMap === "function") {
          // A decorative map failure must not discard working predictor controls.
          try {
            attachCensusDivisionMap(wrapper, input, field);
          } catch (error) {
            console.error("[Housing] Map initialization failed; dropdown remains available", error);
          }
        }
      }
      if (field.input_type === "select") attachBinaryChoices(wrapper, input, field, label);
      const group = sections.get(field.group) || sections.get("other");
      group.fields.append(wrapper);
      group.count += 1;
    });
    for (const [key, group] of sections) {
      if (!group.count) continue;
      fragment.append(group.section);
      if (key === "location") {
        const explanation = document.createElement("div");
        explanation.className = "tool-explanation";
        const heading = document.createElement("h2");
        heading.textContent = "A few details. A broader perspective.";
        const text = document.createElement("p");
        text.textContent = "Describe the home below to explore an estimate and its uncertainty. Leave any field blank if you do not know the answer.";
        explanation.append(heading, text);
        fragment.append(explanation);
      }
    }
    container.replaceChildren(fragment);
    predictors = data.features;
    console.info("[Housing] Fields generated", predictors.length);
    console.debug("Generated predictor fields (required only when explicitly configured):");
    console.table(predictors.map((field) => ({
      name: field.name,
      required: form.elements.namedItem(field.name).required,
    })));
    updateFormSummary();
    resetButton.disabled = false;
    formStatus.hidden = true;
    button.disabled = false;
  } catch (error) {
    console.error("[Housing] Field loading failed", error);
    container.replaceChildren();
    formStatus.textContent = "Property fields could not be loaded. Reload the page to retry.";
    errorMessage.textContent = error.name === "AbortError" ? "Loading form configuration timed out. Check the backend and reload."
      : error instanceof TypeError ? "Cannot load form configuration. Check that Flask is running and reload the page." : error.message;
    errorMessage.hidden = false;
    button.disabled = true;
    fieldCounter.textContent = "Characteristics unavailable — check the API connection.";
  } finally {
    clearTimeout(timeout);
  }
}
loadFields();
resetButton.addEventListener("click", () => {
  if (resetButton.disabled) return;
  predictors.forEach(field => {
    const input = form.elements.namedItem(field.name);
    input.value = "";
    input.removeAttribute("aria-invalid");
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  });
  clearEstimate();
  updateFormSummary();
  if (predictors.length) focusField(form.elements.namedItem(predictors.find(field => field.name === "DIVISION")?.name || predictors[0].name));
});

// Clear outdated results when the user changes property details.
form.addEventListener("input", (event) => {
  event.target.removeAttribute("aria-invalid");
  clearEstimate();
  updateFormSummary();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  console.info("[Housing] Predict button/form submitted", { disabled: button.disabled });
  if (button.disabled) return;
  errorMessage.hidden = true;
  price.hidden = true;
  predictionInterval.hidden = true;
  resultDetails.hidden = true;

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
        focusField(input);
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
      focusField(input);
      return;
    }
    payload[field.name] = parsed;
  }

  // Lock inputs while sending this snapshot; enforce a timeout for stalled requests.
  const controls = [...form.querySelectorAll("input, select, button"), resetButton];
  controls.forEach((control) => { control.disabled = true; });
  button.textContent = "Calculating…";
  result.setAttribute("aria-busy", "true");
  statusMessage.textContent = "Calculating your estimated home value…";
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);
  try {
    console.info("[Housing] Outgoing prediction payload", payload);
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    const data = await response.json().catch(() => null);
    console.info("[Housing] /predict response", { status: response.status, data });
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
    const supplied = Number.isInteger(data.predictors_supplied) ? data.predictors_supplied
      : Object.values(payload).filter(value => value !== null).length;
    const total = Number.isInteger(data.predictors_total) ? data.predictors_total : predictors.length;
    resultCount.textContent = `${supplied} of ${total}`;
    resultDivision.textContent = selectedDivision(payload.DIVISION == null ? "" : String(payload.DIVISION));
    resultDetails.hidden = false;
    statusMessage.textContent = "An estimate based on the home characteristics you provided.";
    result.scrollIntoView?.({ block: "center" });
  } catch (error) {
    console.error("[Housing] Prediction failed", error);
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
    button.textContent = "Estimate Home Value ↗";
    result.removeAttribute("aria-busy");
  }
});
