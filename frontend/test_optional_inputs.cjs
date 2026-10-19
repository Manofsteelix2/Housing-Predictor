// Run with node frontend/test_optional_inputs.cjs (Python backend dependencies required).
// Execute the actual frontend script with a small DOM fixture and Flask test client.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { spawn } = require("node:child_process");
const readline = require("node:readline");
const root = path.resolve(__dirname, "..");
const python = `
import sys, json
sys.path.insert(0, "backend")
from app import app
with app.test_client() as client:
    for line in sys.stdin:
        request = json.loads(line)
        response = client.get("/model-info") if request["method"] == "GET" else client.post("/predict", json=request["payload"])
        print(json.dumps({"status": response.status_code, "body": response.get_json()}), flush=True)
`;
// Reuse one real Flask client so each assertion need not reload Python/sklearn.
const backend = spawn(process.env.PYTHON || "python", ["-B", "-u", "-c", python], {
  cwd: root, stdio: ["pipe", "pipe", "inherit"],
});
const pending = [];
readline.createInterface({ input: backend.stdout }).on("line", line => {
  const request = pending.shift();
  if (!request) return;
  try { request.resolve(JSON.parse(line)); } catch (error) { request.reject(error); }
});
const failPending = error => pending.splice(0).forEach(request => request.reject(error));
backend.on("error", failPending);
backend.on("exit", code => failPending(new Error(`Flask test process exited (${code}).`)));
backend.stdin.on("error", failPending);
function api(method, payload) {
  return new Promise((resolve, reject) => {
    pending.push({ resolve, reject });
    backend.stdin.write(JSON.stringify({ method, payload }) + "\n");
  });
}
async function main() {
class Element {
  constructor() {
    this.children = []; this.value = ""; this.required = false;
    this.validity = { badInput: false }; this.listeners = {}; this.attributes = {};
  }
  append(...children) { this.children.push(...children); }
  add(option) { this.children.push(option); }
  replaceChildren(...children) { this.children = children; }
  setAttribute(name, value) { this.attributes[name] = value; }
  removeAttribute(name) { delete this.attributes[name]; }
  addEventListener(name, handler) { this.listeners[name] = handler; }
  dispatchEvent(event) {
    event.target = this;
    this.listeners[event.type]?.(event);
    if (event.bubbles) form.listeners[event.type]?.(event);
  }
  checkValidity() { return !this.validity.badInput && (!this.required || this.value !== ""); }
  focus() {}
}
const nodes = new Map();
for (const id of ["prediction-form", "predict-button", "error-message", "prediction-result",
  "result-status", "predicted-value", "form-status", "predictor-fields", "model-target-range",
  "prediction-interval", "prediction-interval-value"]) nodes.set(`#${id}`, new Element());
function controls(element) {
  return (element.children ?? []).flatMap(child => [child, ...controls(child)]).filter(child => child.name);
}
const form = nodes.get("#prediction-form");
const fields = () => controls(nodes.get("#predictor-fields"));
form.elements = { namedItem: name => fields().find(field => field.name === name) };
form.querySelectorAll = () => [...fields(), nodes.get("#predict-button")];
const info = await api("GET");
assert.equal(info.status, 200);
assert.ok(info.body.features.every(field => field.required === false));
const requests = [];
let responseOverride;
const context = vm.createContext({
  document: { querySelector: selector => nodes.get(selector),
    createElement: () => new Element(), createElementNS: () => new Element(),
    createDocumentFragment: () => new Element() },
  Event: class { constructor(type, options = {}) { this.type = type; this.bubbles = options.bubbles; } },
  Option: function(text, value) { this.text = text; this.value = value; },
  Intl, AbortController, setTimeout, clearTimeout, console,
  fetch: async (url, options) => {
    const response = options?.method === "POST" ? await (async () => {
      const payload = JSON.parse(options.body);
      requests.push(payload);
      return responseOverride ?? api("POST", payload);
    })() : { ...info, body: { ...info.body, features: info.body.features.map(field =>
      field.input_type === "select" ? { ...field, options: [...field.options,
        -6, "-9", ".M", ".N", { value: -1, label: "Missing" },
        { value: ".M", label: "Not reported" }] } : field) } };
    return { ok: response.status === 200, status: response.status, json: async () => response.body };
  },
});
vm.runInContext(fs.readFileSync(path.join(__dirname, "census-map.js"), "utf8"), context);
vm.runInContext(fs.readFileSync(path.join(__dirname, "script.js"), "utf8"), context);
  // Let asynchronous metadata loading finish before submitting.
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(fields().length, info.body.features.length);
  assert.equal(nodes.get("#model-target-range").hidden, false);
  const currency = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });
  assert.ok(nodes.get("#model-target-range").textContent.includes(currency.format(info.body.target_min)));
  assert.ok(nodes.get("#model-target-range").textContent.includes(currency.format(info.body.target_max)));
  assert.ok(fields().every(field => field.required === false && !("required" in field.attributes)));
  const descendants = element => (element.children ?? []).flatMap(child => [child, ...descendants(child)]);
  const states = descendants(nodes.get("#predictor-fields")).filter(node => node.attributes?.d);
  assert.equal(states.length, 51, "50 states and DC are selectable");
  const division = form.elements.namedItem("DIVISION");
  const divisionGroups = ["CT ME MA NH RI VT", "NJ NY PA", "IL IN MI OH WI",
    "IA KS MN MO NE ND SD", "DE FL GA MD NC SC VA WV DC", "AL KY MS TN",
    "AR LA OK TX", "AZ CO ID MT NV NM UT WY", "AK CA HI OR WA"];
  divisionGroups.forEach((group, index) => {
    const members = group.split(" ");
    members.forEach(code => {
      states.find(state => state.attributes["data-state"] === code).listeners.click();
      assert.equal(division.value, String(index + 1));
      assert.deepEqual(states.filter(state => state.attributes["aria-pressed"] === "true")
        .map(state => state.attributes["data-state"]).sort(), [...members].sort());
    });
    division.value = String(index + 1);
    division.listeners.change();
    assert.equal(states.filter(state => state.attributes["aria-pressed"] === "true").length, members.length);
  });
  const alaska = states.find(state => state.attributes["data-state"] === "AK");
  division.disabled = true;
  division.value = "";
  alaska.listeners.click();
  assert.equal(division.value, "", "map cannot change a request in flight");
  division.disabled = false;
  for (const key of ["Enter", " "]) {
    alaska.listeners.keydown({ key, preventDefault() {} });
    assert.equal(division.value, "9");
  }
  division.value = "";
  division.listeners.change();
  assert.ok(states.every(state => state.attributes["aria-pressed"] === "false"));
  console.log("Passed: all state mappings, division highlighting, dropdown sync, keyboard and blank clearing.");
  for (const metadata of info.body.features.filter(field => field.input_type === "select")) {
    const input = form.elements.namedItem(metadata.name);
    assert.equal(input.children[0].text, "Not specified");
    assert.equal(input.children[0].value, "");
    assert.equal(input.children.length, metadata.options.length + 1,
      `${metadata.name}: survey missing codes must not appear as choices`);
    metadata.options.forEach((option, index) => {
      assert.equal(input.children[index + 1].text, String(option.label ?? option));
      assert.equal(input.children[index + 1].value, String(option.value ?? option));
    });
  }
  // Every mapped choice must submit its code, including bedroom zero.
  for (const metadata of info.body.features.filter(field => field.options?.some(option => typeof option === "object"))) {
    for (const option of metadata.options) {
      fields().forEach(input => { input.value = ""; });
      form.elements.namedItem(metadata.name).value = String(option.value);
      await form.listeners.submit({ preventDefault() {} });
      assert.equal(requests.at(-1)[metadata.name], option.value);
      assert.ok(Object.entries(requests.at(-1)).every(([name, value]) => name === metadata.name || value === null));
      assert.equal(nodes.get("#error-message").hidden, true);
    }
  }
  console.log("Passed: friendly labels, original codes, and null blanks for every mapped choice.");
  for (const [label, filled] of [["all filled", fields().length], ["one blank", fields().length - 1],
    ["several blank", Math.max(1, fields().length - 4)], ["one filled", 1], ["all blank", 0]]) {
    fields().forEach((input, index) => {
      const metadata = info.body.features[index];
      input.value = index < filled ? String(metadata.input_type === "select" ? (metadata.options[0].value ?? metadata.options[0])
        : metadata.min ?? 1) : "";
    });
    const before = requests.length;
    await form.listeners.submit({ preventDefault() {} });
    assert.equal(requests.length, before + 1, label);
    assert.equal(Object.values(requests.at(-1)).filter(value => value !== null).length, filled, label);
    assert.equal(nodes.get("#predicted-value").hidden, false, label);
    assert.equal(nodes.get("#prediction-interval").hidden, false, label);
    assert.match(nodes.get("#prediction-interval-value").textContent, /\$.* – \$/);
    assert.equal(nodes.get("#error-message").hidden, true, label);
    console.log(`Passed: ${label}`);
  }
  const numeric = fields().find(input => input.type === "number");
  numeric.value = "abc";
  const before = requests.length;
  await form.listeners.submit({ preventDefault() {} });
  assert.equal(requests.length, before);
  assert.equal(nodes.get("#error-message").hidden, false);
  numeric.value = ""; numeric.validity.badInput = true;
  await form.listeners.submit({ preventDefault() {} });
  assert.equal(requests.length, before);
  assert.equal(nodes.get("#error-message").hidden, false);
  console.log("Passed: supplied invalid numeric values blocked (including browser badInput).");
  numeric.validity.badInput = false;
  form.listeners.input({ target: numeric });
  assert.equal(nodes.get("#predicted-value").hidden, true);
  assert.equal(nodes.get("#prediction-interval").hidden, true);
  responseOverride = { status: 503, body: { error: "Retrain the model to generate interval calibration." } };
  await form.listeners.submit({ preventDefault() {} });
  assert.equal(nodes.get("#prediction-interval").hidden, true);
  assert.match(nodes.get("#error-message").textContent, /Retrain/);
  responseOverride = { status: 200, body: { predicted_value: 100 } };
  await form.listeners.submit({ preventDefault() {} });
  assert.equal(nodes.get("#predicted-value").hidden, true);
  assert.equal(nodes.get("#prediction-interval").hidden, true);
  assert.match(nodes.get("#error-message").textContent, /prediction range/);
  console.log("Passed: stale ranges cleared, legacy/malformed interval errors displayed.");
}
main().catch(error => { console.error(error); process.exitCode = 1; })
  .finally(() => backend.stdin.end());
