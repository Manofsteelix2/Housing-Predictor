// Run with node frontend/test_optional_inputs.cjs (Python backend dependencies required).
// Execute the actual frontend script with a small DOM fixture and Flask test client.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { execFileSync } = require("node:child_process");
const root = path.resolve(__dirname, "..");
const python = `
import sys, json
sys.path.insert(0, "backend")
from app import app
with app.test_client() as client:
    request = json.load(sys.stdin)
    response = client.get("/model-info") if request["method"] == "GET" else client.post("/predict", json=request["payload"])
    print(json.dumps({"status": response.status_code, "body": response.get_json()}))
`;
function api(method, payload) {
  return JSON.parse(execFileSync("python", ["-B", "-c", python], {
    cwd: root, input: JSON.stringify({ method, payload }), encoding: "utf8",
  }));
}
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
  checkValidity() { return !this.validity.badInput && (!this.required || this.value !== ""); }
  focus() {}
}
const nodes = new Map();
for (const id of ["prediction-form", "predict-button", "error-message", "prediction-result",
  "result-status", "predicted-value", "form-status", "predictor-fields"]) nodes.set(`#${id}`, new Element());
function controls(element) {
  return (element.children ?? []).flatMap(child => [child, ...controls(child)]).filter(child => child.name);
}
const form = nodes.get("#prediction-form");
const fields = () => controls(nodes.get("#predictor-fields"));
form.elements = { namedItem: name => fields().find(field => field.name === name) };
form.querySelectorAll = () => [...fields(), nodes.get("#predict-button")];
const info = api("GET");
assert.equal(info.status, 200);
assert.ok(info.body.features.every(field => field.required === false));
const requests = [];
const context = vm.createContext({
  document: { querySelector: selector => nodes.get(selector),
    createElement: () => new Element(), createDocumentFragment: () => new Element() },
  Option: function(text, value) { this.text = text; this.value = value; },
  Intl, AbortController, setTimeout, clearTimeout, console,
  fetch: async (url, options) => {
    const response = options?.method === "POST" ? (() => {
      const payload = JSON.parse(options.body);
      requests.push(payload);
      return api("POST", payload);
    })() : info;
    return { ok: response.status === 200, status: response.status, json: async () => response.body };
  },
});
vm.runInContext(fs.readFileSync(path.join(__dirname, "script.js"), "utf8"), context);
(async () => {
  // Let asynchronous metadata loading finish before submitting.
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(fields().length, info.body.features.length);
  assert.ok(fields().every(field => field.required === false && !("required" in field.attributes)));
  for (const [label, filled] of [["all filled", fields().length], ["one blank", fields().length - 1],
    ["several blank", Math.max(1, fields().length - 4)], ["one filled", 1], ["all blank", 0]]) {
    fields().forEach((input, index) => {
      const metadata = info.body.features[index];
      input.value = index < filled ? String(metadata.input_type === "select" ? metadata.options[0]
        : metadata.min ?? 1) : "";
    });
    const before = requests.length;
    await form.listeners.submit({ preventDefault() {} });
    assert.equal(requests.length, before + 1, label);
    assert.equal(Object.keys(requests.at(-1)).length, filled, label);
    assert.equal(nodes.get("#predicted-value").hidden, false, label);
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
})().catch(error => { console.error(error); process.exitCode = 1; });
