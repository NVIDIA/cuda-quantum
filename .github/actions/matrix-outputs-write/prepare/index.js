/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 ******************************************************************************/

'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

function getInput(name) {
  return (process.env[`INPUT_${name.toUpperCase()}`] ?? '').trim();
}

function setOutput(name, value) {
  const delimiter = `gh-output-${crypto.randomUUID()}`;
  fs.appendFileSync(process.env.GITHUB_OUTPUT,
                    `${name}<<${delimiter}\n${value}\n${delimiter}\n`);
}

function fail(message) {
  process.stdout.write(`::error::${message}\n`);
  process.exit(1);
}

// Parses the flat `key: value` maps that our workflows pass as the `outputs`
// input. Values that parse as JSON (objects, numbers, booleans) are kept as
// such, matching how the YAML parser of the original cloudposse action
// treated them. Nested or multi-line YAML is not supported.
function parseOutputs(text) {
  const result = {};
  for (const rawLine of text.split('\n')) {
    const line = rawLine.trim();
    if (line === '' || line.startsWith('#'))
      continue;
    const sep = line.indexOf(':');
    if (sep < 0)
      fail(`Expected "key: value" in outputs, got: ${line}`);
    const key = line.slice(0, sep).trim();
    let value = line.slice(sep + 1).trim();
    if (value.length > 1 && /^(['"]).*\1$/.test(value)) {
      value = value.slice(1, -1);
    } else if (value !== '') {
      try {
        value = JSON.parse(value);
      } catch {
        // keep as plain string
      }
    }
    result[key] = value;
  }
  return result;
}

const stepName = getInput('matrix-step-name');
const matrixKey = getInput('matrix-key');
const rawOutputs = getInput('outputs');

if (stepName === '' && matrixKey !== '')
  fail('`matrix-step-name` can not be empty when `matrix-key` is specified');
if (stepName !== '' && matrixKey === '')
  fail('`matrix-key` can not be empty when `matrix-step-name` is specified');

const outputs = rawOutputs === '' ? {} : parseOutputs(rawOutputs);
setOutput('result', JSON.stringify(outputs));

if (stepName !== '' && rawOutputs !== '') {
  const dir = fs.mkdtempSync(path.join(process.env.RUNNER_TEMP, 'matrix-outputs-'));
  const file = path.join(dir, stepName);
  fs.writeFileSync(file, JSON.stringify({[matrixKey]: outputs}));
  // Artifact names may not contain ", :, <, >, |, *, ?, path separators, or
  // line breaks; the name must stay deterministic so that re-runs overwrite
  // the previous upload instead of leaving a stale duplicate behind.
  const artifactName = `matrix-outputs-${stepName}-${matrixKey}`.replace(
      /[":<>|*?\\/\r\n]/g, '-');
  setOutput('artifact-name', artifactName);
  setOutput('artifact-path', file);
  setOutput('upload', 'true');
} else {
  setOutput('upload', 'false');
}
