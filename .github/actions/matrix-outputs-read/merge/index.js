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

function deepMerge(target, source) {
  for (const [key, value] of Object.entries(source)) {
    const mergeable = (v) =>
        v !== null && typeof v === 'object' && !Array.isArray(v);
    if (mergeable(value) && mergeable(target[key]))
      deepMerge(target[key], value);
    else
      target[key] = value;
  }
  return target;
}

const stepName = getInput('matrix-step-name');
const downloadPath = getInput('download-path');

// Depending on how many artifacts matched the download pattern, the output
// files sit either directly in the download path or in one sub-directory per
// artifact, so search the whole tree for files named after the step.
const files = [];
function walk(dir) {
  for (const entry of fs.readdirSync(dir, {withFileTypes: true})) {
    const entryPath = path.join(dir, entry.name);
    if (entry.isDirectory())
      walk(entryPath);
    else if (entry.name === stepName)
      files.push(entryPath);
  }
}
if (fs.existsSync(downloadPath))
  walk(downloadPath);
files.sort();

const result = {};
for (const file of files)
  deepMerge(result, JSON.parse(fs.readFileSync(file, 'utf8')));

setOutput('result', JSON.stringify(result));
