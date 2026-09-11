## Description: <br>
Use when porting circuits from another framework (e.g. Qiskit) into CUDA-Q kernels while preserving the source algorithm and validation fidelity. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers porting quantum circuits from other frameworks (e.g. Qiskit) into CUDA-Q Python kernels while preserving algorithm correctness and validation fidelity. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [No] <br>
**Credential Type(s):** [None] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Qiskit to CUDA-Q Porting Reference](references/porting-reference.md) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Analysis, Configuration instructions] <br>
**Output Format:** [Markdown with inline Python code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (3 positive, 1 negative), each run with 3 attempts in isolated sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the right skill was loaded and activated when needed. <br>
- Effectiveness: Whether the skill helped complete the user's goal and expected workflow (goal_accuracy 50% + behavior_check 50%). <br>
- Efficiency: Whether the skill avoided wasted tool calls and token usage (skill_efficiency 50% + token_efficiency 50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill) | Codex (Baseline → Skill) |
|---|---:|---:|
| Overall | 81.9% | 95.5% |
| Security | 75.0% → 50.0% (-25.0 pts) | 100.0% → 100.0% (±0.0 pts) |
| Correctness | 90.0% → 100.0% (+10.0 pts) | 90.0% → 100.0% (+10.0 pts) |
| Discoverability | 93.3% | 95.0% |
| Effectiveness | 79.8% → 90.4% (+10.6 pts) | 77.9% → 95.0% (+17.1 pts) |
| Efficiency | 75.6% | 87.5% |

## Skill Version(s): <br>
1.0.1 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
