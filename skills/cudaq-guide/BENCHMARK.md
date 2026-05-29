# Evaluation Report

Evaluation of the `cudaq-guide` skill before publication through NVSkills-Eval.

This benchmark summarizes 3-Tier Evaluation from NVSkills-Eval results for the skill. The goal is to document whether the skill is safe, discoverable, effective, and useful for agents before it is published for broader workflow use.

## Evaluation Summary

- Skill: `cudaq-guide`
- Evaluation date: 2026-05-29
- NVSkills-Eval profile: `external`
- Overall verdict: FAIL
- Tier 3 live agent evaluation: not available in this report

## Agents Used

- Tier 3 agent details were not available in this report.

## Metrics Used

Reported benchmark dimensions:

- Security: checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access.
- Correctness: checks whether the agent follows the expected workflow and produces the correct final output.
- Discoverability: checks whether the agent loads the skill when relevant and avoids using it when irrelevant.
- Effectiveness: checks whether the agent performs measurably better with the skill than without it.
- Efficiency: checks whether the agent uses fewer tokens and avoids redundant work.

Underlying evaluation signals used in this run:

- No Tier 3 evaluation signal details were available in this report.

## Test Tasks

Tier 3 evaluation task details were not available in this report.

## Results

Tier 3 dimension rollup was not available in this report.

## Tier 1: Static Validation Summary

Tier 1 validation passed with observations. NVSkills-Eval ran 9 checks and found 4 total findings.

Top findings:

- MEDIUM SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/cudaq-guide/SKILL.md`)
- MEDIUM SECURITY/Unknown (SDI-2): The Bash tool is declared in an onboarding/documentation guide skill, which primarily serves to present installation ins (`SKILL.md:8`)
- MEDIUM SECURITY/Unknown (SQP-2): The QPU section instructs users to set provider credentials as environment variables but does not warn them about the se (`SKILL.md:177`)
- LOW SCHEMA/author_format: Author must be of the form 'Name <email@host>' (`skills/cudaq-guide/SKILL.md`)

## Tier 2: Deduplication Summary

Tier 2 validation reported findings. NVSkills-Eval ran 2 checks and found 1 total findings.

Top findings:

- HIGH DUPLICATE/duplicate: Duplicate content found within SKILL.md:
  "## CUDA-Q Getting Started Guide" in SKILL.md (lines 1-6)
  vs "## Instructions" in SKILL.md (lines 21-28) (`SKILL.md:1`)

## Publication Recommendation

The skill should be reviewed before NVSkills-Eval publication. Skill owners should address the findings above and rerun NVSkills-Eval to refresh this benchmark.
