FROM ghcr.io/nvidia/cuda-quantum-dev:amd64-gcc12-main

# Give the agent the warm CUDA-Q build and the LLVM tools used by focused pass
# tests without requiring a full CUDA-Q or LLVM rebuild.
ENV PATH="/workspaces/cuda-quantum/build/bin:/usr/local/llvm/bin:/usr/local/cudaq/bin:${PATH}"
ENV LD_LIBRARY_PATH="/workspaces/cuda-quantum/build/lib:/usr/local/cudaq/lib:${LD_LIBRARY_PATH}"
ENV CMAKE_PREFIX_PATH="/usr/local/cudaq:/usr/local/llvm:/workspaces/cuda-quantum/build"
ENV CUDAQ_SOURCE_DIR="/workspaces/cuda-quantum"
ENV CUDAQ_BUILD_DIR="/workspaces/cuda-quantum/build"

# Install the standard trajectory grader dependencies in the Ubuntu 24.04
# development image. Naming them here prevents SkillEvaluator from appending an
# install command that lacks the image's required externally-managed override.
RUN pip install --no-cache-dir --break-system-packages \
    "ragas~=0.4.0" \
    "openai>=1.0" \
    "anthropic>=0.40" \
    "boto3>=1.34"

# The moving development image may already contain a synchronized copy of this
# skill. SkillEvaluator must control whether the with-skill or baseline variant
# receives it. Remove Git metadata as well so deleted skill and evaluator files
# cannot be reconstructed from the image's object database.
RUN rm -rf \
    /workspaces/cuda-quantum/.git \
    /workspaces/cuda-quantum/skills/develop-cudaq-pass \
    /workspace/skills/develop-cudaq-pass \
    /root/.claude/skills/develop-cudaq-pass \
    /root/.agents/skills/develop-cudaq-pass \
    /root/.config/opencode/skills/develop-cudaq-pass && \
    mkdir -p /workspace/skills /logs/agent /logs/verifier

# SkillEvaluator supplies this build context separately for the with-skill and
# baseline variants. Copying the generic directory keeps the Dockerfile itself
# independent of the condition being evaluated.
COPY skills/ /workspace/skills/
COPY skills/ /root/.claude/skills/
COPY skills/ /root/.agents/skills/
COPY skills/ /root/.config/opencode/skills/

# Evaluation prompts, reference answers, and expected behaviors are verifier
# inputs. They must not be available to either agent variant.
RUN rm -rf \
    /workspace/skills/develop-cudaq-pass/evals \
    /root/.claude/skills/develop-cudaq-pass/evals \
    /root/.agents/skills/develop-cudaq-pass/evals \
    /root/.config/opencode/skills/develop-cudaq-pass/evals

WORKDIR /workspace
