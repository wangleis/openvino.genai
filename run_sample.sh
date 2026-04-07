#!/usr/bin/env bash
set -e

SCRIPT_DIR_RUN_MY_SAMPLE_GENAI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR_RUN_MY_SAMPLE_GENAI}"

source ../../../python-env/bin/activate
source ../../../source_ov.sh

echo "${SCRIPT_DIR_RUN_MY_SAMPLE_GENAI}"

SAMPLE_BIN="${SCRIPT_DIR_RUN_MY_SAMPLE_GENAI}/build/samples/cpp/custom_vit_cb/custom_vit_cb"
if [[ ! -x "${SAMPLE_BIN}" ]]; then
	echo "Missing sample binary: ${SAMPLE_BIN}"
	exit 1
fi

"${SAMPLE_BIN}"
