#!/bin/bash

## Call: executionScript.sh [instance]

if [[ -f $1 ]]; then
	binaries/gringo $1 | binaries/clasp
fi
