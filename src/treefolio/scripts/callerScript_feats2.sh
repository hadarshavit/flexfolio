#!/bin/bash

## Call: callerScript.sh [instance] [features]

if [[ -f $1 ]]; then
if [[ -f $2 ]]; then
	binaries/gringo $1 | binaries/claspre > $2
fi
fi
