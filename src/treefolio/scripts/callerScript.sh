#!/bin/bash

## Call: callerScript.sh [instance] [features] [output]

if [[ -f $1 ]]; then
if [[ -f $2 ]]; then
if [[ -f $3 ]]; then
	cp $1 $3; 
	binaries/gringo $1 | binaries/claspre > $2
fi
fi
fi
