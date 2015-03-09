#!/bin/bash

## Call: callerScript.sh [instance] [output]

if [[ -f $1 ]]; then
if [[ -f $2 ]]; then
	cp $1 $2
fi
fi
