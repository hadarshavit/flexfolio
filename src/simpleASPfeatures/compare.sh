#!/bin/bash

zcat $1 | python2.7 main_extract.py

zcat $1 | ~/Downloads/featurizer.o3.static -- -human
