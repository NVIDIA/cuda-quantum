#!/bin/bash

ctest --output-on-failure --test-dir . -R GHZSampleTester.checkSimple # GroverTester.checkNISQ 
RESULT_A=$?
/opt/llvm/bin/llvm-lit -v --param nvqpp_site_config=build/test/lit.site.cfg.py build/test
RESULT_B=$?

echo $RESULT_A
echo $RESULT_B

if [ $RESULT_A -eq 0 ] && [ $RESULT_B -eq 0 ]
then
  exit 0
else 
  echo "ctest failure status = " $RESULT_A
  echo "llvm-lit failure status = " $RESULT_B
  exit 1
fi 