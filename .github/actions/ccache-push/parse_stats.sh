#!/bin/bash
# Parses ccache stats from stdin and prints "<cacheable_calls> <cache_misses>".
# Prints nothing when the format is unrecognized. Handles all three formats:
# tab-separated --print-stats (>= 4.1), human 4.x, and human 3.x output.
awk '
  /^direct_cache_hit[ \t]/          { mdh = $2; machine = 1 }
  /^preprocessed_cache_hit[ \t]/    { mph = $2; machine = 1 }
  /^cache_miss[ \t]/                { mm = $2; machine = 1 }
  /^Cacheable calls:/               { if (h4c == "") h4c = $3 }
  /^  Misses:/                      { if (h4m == "") h4m = $2 }
  /^cache hit \(direct\)/           { h3dh = $4 }
  /^cache hit \(preprocessed\)/     { h3ph = $4 }
  /^cache miss/                     { h3m = $3 }
  END {
    if (machine)        print mdh + mph + mm, mm
    else if (h4c != "") print h4c, h4m
    else if (h3m != "") print h3dh + h3ph + h3m, h3m
  }'
