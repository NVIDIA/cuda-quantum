#! /usr/bin/perl

# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This script modifies the linkages on kernels so that we can replace
# them with our own kernel code.

my $modFile = $ARGV[0];
my $llFile = $ARGV[1];
my @functions = ();
my $cnt = 0;

# 1. Get a list of all the mangled kernel names from the .qke file.
open(F, $modFile);
MAP: while (<F>) {
    if ($_ =~ /quake.mangled_name_map/) {
	my $i = index($_, '"');	
	while ($i != -1) {
	    $_ = substr($_, $i + 1);
	    my $j = index($_, '"');
	    my $funcName = substr($_, 0, $j);
	    $functions[$cnt++] = $funcName;
	    $_ = substr($_, $j + 1);
	    $i = index($_, '"');
	}
	last MAP;
    }
}
close F;

die "No mangled name map in the quake file." if $#functions < 0;

# 2. Remove the "internal" linkage from any mangled kernel names in
# the .ll file. This will allow the linker to substitute the
# CUDA Quantum kernel launch code.
rename ($llFile, "$llFile.bak");
open (F, "$llFile.bak");
open (G, ">$llFile");
while (<F>) {
    if ($_ =~ /^define internal /) {
	LINE: for $name (@functions) {
	    if ($_ =~ /^define internal (.*)\Q$name/) {
		$_ = "define linkonce_odr dso_preemptable $1$name$'";
		last LINE;
	    }
	}
    }
    print G;
}
close G;
close F;
unlink ("$llFile.bak");
