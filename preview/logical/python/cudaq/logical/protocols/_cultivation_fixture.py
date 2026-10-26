# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Pinned Gidney--Shutty end-to-end cultivation reference fixture.

The compressed text is the flattened output of
``cultiv.make_end2end_cultivation_circuit(dcolor=3, dsurface=6,
basis="Y", r_growing=2, r_end=2, inject_style="unitary")`` from the
authors' Zenodo archive. The reference is a Cliffordized fault-analysis
envelope: the authors' vector sampler interprets every ``S``/``S_DAG`` in the
injection and double-cat check as a quarter-turn ``T``/``T_DAG``. QLX applies
that same interpretation in its P2 authoring fixture; executable TSim use is
conditional on a separate verified P3 projection.
"""

from __future__ import annotations

import base64
import hashlib
import zlib

UPSTREAM_REVISION = "871e68ff6df2f75190b1bfd6351459d1b5a037e3"
UPSTREAM_ARCHIVE_MD5 = "39a1b3208bc0b41144b7d748f8500433"
REFERENCE_SHA256 = "da0702afdbcb5ed5ad71dff9379205adc878a75b9faf275bea228035ada6e9d8"

# The final d=6 grafted-matchable patch uses these reference-circuit wires.
OUTPUT_CARRIERS = (
    1,
    3,
    5,
    9,
    12,
    14,
    15,
    17,
    18,
    20,
    22,
    25,
    27,
    29,
    31,
    33,
    35,
    37,
    39,
    41,
    43,
    45,
    46,
    48,
    50,
    52,
    54,
    57,
    59,
    61,
    63,
    64,
    66,
    68,
    71,
    73,
    74,
)
OUTPUT_HX = (
    (0, 1),
    (2, 4, 5, 9),
    (3, 6),
    (1, 3, 7),
    (4, 8, 9, 13, 14, 19),
    (10, 14, 15, 20),
    (11, 16),
    (6, 11, 12, 17),
    (7, 8, 12, 13, 18),
    (17, 22, 23, 27),
    (18, 23, 24, 28),
    (19, 24, 25, 29),
    (20, 25, 26, 30),
    (21, 26),
    (28, 31, 32, 34),
    (29, 32, 33, 35),
    (30, 33),
    (35, 36),
)
OUTPUT_HZ = (
    (2, 5),
    (0, 1, 2, 4, 7, 8),
    (4, 9),
    (3, 6, 7, 12),
    (8, 13),
    (5, 9, 10, 14),
    (10, 15),
    (11, 16, 17, 22),
    (12, 17, 18, 23),
    (13, 18, 19, 24),
    (14, 19, 20, 25),
    (15, 20, 21, 26),
    (22, 27),
    (23, 27, 28, 31),
    (24, 28, 29, 32),
    (25, 29, 30, 33),
    (31, 34),
    (32, 34, 35, 36),
)
OUTPUT_LX = ((16, 22, 27, 31, 34, 36),)
OUTPUT_LZ = ((21, 26, 30, 33, 35, 36),)
FRAME_SIZE = 76

_REFERENCE_B85 = (
    "c-rlpO>Z2@4TkUj6?0q<VxjB1`nZ;bz{o1zmA%2F1Hp$Ne<1(=ke96Kk1DdHTCWlKFb6|gHoNL2Sxr7Yn&h|N-"
    "+%n__5RbRrw^Zhn*MTnXWpKsH@`L3>Fvn~BRp+wH06FLthnC`BRqpJ+P9}DjPC6z)kce*V9nBs1!if*US<+z3d;y9`C4Jcy2c4BG"
    "?*qbSEQLB&9p|8X4)_bGl|Q>O1`tOVqrT8D^yq`tR7Y?tO3?0JP&DP_R>si1Zk!XqcoGcB+YbwvM?iTI}0nFs!3R}urtC670wDPB"
    "s&=(;&V-Wp@i*%G}9VUnrXu%&7>|%Glen>Gd&@bu)^0HVa3AU3M*82C#;a{Wq^p!H}S<1pO4Z^D<^3t8CjZ1^DNAWFHFKpEx-sX7"
    "7kWep&~e8h2-Fc74d~8zEt81QJQJxB+VovOEYPng&Fb1Nm$_<jj&RWv%(4$v57Ai712vGtsI1zR77DV-z2QGa28fl(L^^F6={-"
    "YtelK6Lq)Q}O1|x6D5E8LX(sRR=IMe~j5dhwb_yqdis%9-"
    "b4t`Ap3n%<lBt7#`FQ`gH}}PNCU;}#4)p8e6~|ZR%2xb#n<rBwUsu^-^<<5-0XorY53tzA;^|(y-"
    "y1RJ_Vus7fByR6=f7$fnj2`}gpqg<i0i5R_Y?p3jLT1?kR&^rM>3iOb0eKr!ez>d<1Eldls<bD#V#dzynvmKGTFW+y^fkPbI5v=B"
    "~E9zTH$)t@TRU?qeDQCdjis%$1N`(9=<%>fBE#p)1D98fB*CEf4xi3`xTll`<3`y-"
    "DCGF;c$hRWWVC;D~EmD;fkyFvGqNNea!P#Tt0)_*|W!!V<`ES^TSWaA@i2AD=Mc{$gU1`Z7khfm6}U+P4s0uAy1d%VLABdi}dyz("
    ")-xUhL(v>?>|30{p08NzdU^X`1ku?e*f_B)1mS&T#e6bt=B4THQ7(ao$)qDt|kL7{>|6>=c|8y?wvEMoHJKl3H1p)T=B}P<-"
    "A$typ_{ymGd@Ls~*mqwcsgW5}*St0Yt#e#wf;w#TdhM!f?UNKpRKPK26+@@$_<8ES_N$#RC9Ab>Ia+0A@QzHYPDf6($e{3-"
    "|PsJE4py8rnF{L;`Aop8!RG4-"
    "f@rF$OG#9!3kF2^<LS`wr^RTk$*A$L<zngL#f2jnRvViU~z?XFXO^X)@PVk!jC~_T<Ayda)iPV@XmSNHrutETSa|k@Qluh8GSrru"
    "MZR_s`669GfWhPg;;-nkbLhg;E(=X`q}^WD_s$cv)BX+vzmdk3w2ID62+|pp?5Q7t0dXRxO#tUHfvN)Rkw6>cT-"
    "%!Y5v!R6B4=6J?c3z@mK-"
    "?YWMTN=bDwd>kB~h(u|pGEICmESK1Gl_RIjYQ5f51c#1Kstlh2!^g((cF==;AqksOt!FOGMB1PM@*n^Tp#&mPE$GD7bPNJZimWQJ"
    "u*?U;ANA05nK?0s5mT~UDo80dlr0FcoEyvOv78>q>2Yg%0?Q06GqB9aG9$~3ER&v`Kv|*OSdw-qy5q7p!*VONYf4S8E6N~D*Fv`^"
    "JF@h|C7wB%hD+?Y#9=LQQO~*qE2d{)=9C&vrdvxKIkBY&*sV2xYcF{GmIwTW{_WDWMY^?DI=8m+wKv<fIos9Qu5I6Lec!B&f4$wT"
    "qd>O!x;Z;#WYGIHURAhN$5rZd?BaLE(o10Z;hU-"
    "Ax3yz+5BZwA*`*inskr4Xr~W=BHKV7qwrPgfJIg-!Ev+7*_LyE(Y3m!DNLQ85yAnl5-jT=nwOz<FjfV{2Se-KKYdzN24nnYXwSzb"
    "S_84-~WX8rps61vlYa*xGL5MW-!C}_6s5o^}_Aq(phi^xM+nm46`P-bo&H3X4-"
    "i*)CAD>g10JTTjW{Z4Z3iYP}>&#A89kl!+70L@=dy8vc*WNjy!|Y5_yGqHMkZH%4_A`vY{n@P}V2!dJ-gX^1e)ZhQL>b?qo-"
    "5mXZxFAkp?a;1xj(=|e}MC!7Fv%OcT}rg#bPn+Fju>pwo-8gw&-"
    "{6Yv0O?mavAu(Vwq9oyh*Qb7JKg+LZcv2T4IEjq93I4O1G|l|Cz835RpCdB%-"
    "jD@VCx*Z{Y3lrpuCBcfU@w}W==DxQF7cNA{{tnplnq>xzSxfV%5vf49j?+6N;HPNpzDUjB9+Gj~?Nh^_gmY*jktwdXQsPRswGvxh"
    "+rbWfZb9)(*%-DFYCrN|HTBzIdS$m4mT~^+juDIFkz8=r-!=33zCeK2h-"
    "+?zH#mT(+r%99lNbtkpCiBz}1DllQBXAbfyg}yw4KmZ8!pwq(v8I+JKE{cCthsV%?UB}w<+6avSVN`yInNa;n?-"
    "GpKjr@0obG{WiTa2&p%0a@hDyJ;<33%+nl4?RE@MrXvxdQiE}Jq~Ntd-p+&&tdBwfZq5H+G?>nQFLX}RS{mv+xo7c-"
    "lBm_BAUxt(6X%-"
    "A7P1v6uZNF`=Q9Y`f+Mx93`W=5Sy70hhPrUHJ{o{27zO3xmU%I$kVDvzQHq)9E3?_*}vfmC8<)Ol25X4H99!OS?o%~?rj7f3}qbs"
    "(M9(5EhDMx93$%#3}=y<z4JGjEuA!_4oAndvpmOs`^Q?U7<1GgIe-"
    "(I~#slxmuYhnMXZNR$52r3R#_!>*D;oBdzl(8klhfS+j?KO3P_A3xKp_*r}GACxNinRHUs@H6S3E$_%G@iTQk+m+PWc)mcLX`DLK"
    "ICZ9B>P(~5nMSEI4O3_8d=#xvXX*f|q|VggQ%Rkv!>5uuQ-@C#>P*AbnMSEIjZ$YCrq0yCXi1%^!>5uuQ-"
    "@C#>P(~5nTDw|>CAhh&Kq^!sPjgh-"
    "<vw~Yt)%vrOrG|oq3o#^Eh?paq7&&)R~8=Gmlee9;VJbOr7~v>dfQRna8O!|8eTf!_=9FsWXpLXC9}{JWQQ=lsfY$b>?B}%;VIVh"
    "p96UQ)eEg&OA<?d6+u$D0Svh>deE`nTM$}4^w9zr_MY|oq3o#^DET(`0KByiwoz)hx5h9^Ah~|64H5z?tF<JKlAyL%y~)f{7<05#"
    "hfqhbb+(6_y=c<<;#p;1Fj|fgulg43mc~E#lp?eO;`^T_Tuq-nD7sF%+Uo)JM4O1u-"
    "<|N77Y6Z>n?V}KiDwju{ahCWnm2N1&c2Z|FGDC3B6$P#XAlH0h|^N1b*0LiH3tgAdo$CP?s1fsM7^OT_`C0pb+Q|2%sqKHInHX2~"
    ">j;iXcQmVt|GV1T8LI#b5)7;K(S8-"
    "~)(*kohVio)*d|HeO&dp~MRo;n<N!#Ezdcmc=zH3M~!hLk>!;6eCy!iG`eeO(c>`#Y5akXo-!B2AQ~@iduXVu2WHwU-"
    "&H7qArQ@l__6Wss5F(-PfEj-Juc8oj-"
    "#{@pZgepQ_DQ<U@HJXMGk`4;lT1PLZI}LV=Z8M}X>q<o|mCWm%mp?yloy?hHcPI*Zu0KYv!=w6S|^Dqk2u4YNKrt3mHrt`ic}3GX"
    "_!5PuA<5$2A+v@fL_ZQ_!2nd|y%hIHM_1@~#+eml_16*umwXpbFfM9n%J%)(bxdjD=-Y@NlC-w4495prGa3wrD-"
    "<*K7UbV}`&SsC<7Z={e~U+KC%9j5h_5{w4P>?`XwDYOtfWz)f?Q#b8@^cnTCsb3V0df6ronUbF09UWZjEPb;ELu!7vNncTpo`#(t"
    "gtN%Q_Vv9AN_4ex$EW`Rye@-~")


def reference_stim() -> str:
    text = zlib.decompress(base64.b85decode("".join(_REFERENCE_B85))).decode()
    if hashlib.sha256(text.encode()).hexdigest() != REFERENCE_SHA256:
        raise RuntimeError(
            "cultivation reference fixture failed its SHA-256 check")
    return text


__all__ = [
    "FRAME_SIZE",
    "OUTPUT_CARRIERS",
    "OUTPUT_HX",
    "OUTPUT_HZ",
    "OUTPUT_LX",
    "OUTPUT_LZ",
    "REFERENCE_SHA256",
    "UPSTREAM_ARCHIVE_MD5",
    "UPSTREAM_REVISION",
    "reference_stim",
]
