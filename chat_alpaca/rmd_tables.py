"""Versioned IRS lifetime divisors used by the owner-IRA RMD approximation.

Source: IRS Publication 590-B (2025), Appendix B, Tables II and III,
https://www.irs.gov/publications/p590b (retrieved 2026-07-21).  The compressed
Table II payload contains the applicable owner-age/spouse-age cells for owners
age 70 through 120 and spouses more than 10 years younger; 120+ is capped at
120, matching the published table.
"""

from __future__ import annotations

import base64
import lzma

RMD_TABLE_VERSION = "IRS Publication 590-B (2025), Appendix B"

UNIFORM_LIFETIME_FACTORS = dict(
    zip(
        range(72, 121),
        (
            27.4,
            26.5,
            25.5,
            24.6,
            23.7,
            22.9,
            22.0,
            21.1,
            20.2,
            19.4,
            18.5,
            17.7,
            16.8,
            16.0,
            15.2,
            14.4,
            13.7,
            12.9,
            12.2,
            11.5,
            10.8,
            10.1,
            9.5,
            8.9,
            8.4,
            7.8,
            7.3,
            6.8,
            6.4,
            6.0,
            5.6,
            5.2,
            4.9,
            4.6,
            4.3,
            4.1,
            3.9,
            3.7,
            3.5,
            3.4,
            3.3,
            3.1,
            3.0,
            2.9,
            2.8,
            2.7,
            2.5,
            2.3,
            2.0,
        ),
        strict=True,
    )
)

_JOINT_TABLE_PAYLOAD = """{Wp48S^xk9=GL@E0stWa8~^|S5YJf5;FG%xCtUy=i~+i+V_kZ!S)0w3KKc2&LdJiwJ=Tv&Z$X*mM^8%F(-G<VQ1IrrA%N|dV?Lt{7kt*p)(!xcD>Z`uK8#K&&vPXj*L<PM3v%+N5^eYjQ4yE@d1bX^cqWm-Ss36S2)aXYXbx-4rrFV^jSu+_6~lUz1|O{pnbyY_=s?|J0V!iD!APYnxl>7Y32#)!PNi1}fRKU<ewo5$rL(>BU5gmz<kA_sAP1+<OYVn_s!<g!)-Wq4M-au{9833Tp)ATms50d8_L##77RZ7PufQ^sZN6s@uiNDklP6%Z51p&c)EcToWJk^FFh*nt0u>#_o#m3qWiqHYFT?kJ4;UEs&od}C(95kEILGf>_p0~>9%E8R9V%8c0%S63EP9>&cl?>DcdP+vrjEKf^7z>G4~?lJJyCW|IOi@8H(`Dz^bN1p0hKZ$deMW<0Q;QGjmm@M0BJbjOYFiev0GsC@49)hYDL(;l91$4k-&3!ToTp^AA6&w+|0kfUtEmtKrns@*CxS92Idp3@+OMws&WKjL>Fle5h>6PLJ@b6rg3NOjJC1)vD091o_M&y0#I%CCi4SW6X+9O`4m(LMEH&IIcuN1t<WZLyq&g*Ts`|%8MLs`{p#ltA|yb}*3!VwdvxGqIOja1nL(*rWp!4sRTsMVU5wf^O%9LTjEr{%7;hEdGCjOyzBq?Lx)KAYXdzbdNF^4O5E_Tsj$#YD&%B5TE^!Kd&1vh|LGamr4>F<7!7;SHs-gnFbGu)UNjlvqrUs<EdPc0b2j7x)smF<V_dXZj=DO#&6WDte_>?1Osl_$u1oNvIF$-;+ZGnWAEf&7!?{@HTMItzf85!!J@a2rKw}HL5<*uljsZ_{wlZ>hpxg7A;=#MF2oAhbL_7)PIz2=v@-Lujn9&t!#a<6P$41uMAN%W=2$+w}0Ia8mRv+B=&QA{H2X?%4n)?@a;iqV#GgmflEU6)_rNs*Q?>kJWTV#_~6Qy@nD-+@nUrc~{Wx5(j&tq<ZNB+Bd#=n_BtPv=-fFUY*?XZEah99H~PI@Ggw&YbqZ8#;2V{fOHK9{wU1qICA~&O&sgZktxs=5*HB58*3h)6>FGhTUgNOySCAA5pA}$jy_eVXICUFdzaAG?-ppUl%lFYqej&Bm_y5L$SLMpIn0snKw>KZ1?CGdNd+b!P6IFQ@P0D)9#GUZ0tS2_x*`}oIApz1!Bys9RTa+;${Vm85#pln<^S^*r&Vqq|mq6`ay2vEHye6lVEe{>GB-C!!Vp!e7XPxygE|t`|r@gLcGP!|Jk~$oy~1iWJlRJx@1tmRzpA9rL-NIRW?Iu2c?N9r4B&tAdeteuvD)7zfs(-1%;>Z;o5(1-=`*VSsa>Vm0`j)rS4A>g?|w};JR{pd_Jn0imCQD&v#O-Hx4V)MU%uiKb-+VsQ%_aB1&d7Y9`IB1z!i0x%SlicgO$l?OE|&Da3_9L7ay)=njhx+8kRNZr%iVDGjm$2Jj@0qE5Bd&Xo7%LFLI<v{|K#<MR3iL#^;dLNQ84WyAM<#cZ37A;n0Ls>8$u+5zZ)5wx=r#+TCVF*zmEQ=@QY>KqtP_<a4RbvOIZ@QOObsGvzU89XBL{D_Y1Z%$zkGGPl1IG~2UY7{^fQjF?HVq!ts7g<L>uM15Adb6vO%M0NFWG|LNB9>CuSIL<Tskex~x1|BIu#L+GmS>+;901S8%z*PDG0u1qzV4}a@GZdo?m8VM&~Timiqghq$jW5~Jd_t-QvK?A9I99auZ;Q2>K>QvG0Zm2l*s}q{a3}QZxdl?WucD_(!~yH@8FU>9VfD37;fb!xW|)myL0VWN%A>9irI5E`P};4S&&XqhqN-K7VM9b4nAn}f}uNdniXy10d2?y-mI0r=eIjiUXURrLG5>~&Kr3+TbD(XLiWJ`AUtYK5q-^TQ=my!4{BX2ljL)eZf)i?#HPSVx2{rdHnyBwfrcy4^SzVI%KhbO)6>Sa{U`QeOTfK(ggT<sZ&04Ml3?0yFboH?Y0f*!y>mi*Bb(e=>=|zF7abuV7P?Hd<xJPA5^LQMDuu>tt6MxKVuJv*rkF)GcWL-WzNi$JalJ=gPjQ3$9+gCRA&S+{VFxIS3YaK1-;eX6|CcHQDq-3+@vXIBiGJ_Yz;!P0kzmocl3Z!Qwq<a7T@RN?*ve_XT!7dW_e{j!@CJy96F>%##ft}X51TvCDOysVkfQ9gYP&a3%P<qJk>e>+D6p$F3O!3J>%)J?%U#(DQ<fWRsypD-jtzq#*#d|jWl2lp5YtQ8ag`)tdADA$eq!!-!3JuE>raYxbaZbrl3>nD$9IA~pg1Dn)=EoOjD6KBsG)C;y+YD{TS&{KKr+aZxQ7+mRGXd|jkm<<ZcU3NeRSS%^(~iLyw;224rcbKdvK77zekIXf{;Rh<(^v?QwdB?vNpBp$|hA_dD=wRdCfTm=C^)S^J!|+2IbV;rLKTZN{Oj8$+VHqt@&ie`pXG7VLt>M0u?wvpcZME*kcZU`UX3D4K_GNLGmiSYx<}TW<W+qeL6|cp-WZ}qQC6oy?^kOcMxVxDQiC9#ZQ`u-E3yhKvX;BDns)F1hV!|LnWv&xnVklvr6$F856sw{NWwNd}35^+}<q$lXeUot}<uTogKX>M-djVK|L&4!jsyqb)%|%1n74@sqFanlx3YzH);ce7ygxq!0@IjIsuSaAQ>-XA8cK?uAy-A7`k?3qI#U1*YWHYu&6{;Rm8&U(7$+{g5sd~2bd@+^-Z(Yn)$9`;G~tETKmZy1EukT+oFi(qMDWb44v;>v!?<?%ovWyU;Z3;r=4Gwk9lxGDaK26M*pS66rq&enmhI-94@wv%;PY&=n3E1KJ68io)fi&s%g+^BlKAo9<g|W37e=fGMMIo5%le5CPvt&0(2A9dlDS7R_v1(A=P^%^C4g{I#E(RN7lQsl8_@0>P0&C<4Klw|77b_4nOsUNgmnC{qtfw&km>lYM<37N{Rl9R*0Ire%57fcrAjK#m!kHG*g6yQx52cRRiP<6b?r0RBVicJ#bdlP>F{0loklNdLEg8rxObaR<u9w+?(dIg`Hc-TDEbzSd%h1?lzsxn)V;5847`^9sXOIjp4f&YP#((4=Ya(5a=v{-u4+dGY#&r#g?PtDcif)Bmt`)Af82`SknV!G%2vWPWz4x<ep4}Vm8c~AaeHI_mlL|G*Gr}30c)!xmUJr5C~&)HZ|{Xo1Kiq^e@#>-i|H+sUPb~PZOV8uooNw#Od<20!R015<hGO@EcK3*Bbddi!wP5H<=MycQ8m<E<Hu`jW#)AdZNzs?hlMQn1TbUkE05Y#?v(1%S};-|G>RD3=&vY7_w(HR}oiNS?j$L2+fs#nHEXj1pfD8^7I?EKdU{ht6+i9pVm|88v^3gU&90GdYYmts9_8Ccgp`*`8|_C3co1EcqZ4(vCR*uvU44eIu%;S%TE@JX*o2EJ7X34vl!FQG{e!k?vW6}+u^w7ZJ2E-kBaN+Pqt2r)3b{|r;`a1t@XkiYl<N};#wki;sPl4t)18NQN+|nOenaoK8^5!WHtepYt@LU5A&`%883~yz$56|qty|uvnVtDw}2<w_p9~Tfp9$#?t7oS&+C$`!d!OkR|&qz)`A4*S3o$C3(jC4ch|z|z3SK0^g=-i%vA<Gw_}NxL5*2kR{#J2v;YCEl=0yo00F}myr%*HpaLucvBYQl0ssI200dcD"""


def _joint_factors() -> dict[tuple[int, int], float]:
    rows = lzma.decompress(base64.b85decode(_JOINT_TABLE_PAYLOAD)).decode().splitlines()
    return {
        (int(owner), int(spouse)): float(value)
        for row in rows
        for owner, spouse, value in (row.split(","),)
    }


JOINT_LIFE_FACTORS = _joint_factors()


def lifetime_divisor(owner_age: int, spouse_age: int | None = None) -> tuple[float, str]:
    """Return the published divisor and table name for an owner RMD year."""
    owner = min(owner_age, 120)
    if spouse_age is not None and spouse_age < owner_age - 10:
        spouse = min(spouse_age, 120)
        try:
            return JOINT_LIFE_FACTORS[owner, spouse], "Table II"
        except KeyError as exc:
            raise ValueError("The spouse age is outside the versioned IRS Table II data.") from exc
    try:
        return UNIFORM_LIFETIME_FACTORS[owner], "Table III"
    except KeyError as exc:
        raise ValueError("The owner age is outside the versioned IRS Table III data.") from exc
