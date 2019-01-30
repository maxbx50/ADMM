set term cairolatex standalone pdf color solid dashed size 15cm,10cm
set output 'error_curl.tex'

set logscale y
set grid
set format y "$10^{%L}$"

set style line 1 lt 1 lc 1 lw 2 pt 6 pi 10 ps 0.5
set style line 2 lt 1 lc 2 lw 2 pt 6 pi 10 ps 0.5
set style line 3 lt 2 lc 3 lw 1
set style line 4 lt 1 lc 1 lw 2 pt 4 pi 10 ps 0.5
set style line 5 lt 1 lc 2 lw 1 pt 4 pi 10 ps 0.5

plot 'err_curl.csv' every ::1::80 w lp t '$\|y_{k,h}-\bar y_h\|_{L^2(\Omega)}$ for $Y=L^2(\Omega)$' ls 1, \
     'dist_curl.csv' every ::1::80 w lp t '$\|y_{k,h}-y_{k+1,h}\|_{L^2(\Omega)}$ for $Y=L^2(\Omega)$' ls 2, \
     'eps_curl.csv' every ::1::80 w l t '$\varepsilon_k$' ls 3, \
     'err_l2.csv' every ::1::80 w lp t '$\|y_{k,h}-\bar y_h\|_{L^2(\Omega)}$ for $Y=H(\mbox{curl};\Omega)$' ls 4, \
     'dist_l2.csv' every ::1::80 w lp t '$\|y_{k,h}-y_{k+1,h}\|_{L^2(\Omega)}$ for $Y=H(\mbox{curl};\Omega)$' ls 5